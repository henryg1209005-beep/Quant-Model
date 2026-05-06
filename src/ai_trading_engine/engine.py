from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
import time
from zoneinfo import ZoneInfo
from typing import Any

from ai_trading_engine.adapters.alpaca_adapter import AlpacaMarketDataAdapter
from ai_trading_engine.adaptive import AdaptiveLearner
from ai_trading_engine.anti_decay import EdgeMonitor
from ai_trading_engine.adapters.mock_adapter import MockMarketDataAdapter
from ai_trading_engine.bar_builder import BarBuilder
from ai_trading_engine.audit_log import append_audit_event
from ai_trading_engine.config import Settings
from ai_trading_engine.context_builder import SYSTEM_PROMPT, build_llm_context
from ai_trading_engine.economic_calendar import load_economic_calendar, render_economic_calendar
from ai_trading_engine.execution.alpaca_execution import AlpacaPaperExecutionEngine
from ai_trading_engine.finnhub_context import load_finnhub_context, render_finnhub_context
from ai_trading_engine.execution.mock_execution import MockExecutionEngine
from ai_trading_engine.indicators import compute_indicator_set
from ai_trading_engine.llm.base import LlmClient
from ai_trading_engine.llm.gemini_client import GeminiLlmClient
from ai_trading_engine.llm.mock_client import MockLlmClient
from ai_trading_engine.llm.openai_client import OpenAiLlmClient
from ai_trading_engine.llm.parser import parse_decision
from ai_trading_engine.macro import load_macro_snapshot, render_macro_context
from ai_trading_engine.models import AccountState, AiDecision
from ai_trading_engine.pre_trade_risk import evaluate_pre_trade_risk
from ai_trading_engine.risk import apply_risk_rules
from ai_trading_engine.scorer import build_market_context, render_context_dashboard


@dataclass
class CycleResult:
    timestamp: datetime
    dashboard: str
    llm_raw: str
    decision: AiDecision
    note: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _hold_decision(reason: str) -> AiDecision:
    return AiDecision(
        action="hold",
        direction=None,
        confidence=0.0,
        size=1,
        sl_ticks=0,
        tp_ticks=0,
        reasoning=reason,
    )


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class TradingEngine:
    def __init__(self, settings: Settings, initial_balance: float = 50000.0) -> None:
        self.settings = settings
        self.account = AccountState(balance=initial_balance, starting_balance=initial_balance)
        self.data_adapter = self._build_data_adapter()
        self.bar_builder = BarBuilder(self.data_adapter)
        self.execution = self._build_execution_engine()
        self._startup_warnings: list[str] = []
        self.llm = self._build_llm_client()
        self.llm_secondary = self._build_secondary_llm_client()
        self._last_llm_signature: dict[str, Any] | None = None
        self._last_llm_decision: AiDecision | None = None
        self._last_llm_raw: str = ""
        self._last_llm_call_at: datetime | None = None
        self._acceleration_mode_override: bool | None = None
        self._gate_profile = self._build_gate_profile()
        self.learner = AdaptiveLearner(
            state_path=self.settings.adaptive_state_path,
            enabled=self.settings.adaptive_enabled,
        )
        self.edge_monitor = EdgeMonitor(
            state_path=self.settings.edge_monitor_state_path,
            window_trades=self.settings.edge_window_trades,
            min_win_rate=float(self._gate_profile["edge_min_win_rate"]),
            min_expectancy=float(self._gate_profile["edge_min_expectancy"]),
            throttle_size_mult=self.settings.edge_throttle_size_mult,
            shadow_enabled=self.settings.shadow_challenger_enabled,
        )
        self._last_order_key = ""
        self._last_order_submitted_at: datetime | None = None
        self._confidence_controls: dict[str, Any] = {
            "enabled": False,
            "default_min_confidence": 0.55,
            "global_min_confidence": 0.55,
            "symbol_min_confidence": {},
            "symbol_calibration": {},
        }
        self._regime_controls: dict[str, Any] = {
            "enabled": False,
            "allowed_pairs": {},
        }
        self._horizon_controls: dict[str, Any] = {
            "enabled": False,
            "symbol_best_horizon": {},
        }
        self._llm_provider_health: dict[str, dict[str, Any]] = {}
        self._forced_primary_provider: str | None = None
        self._forced_primary_until_ts: float | None = None

    @staticmethod
    def _extract_regime_and_confidence(thesis: str) -> tuple[str, float]:
        regime_match = re.search(r"\[REGIME:([^\]]+)\]", thesis or "")
        conf_match = re.search(r"\[CONF:([0-9]*\.?[0-9]+)\]", thesis or "")
        regime = regime_match.group(1) if regime_match else ""
        confidence = float(conf_match.group(1)) if conf_match else 0.0
        return regime, confidence

    @staticmethod
    def _exc_brief(exc: Exception, max_len: int = 180) -> str:
        msg = str(exc).replace("\n", " ").replace("\r", " ").strip()
        if not msg:
            return type(exc).__name__
        if len(msg) > max_len:
            msg = msg[:max_len].rstrip() + "..."
        return f"{type(exc).__name__}:{msg}"

    def _build_data_adapter(self):
        if self.settings.data_provider == "alpaca":
            return AlpacaMarketDataAdapter(
                key_id=self.settings.alpaca_key_id,
                secret_key=self.settings.alpaca_secret_key,
                data_url=self.settings.alpaca_data_url,
            )
        return MockMarketDataAdapter()

    def _build_execution_engine(self):
        if self.settings.execution_provider == "alpaca":
            return AlpacaPaperExecutionEngine(
                key_id=self.settings.alpaca_key_id,
                secret_key=self.settings.alpaca_secret_key,
                trading_url=self.settings.alpaca_trading_url,
            )
        return MockExecutionEngine(
            cost_model_enabled=bool(self.settings.cost_model_enabled and self.settings.cost_apply_in_mock),
            cost_slippage_bps_per_side=self.settings.cost_slippage_bps_per_side,
            cost_fee_per_share=self.settings.cost_fee_per_share,
            cost_min_fee_per_order=self.settings.cost_min_fee_per_order,
        )

    def _build_llm_client(self) -> LlmClient:
        if self.settings.llm_provider == "gemini":
            return GeminiLlmClient(
                self.settings.gemini_model,
                temperature=self.settings.gemini_temperature,
                timeout_seconds=self.settings.gemini_timeout_seconds,
                max_output_tokens=self.settings.llm_max_output_tokens,
            )
        if self.settings.llm_provider == "openai":
            return OpenAiLlmClient(
                self.settings.openai_model,
                reasoning_effort=self.settings.openai_reasoning_effort,
                temperature=self.settings.openai_temperature,
                max_output_tokens=self.settings.llm_max_output_tokens,
            )
        return MockLlmClient()

    def _build_secondary_llm_client(self) -> LlmClient | None:
        if not bool(self.settings.llm_two_tier_enabled):
            return None
        provider = str(self.settings.llm_two_tier_secondary_provider or "").strip().lower()
        try:
            if provider == "gemini":
                model = str(self.settings.gemini_secondary_model or "").strip() or self.settings.gemini_model
                return GeminiLlmClient(
                    model,
                    temperature=self.settings.gemini_temperature,
                    timeout_seconds=self.settings.gemini_timeout_seconds,
                    max_output_tokens=self.settings.llm_max_output_tokens,
                )
            if provider == "openai":
                return OpenAiLlmClient(
                    self.settings.openai_model,
                    reasoning_effort=self.settings.openai_reasoning_effort,
                    temperature=self.settings.openai_temperature,
                    max_output_tokens=self.settings.llm_max_output_tokens,
                )
        except RuntimeError as exc:
            self._startup_warnings.append(f"secondary_llm_disabled:{provider}:{self._exc_brief(exc)}")
            return None
        return None

    def _should_escalate_to_secondary(self, decision: AiDecision) -> tuple[bool, str]:
        if self.llm_secondary is None:
            return False, "secondary_disabled"
        conf = max(0.0, min(1.0, float(decision.confidence)))
        lo = float(self.settings.llm_two_tier_min_confidence)
        hi = float(self.settings.llm_two_tier_max_confidence)
        in_band = lo <= conf <= hi
        if bool(self.settings.llm_two_tier_escalate_on_trade):
            if decision.action != "trade":
                return False, "not_trade"
            if bool(self.settings.llm_two_tier_require_confidence_band):
                return (in_band, f"trade_confidence_band={lo:.2f}-{hi:.2f}" if in_band else "trade_outside_band")
            return True, "trade_action"
        if in_band:
            return True, f"confidence_band={lo:.2f}-{hi:.2f}"
        return False, "not_triggered"

    def _decide_with_parse_repair(
        self,
        client: LlmClient,
        llm_context: str,
        *,
        max_attempts: int = 2,
    ) -> tuple[str, AiDecision]:
        raw_last = ""
        for attempt in range(max(1, int(max_attempts))):
            if attempt == 0:
                prompt = SYSTEM_PROMPT
            else:
                prompt = (
                    SYSTEM_PROMPT
                    + "\n\nReturn ONLY a valid JSON object that matches the required schema."
                )
            raw = client.decide(prompt, llm_context)
            raw_last = raw
            try:
                return raw, parse_decision(raw)
            except Exception:
                continue
        # Final parse attempt raises the original parser error semantics.
        return raw_last, parse_decision(raw_last)

    def _provider_name_for_client(self, client: LlmClient) -> str:
        if client is self.llm_secondary:
            return str(self.settings.llm_two_tier_secondary_provider or "secondary")
        return str(self.settings.llm_provider or "primary")

    def _record_llm_provider_event(
        self,
        *,
        provider: str,
        success: bool,
        latency_ms: float,
        error: str = "",
    ) -> None:
        key = str(provider or "unknown").strip().lower() or "unknown"
        bucket = self._llm_provider_health.setdefault(key, {"events": [], "cooldown_until": None})
        ev = {
            "ts": datetime.now(tz=timezone.utc),
            "success": bool(success),
            "latency_ms": float(max(0.0, latency_ms)),
            "error": str(error or ""),
        }
        bucket["events"].append(ev)
        window = max(10, int(self.settings.llm_provider_health_window))
        if len(bucket["events"]) > window:
            bucket["events"] = bucket["events"][-window:]
        events = bucket["events"]
        fails = sum(1 for e in events if not bool(e.get("success", False)))
        fail_rate = fails / float(max(1, len(events)))
        if len(events) >= 8 and fail_rate >= float(self.settings.llm_provider_unhealthy_fail_rate):
            cooldown = datetime.now(tz=timezone.utc).timestamp() + float(max(30, int(self.settings.llm_provider_cooldown_seconds)))
            bucket["cooldown_until"] = cooldown

    def _provider_is_unhealthy(self, provider: str) -> bool:
        key = str(provider or "unknown").strip().lower() or "unknown"
        bucket = self._llm_provider_health.get(key) or {}
        cooldown_until = bucket.get("cooldown_until")
        if cooldown_until is None:
            return False
        try:
            return datetime.now(tz=timezone.utc).timestamp() < float(cooldown_until)
        except Exception:
            return False

    def llm_provider_health_snapshot(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        now_ts = datetime.now(tz=timezone.utc).timestamp()
        for provider, bucket in dict(self._llm_provider_health).items():
            events = list(bucket.get("events") or [])
            total = len(events)
            fails = sum(1 for e in events if not bool(e.get("success", False)))
            avg_lat = (sum(float(e.get("latency_ms", 0.0)) for e in events) / float(max(1, total))) if events else 0.0
            cooldown_until = bucket.get("cooldown_until")
            out[str(provider)] = {
                "window_events": int(total),
                "fail_count": int(fails),
                "fail_rate": float(fails / float(max(1, total))) if events else 0.0,
                "avg_latency_ms": float(avg_lat),
                "unhealthy": bool(cooldown_until is not None and now_ts < float(cooldown_until)),
                "cooldown_until": (
                    datetime.fromtimestamp(float(cooldown_until), tz=timezone.utc).isoformat()
                    if cooldown_until is not None else None
                ),
            }
        return out

    def startup_warnings(self) -> list[str]:
        return list(self._startup_warnings)

    def force_primary_provider(self, provider: str | None, *, cooldown_seconds: int = 0) -> None:
        p = (provider or "").strip().lower()
        if p not in {"", "gemini", "openai"}:
            return
        if not p:
            self._forced_primary_provider = None
            self._forced_primary_until_ts = None
            return
        self._forced_primary_provider = p
        self._forced_primary_until_ts = (
            datetime.now(tz=timezone.utc).timestamp() + float(max(0, int(cooldown_seconds)))
            if cooldown_seconds > 0 else None
        )

    def _resolve_forced_primary_provider(self) -> str | None:
        if not self._forced_primary_provider:
            return None
        if self._forced_primary_until_ts is None:
            return self._forced_primary_provider
        if datetime.now(tz=timezone.utc).timestamp() <= float(self._forced_primary_until_ts):
            return self._forced_primary_provider
        self._forced_primary_provider = None
        self._forced_primary_until_ts = None
        return None

    def _decide_with_retries(
        self,
        client: LlmClient,
        llm_context: str,
        *,
        provider: str,
    ) -> tuple[str, AiDecision]:
        retries = max(1, int(self.settings.llm_request_retries))
        backoff_ms = max(0, int(self.settings.llm_retry_backoff_ms))
        last_exc: Exception | None = None
        for attempt in range(retries):
            t0 = time.perf_counter()
            try:
                raw, decision = self._decide_with_parse_repair(client, llm_context, max_attempts=2)
                lat_ms = (time.perf_counter() - t0) * 1000.0
                self._record_llm_provider_event(provider=provider, success=True, latency_ms=lat_ms)
                return raw, decision
            except Exception as exc:
                lat_ms = (time.perf_counter() - t0) * 1000.0
                self._record_llm_provider_event(provider=provider, success=False, latency_ms=lat_ms, error=str(exc))
                last_exc = exc
                if attempt < retries - 1 and backoff_ms > 0:
                    time.sleep((backoff_ms * (attempt + 1)) / 1000.0)
        if last_exc is None:
            raise RuntimeError("LLM request failed with unknown error")
        raise last_exc

    @staticmethod
    def _clone_decision(decision: AiDecision) -> AiDecision:
        return AiDecision(
            action=decision.action,
            direction=decision.direction,
            confidence=float(decision.confidence),
            size=int(decision.size),
            sl_ticks=int(decision.sl_ticks),
            tp_ticks=int(decision.tp_ticks),
            reasoning=str(decision.reasoning),
            forecast_direction=decision.forecast_direction,
            forecast_confidence=float(decision.forecast_confidence),
            forecast_horizon_minutes=int(decision.forecast_horizon_minutes),
        )

    def _build_state_signature(self, quote, context, event_risk: str, news_risk: str) -> dict[str, Any]:
        return {
            "last": float(quote.last),
            "trend": float(self._get_dim_value(context, "Trend", 0.0)),
            "momentum": float(self._get_dim_value(context, "Momentum", 0.0)),
            "event_risk": str(event_risk or ""),
            "news_risk": str(news_risk or ""),
        }

    def _should_skip_llm_due_to_state(self, signature: dict[str, Any], now_utc: datetime) -> tuple[bool, str]:
        if not bool(self.settings.llm_state_change_gate_enabled):
            return False, "state_gate_disabled"
        if self._last_llm_decision is None or self._last_llm_signature is None or self._last_llm_call_at is None:
            return False, "state_gate_cold_start"
        elapsed = (now_utc - self._last_llm_call_at).total_seconds()
        if elapsed >= float(max(1, int(self.settings.llm_state_change_min_seconds))):
            return False, "state_gate_refresh_due"
        prev = self._last_llm_signature
        prev_last = max(1e-9, float(prev.get("last", 0.0)))
        price_bps = abs((float(signature["last"]) - prev_last) / prev_last) * 10000.0
        trend_delta = abs(float(signature["trend"]) - float(prev.get("trend", 0.0)))
        momentum_delta = abs(float(signature["momentum"]) - float(prev.get("momentum", 0.0)))
        unchanged = (
            price_bps <= float(self.settings.llm_state_change_price_bps)
            and trend_delta <= float(self.settings.llm_state_change_trend_delta)
            and momentum_delta <= float(self.settings.llm_state_change_momentum_delta)
            and str(signature["event_risk"]) == str(prev.get("event_risk", ""))
            and str(signature["news_risk"]) == str(prev.get("news_risk", ""))
        )
        if unchanged:
            return True, (
                f"state_unchanged price_bps={price_bps:.2f} trend_d={trend_delta:.2f} "
                f"mom_d={momentum_delta:.2f} elapsed={elapsed:.0f}s"
            )
        return False, "state_changed"

    @staticmethod
    def _parse_hhmm(value: str) -> tuple[int, int]:
        parts = (value or "").split(":")
        if len(parts) != 2:
            return 0, 0
        try:
            hh = max(0, min(23, int(parts[0])))
            mm = max(0, min(59, int(parts[1])))
            return hh, mm
        except ValueError:
            return 0, 0

    def _is_live_execution_mode(self) -> bool:
        if self.settings.execution_provider != "alpaca":
            return False
        endpoint = (self.settings.alpaca_trading_url or "").strip().lower()
        return "paper-api.alpaca.markets" not in endpoint

    def set_confidence_controls(self, controls: dict[str, Any]) -> None:
        payload = dict(controls or {})
        self._confidence_controls = {
            "enabled": bool(payload.get("enabled", False)),
            "default_min_confidence": float(payload.get("default_min_confidence", 0.55)),
            "global_min_confidence": float(payload.get("global_min_confidence", 0.55)),
            "symbol_min_confidence": dict(payload.get("symbol_min_confidence") or {}),
            "symbol_calibration": dict(payload.get("symbol_calibration") or {}),
            "calibration_gate_enabled": bool(payload.get("calibration_gate_enabled", False)),
            "calibration_gate_stress_bps": float(payload.get("calibration_gate_stress_bps", 0.0)),
            "symbol_allowed_confidence_bins": dict(payload.get("symbol_allowed_confidence_bins") or {}),
            "symbol_bin_stressed_bps": dict(payload.get("symbol_bin_stressed_bps") or {}),
            "symbol_direction_policy": dict(payload.get("symbol_direction_policy") or {}),
            "symbol_blocked_directions": dict(payload.get("symbol_blocked_directions") or {}),
        }

    def set_regime_controls(self, controls: dict[str, Any]) -> None:
        payload = dict(controls or {})
        allowed_pairs = payload.get("allowed_pairs") or {}
        self._regime_controls = {
            "enabled": bool(payload.get("enabled", False)),
            "allowed_pairs": {
                str(k).upper(): {str(r).strip().lower() for r in (v or [])}
                for k, v in dict(allowed_pairs).items()
            },
        }

    def set_horizon_controls(self, controls: dict[str, Any]) -> None:
        payload = dict(controls or {})
        self._horizon_controls = {
            "enabled": bool(payload.get("enabled", False)),
            "symbol_best_horizon": {
                str(k).upper(): int(v)
                for k, v in dict(payload.get("symbol_best_horizon") or {}).items()
                if str(k).strip()
            },
            "blocked_symbols": {
                str(s).upper()
                for s in list(payload.get("blocked_symbols") or [])
                if str(s).strip()
            },
        }

    def _passes_regime_gate(self, symbol: str, regime: str, now_utc: datetime) -> tuple[bool, str]:
        controls = self._regime_controls or {}
        if not bool(controls.get("enabled", False)):
            return True, "regime_gate_disabled"
        sym = str(symbol or "").strip().upper()
        reg = str(regime or "unknown").strip().lower()
        bucket = self._session_bucket(now_utc)
        key = f"{reg}|{bucket}"
        allowed = dict(controls.get("allowed_pairs") or {}).get(sym, set())
        if key in allowed:
            return True, f"{sym}/{key} allowed"
        return False, f"{sym}/{key} not in allowed regime set"

    def _passes_horizon_gate(self, symbol: str, horizon_minutes: int) -> tuple[bool, str]:
        controls = self._horizon_controls or {}
        if not bool(controls.get("enabled", False)):
            return True, "horizon_gate_disabled"
        sym = str(symbol or "").strip().upper()
        if sym in set(controls.get("blocked_symbols") or set()):
            return False, f"{sym} blocked_by_auto_prune"
        best = dict(controls.get("symbol_best_horizon") or {}).get(sym)
        if best is None:
            return True, f"{sym} no_best_horizon"
        if int(horizon_minutes) == int(best):
            return True, f"{sym} horizon={horizon_minutes} best={best}"
        return False, f"{sym} horizon={horizon_minutes} best={best}"

    def _apply_confidence_controls(self, decision: AiDecision) -> AiDecision:
        if decision.action != "trade" or decision.direction is None:
            return decision
        controls = self._confidence_controls or {}
        if not bool(controls.get("enabled", False)):
            return decision
        symbol = str(self.settings.symbol).strip().upper()
        direction = str(decision.direction or "").strip().upper()
        blocked_directions = {
            str(d).strip().upper()
            for d in (controls.get("symbol_blocked_directions") or {}).get(symbol, [])
            if str(d).strip()
        }
        if direction in blocked_directions:
            policy = dict((controls.get("symbol_direction_policy") or {}).get(symbol) or {})
            metrics = dict(policy.get(direction) or {})
            signed = float(metrics.get("avg_signed_return_bps", 0.0) or 0.0)
            stressed = float(metrics.get("stressed_signed_bps", signed) or signed)
            count = int(metrics.get("count", 0) or 0)
            return _hold_decision(
                f"Direction policy ({symbol}): {direction} blocked "
                f"(n={count}, signed_bps={signed:.2f}, stressed_signed_bps={stressed:.2f})."
            )
        c_raw = _clip(float(decision.confidence), 0.0, 1.0)
        low = int(c_raw * 10) / 10.0
        bin_key = f"{low:.1f}-{low + 0.1:.1f}"
        sym_cal = dict((controls.get("symbol_calibration") or {}).get(symbol) or {})
        calibrated = _clip(float(sym_cal.get(bin_key, c_raw)), 0.0, 1.0)
        decision.confidence = calibrated
        decision.forecast_confidence = _clip(float(decision.forecast_confidence or calibrated), 0.0, 1.0)
        sym_min = (controls.get("symbol_min_confidence") or {}).get(symbol)
        min_conf = float(
            sym_min if sym_min is not None else controls.get("global_min_confidence", controls.get("default_min_confidence", 0.55))
        )
        if calibrated < min_conf:
            return _hold_decision(f"Confidence gate ({symbol}): calibrated {calibrated:.2f} < min {min_conf:.2f}.")
        if bool(controls.get("calibration_gate_enabled", False)):
            allowed_bins = set((controls.get("symbol_allowed_confidence_bins") or {}).get(symbol) or [])
            stressed_map = dict((controls.get("symbol_bin_stressed_bps") or {}).get(symbol) or {})
            has_bin_history = bool(sym_cal)
            if has_bin_history and bin_key not in allowed_bins:
                stressed = stressed_map.get(bin_key)
                if stressed is None:
                    return _hold_decision(
                        f"Calibration gate ({symbol}): bin {bin_key} not approved (insufficient/negative bin edge)."
                    )
                return _hold_decision(
                    f"Calibration gate ({symbol}): bin {bin_key} stressed_signed_bps {float(stressed):.2f} <= 0."
                )
        decision.reasoning = f"{decision.reasoning} [cal_conf={calibrated:.2f} min_conf={min_conf:.2f}]".strip()
        return decision

    def _build_gate_profile(self) -> dict[str, Any]:
        requested = (
            bool(self._acceleration_mode_override)
            if self._acceleration_mode_override is not None
            else bool(self.settings.data_acceleration_mode)
        )
        blocked_live = requested and self._is_live_execution_mode()
        active = requested and not blocked_live

        session_start = self.settings.session_start_et
        session_end = self.settings.session_end_et
        max_spread_bps = float(self.settings.max_spread_bps)
        entry_confluence_min = float(self.settings.entry_confluence_min)
        ev_min_ticks = float(self.settings.ev_min_ticks)
        edge_min_win_rate = float(self.settings.edge_min_win_rate)
        edge_min_expectancy = float(self.settings.edge_min_expectancy)

        if active:
            session_start = self.settings.accel_session_start_et
            session_end = self.settings.accel_session_end_et
            max_spread_bps = max(
                max_spread_bps,
                max_spread_bps * max(1.0, float(self.settings.accel_spread_mult)),
            )
            entry_confluence_min = max(
                0.0,
                entry_confluence_min - max(0.0, float(self.settings.accel_relax_confluence)),
            )
            ev_min_ticks = max(
                0.0,
                ev_min_ticks - max(0.0, float(self.settings.accel_relax_ev_ticks)),
            )
            edge_min_win_rate = max(
                0.05,
                edge_min_win_rate - max(0.0, float(self.settings.accel_edge_min_win_rate_delta)),
            )
            edge_min_expectancy = edge_min_expectancy - max(
                0.0, float(self.settings.accel_edge_min_expectancy_delta)
            )

        mode = "accelerated" if active else "standard"
        if blocked_live:
            mode = "standard_live_guard"

        return {
            "requested": requested,
            "active": active,
            "blocked_live": blocked_live,
            "mode": mode,
            "session_start_et": session_start,
            "session_end_et": session_end,
            "max_spread_bps": max_spread_bps,
            "entry_confluence_min": entry_confluence_min,
            "ev_min_ticks": ev_min_ticks,
            "edge_min_win_rate": edge_min_win_rate,
            "edge_min_expectancy": edge_min_expectancy,
        }

    def acceleration_snapshot(self) -> dict[str, Any]:
        return dict(self._gate_profile)

    def set_acceleration_mode(self, enabled: bool | None) -> dict[str, Any]:
        self._acceleration_mode_override = None if enabled is None else bool(enabled)
        self._gate_profile = self._build_gate_profile()
        self.edge_monitor.configure_thresholds(
            min_win_rate=float(self._gate_profile["edge_min_win_rate"]),
            min_expectancy=float(self._gate_profile["edge_min_expectancy"]),
        )
        return self.acceleration_snapshot()

    def _market_tz(self) -> ZoneInfo:
        try:
            return ZoneInfo(str(self.settings.timezone or "America/New_York"))
        except Exception:
            return ZoneInfo("America/New_York")

    def _passes_session_filter(self, now_utc: datetime) -> tuple[bool, str]:
        if not self.settings.enable_session_filter:
            return True, "session_filter_disabled"
        et = now_utc.astimezone(self._market_tz())
        start = str(self._gate_profile["session_start_et"])
        end = str(self._gate_profile["session_end_et"])
        start_h, start_m = self._parse_hhmm(start)
        end_h, end_m = self._parse_hhmm(end)
        start_minutes = start_h * 60 + start_m
        end_minutes = end_h * 60 + end_m
        now_minutes = et.hour * 60 + et.minute
        if start_minutes <= end_minutes:
            ok = start_minutes <= now_minutes <= end_minutes
        else:
            ok = now_minutes >= start_minutes or now_minutes <= end_minutes
        blocked, block_meta = self._in_no_trade_window(et)
        if blocked:
            return False, f"session={et.strftime('%H:%M')} no_trade_window={block_meta}"
        return ok, f"session={et.strftime('%H:%M')} window={start}-{end}"

    def _in_no_trade_window(self, et: datetime) -> tuple[bool, str]:
        raw = str(self.settings.no_trade_windows_et or "").strip()
        if not raw:
            return False, ""
        now_m = et.hour * 60 + et.minute
        for chunk in raw.split(","):
            token = chunk.strip()
            if not token or "-" not in token:
                continue
            start, end = token.split("-", 1)
            sh, sm = self._parse_hhmm(start.strip())
            eh, em = self._parse_hhmm(end.strip())
            s = sh * 60 + sm
            e = eh * 60 + em
            inside = (s <= now_m <= e) if s <= e else (now_m >= s or now_m <= e)
            if inside:
                return True, token
        return False, ""

    @staticmethod
    def _get_dim_value(context, dim_name: str, default: float = 0.0) -> float:
        for d in context.dimensions:
            if d.name == dim_name:
                try:
                    return float(d.value)
                except (TypeError, ValueError):
                    return default
        return default

    @staticmethod
    def _get_dim_label(context, dim_name: str, default: str = "") -> str:
        for d in context.dimensions:
            if d.name == dim_name:
                return str(d.label or default)
        return default

    def _confluence_score(self, decision: AiDecision, context) -> tuple[float, str]:
        trend = self._get_dim_value(context, "Trend", 0.0)
        momentum = self._get_dim_value(context, "Momentum", 0.0)
        mean_rev = self._get_dim_value(context, "Mean Reversion", 0.0)
        vol_label = self._get_dim_label(context, "Volatility", "NORMAL")
        volm_label = self._get_dim_label(context, "Volume", "NORMAL")
        sr_label = self._get_dim_label(context, "S/R Proximity", "")
        sr_dist = self._get_dim_value(context, "S/R Proximity", 99.0)

        score = 25.0
        aligned = 1 if decision.direction == "LONG" else -1

        score += max(0.0, aligned * trend) * 0.35
        score += max(0.0, aligned * momentum) * 0.25
        score -= max(0.0, -aligned * trend) * 0.18
        score -= max(0.0, -aligned * momentum) * 0.12

        if volm_label == "SURGE":
            score += 12
        elif volm_label == "ABOVE_AVG":
            score += 8
        elif volm_label == "NORMAL":
            score += 4
        elif volm_label == "BELOW_AVG":
            score -= 5
        elif volm_label == "DEAD":
            score -= 10

        if vol_label == "NORMAL":
            score += 5
        elif vol_label == "LOW":
            score += 1
        elif vol_label == "HIGH":
            score -= 6
        elif vol_label == "EXTREME":
            score -= 14

        if decision.direction == "LONG" and "RESISTANCE" in sr_label and sr_dist <= 0.5:
            score -= 12
        if decision.direction == "SHORT" and "SUPPORT" in sr_label and sr_dist <= 0.5:
            score -= 12

        if mean_rev >= 65:
            snap = self._get_dim_label(context, "Mean Reversion", "")
            if decision.direction == "LONG" and "SNAP_DOWN" in snap:
                score -= 8
            if decision.direction == "SHORT" and "SNAP_UP" in snap:
                score -= 8

        score = max(0.0, min(100.0, score))
        detail = f"trend={trend:.1f} momentum={momentum:.1f} vol={vol_label} volume={volm_label} sr={sr_label}@{sr_dist:.2f}"
        return score, detail

    @staticmethod
    def _spread_bps(quote) -> float:
        if quote.last <= 0:
            return 0.0
        spread = max(0.0, quote.ask - quote.bid)
        return (spread / quote.last) * 10000.0

    def _session_bucket(self, now_utc: datetime) -> str:
        if not self.settings.enable_session_filter:
            return "session_filter_disabled"
        et = now_utc.astimezone(self._market_tz())
        minutes = et.hour * 60 + et.minute
        open_start = 9 * 60 + 30
        if minutes < open_start or minutes > 16 * 60:
            return "outside"
        if minutes <= 10 * 60 + 30:
            return "open"
        if minutes >= 15 * 60:
            return "close"
        return "midday"

    def _ev_ticks(self, decision: AiDecision) -> float:
        p = max(0.0, min(1.0, float(decision.confidence)))
        tp = max(0.0, float(decision.tp_ticks))
        sl = max(0.0, float(decision.sl_ticks))
        return (p * tp) - ((1.0 - p) * sl)

    def _atr_target_size(self, decision: AiDecision, atr_value: float) -> int:
        tick = max(1e-6, self.settings.price_tick_size)
        risk_per_unit = max(tick, float(decision.sl_ticks) * tick, atr_value * 0.35)
        target = int(self.settings.risk_per_trade_usd / risk_per_unit) if risk_per_unit > 0 else 1
        return max(1, min(self.settings.max_position_size, target))

    def run_cycle(self, *, collect_only: bool = False) -> CycleResult:
        quote = self.data_adapter.get_latest_quote(self.settings.symbol)

        closed = self.execution.process_open_position(quote, self.account)
        closed_note = f"Closed position PnL={closed.pnl:.2f}" if closed else "No bracket exit this cycle"
        if closed:
            closed_regime, closed_conf = self._extract_regime_and_confidence(closed.thesis)
            closed.regime = closed.regime or closed_regime
            if closed.confidence <= 0:
                closed.confidence = closed_conf
            self.learner.update_from_trade(closed)
            self.edge_monitor.update_trade(closed)

        bars = self.bar_builder.build_multi_timeframe(
            symbol=self.settings.symbol,
            primary_min=self.settings.primary_timeframe_min,
            short_min=self.settings.short_timeframe_min,
            long_min=self.settings.long_timeframe_min,
            primary_count=self.settings.primary_bars,
            short_count=self.settings.short_bars,
            long_count=self.settings.long_bars,
        )
        if not bars.primary or not bars.short or not bars.long:
            decision = _hold_decision("Insufficient completed bars from data provider.")
            return CycleResult(
                timestamp=datetime.now(tz=timezone.utc),
                dashboard="MARKET CONTEXT DASHBOARD\nNo completed bars available.",
                llm_raw="",
                decision=decision,
                note=f"{closed_note}; Hold (no market data)",
            )

        indicators = compute_indicator_set(bars.primary)
        context = build_market_context(indicators, bars.primary)
        dashboard = render_context_dashboard(context)
        macro_snapshot = load_macro_snapshot(self.settings)
        macro_text = render_macro_context(macro_snapshot)
        economic_calendar = load_economic_calendar(self.settings)
        calendar_text = render_economic_calendar(economic_calendar)
        finnhub_context = load_finnhub_context(self.settings, self.settings.symbol)
        finnhub_text = render_finnhub_context(finnhub_context)
        dashboard_with_macro = "\n\n".join([dashboard, macro_text, calendar_text, finnhub_text])
        llm_context = build_llm_context(
            dashboard_with_macro,
            bars,
            self.account,
            recent_bars=max(1, int(self.settings.llm_context_recent_bars)),
        )
        max_chars = max(2000, int(self.settings.llm_max_context_chars))
        context_chars_before_truncation = len(llm_context)
        if len(llm_context) > max_chars:
            llm_context = llm_context[:max_chars]

        now_utc = datetime.now(tz=timezone.utc)
        state_signature = self._build_state_signature(
            quote,
            context,
            str(economic_calendar.event_risk),
            str(finnhub_context.news_risk),
        )
        llm_routing: dict[str, Any] = {
            "two_tier_enabled": bool(self.settings.llm_two_tier_enabled),
            "primary_provider": str(self.settings.llm_provider),
            "primary_model": str(self.settings.gemini_model if self.settings.llm_provider == "gemini" else self.settings.openai_model),
            "secondary_invoked": False,
            "secondary_reason": "",
            "active_tier": "primary",
            "state_gate_used_cache": False,
            "state_gate_reason": "",
            "llm_fresh": True,
            "cache_age_seconds": 0.0,
            "cache_source": "",
            "primary_call_count": 0,
            "secondary_call_count": 0,
            "context_chars": len(llm_context),
            "context_chars_before_truncation": context_chars_before_truncation,
            "context_truncated": bool(context_chars_before_truncation > len(llm_context)),
            "context_recent_bars": max(1, int(self.settings.llm_context_recent_bars)),
            "max_output_tokens": int(self.settings.llm_max_output_tokens),
            "provider_health": self.llm_provider_health_snapshot(),
        }
        skip_llm, skip_reason = self._should_skip_llm_due_to_state(state_signature, now_utc)
        if skip_llm and self._last_llm_decision is not None:
            decision = self._clone_decision(self._last_llm_decision)
            raw = self._last_llm_raw
            decision.reasoning = f"{decision.reasoning} [cached:{skip_reason}]".strip()
            llm_routing["state_gate_used_cache"] = True
            llm_routing["state_gate_reason"] = skip_reason
            llm_routing["llm_fresh"] = False
            llm_routing["cache_source"] = "state_gate"
            if self._last_llm_call_at is not None:
                llm_routing["cache_age_seconds"] = max(0.0, (now_utc - self._last_llm_call_at).total_seconds())
        else:
            primary_provider = str(self.settings.llm_provider)
            secondary_provider = str(self.settings.llm_two_tier_secondary_provider)
            forced_provider = self._resolve_forced_primary_provider()
            primary_client = self.llm
            if forced_provider == "gemini":
                primary_client = self.llm if str(self.settings.llm_provider) == "gemini" else (self.llm_secondary or self.llm)
                primary_provider = "gemini"
            elif forced_provider == "openai":
                primary_client = self.llm if str(self.settings.llm_provider) == "openai" else (self.llm_secondary or self.llm)
                primary_provider = "openai"
            try:
                if self._provider_is_unhealthy(primary_provider) and self.llm_secondary is not None:
                    llm_routing["secondary_call_count"] = int(llm_routing["secondary_call_count"]) + 1
                    raw_primary, decision = self._decide_with_retries(
                        self.llm_secondary, llm_context, provider=secondary_provider
                    )
                    llm_routing["active_tier"] = "secondary"
                    llm_routing["secondary_invoked"] = True
                    llm_routing["secondary_reason"] = "primary_unhealthy_failover"
                else:
                    llm_routing["primary_call_count"] = int(llm_routing["primary_call_count"]) + 1
                    raw_primary, decision = self._decide_with_retries(
                        primary_client, llm_context, provider=primary_provider
                    )
                raw = raw_primary
                self._last_llm_signature = state_signature
                self._last_llm_call_at = now_utc
                self._last_llm_raw = raw_primary
                self._last_llm_decision = self._clone_decision(decision)
                llm_routing["state_gate_reason"] = skip_reason
            except Exception as primary_exc:
                failover_succeeded = False
                if self.llm_secondary is not None:
                    try:
                        llm_routing["secondary_call_count"] = int(llm_routing["secondary_call_count"]) + 1
                        raw_secondary, decision = self._decide_with_retries(
                            self.llm_secondary, llm_context, provider=secondary_provider
                        )
                        raw = raw_secondary
                        self._last_llm_signature = state_signature
                        self._last_llm_call_at = now_utc
                        self._last_llm_raw = raw_secondary
                        self._last_llm_decision = self._clone_decision(decision)
                        llm_routing["active_tier"] = "secondary"
                        llm_routing["secondary_invoked"] = True
                        llm_routing["secondary_reason"] = "primary_exception_failover"
                        llm_routing["state_gate_reason"] = f"primary_exception:{self._exc_brief(primary_exc)}"
                        failover_succeeded = True
                    except Exception as secondary_exc:
                        llm_routing["secondary_invoked"] = True
                        llm_routing["secondary_reason"] = (
                            f"primary_exception_failover_failed:{self._exc_brief(primary_exc)}->{self._exc_brief(secondary_exc)}"
                        )
                if not failover_succeeded:
                    if self._last_llm_decision is not None:
                        decision = self._clone_decision(self._last_llm_decision)
                        raw = self._last_llm_raw
                        decision.reasoning = f"{decision.reasoning} [fallback:last_valid_decision]".strip()
                        llm_routing["state_gate_used_cache"] = True
                        llm_routing["state_gate_reason"] = f"primary_exception_fallback_cache:{self._exc_brief(primary_exc)}"
                        llm_routing["llm_fresh"] = False
                        llm_routing["cache_source"] = "last_valid_decision"
                        if self._last_llm_call_at is not None:
                            llm_routing["cache_age_seconds"] = max(0.0, (now_utc - self._last_llm_call_at).total_seconds())
                    else:
                        decision = _hold_decision("LLM unavailable; fallback deterministic hold.")
                        decision.forecast_direction = "LONG" if self._get_dim_value(context, "Trend", 0.0) >= 0 else "SHORT"
                        decision.forecast_confidence = 0.5
                        decision.forecast_horizon_minutes = 15
                        raw = ""
                        llm_routing["state_gate_used_cache"] = True
                        llm_routing["state_gate_reason"] = f"llm_unavailable_deterministic_hold:{self._exc_brief(primary_exc)}"
                        llm_routing["llm_fresh"] = False
                        llm_routing["cache_source"] = "deterministic_hold"
        escalate, reason = self._should_escalate_to_secondary(decision)
        if (not llm_routing["state_gate_used_cache"]) and escalate and self.llm_secondary is not None:
            try:
                llm_routing["secondary_call_count"] = int(llm_routing["secondary_call_count"]) + 1
                raw_secondary, decision = self._decide_with_retries(
                    self.llm_secondary, llm_context, provider=str(self.settings.llm_two_tier_secondary_provider)
                )
                raw = raw_secondary
                self._last_llm_raw = raw_secondary
                self._last_llm_decision = self._clone_decision(decision)
                llm_routing["secondary_invoked"] = True
                llm_routing["secondary_reason"] = reason
                llm_routing["active_tier"] = "secondary"
                llm_routing["secondary_provider"] = str(self.settings.llm_two_tier_secondary_provider)
                llm_routing["secondary_model"] = str(
                    self.settings.gemini_secondary_model
                    if str(self.settings.llm_two_tier_secondary_provider) == "gemini"
                    else self.settings.openai_model
                )
            except Exception as exc:
                llm_routing["secondary_invoked"] = True
                llm_routing["secondary_reason"] = f"{reason}; fallback_primary={self._exc_brief(exc)}"
        llm_routing["provider_health"] = self.llm_provider_health_snapshot()
        candidate_trade = decision.action == "trade" and decision.direction is not None
        challenger_trade = False

        adaptive = self.learner.adapt(decision, indicators.regime)
        decision = adaptive.decision
        if adaptive.note not in {"adaptive_bypass", ""}:
            decision.reasoning = f"{decision.reasoning} [{adaptive.note}]".strip()
        decision = self._apply_confidence_controls(decision)

        in_session, session_meta = self._passes_session_filter(now_utc)
        if decision.action == "trade" and decision.direction is not None:
            horizon_ok, horizon_meta = self._passes_horizon_gate(
                self.settings.symbol,
                int(decision.forecast_horizon_minutes or 15),
            )
            regime_ok, regime_meta = self._passes_regime_gate(self.settings.symbol, indicators.regime, now_utc)
            if not horizon_ok:
                decision = _hold_decision(f"Horizon gate: {horizon_meta}.")
            elif not regime_ok:
                decision = _hold_decision(f"Regime gate: {regime_meta}.")
            elif self.settings.symbol.strip().upper() in set(self.settings.trade_blocked_symbols or ()):
                decision = _hold_decision(
                    f"Symbol gate: trading disabled for {self.settings.symbol.strip().upper()}."
                )
            elif not in_session:
                decision = _hold_decision(
                    f"Session gate: outside trading window ({session_meta})."
                )
            elif (
                self.settings.economic_calendar_block_high_impact
                and economic_calendar.event_risk == "high_impact_window"
            ):
                decision = _hold_decision(
                    "Economic calendar gate: high-impact event inside configured risk window."
                )
            else:
                spread_bps = self._spread_bps(quote)
                if spread_bps > float(self._gate_profile["max_spread_bps"]):
                    decision = _hold_decision(
                        f"Spread gate: {spread_bps:.2f} bps exceeds max {float(self._gate_profile['max_spread_bps']):.2f}."
                    )
                else:
                    score, detail = self._confluence_score(decision, context)
                    if score < float(self._gate_profile["entry_confluence_min"]):
                        decision = _hold_decision(
                            f"Confluence gate: score {score:.1f} < min {float(self._gate_profile['entry_confluence_min']):.1f}. {detail}"
                        )
                    else:
                        ev_ticks = self._ev_ticks(decision)
                        if ev_ticks < float(self._gate_profile["ev_min_ticks"]):
                            decision = _hold_decision(
                                f"EV gate: ev_ticks {ev_ticks:.2f} < min {float(self._gate_profile['ev_min_ticks']):.2f}."
                            )
                        else:
                            sized = self._atr_target_size(decision, indicators.atr14)
                            decision.size = max(1, min(self.settings.max_position_size, sized))
                            decision.reasoning = (
                                f"{decision.reasoning} "
                                f"[confluence={score:.1f}/{float(self._gate_profile['entry_confluence_min']):.1f}] "
                                f"[ev_ticks={ev_ticks:.2f}/{float(self._gate_profile['ev_min_ticks']):.2f}] "
                                f"[atr={indicators.atr14:.4f} size={decision.size}]"
                            )
                            if candidate_trade and self.settings.shadow_challenger_enabled:
                                challenger_trade = (
                                    score >= (
                                        float(self._gate_profile["entry_confluence_min"])
                                        + self.settings.shadow_confluence_bump
                                    )
                                    and ev_ticks >= (
                                        float(self._gate_profile["ev_min_ticks"]) + self.settings.shadow_ev_bump
                                    )
                                )

        edge_action = self.edge_monitor.throttle(decision)
        decision = edge_action.decision
        if edge_action.note not in {"edge_bypass", "edge_warmup", "edge_ok", ""}:
            decision.reasoning = f"{decision.reasoning} [{edge_action.note}]".strip()

        decision = apply_risk_rules(decision, self.account, self.settings)
        self.edge_monitor.update_shadow(
            champion_trade=(decision.action == "trade"),
            challenger_trade=challenger_trade,
        )

        if decision.action == "trade" and bool(collect_only):
            decision.action = "hold"
            decision.direction = None
            decision.size = 1
            decision.sl_ticks = 0
            decision.tp_ticks = 0
            decision.reasoning = f"{decision.reasoning} [collection_only]".strip()
            note = "Hold (collection_only)"
        elif decision.action == "trade":
            accel_tag = "[ACCEL:ON] " if bool(self._gate_profile["active"]) else ""
            decision.reasoning = (
                f"[REGIME:{indicators.regime}][CONF:{decision.confidence:.2f}] {accel_tag}"
                f"{decision.reasoning}"
            )
            order_key = (
                f"{self.settings.symbol}|{decision.direction}|{decision.size}|{decision.sl_ticks}|"
                f"{decision.tp_ticks}|{quote.timestamp.isoformat()}"
            )
            risk_result = evaluate_pre_trade_risk(
                account=self.account,
                decision=decision,
                quote=quote,
                max_order_notional_usd=self.settings.pretrade_max_order_notional_usd,
                max_total_exposure_usd=self.settings.pretrade_max_total_exposure_usd,
                max_quote_age_seconds=self.settings.pretrade_max_quote_age_seconds,
                max_pretrade_spread_bps=self.settings.pretrade_max_spread_bps,
                min_buying_power_usd=self.settings.pretrade_min_buying_power_usd,
                duplicate_cooldown_seconds=self.settings.pretrade_duplicate_cooldown_seconds,
                last_order_key=self._last_order_key,
                last_order_submitted_at=self._last_order_submitted_at,
                current_order_key=order_key,
            )
            append_audit_event(
                self.settings.audit_log_path,
                "pretrade_check",
                {
                    "symbol": self.settings.symbol,
                    "decision": decision.__dict__,
                    "allowed": risk_result.allowed,
                    "reason": risk_result.reason,
                    "checks": risk_result.checks,
                },
            )
            if not risk_result.allowed:
                decision = _hold_decision(f"Pre-trade risk gateway blocked order: {risk_result.reason}.")
                note = f"Hold (pretrade_block={risk_result.reason})"
            else:
                try:
                    self.execution.submit_bracket(self.settings.symbol, decision, quote, self.account)
                    self._last_order_key = order_key
                    self._last_order_submitted_at = datetime.now(tz=timezone.utc)
                    self.account.todays_trade_count += 1
                    note = f"Placed {decision.direction} x{decision.size} @ {quote.last:.2f}"
                    append_audit_event(
                        self.settings.audit_log_path,
                        "order_submitted",
                        {
                            "symbol": self.settings.symbol,
                            "decision": decision.__dict__,
                            "quote_last": quote.last,
                        },
                    )
                except Exception as exc:
                    append_audit_event(
                        self.settings.audit_log_path,
                        "order_submit_error",
                        {
                            "symbol": self.settings.symbol,
                            "decision": decision.__dict__,
                            "error": str(exc),
                        },
                    )
                    decision = _hold_decision(f"Execution submit failed: {exc}")
                    note = "Hold (execution_submit_error)"
        else:
            note = "Hold"

        result_ts = datetime.now(tz=timezone.utc)
        quote_age_seconds = max(0.0, (result_ts - quote.timestamp).total_seconds())
        spread_bps = self._spread_bps(quote)
        dim_features = {
            d.name.lower().replace(" ", "_"): {
                "value": d.value,
                "label": d.label,
                "detail": d.detail,
            }
            for d in context.dimensions
        }
        quality_flags = {
            "outside_session": not in_session,
            "quote_stale": quote_age_seconds > float(self.settings.sample_max_quote_age_seconds),
            "spread_too_wide": spread_bps > float(self._gate_profile["max_spread_bps"]),
            "macro_fallback": macro_snapshot.provider != "fred" and self.settings.macro_provider == "fred",
            "economic_calendar_error": economic_calendar.event_risk == "error",
            "finnhub_context_error": "error=" in (finnhub_context.notes or ""),
            "missing_forecast": decision.forecast_direction not in {"LONG", "SHORT"},
        }
        hard_fail_flags = {
            "economic_calendar_error": bool(quality_flags["economic_calendar_error"]),
            "finnhub_context_error": bool(quality_flags["finnhub_context_error"]),
            "missing_forecast": bool(quality_flags["missing_forecast"]),
        }
        sample_quality_good = not any(hard_fail_flags.values())

        return CycleResult(
            timestamp=result_ts,
            dashboard=dashboard_with_macro,
            llm_raw=raw,
            decision=decision,
            note=f"{closed_note}; {note}",
            metadata={
                "macro": macro_snapshot.to_record(),
                "economic_calendar": economic_calendar.to_record(),
                "finnhub_context": finnhub_context.to_record(),
                "quote": {
                    "timestamp": quote.timestamp.isoformat(),
                    "bid": float(quote.bid),
                    "ask": float(quote.ask),
                    "last": float(quote.last),
                    "size": float(quote.size),
                    "spread_bps": float(spread_bps),
                    "age_seconds": float(quote_age_seconds),
                },
                "indicators": {
                    "regime": indicators.regime,
                    "atr14": float(indicators.atr14),
                    "rsi14": float(indicators.rsi14),
                    "macd_hist": float(indicators.macd_hist),
                },
                "features": {
                    "dimensions": dim_features,
                    "key_levels": dict(context.key_levels or {}),
                    "patterns": dict(context.pattern_features or {}),
                    "structure": dict(context.structure_features or {}),
                    "trend_score": float(self._get_dim_value(context, "Trend", 0.0)),
                    "momentum_score": float(self._get_dim_value(context, "Momentum", 0.0)),
                    "mean_reversion_score": float(self._get_dim_value(context, "Mean Reversion", 0.0)),
                    "sr_distance_atr": float(self._get_dim_value(context, "S/R Proximity", 99.0)),
                    "volatility_label": self._get_dim_label(context, "Volatility", ""),
                    "volume_label": self._get_dim_label(context, "Volume", ""),
                    "sr_label": self._get_dim_label(context, "S/R Proximity", ""),
                },
                "sample_quality": {
                    "good": bool(sample_quality_good),
                    "flags": quality_flags,
                    "hard_fail_flags": hard_fail_flags,
                    "session_bucket": self._session_bucket(result_ts),
                    "session_meta": session_meta,
                },
                "llm_routing": llm_routing,
            },
        )
