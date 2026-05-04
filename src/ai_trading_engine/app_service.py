from __future__ import annotations

import json
import math
import random
import re
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from ai_trading_engine.config import Settings
from ai_trading_engine.economic_calendar import load_economic_calendar
from ai_trading_engine.engine import CycleResult, TradingEngine
from ai_trading_engine.finnhub_context import load_finnhub_context
from ai_trading_engine.metadata import MODEL_NAME, MODEL_VERSION
from ai_trading_engine.models import AccountState, AiDecision, ClosedTrade, Position
from ai_trading_engine.notifications import append_notification, list_notifications
from ai_trading_engine.model_monitoring import build_model_decay_report
from ai_trading_engine.persistence import Persistence, RuntimeConfig
from ai_trading_engine.portfolio import optimise_portfolio
from ai_trading_engine.prediction_quality import (
    build_prediction_labels,
    build_confidence_control_report,
    build_feature_ablation_report,
    build_prediction_quality_report,
)
from ai_trading_engine.predictive_research import (
    build_predictive_dataset,
    run_predictive_walk_forward,
    save_predictive_report,
)
from ai_trading_engine.promotion import (
    PromotionPolicy,
    evaluate_predictive_report,
    load_json,
    save_json,
    utc_now_iso,
)
from ai_trading_engine.research import (
    build_trade_dataset,
    run_walk_forward,
    save_walk_forward_report,
)
from ai_trading_engine.sample_coverage import build_sample_coverage_report
from ai_trading_engine.serialization import account_to_record
from ai_trading_engine.trade_costs import estimate_round_trip_cost, should_apply_cost_model


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _restore_position(payload: dict[str, Any] | None) -> Position | None:
    if not payload:
        return None
    return Position(
        symbol=str(payload["symbol"]),
        direction=str(payload["direction"]),
        size=int(payload["size"]),
        entry_price=float(payload["entry_price"]),
        sl_ticks=int(payload["sl_ticks"]),
        tp_ticks=int(payload["tp_ticks"]),
        opened_at=_parse_dt(str(payload["opened_at"])),
        thesis=str(payload.get("thesis", "")),
    )


def _restore_trade(payload: dict[str, Any]) -> ClosedTrade:
    return ClosedTrade(
        symbol=str(payload["symbol"]),
        direction=str(payload["direction"]),
        size=int(payload["size"]),
        entry_price=float(payload["entry_price"]),
        exit_price=float(payload["exit_price"]),
        pnl=float(payload["pnl"]),
        opened_at=_parse_dt(str(payload["opened_at"])),
        closed_at=_parse_dt(str(payload["closed_at"])),
        thesis=str(payload.get("thesis", "")),
        regime=str(payload.get("regime", "")),
        confidence=float(payload.get("confidence", 0.0)),
        metadata=dict(payload.get("metadata", {}) or {}),
    )


def _extract_regime_and_confidence(thesis: str) -> tuple[str, float]:
    regime_match = re.search(r"\[REGIME:([^\]]+)\]", thesis or "")
    conf_match = re.search(r"\[CONF:([0-9]*\.?[0-9]+)\]", thesis or "")
    regime = regime_match.group(1) if regime_match else ""
    confidence = float(conf_match.group(1)) if conf_match else 0.0
    return regime, confidence


def _enrich_trade_for_learning(trade: ClosedTrade) -> tuple[ClosedTrade, bool, bool]:
    regime_backfilled = False
    confidence_backfilled = False
    if not trade.regime:
        regime, conf = _extract_regime_and_confidence(trade.thesis)
        if regime:
            trade.regime = regime
        else:
            trade.regime = "unknown"
            regime_backfilled = True
        if trade.confidence <= 0.0:
            if conf > 0.0:
                trade.confidence = conf
            else:
                trade.confidence = 0.5
                confidence_backfilled = True
    return trade, regime_backfilled, confidence_backfilled


def restore_account(snapshot: dict[str, Any] | None, default_balance: float = 50000.0) -> AccountState:
    if not snapshot:
        return AccountState(balance=default_balance, starting_balance=default_balance)
    return AccountState(
        balance=float(snapshot.get("balance", default_balance)),
        starting_balance=float(snapshot.get("starting_balance", default_balance)),
        trading_day_et=str(snapshot.get("trading_day_et")) if snapshot.get("trading_day_et") else None,
        todays_trade_count=int(snapshot.get("todays_trade_count", 0)),
        daily_realized_pnl=float(snapshot.get("daily_realized_pnl", 0.0)),
        open_position=_restore_position(snapshot.get("open_position")),
        closed_trades_today=[_restore_trade(t) for t in snapshot.get("closed_trades_today", [])],
    )


@dataclass
class WorkerStatus:
    running: bool
    cycles_completed: int
    last_cycle_at: str | None
    last_note: str
    last_error: str | None
    model_monitoring: dict[str, Any] | None = None
    kill_switch: dict[str, Any] | None = None
    startup_warnings: list[str] | None = None


@dataclass
class GoLiveCheck:
    name: str
    actual: float
    threshold: float
    comparator: str
    passed: bool


class TradingAppService:
    def __init__(self, settings: Settings, persistence: Persistence) -> None:
        self._settings = settings
        self._persistence = persistence
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._cycles_completed = 0
        self._last_cycle_at: str | None = None
        self._last_note = "idle"
        self._last_error: str | None = None
        self._last_broker_sync_at: datetime | None = None
        self._automation_state_path = Path(self._settings.auto_retrain_state_path)
        self._automation_state_path.parent.mkdir(parents=True, exist_ok=True)
        self._promotion_state_path = Path(self._settings.promotion_state_path)
        self._promotion_state_path.parent.mkdir(parents=True, exist_ok=True)
        self._automation_lock = threading.Lock()
        self._automation_stop_event = threading.Event()
        self._automation_thread: threading.Thread | None = None
        self._autonomy_lock = threading.Lock()
        self._autonomy_stop_event = threading.Event()
        self._autonomy_thread: threading.Thread | None = None
        self._signed_bps_gate_state: dict[str, Any] = {
            "evaluated_at": None,
            "blocked": False,
            "reason": "not_evaluated",
            "report": {},
        }
        self._symbol_gate_state: dict[str, Any] = {
            "evaluated_at": None,
            "enabled": bool(self._settings.symbol_gate_enabled),
            "allowed_symbols": [],
            "blocked": False,
            "reason": "not_evaluated",
            "report": {},
        }
        self._confidence_control_state: dict[str, Any] = {
            "evaluated_at": None,
            "enabled": bool(self._settings.confidence_control_enabled),
            "controls": {},
            "report": {},
        }
        self._regime_gate_state: dict[str, Any] = {
            "evaluated_at": None,
            "enabled": bool(self._settings.regime_gate_enabled),
            "controls": {},
            "report": {},
        }
        self._horizon_gate_state: dict[str, Any] = {
            "evaluated_at": None,
            "enabled": bool(self._settings.best_horizon_gate_enabled),
            "controls": {},
            "report": {},
        }
        self._quality_guard_state: dict[str, Any] = {
            "evaluated_at": None,
            "enabled": bool(self._settings.automation_quality_checks_enabled),
            "active": False,
            "reason": "not_evaluated",
            "metrics": {},
        }
        self._cost_guard_state: dict[str, Any] = {
            "evaluated_at": None,
            "enabled": bool(self._settings.automation_cost_guard_enabled),
            "active": False,
            "warning": False,
            "reason": "not_evaluated",
            "metrics": {},
        }
        self._sample_flow_guard_state: dict[str, Any] = {
            "evaluated_at": None,
            "enabled": bool(self._settings.sample_flow_guard_enabled),
            "active": False,
            "reason": "not_evaluated",
            "metrics": {},
        }
        self._coverage_guard_state: dict[str, Any] = {
            "active": False,
            "daily_labels": 0,
            "target": int(self._settings.coverage_guard_daily_label_target),
            "label_factor": 1.0,
        }
        self._model_monitor_state_path = Path(self._settings.model_monitor_state_path)
        self._model_monitor_state_path.parent.mkdir(parents=True, exist_ok=True)

        self._engine = TradingEngine(settings)
        snapshot = self._persistence.latest_account_snapshot()
        self._engine.account = restore_account(snapshot)
        self._shadow_engines = self._build_shadow_engines()
        self._shadow_symbols = tuple(self._shadow_engines.keys())
        self._shadow_index = 0
        self._last_shadow_symbol: str | None = None
        self._sync_account_from_broker(force_baseline=True)
        self._rollover_daily_state_if_needed()
        self._persistence.save_account(self._engine.account)
        self._save_runtime_config()
        if self._settings.auto_retrain_enabled:
            self._automation_thread = threading.Thread(
                target=self._automation_loop,
                daemon=True,
                name="automation-worker",
            )
            self._automation_thread.start()
        if self._settings.autonomous_research_enabled:
            self.start_autonomous_research()

    def _market_tz(self) -> ZoneInfo:
        try:
            return ZoneInfo(str(self._settings.timezone or "America/New_York"))
        except Exception:
            return ZoneInfo("America/New_York")

    def _now_utc(self) -> datetime:
        return datetime.now(tz=timezone.utc)

    def _is_weekend_market_day(self) -> bool:
        now_et = self._now_utc().astimezone(self._market_tz())
        return now_et.weekday() >= 5

    def _is_in_session_window(self) -> bool:
        if not bool(self._settings.enable_session_filter):
            return True
        now_et = self._now_utc().astimezone(self._market_tz())
        minutes = (now_et.hour * 60) + now_et.minute

        def _parse_hhmm(raw: str, fallback: int) -> int:
            try:
                hh, mm = str(raw or "").split(":", 1)
                return max(0, min(23, int(hh))) * 60 + max(0, min(59, int(mm)))
            except Exception:
                return fallback

        start = _parse_hhmm(str(self._settings.session_start_et), 9 * 60 + 35)
        end = _parse_hhmm(str(self._settings.session_end_et), 15 * 60 + 45)
        if start <= end:
            return start <= minutes <= end
        return minutes >= start or minutes <= end

    @staticmethod
    def _to_non_negative_int(value: Any) -> int:
        try:
            n = int(float(value))
        except Exception:
            return 0
        return max(0, n)

    def _estimate_llm_calls_recent(self, *, window_minutes: int = 60, sample_limit: int = 5000) -> int:
        start = datetime.now(tz=timezone.utc) - timedelta(minutes=max(1, int(window_minutes)))
        rows = self._persistence.list_data_samples(max(100, int(sample_limit)))
        total_calls = 0
        for r in rows:
            try:
                ts = datetime.fromisoformat(str(r.get("timestamp") or ""))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except Exception:
                continue
            if ts < start:
                continue
            md = dict(r.get("metadata") or {})
            lr = dict(md.get("llm_routing") or {})
            if not lr:
                continue
            if "primary_call_count" in lr or "secondary_call_count" in lr:
                total_calls += self._to_non_negative_int(lr.get("primary_call_count", 0))
                total_calls += self._to_non_negative_int(lr.get("secondary_call_count", 0))
                continue
            if not bool(lr.get("state_gate_used_cache", False)):
                total_calls += 1
            if bool(lr.get("secondary_invoked", False)):
                total_calls += 1
        return total_calls

    def _run_engine_cycle_with_budget_lock(self, *, collect_only: bool = False) -> CycleResult:
        if not bool(self._settings.budget_lock_enabled):
            return self._engine.run_cycle(collect_only=collect_only)
        locked_settings = replace(
            self._engine.settings,
            llm_state_change_min_seconds=max(
                int(self._engine.settings.llm_state_change_min_seconds),
                int(self._settings.budget_lock_state_change_min_seconds),
            ),
            llm_state_change_price_bps=max(float(self._engine.settings.llm_state_change_price_bps), 14.0),
            llm_state_change_trend_delta=max(float(self._engine.settings.llm_state_change_trend_delta), 18.0),
            llm_state_change_momentum_delta=max(float(self._engine.settings.llm_state_change_momentum_delta), 18.0),
            llm_two_tier_escalate_on_trade=(
                False
                if bool(self._settings.budget_lock_disable_secondary_escalation)
                else bool(self._engine.settings.llm_two_tier_escalate_on_trade)
            ),
            finnhub_context_enabled=(
                False
                if bool(self._settings.automation_cost_mode_disable_finnhub_context)
                else bool(self._engine.settings.finnhub_context_enabled)
            ),
            economic_calendar_enabled=(
                False
                if bool(self._settings.automation_cost_mode_disable_economic_calendar)
                else bool(self._engine.settings.economic_calendar_enabled)
            ),
        )
        try:
            self._engine.settings = locked_settings
            return self._engine.run_cycle(collect_only=collect_only)
        finally:
            self._engine.settings = self._settings

    def _load_model_monitor_state(self) -> dict[str, Any]:
        state = load_json(self._model_monitor_state_path)
        if not isinstance(state, dict):
            state = {}
        state.setdefault("breach_streak", 0)
        state.setdefault("safe_mode_active", False)
        state.setdefault("last_report", {})
        state.setdefault("last_evaluated_at", None)
        state.setdefault("last_breached_at", None)
        state.setdefault("safe_mode_triggered_at", None)
        return state

    def _save_model_monitor_state(self, state: dict[str, Any]) -> None:
        save_json(self._model_monitor_state_path, state)

    def _evaluate_model_monitoring(self) -> dict[str, Any]:
        if not self._settings.model_monitoring_enabled:
            return {
                "enabled": False,
                "safe_mode_active": False,
                "breach_streak": 0,
                "required_breach_streak": int(self._settings.model_monitor_breach_streak),
                "state_path": str(self._model_monitor_state_path),
            }
        state = self._load_model_monitor_state()
        rows = self._persistence.list_data_samples(max(1, min(100000, int(self._settings.model_monitor_lookback))))
        report = build_model_decay_report(
            list(reversed(rows)),
            horizon_minutes=int(self._settings.model_monitor_horizon_minutes),
            min_confidence=float(self._settings.model_monitor_min_confidence),
            short_window_days=int(self._settings.model_monitor_short_window_days),
            long_window_days=int(self._settings.model_monitor_long_window_days),
            min_labels=int(self._settings.model_monitor_min_labels),
            min_accuracy_delta=float(self._settings.model_monitor_min_accuracy_delta),
            min_signed_bps_delta=float(self._settings.model_monitor_min_signed_bps_delta),
            max_brier_delta=float(self._settings.model_monitor_max_brier_delta),
        )
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        breached = bool(report.get("breached", False))
        if breached:
            state["breach_streak"] = int(state.get("breach_streak", 0)) + 1
            state["last_breached_at"] = now_iso
        else:
            state["breach_streak"] = 0
        required_streak = max(1, int(self._settings.model_monitor_breach_streak))
        safe_mode = bool(
            self._settings.model_monitor_safe_mode_enabled
            and int(state.get("breach_streak", 0)) >= required_streak
            and bool(report.get("sufficient_sample", False))
        )
        if safe_mode and not bool(state.get("safe_mode_active", False)):
            state["safe_mode_triggered_at"] = now_iso
        if not safe_mode and bool(state.get("safe_mode_active", False)):
            state["safe_mode_triggered_at"] = None
        state["safe_mode_active"] = safe_mode
        state["last_report"] = report
        state["last_evaluated_at"] = now_iso
        self._save_model_monitor_state(state)
        return {
            "enabled": True,
            "state_path": str(self._model_monitor_state_path),
            "required_breach_streak": required_streak,
            "breach_streak": int(state.get("breach_streak", 0)),
            "safe_mode_active": bool(state.get("safe_mode_active", False)),
            "last_evaluated_at": state.get("last_evaluated_at"),
            "last_breached_at": state.get("last_breached_at"),
            "safe_mode_triggered_at": state.get("safe_mode_triggered_at"),
            "report": report,
        }

    def _sync_account_from_broker(self, force_baseline: bool = False, record_closures: bool = False) -> bool:
        fetcher = getattr(self._engine.execution, "fetch_account_summary", None)
        if not callable(fetcher):
            return False
        prior_balance = float(self._engine.account.balance)
        prior_position = self._engine.account.open_position
        try:
            summary = fetcher()
        except Exception:
            return False
        if not summary:
            return False

        equity = float(summary.get("equity") or summary.get("portfolio_value") or 0.0)
        last_equity = float(summary.get("last_equity") or equity or 0.0)
        if equity <= 0:
            return False

        self._engine.account.balance = equity
        if force_baseline:
            self._engine.account.starting_balance = last_equity if last_equity > 0 else equity
        elif self._engine.account.starting_balance <= 0:
            self._engine.account.starting_balance = last_equity if last_equity > 0 else equity

        position_fetcher = getattr(self._engine.execution, "fetch_symbol_position", None)
        if callable(position_fetcher):
            try:
                broker_position = position_fetcher(self._settings.symbol)
            except Exception:
                return False
            if broker_position is not None:
                if (
                    prior_position is not None
                    and prior_position.symbol == broker_position.symbol
                    and prior_position.direction == broker_position.direction
                ):
                    broker_position.sl_ticks = prior_position.sl_ticks
                    broker_position.tp_ticks = prior_position.tp_ticks
                    broker_position.opened_at = prior_position.opened_at
                    broker_position.thesis = prior_position.thesis
                self._engine.account.open_position = broker_position
                self._last_broker_sync_at = datetime.now(tz=timezone.utc)
                return True

            self._engine.account.open_position = None
            if not record_closures or prior_position is None or force_baseline:
                self._last_broker_sync_at = datetime.now(tz=timezone.utc)
                return True

            pnl = float(equity - prior_balance)
            exit_price = float(prior_position.entry_price)
            if prior_position.size > 0:
                move = pnl / float(prior_position.size)
                if prior_position.direction == "SHORT":
                    move *= -1.0
                exit_price = float(prior_position.entry_price + move)

            if should_apply_cost_model(
                self._settings,
                execution_provider=self._settings.execution_provider,
                is_live=self._is_live_execution_mode(),
            ):
                cost = estimate_round_trip_cost(
                    entry_price=prior_position.entry_price,
                    exit_price=exit_price,
                    size=prior_position.size,
                    slippage_bps_per_side=self._settings.cost_slippage_bps_per_side,
                    fee_per_share=self._settings.cost_fee_per_share,
                    min_fee_per_order=self._settings.cost_min_fee_per_order,
                )
                pnl -= cost

            closed = ClosedTrade(
                symbol=prior_position.symbol,
                direction=prior_position.direction,
                size=prior_position.size,
                entry_price=prior_position.entry_price,
                exit_price=exit_price,
                pnl=pnl,
                opened_at=prior_position.opened_at,
                closed_at=datetime.now(tz=timezone.utc),
                thesis=prior_position.thesis,
            )
            regime, confidence = _extract_regime_and_confidence(closed.thesis)
            closed.regime = regime
            closed.confidence = confidence
            closed.metadata = self._decision_metadata()
            self._engine.account.daily_realized_pnl += pnl
            self._engine.account.closed_trades_today.append(closed)
            self._engine.learner.update_from_trade(closed)
            self._engine.edge_monitor.update_trade(closed)
            self._persistence.save_closed_trade(closed)
            self._notify(
                "position_closed",
                {
                    "symbol": closed.symbol,
                    "direction": closed.direction,
                    "pnl": closed.pnl,
                    "metadata": closed.metadata,
                },
            )
            self._last_broker_sync_at = datetime.now(tz=timezone.utc)
            return True
        self._last_broker_sync_at = datetime.now(tz=timezone.utc)
        return True

    def _position_age_seconds(self, pos: Position | None) -> float:
        if pos is None:
            return 0.0
        opened = pos.opened_at
        if opened.tzinfo is None:
            opened = opened.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(tz=timezone.utc) - opened).total_seconds())

    def _build_guard_cycle(self, reason: str, note: str | None = None) -> CycleResult:
        decision = AiDecision(
            action="hold",
            direction=None,
            confidence=0.0,
            size=1,
            sl_ticks=0,
            tp_ticks=0,
            reasoning=reason,
        )
        return CycleResult(
            timestamp=datetime.now(tz=timezone.utc),
            dashboard=f"GUARD HOLD\n{reason}",
            llm_raw="",
            decision=decision,
            note=note or reason,
        )

    def kill_switch_snapshot(self) -> dict[str, Any]:
        return {
            "enabled": bool(self._settings.trading_kill_switch_enabled),
            "reason": str(self._settings.trading_kill_switch_reason or "manual_kill_switch"),
            "blocks_worker_start": True,
            "blocks_run_once": True,
        }

    def _ensure_position_safety(self) -> tuple[bool, str]:
        pos = self._engine.account.open_position
        if pos is None:
            return False, "no_open_position"

        age_seconds = self._position_age_seconds(pos)
        max_age = max(1, int(self._settings.max_position_age_minutes)) * 60
        if age_seconds > max_age:
            if bool(self._settings.stale_position_force_close):
                if self._settings.execution_provider == "alpaca":
                    close = self.close_open_position_now()
                else:
                    self._engine.account.open_position = None
                    self._persistence.save_account(self._engine.account)
                    close = {"ok": True, "symbol": pos.symbol, "result": {"closed": True, "reason": "local_mock_clear"}}
                self._notify(
                    "stale_position_force_closed",
                    {
                        "symbol": pos.symbol,
                        "age_seconds": age_seconds,
                        "max_age_seconds": max_age,
                        "result": close.get("result"),
                    },
                )
                return True, f"stale position aged {age_seconds:.0f}s > {max_age}s; forced close requested"
            return True, f"stale position aged {age_seconds:.0f}s > {max_age}s"

        if bool(self._settings.require_broker_protection_orders) and self._settings.execution_provider == "alpaca":
            grace = max(0, int(self._settings.protection_check_grace_seconds))
            if age_seconds >= float(grace):
                checker = getattr(self._engine.execution, "has_protection_orders", None)
                if callable(checker):
                    try:
                        has_protection = bool(checker(pos.symbol))
                    except Exception:
                        has_protection = True
                    if not has_protection:
                        if bool(self._settings.stale_position_force_close):
                            close = self.close_open_position_now()
                            self._notify(
                                "missing_protection_force_closed",
                                {
                                    "symbol": pos.symbol,
                                    "age_seconds": age_seconds,
                                    "grace_seconds": grace,
                                    "result": close.get("result"),
                                },
                            )
                            return True, "position missing protection orders; forced close requested"
                        return True, "position missing protection orders"
        return False, "ok"

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.run_cycle_once()
            except Exception as exc:
                with self._lock:
                    self._last_error = str(exc)
                    self._last_note = "cycle failed"
                self._notify("worker_cycle_failed", {"error": str(exc), "model_name": MODEL_NAME})
            self._stop_event.wait(timeout=self._next_cycle_wait_seconds())

    def _next_cycle_wait_seconds(self) -> int:
        wait_seconds = max(1, int(self._settings.cycle_seconds))
        if not bool(self._settings.automation_cost_mode_enabled):
            return wait_seconds
        quality = dict(self._quality_guard_state or {})
        cost = dict(self._cost_guard_state or {})
        if bool(quality.get("active", False)):
            wait_seconds = max(wait_seconds, int(self._settings.automation_quality_throttle_cycle_seconds))
        if bool(cost.get("warning", False)) or bool(cost.get("active", False)):
            wait_seconds = max(wait_seconds, int(self._settings.automation_cost_mode_cycle_seconds))
        return max(1, wait_seconds)

    def _evaluate_signed_bps_gate(self, *, force: bool = False) -> dict[str, Any]:
        now = datetime.now(tz=timezone.utc)
        if not bool(self._settings.signed_bps_kill_enabled):
            self._signed_bps_gate_state.update(
                {
                    "evaluated_at": now.isoformat(),
                    "blocked": False,
                    "reason": "disabled",
                    "report": {},
                }
            )
            return dict(self._signed_bps_gate_state)

        last_raw = self._signed_bps_gate_state.get("evaluated_at")
        if not force and last_raw:
            try:
                last = datetime.fromisoformat(str(last_raw))
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                if (now - last).total_seconds() < float(max(10, int(self._settings.signed_bps_kill_refresh_seconds))):
                    return dict(self._signed_bps_gate_state)
            except Exception:
                pass

        report = self.prediction_quality_report(
            lookback=100000,
            horizons_minutes=(max(1, min(390, int(self._settings.signed_bps_kill_horizon_minutes))),),
            min_confidence=0.0,
            quality_mode="good_only",
        )
        horizon_key = str(max(1, min(390, int(self._settings.signed_bps_kill_horizon_minutes))))
        h = (report.get("horizons") or {}).get(horizon_key, {})
        overall = h.get("overall") or {}
        label_count = int(h.get("label_count", 0) or 0)
        signed_bps = float(overall.get("avg_signed_return_bps", 0.0) or 0.0)

        blocked = bool(
            label_count >= int(self._settings.signed_bps_kill_min_labels)
            and signed_bps <= float(self._settings.signed_bps_kill_threshold)
        )
        reason = (
            f"signed_bps={signed_bps:.2f} <= {float(self._settings.signed_bps_kill_threshold):.2f} "
            f"at n={label_count}"
            if blocked
            else "ok"
        )
        self._signed_bps_gate_state.update(
            {
                "evaluated_at": now.isoformat(),
                "blocked": blocked,
                "reason": reason,
                "report": {
                    "horizon_minutes": int(self._settings.signed_bps_kill_horizon_minutes),
                    "label_count": label_count,
                    "avg_signed_return_bps": signed_bps,
                    "threshold": float(self._settings.signed_bps_kill_threshold),
                    "min_labels": int(self._settings.signed_bps_kill_min_labels),
                },
            }
        )
        return dict(self._signed_bps_gate_state)

    def _evaluate_symbol_gate(self, *, force: bool = False) -> dict[str, Any]:
        now = datetime.now(tz=timezone.utc)
        symbol = str(self._settings.symbol).strip().upper()
        if not bool(self._settings.symbol_gate_enabled):
            self._symbol_gate_state.update(
                {
                    "evaluated_at": now.isoformat(),
                    "enabled": False,
                    "allowed_symbols": [],
                    "blocked": False,
                    "reason": "disabled",
                    "report": {},
                }
            )
            return dict(self._symbol_gate_state)

        last_raw = self._symbol_gate_state.get("evaluated_at")
        if not force and last_raw:
            try:
                last = datetime.fromisoformat(str(last_raw))
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                if (now - last).total_seconds() < float(max(10, int(self._settings.symbol_gate_refresh_seconds))):
                    return dict(self._symbol_gate_state)
            except Exception:
                pass

        horizon = max(1, min(390, int(self._settings.symbol_gate_horizon_minutes)))
        report = self.prediction_quality_report(
            lookback=100000,
            horizons_minutes=(horizon,),
            min_confidence=0.0,
            quality_mode="good_only",
        )
        h = (report.get("horizons") or {}).get(str(horizon), {})
        by_symbol = h.get("by_symbol") or {}
        min_labels = int(self._settings.symbol_gate_min_labels)
        min_signed = float(self._settings.symbol_gate_min_signed_bps)
        min_acc = float(self._settings.symbol_gate_min_accuracy)

        allowed: list[str] = []
        for sym, m in by_symbol.items():
            n = int(m.get("count", 0) or 0)
            signed = float(m.get("avg_signed_return_bps", 0.0) or 0.0)
            acc = float(m.get("accuracy", 0.0) or 0.0)
            if n >= min_labels and signed >= min_signed and acc >= min_acc:
                allowed.append(str(sym).upper())

        metrics = by_symbol.get(symbol) or {}
        n_cur = int(metrics.get("count", 0) or 0)
        signed_cur = float(metrics.get("avg_signed_return_bps", 0.0) or 0.0)
        acc_cur = float(metrics.get("accuracy", 0.0) or 0.0)
        bootstrap_collecting = n_cur < min_labels
        blocked = (not bootstrap_collecting) and (symbol not in set(allowed))
        if bootstrap_collecting:
            reason = (
                f"{symbol} bootstrap collect mode (n={n_cur} < min_labels={min_labels}); "
                "symbol gate not blocking while labels accumulate"
            )
        else:
            reason = (
                f"{symbol} blocked by symbol gate (n={n_cur}, signed_bps={signed_cur:.2f}, acc={acc_cur:.2%})"
                if blocked
                else f"{symbol} allowed"
            )
        self._symbol_gate_state.update(
            {
                "evaluated_at": now.isoformat(),
                "enabled": True,
                "allowed_symbols": sorted(set(allowed)),
                "blocked": blocked,
                "reason": reason,
                "report": {
                    "horizon_minutes": horizon,
                    "min_labels": min_labels,
                    "min_signed_bps": min_signed,
                    "min_accuracy": min_acc,
                    "bootstrap_collecting": bootstrap_collecting,
                    "current_symbol": symbol,
                    "current_symbol_metrics": {
                        "count": n_cur,
                        "avg_signed_return_bps": signed_cur,
                        "accuracy": acc_cur,
                    },
                },
            }
        )
        return dict(self._symbol_gate_state)

    def _load_automation_state(self) -> dict[str, Any]:
        return load_json(self._automation_state_path)

    def _session_bucket_from_ts(self, ts: str) -> str:
        try:
            dt = datetime.fromisoformat(str(ts))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            et = dt.astimezone(self._market_tz())
            minutes = et.hour * 60 + et.minute
            if minutes < 9 * 60 + 30 or minutes > 16 * 60:
                return "outside"
            if minutes <= 10 * 60 + 30:
                return "open"
            if minutes >= 15 * 60:
                return "close"
            return "midday"
        except Exception:
            return "unknown"

    def _research_label_filters(self) -> dict[str, Any]:
        buckets = tuple(
            s.strip().lower()
            for s in str(self._settings.research_allowed_session_buckets or "").split(",")
            if s.strip()
        )
        return {
            "max_quote_age_seconds": float(self._settings.research_max_quote_age_seconds),
            "allowed_session_buckets": buckets or None,
        }

    def _week_key_et(self, now_utc: datetime) -> str:
        et = now_utc.astimezone(self._market_tz())
        iso = et.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"

    def _run_weekly_cell_pruning_if_due(self, *, force: bool = False) -> dict[str, Any] | None:
        now = datetime.now(tz=timezone.utc)
        state = self._load_automation_state()
        week_key = self._week_key_et(now)
        if (not force) and str(state.get("weekly_pruning_last_week", "") or "") == week_key:
            return None
        report = self.prediction_quality_report(
            lookback=100000,
            horizons_minutes=(15,),
            min_confidence=0.0,
            quality_mode="good_only",
        )
        horizons = dict(report.get("horizons") or {})
        h15 = dict(horizons.get("15") or {})
        by_symbol = dict(h15.get("by_symbol") or {})
        cells: list[tuple[str, float]] = []
        for sym, m in by_symbol.items():
            n = int((m or {}).get("count", 0) or 0)
            if n < 20:
                continue
            signed = float((m or {}).get("avg_signed_return_bps", 0.0) or 0.0)
            cells.append((str(sym).upper(), signed))
        cells.sort(key=lambda x: x[1])
        cut = int(math.floor(len(cells) * 0.30))
        blocked = sorted({sym for sym, _ in cells[:cut]})
        result = {
            "ok": True,
            "week_key_et": week_key,
            "evaluated_cells": int(len(cells)),
            "blocked_symbols": blocked,
        }
        state["weekly_pruning_last_week"] = week_key
        state["auto_pruned_symbols"] = blocked
        state["weekly_pruning_last"] = result
        self._save_automation_state(state)
        self._notify("weekly_cell_pruning_completed", result)
        return result

    def _run_weekly_drift_rollback_if_due(self, *, force: bool = False) -> dict[str, Any] | None:
        now = datetime.now(tz=timezone.utc)
        state = self._load_automation_state()
        week_key = self._week_key_et(now)
        if (not force) and str(state.get("weekly_drift_last_week", "") or "") == week_key:
            return None
        monitor = self._evaluate_model_monitoring()
        report = dict(monitor.get("report") or {})
        breached = bool(report.get("breached", False)) and bool(report.get("sufficient_sample", False))
        rolled_back = False
        reason = "stable"
        if breached:
            self.set_acceleration_mode("standard")
            stable = dict(state.get("stable_controls") or {})
            if isinstance(stable.get("auto_pruned_symbols"), list):
                state["auto_pruned_symbols"] = list(stable.get("auto_pruned_symbols") or [])
            rolled_back = True
            reason = "model_drift_breached"
        else:
            state["stable_controls"] = {
                "captured_at": now.isoformat(),
                "auto_pruned_symbols": list(state.get("auto_pruned_symbols") or []),
                "acceleration_mode": "standard",
            }
        result = {
            "ok": True,
            "week_key_et": week_key,
            "rolled_back": rolled_back,
            "reason": reason,
            "monitoring": monitor,
        }
        state["weekly_drift_last_week"] = week_key
        state["weekly_drift_last"] = result
        self._save_automation_state(state)
        if rolled_back:
            self._notify("weekly_drift_rollback_applied", result)
        return result

    def _evaluate_data_quality_guard(self, *, force: bool = False) -> dict[str, Any]:
        now = datetime.now(tz=timezone.utc)
        if not bool(self._settings.automation_quality_checks_enabled):
            self._quality_guard_state = {
                "evaluated_at": now.isoformat(),
                "enabled": False,
                "active": False,
                "reason": "disabled",
                "metrics": {},
            }
            return dict(self._quality_guard_state)
        last_raw = self._quality_guard_state.get("evaluated_at")
        if not force and last_raw:
            try:
                last = datetime.fromisoformat(str(last_raw))
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                if (now - last).total_seconds() < float(max(30, int(self._settings.automation_quality_check_interval_seconds))):
                    return dict(self._quality_guard_state)
            except Exception:
                pass

        window_mins = max(5, int(self._settings.automation_quality_window_minutes))
        start = now - timedelta(minutes=window_mins)
        rows = self._persistence.list_data_samples(20000)
        total = 0
        stale = 0
        missing_forecast = 0
        bad = 0
        for r in rows:
            try:
                ts = datetime.fromisoformat(str(r.get("timestamp") or ""))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except Exception:
                continue
            if ts < start:
                continue
            total += 1
            md = dict(r.get("metadata") or {})
            sq = dict(md.get("sample_quality") or {})
            flags = dict(sq.get("flags") or {})
            if bool(flags.get("quote_stale", False)):
                stale += 1
            if bool(flags.get("missing_forecast", False)):
                missing_forecast += 1
            if not bool(sq.get("good", True)):
                bad += 1

        base_min_samples = max(1, int(self._settings.automation_quality_min_samples))
        et = now.astimezone(self._market_tz())
        minutes = et.hour * 60 + et.minute
        in_session = (9 * 60 + 30) <= minutes <= (16 * 60)
        if in_session:
            min_samples = base_min_samples
        else:
            factor = max(0.1, min(1.0, float(self._settings.automation_quality_min_samples_offsession_factor)))
            floor = max(1, int(self._settings.automation_quality_min_samples_floor))
            min_samples = max(floor, int(round(base_min_samples * factor)))
        stale_ratio = stale / float(max(1, total))
        missing_ratio = missing_forecast / float(max(1, total))
        bad_ratio = bad / float(max(1, total))
        breaches: list[str] = []
        enforce_volume = bool(in_session or bool(self._settings.automation_quality_enforce_volume_offsession))
        if enforce_volume and total < min_samples:
            breaches.append(f"low_sample_volume:{total}<{min_samples}")
        if stale_ratio > float(self._settings.automation_quality_max_stale_quote_ratio):
            breaches.append(
                f"stale_quote_ratio:{stale_ratio:.2f}>{float(self._settings.automation_quality_max_stale_quote_ratio):.2f}"
            )
        if missing_ratio > float(self._settings.automation_quality_max_missing_forecast_ratio):
            breaches.append(
                f"missing_forecast_ratio:{missing_ratio:.2f}>{float(self._settings.automation_quality_max_missing_forecast_ratio):.2f}"
            )
        if bad_ratio > float(self._settings.automation_quality_max_bad_sample_ratio):
            breaches.append(
                f"bad_sample_ratio:{bad_ratio:.2f}>{float(self._settings.automation_quality_max_bad_sample_ratio):.2f}"
            )
        active = len(breaches) > 0
        self._quality_guard_state = {
            "evaluated_at": now.isoformat(),
            "enabled": True,
            "active": active,
            "reason": "; ".join(breaches) if breaches else "ok",
            "metrics": {
                "window_minutes": window_mins,
                "samples": total,
                "base_min_samples": int(base_min_samples),
                "effective_min_samples": int(min_samples),
                "in_session": bool(in_session),
                "volume_enforced": bool(enforce_volume),
                "stale_quote_ratio": stale_ratio,
                "missing_forecast_ratio": missing_ratio,
                "bad_sample_ratio": bad_ratio,
                "throttle_cycle_seconds": int(self._settings.automation_quality_throttle_cycle_seconds),
            },
        }
        return dict(self._quality_guard_state)

    def autonomy_error_causes_report(self, *, lookback: int = 5000, window_minutes: int = 1440) -> dict[str, Any]:
        now = datetime.now(tz=timezone.utc)
        start = now - timedelta(minutes=max(15, int(window_minutes)))
        rows = self._persistence.list_data_samples(max(100, min(100000, int(lookback))))
        decisions = self._persistence.list_decisions(max(100, min(100000, int(lookback))))
        causes: dict[str, int] = {}

        def bump(name: str) -> None:
            causes[name] = int(causes.get(name, 0)) + 1

        considered = 0
        for r in rows:
            raw_ts = str(r.get("timestamp") or r.get("ts") or "")
            try:
                ts = datetime.fromisoformat(raw_ts)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except Exception:
                continue
            if ts < start:
                continue
            considered += 1
            if bool(r.get("quality_quote_stale", False)):
                bump("stale_quote")
            if bool(r.get("quality_missing_forecast", False)):
                bump("missing_forecast")
            note = str(r.get("note") or "").lower()
            reasoning = str(r.get("reasoning") or "").lower()
            if "llm unavailable" in note or "llm unavailable" in reasoning:
                bump("llm_unavailable")
            if note.startswith("guard hold"):
                bump("guard_hold")
                if "session gate" in note:
                    bump("guard_hold_session")
                elif "data quality guard" in note or "data quality" in note:
                    bump("guard_hold_quality")
                elif "cost guard" in note or "cost budget" in note:
                    bump("guard_hold_cost")
                elif "symbol gate" in note:
                    bump("guard_hold_symbol_gate")
                elif "signed-bps kill switch" in note:
                    bump("guard_hold_signed_bps_kill")
                elif "model decay safe mode" in note or "model monitoring safe mode" in note:
                    bump("guard_hold_model_decay")
                elif "regime gate" in note:
                    bump("guard_hold_regime_gate")
                elif "horizon gate" in note:
                    bump("guard_hold_horizon_gate")
                elif "broker sync stale" in note:
                    bump("guard_hold_broker_sync")
                else:
                    bump("guard_hold_other")

        for d in decisions:
            raw_ts = str(d.get("timestamp") or d.get("ts") or "")
            try:
                ts = datetime.fromisoformat(raw_ts)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except Exception:
                continue
            if ts < start:
                continue
            note = str(d.get("note") or "").lower()
            if "cycle execution failed" in note:
                bump("cycle_execution_failed")

        ranked = sorted(
            [{"cause": k, "count": int(v), "share": (float(v) / float(max(1, considered)))} for k, v in causes.items()],
            key=lambda x: x["count"],
            reverse=True,
        )
        hard_error_causes = [
            c for c in ranked
            if not str(c.get("cause", "")).startswith("guard_hold")
        ]
        return {
            "ok": True,
            "window_minutes": int(window_minutes),
            "considered_samples": int(considered),
            "top_cause": ranked[0] if ranked else {"cause": "none", "count": 0, "share": 0.0},
            "top_hard_error_cause": hard_error_causes[0] if hard_error_causes else {"cause": "none", "count": 0, "share": 0.0},
            "causes": ranked,
            "hard_error_causes": hard_error_causes,
        }

    def _evaluate_cost_guard(self, *, force: bool = False) -> dict[str, Any]:
        now = datetime.now(tz=timezone.utc)
        if not bool(self._settings.automation_cost_guard_enabled):
            self._cost_guard_state = {
                "evaluated_at": now.isoformat(),
                "enabled": False,
                "active": False,
                "warning": False,
                "reason": "disabled",
                "metrics": {},
            }
            return dict(self._cost_guard_state)
        last_raw = self._cost_guard_state.get("evaluated_at")
        if not force and last_raw:
            try:
                last = datetime.fromisoformat(str(last_raw))
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                if (now - last).total_seconds() < 60.0:
                    return dict(self._cost_guard_state)
            except Exception:
                pass

        day_et = self._current_trading_day_et_from(now)
        rows = self._persistence.list_data_samples(100000)
        primary_calls = 0
        secondary_calls = 0
        total_samples = 0
        for r in rows:
            try:
                ts = datetime.fromisoformat(str(r.get("timestamp") or ""))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except Exception:
                continue
            if ts.astimezone(self._market_tz()).date().isoformat() != day_et:
                continue
            total_samples += 1
            md = dict(r.get("metadata") or {})
            lr = dict(md.get("llm_routing") or {})
            if bool(lr):
                if "primary_call_count" in lr or "secondary_call_count" in lr:
                    primary_calls += self._to_non_negative_int(lr.get("primary_call_count", 0))
                    secondary_calls += self._to_non_negative_int(lr.get("secondary_call_count", 0))
                else:
                    if not bool(lr.get("state_gate_used_cache", False)):
                        primary_calls += 1
                    if bool(lr.get("secondary_invoked", False)):
                        secondary_calls += 1

        est = (
            float(primary_calls) * float(self._settings.automation_cost_est_primary_call_usd)
            + float(secondary_calls) * float(self._settings.automation_cost_est_secondary_call_usd)
        )
        budget = max(0.01, float(self._settings.automation_cost_daily_budget_usd))
        warn_threshold = budget * max(0.1, min(1.0, float(self._settings.automation_cost_warn_fraction)))
        warning = est >= warn_threshold
        active = bool(self._settings.automation_cost_hard_block) and est >= budget
        reason = "budget_exceeded" if active else ("near_budget" if warning else "ok")
        self._cost_guard_state = {
            "evaluated_at": now.isoformat(),
            "enabled": True,
            "active": active,
            "warning": warning,
            "reason": reason,
            "metrics": {
                "trading_day_et": day_et,
                "sample_rows": total_samples,
                "estimated_primary_calls": primary_calls,
                "estimated_secondary_calls": secondary_calls,
                "estimated_cost_usd": est,
                "daily_budget_usd": budget,
                "warning_threshold_usd": warn_threshold,
            },
        }
        return dict(self._cost_guard_state)

    def _evaluate_sample_flow_guard(self, *, force: bool = False) -> dict[str, Any]:
        now = datetime.now(tz=timezone.utc)
        if not bool(self._settings.sample_flow_guard_enabled):
            self._sample_flow_guard_state = {
                "evaluated_at": now.isoformat(),
                "enabled": False,
                "active": False,
                "reason": "disabled",
                "metrics": {},
            }
            return dict(self._sample_flow_guard_state)
        last_raw = self._sample_flow_guard_state.get("evaluated_at")
        if not force and last_raw:
            try:
                last = datetime.fromisoformat(str(last_raw))
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                if (now - last).total_seconds() < 60.0:
                    return dict(self._sample_flow_guard_state)
            except Exception:
                pass
        et = now.astimezone(self._market_tz())
        minutes = et.hour * 60 + et.minute
        in_session = (9 * 60 + 30) <= minutes <= (16 * 60)
        rows = self._persistence.list_data_samples(2000)
        latest_ts: datetime | None = None
        for r in rows:
            try:
                ts = datetime.fromisoformat(str(r.get("timestamp") or ""))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except Exception:
                continue
            if latest_ts is None or ts > latest_ts:
                latest_ts = ts
        stall_minutes = max(1, int(self._settings.sample_flow_guard_stall_minutes_in_session))
        age_minutes = None if latest_ts is None else ((now - latest_ts).total_seconds() / 60.0)
        active = bool(in_session and (age_minutes is None or age_minutes > float(stall_minutes)))
        reason = "ok"
        if active:
            if age_minutes is None:
                reason = f"no_samples_seen_in_session>{stall_minutes}m"
            else:
                reason = f"sample_stall_in_session:{age_minutes:.1f}m>{stall_minutes}m"
        self._sample_flow_guard_state = {
            "evaluated_at": now.isoformat(),
            "enabled": True,
            "active": active,
            "reason": reason,
            "metrics": {
                "in_session": bool(in_session),
                "stall_minutes_threshold": int(stall_minutes),
                "latest_sample_at": latest_ts.isoformat() if latest_ts else None,
                "latest_sample_age_minutes": (None if age_minutes is None else float(age_minutes)),
            },
        }
        return dict(self._sample_flow_guard_state)

    def _sample_flow_auto_recovery(self, *, force: bool = False) -> dict[str, Any]:
        guard = self._evaluate_sample_flow_guard(force=force)
        now = datetime.now(tz=timezone.utc)
        if not bool(guard.get("enabled", False)):
            return {"ok": True, "attempted": False, "reason": "guard_disabled"}
        if not bool(guard.get("active", False)):
            return {"ok": True, "attempted": False, "reason": "no_stall"}
        state = self._load_automation_state()
        last_raw = str(state.get("sample_flow_recovery_last_at", "") or "")
        if last_raw:
            try:
                last = datetime.fromisoformat(last_raw)
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                if (now - last).total_seconds() < 300.0:
                    return {"ok": True, "attempted": False, "reason": "cooldown"}
            except Exception:
                pass
        restarted = False
        run_once_ok = False
        if not bool(self.status().running):
            start_result = self.start_with_guard()
            restarted = bool(start_result.get("started", False))
        try:
            _ = self.run_cycle_once()
            run_once_ok = True
        except Exception:
            run_once_ok = False
        state["sample_flow_recovery_last_at"] = now.isoformat()
        state["sample_flow_recovery_last"] = {
            "at": now.isoformat(),
            "guard_reason": guard.get("reason"),
            "restarted_worker": restarted,
            "run_once_ok": run_once_ok,
        }
        self._save_automation_state(state)
        self._notify("sample_flow_recovery_attempt", state["sample_flow_recovery_last"])
        return {"ok": True, "attempted": True, **state["sample_flow_recovery_last"]}

    def _auto_recovery_check(self, *, force: bool = False) -> dict[str, Any]:
        now = datetime.now(tz=timezone.utc)
        if not bool(self._settings.automation_auto_recovery_enabled):
            return {"enabled": False, "attempted": False, "recovered": False, "reason": "disabled"}
        state = self._load_automation_state()
        last_raw = str(state.get("auto_recovery_last_at", "") or "")
        if not force and last_raw:
            try:
                last = datetime.fromisoformat(last_raw)
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                if (now - last).total_seconds() < float(max(30, int(self._settings.automation_auto_recovery_cooldown_seconds))):
                    return {"enabled": True, "attempted": False, "recovered": False, "reason": "cooldown"}
            except Exception:
                pass
        running = bool(self._thread and self._thread.is_alive())
        attempted = False
        recovered = False
        reason = "worker_running"
        if (not running) and bool(self._settings.auto_start_worker):
            attempted = True
            started = self.start_with_guard()
            recovered = bool(started.get("started", False))
            reason = "worker_restart_success" if recovered else str(started.get("reason", "worker_restart_failed"))
            state["auto_recovery_last_result"] = started
        state["auto_recovery_last_at"] = now.isoformat()
        state["auto_recovery_last_reason"] = reason
        self._save_automation_state(state)
        if attempted:
            self._notify(
                "auto_recovery_attempt",
                {"attempted": attempted, "recovered": recovered, "reason": reason},
            )
        return {"enabled": True, "attempted": attempted, "recovered": recovered, "reason": reason}

    def _evaluate_coverage_guard(self, *, force: bool = False) -> dict[str, Any]:
        if not bool(self._settings.coverage_guard_enabled):
            self._coverage_guard_state = {
                "active": False,
                "daily_labels": 0,
                "target": int(self._settings.coverage_guard_daily_label_target),
                "label_factor": 1.0,
            }
            return dict(self._coverage_guard_state)
        rows = self._persistence.list_data_samples(100000)
        now = datetime.now(tz=timezone.utc)
        start = now - timedelta(days=1)
        recent = []
        for r in rows:
            try:
                ts = datetime.fromisoformat(str(r.get("timestamp") or ""))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts >= start:
                    recent.append(r)
            except Exception:
                continue
        labels = build_prediction_labels(
            list(reversed(recent)),
            horizons_minutes=(15,),
            min_confidence=0.0,
            quality_mode="good_only",
            **self._research_label_filters(),
        ).get("labels_by_horizon", {}).get(15, [])
        daily_labels = len(labels)
        target = max(1, int(self._settings.coverage_guard_daily_label_target))
        active = daily_labels < target
        factor = float(self._settings.coverage_guard_relax_label_factor) if active else 1.0
        factor = max(0.3, min(1.0, factor))
        self._coverage_guard_state = {
            "active": active,
            "daily_labels": daily_labels,
            "target": target,
            "label_factor": factor,
        }
        return dict(self._coverage_guard_state)

    def _resolve_policy_tier_from_coverage(self, coverage: dict[str, Any]) -> str:
        if not bool(self._settings.robust_policy_tiering_enabled):
            return "balanced"
        daily_labels = max(0.0, float((coverage or {}).get("daily_labels", 0.0) or 0.0))
        per_hour = daily_labels / 24.0
        strict_cut = max(0.01, float(self._settings.robust_policy_tier_coverage_hours_strict))
        balanced_cut = max(0.01, float(self._settings.robust_policy_tier_coverage_hours_balanced))
        if per_hour >= strict_cut:
            return "strict"
        if per_hour >= balanced_cut:
            return "balanced"
        return "explore"

    def _tier_parameters(self, tier: str) -> dict[str, Any]:
        normalized = str(tier or "balanced").strip().lower()
        min_labels_default = max(1, int(self._settings.robust_allowlist_min_labels))
        stress_default = max(0.0, float(self._settings.robust_allowlist_cost_stress_multiplier))
        table = {
            "strict": {
                "min_labels": max(1, int(self._settings.robust_policy_tier_strict_min_labels)),
                "stress_mult": max(0.0, float(self._settings.robust_policy_tier_strict_cost_stress_multiplier)),
            },
            "balanced": {
                "min_labels": max(1, int(self._settings.robust_policy_tier_balanced_min_labels)),
                "stress_mult": max(0.0, float(self._settings.robust_policy_tier_balanced_cost_stress_multiplier)),
            },
            "explore": {
                "min_labels": max(1, int(self._settings.robust_policy_tier_explore_min_labels)),
                "stress_mult": max(0.0, float(self._settings.robust_policy_tier_explore_cost_stress_multiplier)),
            },
        }
        out = dict(table.get(normalized) or {})
        if not out:
            out = {"min_labels": min_labels_default, "stress_mult": stress_default}
        out["tier"] = normalized if normalized in table else "balanced"
        return out

    def _evaluate_confidence_controls(self, *, force: bool = False) -> dict[str, Any]:
        now = datetime.now(tz=timezone.utc)
        if not bool(self._settings.confidence_control_enabled):
            controls = {
                "enabled": False,
                "default_min_confidence": float(self._settings.confidence_control_default_min_confidence),
                "global_min_confidence": float(self._settings.confidence_control_default_min_confidence),
                "symbol_min_confidence": {},
                "symbol_calibration": {},
                "calibration_gate_enabled": bool(self._settings.calibration_gate_enabled),
                "calibration_gate_stress_bps": float(self._settings.calibration_gate_stress_bps),
                "symbol_allowed_confidence_bins": {},
                "symbol_bin_stressed_bps": {},
            }
            self._engine.set_confidence_controls(controls)
            self._confidence_control_state.update(
                {"evaluated_at": now.isoformat(), "enabled": False, "controls": controls, "report": {}}
            )
            return dict(self._confidence_control_state)

        last_raw = self._confidence_control_state.get("evaluated_at")
        if not force and last_raw:
            try:
                last = datetime.fromisoformat(str(last_raw))
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                if (now - last).total_seconds() < float(max(10, int(self._settings.confidence_control_refresh_seconds))):
                    controls = dict(self._confidence_control_state.get("controls") or {})
                    if controls:
                        self._engine.set_confidence_controls(controls)
                    return dict(self._confidence_control_state)
            except Exception:
                pass

        rows = self._persistence.list_data_samples(100000)
        report = build_confidence_control_report(
            list(reversed(rows)),
            horizon_minutes=max(1, min(390, int(self._settings.confidence_control_horizon_minutes))),
            quality_mode="good_only",
            min_bin_count=max(1, int(self._settings.confidence_control_min_bin_count)),
            min_symbol_labels=max(1, int(self._settings.confidence_control_min_symbol_labels)),
            min_threshold_labels=max(1, int(self._settings.confidence_control_min_threshold_labels)),
            default_min_confidence=max(0.0, min(1.0, float(self._settings.confidence_control_default_min_confidence))),
            **self._research_label_filters(),
        )
        symbol_controls = dict(report.get("symbol_controls") or {})
        stress_bps = float(self._settings.calibration_gate_stress_bps)
        symbol_allowed_confidence_bins: dict[str, list[str]] = {}
        symbol_bin_stressed_bps: dict[str, dict[str, float]] = {}
        for sym, cfg in symbol_controls.items():
            sym_u = str(sym).upper()
            cal = dict((cfg or {}).get("calibration") or {})
            stressed_map: dict[str, float] = {}
            allowed_bins: list[str] = []
            for bin_key, bin_row in cal.items():
                signed = float((bin_row or {}).get("avg_signed_return_bps", 0.0) or 0.0)
                stressed = signed - stress_bps
                stressed_map[str(bin_key)] = float(stressed)
                if stressed > 0.0:
                    allowed_bins.append(str(bin_key))
            symbol_allowed_confidence_bins[sym_u] = sorted(set(allowed_bins))
            symbol_bin_stressed_bps[sym_u] = stressed_map
        controls = {
            "enabled": True,
            "default_min_confidence": float(report.get("default_min_confidence", 0.55)),
            "global_min_confidence": float(report.get("global_threshold", report.get("default_min_confidence", 0.55))),
            "symbol_min_confidence": {
                str(sym).upper(): float((cfg or {}).get("min_confidence", report.get("default_min_confidence", 0.55)))
                for sym, cfg in symbol_controls.items()
            },
            "symbol_calibration": {
                str(sym).upper(): {
                    str(k): float((v or {}).get("calibrated_confidence", 0.0))
                    for k, v in dict((cfg or {}).get("calibration") or {}).items()
                }
                for sym, cfg in symbol_controls.items()
            },
            "calibration_gate_enabled": bool(self._settings.calibration_gate_enabled),
            "calibration_gate_stress_bps": stress_bps,
            "symbol_allowed_confidence_bins": symbol_allowed_confidence_bins,
            "symbol_bin_stressed_bps": symbol_bin_stressed_bps,
        }
        self._engine.set_confidence_controls(controls)
        self._confidence_control_state.update(
            {
                "evaluated_at": now.isoformat(),
                "enabled": True,
                "controls": controls,
                "report": report,
            }
        )
        return dict(self._confidence_control_state)

    def _evaluate_regime_gate(self, *, force: bool = False) -> dict[str, Any]:
        now = datetime.now(tz=timezone.utc)
        coverage = self._evaluate_coverage_guard()
        if not bool(self._settings.regime_gate_enabled):
            controls = {"enabled": False, "allowed_pairs": {}}
            self._engine.set_regime_controls(controls)
            self._regime_gate_state.update(
                {"evaluated_at": now.isoformat(), "enabled": False, "controls": controls, "report": {}}
            )
            return dict(self._regime_gate_state)

        last_raw = self._regime_gate_state.get("evaluated_at")
        if not force and last_raw:
            try:
                last = datetime.fromisoformat(str(last_raw))
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                if (now - last).total_seconds() < float(max(10, int(self._settings.regime_gate_refresh_seconds))):
                    controls = dict(self._regime_gate_state.get("controls") or {})
                    if controls:
                        self._engine.set_regime_controls(controls)
                    return dict(self._regime_gate_state)
            except Exception:
                pass

        if bool(self._settings.robust_allowlist_enabled):
            policy = self._build_robust_allowlist_policy(force=force)
            controls = {
                "enabled": True,
                "allowed_pairs": dict(policy.get("allowed_pairs") or {}),
            }
            self._engine.set_regime_controls(controls)
            self._regime_gate_state.update(
                {
                    "evaluated_at": now.isoformat(),
                    "enabled": True,
                    "controls": controls,
                    "report": {
                        "mode": "robust_allowlist",
                        "policy": policy,
                    },
                }
            )
            return dict(self._regime_gate_state)

        horizon = max(1, min(390, int(self._settings.regime_gate_horizon_minutes)))
        rows = self._persistence.list_data_samples(100000)
        labels = build_prediction_labels(
            list(reversed(rows)),
            horizons_minutes=(horizon,),
            min_confidence=0.0,
            quality_mode="good_only",
            **self._research_label_filters(),
        ).get("labels_by_horizon", {}).get(horizon, [])
        min_labels_base = max(1, int(self._settings.regime_gate_min_labels))
        min_labels = max(1, int(round(min_labels_base * float(coverage.get("label_factor", 1.0)))))
        min_signed = float(self._settings.regime_gate_min_signed_bps)
        min_acc = float(self._settings.regime_gate_min_accuracy)
        combo: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for r in labels:
            sym = str(r.get("symbol") or "").upper()
            reg = str(r.get("indicator_regime") or "unknown").strip().lower()
            bucket = self._session_bucket_from_ts(str(r.get("timestamp") or ""))
            if not sym:
                continue
            combo.setdefault((sym, reg, bucket), []).append(r)
        allowed_pairs: dict[str, set[str]] = {}
        stats: list[dict[str, Any]] = []
        for (sym, reg, session_bucket), cell_rows in combo.items():
            n = len(cell_rows)
            wins = sum(1.0 for b in cell_rows if float(b.get("signed_return_bps", 0.0)) > 0.0)
            acc = wins / float(max(1, n))
            signed = sum(float(b.get("signed_return_bps", 0.0)) for b in cell_rows) / float(max(1, n))
            allow = n >= min_labels and signed >= min_signed and acc >= min_acc
            if allow:
                allowed_pairs.setdefault(sym, set()).add(f"{reg}|{session_bucket}")
            stats.append(
                {
                    "symbol": sym,
                    "regime": reg,
                    "session_bucket": session_bucket,
                    "count": n,
                    "accuracy": acc,
                    "avg_signed_return_bps": signed,
                    "allow": allow,
                }
            )
        controls = {
            "enabled": True,
            "allowed_pairs": {k: sorted(v) for k, v in allowed_pairs.items()},
        }
        self._engine.set_regime_controls(controls)
        self._regime_gate_state.update(
            {
                "evaluated_at": now.isoformat(),
                "enabled": True,
                "controls": controls,
                "report": {
                    "horizon_minutes": horizon,
                    "coverage_guard": coverage,
                    "min_labels_base": min_labels_base,
                    "min_labels": min_labels,
                    "min_signed_bps": min_signed,
                    "min_accuracy": min_acc,
                    "rows": sorted(stats, key=lambda x: (not x["allow"], -x["avg_signed_return_bps"])),
                },
            }
        )
        return dict(self._regime_gate_state)

    def _build_robust_allowlist_policy(self, *, force: bool = False) -> dict[str, Any]:
        now = datetime.now(tz=timezone.utc)
        coverage = self._evaluate_coverage_guard()
        active_tier = self._resolve_policy_tier_from_coverage(coverage)
        tier_params = self._tier_parameters(active_tier)
        state = self._load_automation_state()
        if str(state.get("policy_tier_active") or "") != active_tier:
            state["policy_tier_last_changed_at"] = now.isoformat()
        state["policy_tier_active"] = active_tier
        self._save_automation_state(state)
        horizons = tuple(
            max(1, min(390, int(x.strip())))
            for x in str(self._settings.best_horizon_choices or "5,15,30").split(",")
            if x.strip()
        ) or (5, 15, 30)
        min_labels = max(1, int(tier_params.get("min_labels", self._settings.robust_allowlist_min_labels)))
        n_boot = max(50, int(self._settings.robust_allowlist_n_bootstrap))
        stress_mult = max(0.0, float(tier_params.get("stress_mult", self._settings.robust_allowlist_cost_stress_multiplier)))
        base_rtt = 2.0 * float(self._settings.cost_slippage_bps_per_side)
        stress_cost = base_rtt * stress_mult

        oos = self.champion_challenger_daily_report(
            lookback=100000,
            horizon_minutes=max(1, int(self._settings.robust_allowlist_horizon_minutes)),
            quality_mode="good_only",
            min_train_labels=150,
            min_cell_labels=max(8, min_labels // 2),
            challenger_min_confidence=0.55,
            min_daily_selections=10,
        )
        oos_positive = float(oos.get("overall_delta_signed_bps", 0.0) or 0.0) > 0.0
        oos_ok = bool(oos_positive) if bool(self._settings.robust_allowlist_require_oos_positive) else True

        boot = self.cell_leaderboard_bootstrap_report(
            lookback=100000,
            horizons_minutes=horizons,
            quality_mode="good_only",
            min_labels=min_labels,
            n_bootstrap=n_boot,
            robust_only=False,
        )
        rows = list(boot.get("rows") or [])
        allowed_pairs: dict[str, set[str]] = {}
        symbol_best: dict[str, tuple[int, float]] = {}
        robust_cells: list[dict[str, Any]] = []
        for r in rows:
            sym = str(r.get("symbol") or "").upper()
            reg = str(r.get("regime") or "unknown").strip().lower()
            sess = str(r.get("session_bucket") or "unknown").strip().lower()
            h = max(1, int(r.get("horizon_minutes", 0) or 0))
            n = int(r.get("count", 0) or 0)
            ci_low = float(r.get("ci95_low_signed_bps", 0.0) or 0.0)
            avg = float(r.get("avg_signed_return_bps", 0.0) or 0.0)
            stressed = avg - stress_cost
            robust = bool(r.get("robust_positive", False))
            allow = bool(sym and robust and n >= min_labels and ci_low > 0.0 and stressed > 0.0 and oos_ok)
            if not allow:
                continue
            allowed_pairs.setdefault(sym, set()).add(f"{reg}|{sess}")
            cur = symbol_best.get(sym)
            if cur is None or stressed > float(cur[1]):
                symbol_best[sym] = (h, stressed)
            robust_cells.append(
                {
                    "symbol": sym,
                    "regime": reg,
                    "session_bucket": sess,
                    "horizon_minutes": h,
                    "count": n,
                    "ci95_low_signed_bps": ci_low,
                    "avg_signed_return_bps": avg,
                    "stressed_signed_bps": stressed,
                }
            )

        cooldown_report = self._update_cell_cooldowns(
            horizon_minutes=max(1, int(self._settings.robust_allowlist_horizon_minutes)),
            force=force,
        )
        quarantine_report = self._update_symbol_quarantines(force=force)
        quarantined_symbols = {
            str(s).upper()
            for s in list((quarantine_report.get("active") or {}).keys())
            if str(s).strip()
        }
        active_cooldowns = set(str(k) for k in list((cooldown_report.get("active") or {}).keys()))
        filtered_pairs: dict[str, list[str]] = {}
        for sym, pairs in allowed_pairs.items():
            if sym in quarantined_symbols:
                continue
            keep = []
            for pair in sorted(pairs):
                key = f"{sym}|{pair}"
                if key in active_cooldowns:
                    continue
                keep.append(pair)
            if keep:
                filtered_pairs[sym] = keep
        filtered_symbol_best = {
            sym: int(v[0]) for sym, v in symbol_best.items()
            if sym in filtered_pairs
        }
        allowed_symbols = sorted(filtered_pairs.keys())
        all_symbols = sorted({str(r.get("symbol") or "").upper() for r in rows if str(r.get("symbol") or "").strip()})
        blocked_symbols = sorted([s for s in all_symbols if s and s not in set(allowed_symbols)])
        return {
            "generated_at": now.isoformat(),
            "policy_tier": active_tier,
            "policy_tiering_enabled": bool(self._settings.robust_policy_tiering_enabled),
            "policy_tier_params": dict(tier_params),
            "coverage_guard": coverage,
            "coverage_labels_per_hour": float(max(0.0, float((coverage or {}).get("daily_labels", 0.0)) / 24.0)),
            "oos_ok": bool(oos_ok),
            "oos_delta_signed_bps": float(oos.get("overall_delta_signed_bps", 0.0) or 0.0),
            "stress_cost_bps": float(stress_cost),
            "min_labels": int(min_labels),
            "allowed_pairs": filtered_pairs,
            "symbol_best_horizon": filtered_symbol_best,
            "allowed_symbols": allowed_symbols,
            "blocked_symbols": blocked_symbols,
            "cooldowns": cooldown_report,
            "symbol_quarantine": quarantine_report,
            "robust_cells": robust_cells[:200],
        }

    def _update_cell_cooldowns(self, *, horizon_minutes: int, force: bool = False) -> dict[str, Any]:
        state = self._load_automation_state()
        now = datetime.now(tz=timezone.utc)
        active = dict(state.get("cell_cooldowns") or {})
        # prune expired
        pruned: dict[str, Any] = {}
        for key, payload in active.items():
            until_raw = str((payload or {}).get("until") or "")
            try:
                until = datetime.fromisoformat(until_raw)
                if until.tzinfo is None:
                    until = until.replace(tzinfo=timezone.utc)
                if until > now:
                    pruned[key] = payload
            except Exception:
                continue
        active = pruned
        if bool(self._settings.cell_cooldown_enabled):
            rows = self._persistence.list_data_samples(100000)
            labels = build_prediction_labels(
                list(reversed(rows)),
                horizons_minutes=(int(horizon_minutes),),
                min_confidence=0.0,
                quality_mode="good_only",
                **self._research_label_filters(),
            ).get("labels_by_horizon", {}).get(int(horizon_minutes), [])
            window_start = now - timedelta(days=1)
            cell_rows: dict[str, list[float]] = {}
            for r in labels:
                try:
                    ts = datetime.fromisoformat(str(r.get("timestamp") or ""))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                except Exception:
                    continue
                if ts < window_start:
                    continue
                sym = str(r.get("symbol") or "").upper()
                reg = str(r.get("indicator_regime") or "unknown").strip().lower()
                sess = self._session_bucket_from_ts(str(r.get("timestamp") or ""))
                key = f"{sym}|{reg}|{sess}"
                cell_rows.setdefault(key, []).append(float(r.get("signed_return_bps", 0.0) or 0.0))
            min_n = max(1, int(self._settings.cell_cooldown_min_labels))
            kill = float(self._settings.cell_cooldown_signed_bps_kill)
            mins = max(5, int(self._settings.cell_cooldown_minutes))
            for key, vals in cell_rows.items():
                if len(vals) < min_n:
                    continue
                avg = sum(vals) / float(len(vals))
                if avg <= kill:
                    until = now + timedelta(minutes=mins)
                    active[key] = {
                        "set_at": now.isoformat(),
                        "until": until.isoformat(),
                        "avg_signed_bps": float(avg),
                        "count": int(len(vals)),
                        "reason": f"avg_signed_bps {avg:.2f} <= {kill:.2f}",
                    }
        state["cell_cooldowns"] = active
        self._save_automation_state(state)
        return {"active": active, "count": int(len(active))}

    def _update_symbol_quarantines(self, *, force: bool = False) -> dict[str, Any]:
        state = self._load_automation_state()
        now = datetime.now(tz=timezone.utc)
        active = dict(state.get("symbol_quarantines") or {})
        kept: dict[str, Any] = {}
        for sym, payload in active.items():
            until_raw = str((payload or {}).get("until") or "")
            try:
                until = datetime.fromisoformat(until_raw)
                if until.tzinfo is None:
                    until = until.replace(tzinfo=timezone.utc)
                if until > now:
                    kept[str(sym).upper()] = payload
            except Exception:
                continue
        active = kept
        if not bool(self._settings.symbol_quarantine_enabled):
            state["symbol_quarantines"] = active
            self._save_automation_state(state)
            return {"enabled": False, "active": active, "count": int(len(active))}

        horizon = max(1, int(self._settings.symbol_quarantine_horizon_minutes))
        rows = self._persistence.list_data_samples(100000)
        labels = build_prediction_labels(
            list(reversed(rows)),
            horizons_minutes=(horizon,),
            min_confidence=0.0,
            quality_mode="good_only",
            **self._research_label_filters(),
        ).get("labels_by_horizon", {}).get(horizon, [])
        window_start = now - timedelta(days=1)
        by_symbol: dict[str, list[float]] = {}
        for r in labels:
            try:
                ts = datetime.fromisoformat(str(r.get("timestamp") or ""))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except Exception:
                continue
            if ts < window_start:
                continue
            sym = str(r.get("symbol") or "").upper()
            if not sym:
                continue
            by_symbol.setdefault(sym, []).append(float(r.get("signed_return_bps", 0.0) or 0.0))

        min_n = max(1, int(self._settings.symbol_quarantine_min_labels))
        kill = float(self._settings.symbol_quarantine_signed_bps_kill)
        mins = max(5, int(self._settings.symbol_quarantine_minutes))
        rec_min_n = max(1, int(self._settings.symbol_quarantine_recover_min_labels))
        rec_signed = float(self._settings.symbol_quarantine_recover_signed_bps)
        recovered: list[str] = []
        for sym, payload in list(active.items()):
            vals = by_symbol.get(sym, [])
            if len(vals) < rec_min_n:
                continue
            avg = sum(vals) / float(len(vals))
            if avg >= rec_signed:
                active.pop(sym, None)
                recovered.append(sym)

        for sym, vals in by_symbol.items():
            if len(vals) < min_n:
                continue
            avg = sum(vals) / float(len(vals))
            if avg <= kill:
                until = now + timedelta(minutes=mins)
                active[sym] = {
                    "set_at": now.isoformat(),
                    "until": until.isoformat(),
                    "avg_signed_bps": float(avg),
                    "count": int(len(vals)),
                    "reason": f"avg_signed_bps {avg:.2f} <= {kill:.2f}",
                }
        state["symbol_quarantines"] = active
        state["symbol_quarantine_last"] = {
            "at": now.isoformat(),
            "horizon_minutes": int(horizon),
            "active_count": int(len(active)),
            "recovered": recovered,
        }
        self._save_automation_state(state)
        return {
            "enabled": True,
            "horizon_minutes": int(horizon),
            "active": active,
            "count": int(len(active)),
            "recovered": recovered,
        }

    def _evaluate_horizon_gate(self, *, force: bool = False) -> dict[str, Any]:
        now = datetime.now(tz=timezone.utc)
        auto_state = self._load_automation_state()
        quarantine_report = self._update_symbol_quarantines(force=force)
        blocked_symbols = {
            str(s).upper()
            for s in list(auto_state.get("auto_pruned_symbols") or [])
            if str(s).strip()
        }
        blocked_symbols.update(
            {
                str(s).upper()
                for s in list((quarantine_report.get("active") or {}).keys())
                if str(s).strip()
            }
        )
        if bool(self._settings.robust_allowlist_enabled):
            policy = self._build_robust_allowlist_policy(force=force)
            blocked_symbols.update({str(s).upper() for s in list(policy.get("blocked_symbols") or []) if str(s).strip()})
            controls = {
                "enabled": True,
                "symbol_best_horizon": dict(policy.get("symbol_best_horizon") or {}),
                "blocked_symbols": sorted(blocked_symbols),
            }
            self._engine.set_horizon_controls(controls)
            self._horizon_gate_state.update(
                {
                    "evaluated_at": now.isoformat(),
                    "enabled": True,
                    "controls": controls,
                    "report": {"mode": "robust_allowlist", "policy": policy, "blocked_symbols": sorted(blocked_symbols)},
                }
            )
            return dict(self._horizon_gate_state)
        coverage = self._evaluate_coverage_guard()
        if not bool(self._settings.best_horizon_gate_enabled):
            controls = {"enabled": False, "symbol_best_horizon": {}, "blocked_symbols": sorted(blocked_symbols)}
            self._engine.set_horizon_controls(controls)
            self._horizon_gate_state.update(
                {"evaluated_at": now.isoformat(), "enabled": False, "controls": controls, "report": {}}
            )
            return dict(self._horizon_gate_state)

        last_raw = self._horizon_gate_state.get("evaluated_at")
        if not force and last_raw:
            try:
                last = datetime.fromisoformat(str(last_raw))
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                if (now - last).total_seconds() < float(max(10, int(self._settings.best_horizon_gate_refresh_seconds))):
                    controls = dict(self._horizon_gate_state.get("controls") or {})
                    if controls:
                        self._engine.set_horizon_controls(controls)
                    return dict(self._horizon_gate_state)
            except Exception:
                pass

        horizon_choices = tuple(
            max(1, min(390, int(x.strip())))
            for x in str(self._settings.best_horizon_choices or "5,15,30").split(",")
            if x.strip()
        ) or (5, 15, 30)
        report = self.prediction_quality_report(
            lookback=100000,
            horizons_minutes=horizon_choices,
            min_confidence=0.0,
            quality_mode="good_only",
        )
        min_labels_base = max(1, int(self._settings.best_horizon_gate_min_labels))
        min_labels = max(1, int(round(min_labels_base * float(coverage.get("label_factor", 1.0)))))
        symbol_best: dict[str, int] = {}
        rows: list[dict[str, Any]] = []
        symbols = set()
        for h in horizon_choices:
            by_symbol = ((report.get("horizons") or {}).get(str(h), {}) or {}).get("by_symbol", {}) or {}
            symbols.update(str(s).upper() for s in by_symbol.keys())
        for sym in sorted(symbols):
            best_h = None
            best_signed = -1e18
            for h in horizon_choices:
                m = (((report.get("horizons") or {}).get(str(h), {}) or {}).get("by_symbol", {}) or {}).get(sym, {}) or {}
                n = int(m.get("count", 0) or 0)
                signed = float(m.get("avg_signed_return_bps", 0.0) or 0.0)
                rows.append({"symbol": sym, "horizon_minutes": h, "count": n, "avg_signed_return_bps": signed})
                if n >= min_labels and signed > best_signed:
                    best_signed = signed
                    best_h = h
            if best_h is not None:
                symbol_best[sym] = int(best_h)
        controls = {"enabled": True, "symbol_best_horizon": symbol_best, "blocked_symbols": sorted(blocked_symbols)}
        self._engine.set_horizon_controls(controls)
        self._horizon_gate_state.update(
            {
                "evaluated_at": now.isoformat(),
                "enabled": True,
                "controls": controls,
                "report": {"min_labels": min_labels, "rows": rows, "blocked_symbols": sorted(blocked_symbols)},
            }
        )
        return dict(self._horizon_gate_state)

    def _should_store_sample(self, symbol: str) -> tuple[bool, str]:
        if not bool(self._settings.sample_balance_enabled):
            return True, "sample_balance_disabled"
        # In effectively single-symbol collection mode, share-cap logic would
        # permanently block writes after min_daily is met.
        active_collectors = 1 + (len(self._shadow_symbols) if self._shadow_symbols else 0)
        if active_collectors <= 1:
            return True, "single_symbol_collection_mode"
        rows = self._persistence.list_data_samples(5000)
        now_et = datetime.now(tz=timezone.utc).astimezone(self._market_tz()).date().isoformat()
        counts: dict[str, int] = {}
        total = 0
        for r in rows:
            ts = str(r.get("timestamp") or "")
            try:
                dt = datetime.fromisoformat(ts)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                day = dt.astimezone(self._market_tz()).date().isoformat()
            except Exception:
                continue
            if day != now_et:
                continue
            sym = str(r.get("symbol") or "").upper()
            if not sym:
                continue
            counts[sym] = counts.get(sym, 0) + 1
            total += 1
        sym = str(symbol or "").upper()
        cur = counts.get(sym, 0)
        min_daily = max(0, int(self._settings.sample_balance_min_per_symbol_per_day))
        if cur < min_daily:
            return True, f"under_min_daily_{cur}/{min_daily}"
        projected_share = (cur + 1) / float(max(1, total + 1))
        if projected_share > float(self._settings.sample_balance_max_share):
            return False, f"share_cap_{projected_share:.2f}>{float(self._settings.sample_balance_max_share):.2f}"
        return True, "ok"

    def _save_automation_state(self, state: dict[str, Any]) -> None:
        save_json(self._automation_state_path, state)

    def _current_trading_day_et_from(self, now_utc: datetime) -> str:
        return now_utc.astimezone(self._market_tz()).date().isoformat()

    @staticmethod
    def _daily_research_report_dir() -> Path:
        p = Path("data/research/daily")
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def _self_scan_report_dir() -> Path:
        p = Path("data/research/self_scan")
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def _weekly_experiment_report_dir() -> Path:
        p = Path("data/research/weekly_experiments")
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _weekly_experiment_registry(self) -> list[dict[str, Any]]:
        # One-variable experiments only.
        return [
            {"id": "min_conf_0.50", "label": "min_confidence=0.50", "min_confidence": 0.50},
            {"id": "min_conf_0.55", "label": "min_confidence=0.55", "min_confidence": 0.55},
            {"id": "min_conf_0.60", "label": "min_confidence=0.60", "min_confidence": 0.60},
        ]

    def _evaluate_weekly_experiment(self, exp: dict[str, Any], *, lookback: int = 100000) -> dict[str, Any]:
        min_conf = max(0.0, min(1.0, float(exp.get("min_confidence", 0.0))))
        quality = self.prediction_quality_report(
            lookback=lookback,
            horizons_minutes=(15,),
            min_confidence=min_conf,
            quality_mode="good_only",
        )
        h15 = dict((quality.get("horizons") or {}).get("15") or {})
        overall = dict(h15.get("overall") or {})
        label_count = int(overall.get("count", 0) or 0)
        signed_bps = float(overall.get("avg_signed_return_bps", 0.0) or 0.0)
        brier = float(overall.get("brier_score", 0.0) or 0.0)
        stress_mult = max(0.0, float(self._settings.weekly_experiment_stress_multiplier))
        stress_cost = 2.0 * float(self._settings.cost_slippage_bps_per_side) * stress_mult
        stressed_signed = signed_bps - stress_cost
        min_labels = max(1, int(self._settings.weekly_experiment_min_labels))
        sufficient = label_count >= min_labels
        score = stressed_signed if sufficient else -1e9
        return {
            "id": str(exp.get("id") or ""),
            "label": str(exp.get("label") or ""),
            "min_confidence": min_conf,
            "label_count": label_count,
            "signed_bps": signed_bps,
            "brier_score": brier,
            "stress_multiplier": stress_mult,
            "stress_cost_bps": stress_cost,
            "stressed_signed_bps": stressed_signed,
            "sufficient_sample": bool(sufficient),
            "score": float(score),
        }

    def _run_weekly_experiments_if_due(
        self,
        *,
        state: dict[str, Any],
        now: datetime,
        force: bool = False,
    ) -> dict[str, Any] | None:
        if not bool(self._settings.weekly_experiments_enabled):
            return None
        week_key = self._week_key_et(now)
        if (not force) and str(state.get("weekly_experiments_last_week", "") or "") == week_key:
            return None
        registry = self._weekly_experiment_registry()[: max(1, int(self._settings.weekly_experiments_max_per_week))]
        baseline_exp = {"id": "baseline", "label": "min_confidence=0.00", "min_confidence": 0.0}
        baseline = self._evaluate_weekly_experiment(baseline_exp, lookback=100000)
        rows = [self._evaluate_weekly_experiment(exp, lookback=100000) for exp in registry]
        rows.sort(key=lambda r: (not bool(r.get("sufficient_sample", False)), -float(r.get("score", -1e9))))
        winner = rows[0] if rows else baseline
        improved = bool(
            winner
            and bool(winner.get("sufficient_sample", False))
            and float(winner.get("stressed_signed_bps", -1e9)) > float(baseline.get("stressed_signed_bps", -1e9))
        )
        selected = winner if improved else baseline
        active_policy = dict(state.get("research_policy") or {})
        active_policy["min_confidence"] = float(selected.get("min_confidence", 0.0) or 0.0)
        active_policy["selected_by"] = "weekly_experiment_runner"
        active_policy["selected_at"] = now.isoformat()
        active_policy["selected_experiment_id"] = str(selected.get("id") or "baseline")
        state["research_policy"] = active_policy
        result = {
            "ok": True,
            "week_key_et": week_key,
            "freeze_enabled": bool(self._settings.research_freeze_enabled),
            "max_experiments": int(self._settings.weekly_experiments_max_per_week),
            "baseline": baseline,
            "experiments": rows,
            "selected": selected,
            "improved_vs_baseline": improved,
        }
        out_path = self._weekly_experiment_report_dir() / f"weekly_experiments_{week_key}.json"
        save_json(out_path, result)
        state["weekly_experiments_last_week"] = week_key
        state["weekly_experiments_last"] = {**result, "report_path": str(out_path)}
        self._notify(
            "weekly_experiments_completed",
            {
                "week_key_et": week_key,
                "selected_experiment_id": selected.get("id"),
                "selected_min_confidence": selected.get("min_confidence"),
                "improved_vs_baseline": improved,
            },
        )
        return state["weekly_experiments_last"]

    def _strict_auto_promotion_gate(self, payload: dict[str, Any]) -> dict[str, Any]:
        summary = dict(payload.get("summary") or {})
        promotion_eval = dict(payload.get("promotion_policy_evaluation") or {})
        evaluation = dict(promotion_eval.get("evaluation") or {})
        promotions = dict(payload.get("promotion_candidates") or {})
        oos_gate = dict(promotions.get("oos_gate") or {})
        min_labels = max(1, int(self._settings.auto_promotion_min_labels))
        min_stressed = float(self._settings.auto_promotion_min_stressed_signed_bps)
        labels = int(summary.get("labelled_predictions", 0) or 0)
        stressed = float(summary.get("cost_stress_1p5x_signed_bps", 0.0) or 0.0)
        oos_ok = bool(oos_gate.get("passed", False))
        promote_policy_ok = bool(evaluation.get("passes", False))
        checks = {
            "promotion_policy_passes": promote_policy_ok,
            "min_labels": labels >= min_labels,
            "stressed_signed_bps": stressed >= min_stressed,
            "oos_positive": (oos_ok if bool(self._settings.auto_promotion_require_oos_positive) else True),
        }
        passed = all(bool(v) for v in checks.values())
        return {
            "enabled": bool(self._settings.auto_promotion_gate_enabled),
            "passed": bool(passed),
            "checks": checks,
            "thresholds": {
                "min_labels": int(min_labels),
                "min_stressed_signed_bps": float(min_stressed),
                "require_oos_positive": bool(self._settings.auto_promotion_require_oos_positive),
            },
            "actuals": {
                "labels": labels,
                "stressed_signed_bps": stressed,
                "oos_gate_passed": oos_ok,
            },
        }

    def _write_daily_research_markdown(self, out_path: Path, payload: dict[str, Any]) -> None:
        summary = dict(payload.get("summary") or {})
        promotion = dict(payload.get("promotion_candidates") or {})
        promotion_eval = dict(payload.get("promotion_policy_evaluation") or {})
        promotion_eval_ok = bool(promotion_eval.get("ok", False))
        promote_allowed = bool(((promotion_eval.get("evaluation") or {}).get("passes", False))) if promotion_eval_ok else False
        lines = [
            f"# Daily Research {payload.get('trading_day_et', '')}",
            "",
            f"- generated_at_utc: {payload.get('generated_at_utc', '')}",
            f"- labelled_predictions: {summary.get('labelled_predictions', 0)}",
            f"- signed_bps: {summary.get('signed_bps', 0.0):.2f}",
            f"- brier_score: {summary.get('brier_score', 0.0):.3f}",
            f"- cost_stress_1p5x_signed_bps: {summary.get('cost_stress_1p5x_signed_bps', 0.0):.2f}",
            f"- champion_delta_signed_bps: {summary.get('champion_delta_signed_bps', 0.0):.2f}",
            f"- bootstrap_robust_cells: {summary.get('bootstrap_robust_cells', 0)}",
            f"- promotion_candidate_count: {promotion.get('candidate_count', 0)}",
            f"- predictive_report_ok: {bool((payload.get('predictive_walk_forward') or {}).get('ok', False))}",
            f"- walk_forward_report_ok: {bool((payload.get('walk_forward_report') or {}).get('ok', False))}",
            "",
            "## Promotion Gate",
            "",
            f"- oos_gate_passed: {bool((promotion.get('oos_gate') or {}).get('passed', False))}",
            f"- promotion_policy_eval_ok: {promotion_eval_ok}",
            f"- promotion_policy_passes: {promote_allowed}",
            "",
        ]
        out_path.write_text("\n".join(lines), encoding="utf-8")

    def _run_daily_research_if_due(
        self,
        *,
        state: dict[str, Any],
        now: datetime,
        force: bool = False,
    ) -> dict[str, Any] | None:
        if not bool(self._settings.auto_research_enabled):
            return None
        day_et = self._current_trading_day_et_from(now)
        if not force and str(state.get("daily_research_last_day", "") or "") == day_et:
            return None

        research_policy = dict(state.get("research_policy") or {})
        policy_min_conf = max(0.0, min(1.0, float(research_policy.get("min_confidence", 0.0) or 0.0)))
        quality = self.prediction_quality_report(
            lookback=100000,
            horizons_minutes=(15,),
            min_confidence=policy_min_conf,
            quality_mode="good_only",
        )
        cost_stress = self.cost_stress_report(
            lookback=100000,
            horizon_minutes=15,
            quality_mode="good_only",
            multipliers=(1.0, 1.5, 2.0),
        )
        bootstrap = self.cell_leaderboard_bootstrap_report(
            lookback=100000,
            horizons_minutes=(5, 15, 30),
            quality_mode="good_only",
            min_labels=12,
            n_bootstrap=300,
            robust_only=False,
        )
        champion = self.champion_challenger_daily_report(
            lookback=100000,
            horizon_minutes=15,
            quality_mode="good_only",
            min_train_labels=150,
            min_cell_labels=12,
            challenger_min_confidence=0.55,
            min_daily_selections=10,
        )
        promotions = self.promotion_candidates_report(
            lookback=100000,
            horizons_minutes=(5, 15, 30),
            quality_mode="good_only",
            n_bootstrap=300,
        )
        walk_forward = self.research_walk_forward_report(
            lookback=int(self._settings.auto_research_lookback),
            folds=int(self._settings.auto_research_folds),
            min_train=int(self._settings.auto_research_min_train),
            min_test=int(self._settings.auto_research_min_test),
            bins=int(self._settings.auto_research_bins),
        )
        predictive = self.research_predictive_model_report(
            lookback=int(self._settings.auto_research_lookback),
            folds=int(self._settings.auto_research_folds),
            min_train=int(self._settings.auto_research_min_train),
            min_test=int(self._settings.auto_research_min_test),
            n_estimators=80,
            learning_rate=0.1,
            max_bins=16,
        )
        promotion_eval = self.evaluate_promotion_policy()

        horizons = dict((quality.get("horizons") or {}))
        h15 = dict(horizons.get("15") or {})
        h15_overall = dict(h15.get("overall") or {})
        stress_rows = list(cost_stress.get("rows") or [])
        row_1p5 = next((r for r in stress_rows if abs(float(r.get("slippage_multiplier", 0.0)) - 1.5) < 1e-9), None)

        summary = {
            "labelled_predictions": int(h15_overall.get("count", 0) or 0),
            "signed_bps": float(h15_overall.get("avg_signed_return_bps", 0.0) or 0.0),
            "brier_score": float(h15_overall.get("brier_score", 0.0) or 0.0),
            "cost_stress_1p5x_signed_bps": float((row_1p5 or {}).get("avg_signed_bps_after_cost", 0.0) or 0.0),
            "champion_delta_signed_bps": float(champion.get("overall_delta_signed_bps", 0.0) or 0.0),
            "bootstrap_robust_cells": int(sum(1 for r in list(bootstrap.get("rows") or []) if bool(r.get("robust_positive")))),
            "walk_forward_ok": bool(walk_forward.get("ok", False)),
            "predictive_ok": bool(predictive.get("ok", False)),
            "promotion_policy_passes": bool(((promotion_eval.get("evaluation") or {}).get("passes", False))),
        }

        payload = {
            "ok": True,
            "trading_day_et": day_et,
            "generated_at_utc": now.isoformat(),
            "research_policy": {"min_confidence": policy_min_conf},
            "summary": summary,
            "prediction_quality": quality,
            "cost_stress": cost_stress,
            "champion_challenger_daily": champion,
            "cell_leaderboard_bootstrap": bootstrap,
            "promotion_candidates": promotions,
            "walk_forward_report": walk_forward,
            "predictive_walk_forward": predictive,
            "promotion_policy_evaluation": promotion_eval,
        }
        strict_gate = self._strict_auto_promotion_gate(payload)
        payload["auto_promotion_gate"] = strict_gate
        auto_promotion_result: dict[str, Any] | None = None
        if bool(self._settings.auto_promotion_gate_enabled) and bool(strict_gate.get("passed", False)):
            auto_promotion_result = self.promote_predictive_candidate(note=f"auto_promotion_gate {day_et}")
            payload["auto_promotion_result"] = auto_promotion_result
        out_dir = self._daily_research_report_dir()
        json_path = out_dir / f"daily_research_{day_et}.json"
        md_path = out_dir / f"daily_research_{day_et}.md"
        save_json(json_path, payload)
        self._write_daily_research_markdown(md_path, payload)

        result = {
            "ok": True,
            "trading_day_et": day_et,
            "generated_at_utc": now.isoformat(),
            "report_path": str(json_path),
            "markdown_path": str(md_path),
            "summary": summary,
            "promotion_candidate_count": int(promotions.get("candidate_count", 0) or 0),
            "promotion_decision": "promote" if int(promotions.get("candidate_count", 0) or 0) > 0 else "hold",
            "research_policy": {"min_confidence": policy_min_conf},
            "auto_promotion_gate": strict_gate,
            "auto_promotion_result": auto_promotion_result,
        }
        state.update(
            {
                "daily_research_last_day": day_et,
                "daily_research_last_at": now.isoformat(),
                "daily_research_last_error": None,
                "daily_research_last": result,
            }
        )
        self._notify(
            "daily_research_completed",
            {
                "trading_day_et": day_et,
                "report_path": str(json_path),
                "promotion_decision": result.get("promotion_decision"),
                "promotion_candidate_count": result.get("promotion_candidate_count", 0),
                "signed_bps": summary.get("signed_bps", 0.0),
                "brier_score": summary.get("brier_score", 0.0),
                "promotion_policy_passes": summary.get("promotion_policy_passes", False),
            },
        )
        return result

    def _repair_daily_research_summary_if_inconsistent(self, state: dict[str, Any]) -> dict[str, Any] | None:
        last = dict(state.get("daily_research_last") or {})
        if not last:
            return None
        summary = dict(last.get("summary") or {})
        labels = int(summary.get("labelled_predictions", 0) or 0)
        if labels > 0:
            return None
        quality = self.prediction_quality_report(
            lookback=100000,
            horizons_minutes=(15,),
            min_confidence=0.0,
            quality_mode="good_only",
        )
        h15 = dict((quality.get("horizons") or {}).get("15") or {})
        overall = dict(h15.get("overall") or {})
        real_labels = int(overall.get("count", 0) or 0)
        if real_labels <= 0:
            return None
        stress = self.cost_stress_report(
            lookback=100000,
            horizon_minutes=15,
            quality_mode="good_only",
            multipliers=(1.0, 1.5, 2.0),
        )
        row_1p5 = next(
            (r for r in list(stress.get("rows") or []) if abs(float(r.get("slippage_multiplier", 0.0)) - 1.5) < 1e-9),
            None,
        )
        summary["labelled_predictions"] = real_labels
        summary["signed_bps"] = float(overall.get("avg_signed_return_bps", 0.0) or 0.0)
        summary["brier_score"] = float(overall.get("brier_score", 0.0) or 0.0)
        summary["cost_stress_1p5x_signed_bps"] = float((row_1p5 or {}).get("avg_signed_bps_after_cost", 0.0) or 0.0)
        last["summary"] = summary
        state["daily_research_last"] = last
        state["daily_research_last_at"] = datetime.now(tz=timezone.utc).isoformat()
        self._save_automation_state(state)
        self._notify(
            "daily_research_summary_repaired",
            {"labelled_predictions": real_labels, "signed_bps": summary["signed_bps"], "brier_score": summary["brier_score"]},
        )
        return {"ok": True, "repaired": True, "labelled_predictions": real_labels}

    def _closed_trade_count(self, lookback: int = 50000) -> int:
        rows = self._persistence.list_closed_trades(max(1, min(50000, lookback)))
        return len(rows)

    def _run_automation_once(self) -> dict[str, Any]:
        with self._automation_lock:
            state = self._load_automation_state()
            now = datetime.now(tz=timezone.utc)
            last_run_raw = str(state.get("last_run_at", "") or "")
            last_run = None
            if last_run_raw:
                try:
                    last_run = datetime.fromisoformat(last_run_raw)
                    if last_run.tzinfo is None:
                        last_run = last_run.replace(tzinfo=timezone.utc)
                except Exception:
                    last_run = None

            daily_research_result: dict[str, Any] | None = None
            weekly_experiments_result: dict[str, Any] | None = None
            try:
                weekly_experiments_result = self._run_weekly_experiments_if_due(state=state, now=now)
                daily_research_result = self._run_daily_research_if_due(state=state, now=now)
            except Exception as exc:
                state["daily_research_last_error"] = str(exc)

            current_trade_count = self._closed_trade_count()
            previous_trade_count = int(state.get("last_trade_count", 0) or 0)
            new_trades = max(0, current_trade_count - previous_trade_count)

            interval_due = (
                last_run is None
                or (now - last_run) >= timedelta(hours=max(1, int(self._settings.auto_retrain_interval_hours)))
            )
            new_trades_due = new_trades >= max(1, int(self._settings.auto_retrain_min_new_trades))

            neg_lb = max(5, int(self._settings.auto_retrain_neg_expectancy_lookback))
            recent_metrics = self.metrics_snapshot(limit=neg_lb)
            neg_due = (
                float(recent_metrics.get("trade_count", 0.0)) >= float(neg_lb)
                and float(recent_metrics.get("expectancy", 0.0)) < 0.0
            )

            if not (interval_due or new_trades_due or neg_due):
                state["last_check_at"] = now.isoformat()
                state["last_trade_count_observed"] = current_trade_count
                self._save_automation_state(state)
                return {
                    "triggered": False,
                    "interval_due": interval_due,
                    "new_trades_due": new_trades_due,
                    "negative_expectancy_due": neg_due,
                    "new_trades": new_trades,
                    "trade_count": current_trade_count,
                    "daily_research": daily_research_result,
                    "weekly_experiments": weekly_experiments_result,
                }

            reason = "interval"
            if neg_due:
                reason = "negative_expectancy"
            elif new_trades_due:
                reason = "new_trades"

            retrain_result = self.retrain_adaptive_from_history(limit=int(self._settings.auto_retrain_lookback))
            research_result: dict[str, Any] | None = None
            if self._settings.auto_research_enabled:
                research_result = self.research_walk_forward_report(
                    lookback=int(self._settings.auto_research_lookback),
                    folds=int(self._settings.auto_research_folds),
                    min_train=int(self._settings.auto_research_min_train),
                    min_test=int(self._settings.auto_research_min_test),
                    bins=int(self._settings.auto_research_bins),
                )

            current_trade_count = self._closed_trade_count()
            state.update(
                {
                    "last_check_at": now.isoformat(),
                    "last_run_at": now.isoformat(),
                    "last_reason": reason,
                    "last_trade_count": current_trade_count,
                    "last_retrain": retrain_result,
                    "last_research": research_result,
                }
            )
            self._save_automation_state(state)
            return {
                "triggered": True,
                "reason": reason,
                "new_trades": new_trades,
                "trade_count": current_trade_count,
                "retrain": retrain_result,
                "research": research_result,
                "daily_research": daily_research_result,
                "weekly_experiments": weekly_experiments_result,
            }

    def run_daily_research_automation(self, *, force: bool = True) -> dict[str, Any]:
        with self._automation_lock:
            state = self._load_automation_state()
            now = datetime.now(tz=timezone.utc)
            try:
                result = self._run_daily_research_if_due(state=state, now=now, force=force)
            except Exception as exc:
                state["daily_research_last_error"] = str(exc)
                self._save_automation_state(state)
                raise
            self._save_automation_state(state)
            if result is None:
                return {
                    "ok": True,
                    "skipped": True,
                    "reason": "already_ran_today",
                    "trading_day_et": self._current_trading_day_et_from(now),
                    "last": state.get("daily_research_last"),
                }
            return result

    def run_weekly_experiments(self, *, force: bool = True) -> dict[str, Any]:
        with self._automation_lock:
            state = self._load_automation_state()
            now = datetime.now(tz=timezone.utc)
            result = self._run_weekly_experiments_if_due(state=state, now=now, force=force)
            self._save_automation_state(state)
            if result is None:
                return {
                    "ok": True,
                    "skipped": True,
                    "reason": "already_ran_this_week",
                    "week_key_et": self._week_key_et(now),
                    "last": state.get("weekly_experiments_last"),
                }
            return result

    def run_automation_once_manual(self) -> dict[str, Any]:
        result = self._run_automation_once()
        self._notify(
            "automation_manual_run",
            {
                "triggered": bool(result.get("triggered", False)),
                "reason": result.get("reason"),
                "new_trades": int(result.get("new_trades", 0) or 0),
                "trade_count": int(result.get("trade_count", 0) or 0),
                "daily_research_ran": bool(result.get("daily_research")),
            },
        )
        return result

    def run_research_automation_suite(self, *, force_daily: bool = True) -> dict[str, Any]:
        state = self._load_automation_state()
        now = datetime.now(tz=timezone.utc)
        weekly = self._run_weekly_experiments_if_due(state=state, now=now, force=force_daily)
        self._save_automation_state(state)
        policy = dict(state.get("research_policy") or {})
        policy_min_conf = max(0.0, min(1.0, float(policy.get("min_confidence", 0.0) or 0.0)))
        automation = self.run_automation_once_manual()
        daily = self.run_daily_research_automation(force=force_daily)
        promotion = self.promotion_candidates_report(
            lookback=100000,
            horizons_minutes=(5, 15, 30),
            quality_mode="good_only",
            n_bootstrap=300,
        )
        return {
            "ok": True,
            "automation": automation,
            "daily_research": daily,
            "weekly_experiments": weekly,
            "research_policy": {"min_confidence": policy_min_conf},
            "promotion_candidates": {
                "candidate_count": int(promotion.get("candidate_count", 0) or 0),
                "oos_gate": promotion.get("oos_gate", {}),
                "thresholds": promotion.get("thresholds", {}),
            },
        }

    def run_7day_edge_sprint(self, *, force_daily: bool = True) -> dict[str, Any]:
        now = datetime.now(tz=timezone.utc)
        state = self._load_automation_state()
        # Step 1: Recompute confidence controls from latest labels (signed-bps-first thresholding).
        confidence_controls = self._evaluate_confidence_controls(force=True)
        cc_controls = dict(confidence_controls.get("controls") or {})
        # Step 2: Refresh symbol quarantine to prune weak symbols automatically.
        quarantine = self._update_symbol_quarantines(force=True)
        # Step 3: Recompute promotion candidates + strict gate with OOS requirement.
        promotion = self.promotion_candidates_report(
            lookback=100000,
            horizons_minutes=(5, 15, 30),
            quality_mode="good_only",
            n_bootstrap=300,
        )
        strict_gate = self._strict_auto_promotion_gate(
            {
                "promotion_candidates": promotion,
                "promotion_policy_evaluation": self.evaluate_promotion_policy(),
            }
        )
        # Step 4: Run the normal suite so retrain/research loop still progresses.
        suite = self.run_research_automation_suite(force_daily=force_daily)
        result = {
            "ok": True,
            "at": now.isoformat(),
            "plan": "7day_edge_sprint",
            "calibration_gate": {
                "enabled": bool(cc_controls.get("calibration_gate_enabled", False)),
                "stress_bps": float(cc_controls.get("calibration_gate_stress_bps", 0.0) or 0.0),
                "symbols_with_bin_controls": int(len(dict(cc_controls.get("symbol_allowed_confidence_bins") or {}))),
            },
            "confidence_controls": {
                "evaluated_at": confidence_controls.get("evaluated_at"),
                "global_min_confidence": float(cc_controls.get("global_min_confidence", 0.0) or 0.0),
            },
            "symbol_pruning": {
                "active_quarantines": int((quarantine.get("count") or 0)),
                "symbols": sorted(list((quarantine.get("active") or {}).keys())),
            },
            "promotion_gate": {
                "passed": bool(strict_gate.get("passed", False)),
                "oos_positive": bool((strict_gate.get("checks") or {}).get("oos_positive", False)),
                "stressed_signed_bps": float((strict_gate.get("checks") or {}).get("stressed_signed_bps", 0.0) or 0.0),
                "label_count": int((strict_gate.get("checks") or {}).get("label_count", 0) or 0),
            },
            "suite": suite,
        }
        state["edge_sprint_7d_last_at"] = now.isoformat()
        state["edge_sprint_7d_last"] = result
        self._save_automation_state(state)
        self._notify(
            "edge_sprint_7d_completed",
            {
                "at": now.isoformat(),
                "oos_positive": bool((result.get("promotion_gate") or {}).get("oos_positive", False)),
                "active_quarantines": int((result.get("symbol_pruning") or {}).get("active_quarantines", 0)),
            },
        )
        return result

    def _automation_loop(self) -> None:
        while not self._automation_stop_event.is_set():
            try:
                self._run_automation_once()
                self._auto_recovery_check()
            except Exception as exc:
                with self._lock:
                    self._last_error = str(exc)
                    self._last_note = "automation check failed"
            self._automation_stop_event.wait(timeout=max(60, int(self._settings.auto_retrain_check_seconds)))

    def _autonomy_loop(self) -> None:
        while not self._autonomy_stop_event.is_set():
            try:
                now = datetime.now(tz=timezone.utc)
                self._evaluate_data_quality_guard()
                self._evaluate_sample_flow_guard()
                self._evaluate_cost_guard()
                self._auto_recovery_check()
                self._sample_flow_auto_recovery()
                self._run_automation_once()
                self._run_weekly_cell_pruning_if_due()
                self._run_weekly_drift_rollback_if_due()
                self.run_self_scan(force=False)
                self._apply_llm_hard_error_guard(window_minutes=60)
                self._update_llm_remediation_outcome()
                health = self.autonomy_health_score()
                self._apply_autonomy_auto_remediation(health)
                state = self._load_automation_state()
                last_suite_raw = str(state.get("autonomy_last_suite_at", "") or "")
                run_suite = True
                backoff_raw = str(state.get("autonomy_suite_backoff_until", "") or "")
                if backoff_raw:
                    try:
                        backoff_dt = datetime.fromisoformat(backoff_raw)
                        if backoff_dt.tzinfo is None:
                            backoff_dt = backoff_dt.replace(tzinfo=timezone.utc)
                        if now < backoff_dt:
                            run_suite = False
                    except Exception:
                        pass
                if last_suite_raw:
                    try:
                        last_suite = datetime.fromisoformat(last_suite_raw)
                        if last_suite.tzinfo is None:
                            last_suite = last_suite.replace(tzinfo=timezone.utc)
                        mins = max(5, int(self._settings.autonomous_research_suite_interval_minutes))
                        run_suite = (now - last_suite) >= timedelta(minutes=mins)
                    except Exception:
                        run_suite = True
                if run_suite:
                    suite = self.run_research_automation_suite(force_daily=False)
                    state["autonomy_last_suite_at"] = now.isoformat()
                    state["autonomy_last_suite"] = suite
                    self._notify(
                        "autonomy_suite_completed",
                        {
                            "at": now.isoformat(),
                            "promotion_candidates": int(
                                ((suite.get("promotion_candidates") or {}).get("candidate_count", 0) or 0)
                            ),
                        },
                    )
                state["autonomy_last_check_at"] = now.isoformat()
                state["autonomy_last_error"] = None
                self._save_automation_state(state)
            except Exception as exc:
                with self._automation_lock:
                    state = self._load_automation_state()
                    state["autonomy_last_check_at"] = datetime.now(tz=timezone.utc).isoformat()
                    state["autonomy_last_error"] = str(exc)
                    self._save_automation_state(state)
                with self._lock:
                    self._last_error = str(exc)
                    self._last_note = "autonomy loop failed"
            self._autonomy_stop_event.wait(timeout=max(30, int(self._settings.autonomous_research_check_seconds)))

    def start_autonomous_research(self) -> dict[str, Any]:
        with self._autonomy_lock:
            if self._autonomy_thread and self._autonomy_thread.is_alive():
                return {"started": False, "reason": "autonomy_already_running", "status": self.autonomous_research_status()}
            self._autonomy_stop_event.clear()
            self._autonomy_thread = threading.Thread(target=self._autonomy_loop, daemon=True, name="autonomy-worker")
            self._autonomy_thread.start()
        self._notify("autonomy_started", {"status": "running"})
        return {"started": True, "reason": "started", "status": self.autonomous_research_status()}

    def stop_autonomous_research(self) -> dict[str, Any]:
        with self._autonomy_lock:
            thread = self._autonomy_thread
            if not thread or not thread.is_alive():
                return {"stopped": False, "reason": "autonomy_not_running", "status": self.autonomous_research_status()}
            self._autonomy_stop_event.set()
        thread.join(timeout=5)
        self._notify("autonomy_stopped", {"status": "stopped"})
        return {"stopped": True, "reason": "stopped", "status": self.autonomous_research_status()}

    def autonomous_research_run_once(self) -> dict[str, Any]:
        now = datetime.now(tz=timezone.utc)
        state = self._load_automation_state()
        quality = self._evaluate_data_quality_guard(force=True)
        cost = self._evaluate_cost_guard(force=True)
        recovery = self._auto_recovery_check(force=True)
        automation = self._run_automation_once()
        weekly_pruning = self._run_weekly_cell_pruning_if_due(force=True)
        weekly_drift = self._run_weekly_drift_rollback_if_due(force=True)
        self_scan = self.run_self_scan(force=True)
        llm_guard = self._apply_llm_hard_error_guard(window_minutes=60)
        llm_outcome = self._update_llm_remediation_outcome()
        health = self.autonomy_health_score()
        remediation = self._apply_autonomy_auto_remediation(health)
        suite = self.run_research_automation_suite(force_daily=False)
        state["autonomy_last_check_at"] = now.isoformat()
        state["autonomy_last_suite_at"] = now.isoformat()
        state["autonomy_last_suite"] = suite
        state["autonomy_last_error"] = None
        self._save_automation_state(state)
        return {
            "ok": True,
            "at": now.isoformat(),
            "quality_guard": quality,
            "cost_guard": cost,
            "auto_recovery": recovery,
            "automation": automation,
            "weekly_pruning": weekly_pruning,
            "weekly_drift": weekly_drift,
            "self_scan": self_scan,
            "llm_guard": llm_guard,
            "llm_outcome": llm_outcome,
            "autonomy_health": health,
            "remediation": remediation,
            "suite": suite,
        }

    def autonomous_research_status(self) -> dict[str, Any]:
        state = self._load_automation_state()
        running = bool(self._autonomy_thread and self._autonomy_thread.is_alive())
        return {
            "enabled": bool(self._settings.autonomous_research_enabled),
            "running": running,
            "check_seconds": int(self._settings.autonomous_research_check_seconds),
            "suite_interval_minutes": int(self._settings.autonomous_research_suite_interval_minutes),
            "last_check_at": state.get("autonomy_last_check_at"),
            "last_suite_at": state.get("autonomy_last_suite_at"),
            "last_error": state.get("autonomy_last_error"),
            "last_suite": state.get("autonomy_last_suite"),
            "weekly_pruning_last_week": state.get("weekly_pruning_last_week"),
            "weekly_pruning_last": state.get("weekly_pruning_last"),
            "weekly_drift_last_week": state.get("weekly_drift_last_week"),
            "weekly_drift_last": state.get("weekly_drift_last"),
            "auto_pruned_symbols": list(state.get("auto_pruned_symbols") or []),
            "self_scan_last_at": state.get("self_scan_last_at"),
            "self_scan_last": state.get("self_scan_last"),
            "suite_backoff_until": state.get("autonomy_suite_backoff_until"),
            "remediation_last": state.get("autonomy_remediation_last"),
            "llm_guard_last": state.get("llm_guard_last"),
            "llm_guard_outcome": state.get("llm_guard_outcome"),
        }

    def _build_self_scan_report(self, *, lookback: int = 2000) -> dict[str, Any]:
        rows = self._persistence.list_data_samples(max(100, min(100000, int(lookback))))
        decisions = self._persistence.list_decisions(max(100, min(100000, int(lookback))))
        now = datetime.now(tz=timezone.utc)
        window_mins = max(15, int(self._settings.automation_quality_window_minutes))
        start = now - timedelta(minutes=window_mins)
        recent_rows: list[dict[str, Any]] = []
        for r in rows:
            raw_ts = str(r.get("timestamp") or r.get("ts") or "")
            try:
                ts = datetime.fromisoformat(raw_ts)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except Exception:
                continue
            if ts >= start:
                recent_rows.append(r)
        recent_decisions: list[dict[str, Any]] = []
        for d in decisions:
            raw_ts = str(d.get("timestamp") or d.get("ts") or "")
            try:
                ts = datetime.fromisoformat(raw_ts)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
            except Exception:
                continue
            if ts >= start:
                recent_decisions.append(d)
        total = len(recent_rows)
        stale = 0
        missing_fc = 0
        llm_unavail = 0
        guard_holds = 0
        sec_invoked = 0
        for r in recent_rows:
            if bool(r.get("quality_quote_stale", False)):
                stale += 1
            if bool(r.get("quality_missing_forecast", False)):
                missing_fc += 1
            note = str(r.get("note") or "").lower()
            reason = str(r.get("reasoning") or "").lower()
            if "llm unavailable" in note or "llm unavailable" in reason:
                llm_unavail += 1
            if note.startswith("guard hold"):
                guard_holds += 1
            md = dict(r.get("metadata") or {})
            lr = dict(md.get("llm_routing") or {})
            if bool(lr.get("secondary_invoked", False)):
                sec_invoked += 1

        cycle_fail = 0
        for d in recent_decisions:
            n = str(d.get("note") or "").lower()
            if "cycle execution failed" in n:
                cycle_fail += 1

        stale_ratio = stale / float(max(1, total))
        missing_ratio = missing_fc / float(max(1, total))
        llm_unavail_ratio = llm_unavail / float(max(1, total))
        guard_ratio = guard_holds / float(max(1, total))
        sec_ratio = sec_invoked / float(max(1, total))
        findings: list[dict[str, Any]] = []
        if llm_unavail_ratio > 0.10:
            findings.append({"severity": "high", "code": "llm_unavailable_rate_high", "detail": f"{llm_unavail_ratio:.2%}"})
        elif llm_unavail_ratio > 0.03:
            findings.append({"severity": "medium", "code": "llm_unavailable_rate_elevated", "detail": f"{llm_unavail_ratio:.2%}"})
        if stale_ratio > float(self._settings.automation_quality_max_stale_quote_ratio):
            findings.append({"severity": "high", "code": "stale_quote_ratio_high", "detail": f"{stale_ratio:.2%}"})
        if missing_ratio > float(self._settings.automation_quality_max_missing_forecast_ratio):
            findings.append({"severity": "high", "code": "missing_forecast_ratio_high", "detail": f"{missing_ratio:.2%}"})
        if cycle_fail > 0:
            findings.append({"severity": "medium", "code": "cycle_failures_seen", "detail": str(cycle_fail)})
        if guard_ratio > 0.85 and total >= 100:
            findings.append({"severity": "medium", "code": "excessive_guard_hold_rate", "detail": f"{guard_ratio:.2%}"})

        severity = "healthy"
        if any(f.get("severity") == "high" for f in findings):
            severity = "critical"
        elif findings:
            severity = "warning"

        state = self._load_automation_state()
        return {
            "ok": True,
            "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "severity": severity,
            "lookback": int(lookback),
            "metrics": {
                "window_minutes": int(window_mins),
                "sample_count": int(total),
                "decision_count": int(len(recent_decisions)),
                "stale_quote_ratio": float(stale_ratio),
                "missing_forecast_ratio": float(missing_ratio),
                "llm_unavailable_ratio": float(llm_unavail_ratio),
                "guard_hold_ratio": float(guard_ratio),
                "secondary_invocation_ratio": float(sec_ratio),
                "cycle_fail_count": int(cycle_fail),
                "worker_last_error": self._last_error,
                "autonomy_last_error": state.get("autonomy_last_error"),
                "daily_research_last_error": state.get("daily_research_last_error"),
            },
            "findings": findings,
        }

    def run_self_scan(self, *, force: bool = False) -> dict[str, Any]:
        now = datetime.now(tz=timezone.utc)
        state = self._load_automation_state()
        if not bool(self._settings.autonomous_self_scan_enabled):
            return {"ok": False, "reason": "self_scan_disabled"}
        if not force:
            last_raw = str(state.get("self_scan_last_at", "") or "")
            if last_raw:
                try:
                    last = datetime.fromisoformat(last_raw)
                    if last.tzinfo is None:
                        last = last.replace(tzinfo=timezone.utc)
                    if (now - last) < timedelta(minutes=max(5, int(self._settings.autonomous_self_scan_interval_minutes))):
                        return {"ok": True, "skipped": True, "reason": "interval_not_due", "last": state.get("self_scan_last")}
                except Exception:
                    pass

        report = self._build_self_scan_report(lookback=int(self._settings.autonomous_self_scan_lookback))
        out_dir = self._self_scan_report_dir()
        ts = now.strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"self_scan_{ts}.json"
        save_json(out_path, report)
        state["self_scan_last_at"] = now.isoformat()
        state["self_scan_last"] = {
            "severity": report.get("severity"),
            "generated_at_utc": report.get("generated_at_utc"),
            "report_path": str(out_path),
            "finding_count": len(list(report.get("findings") or [])),
        }
        self._save_automation_state(state)
        if report.get("severity") in {"warning", "critical"}:
            self._notify("autonomy_self_scan_alert", state["self_scan_last"])
        return {"ok": True, "report": report, "report_path": str(out_path)}

    def _apply_autonomy_auto_remediation(self, health: dict[str, Any]) -> dict[str, Any]:
        score = int(health.get("score", 100) or 100)
        now = datetime.now(tz=timezone.utc)
        actions: list[str] = []
        state = self._load_automation_state()
        applied = False
        last = dict(state.get("autonomy_remediation_last") or {})
        last_score = int(last.get("score", 100) or 100)
        last_at_raw = str(last.get("at", "") or "")
        cooldown_ok = True
        if last_at_raw:
            try:
                last_at = datetime.fromisoformat(last_at_raw)
                if last_at.tzinfo is None:
                    last_at = last_at.replace(tzinfo=timezone.utc)
                # Avoid repeating identical remediation too frequently.
                if score == last_score and (now - last_at) < timedelta(minutes=30):
                    cooldown_ok = False
            except Exception:
                pass
        if not cooldown_ok:
            state["autonomy_remediation_last"] = {
                "at": now.isoformat(),
                "score": score,
                "applied": False,
                "actions": [],
                "reason": "cooldown",
            }
            self._save_automation_state(state)
            return dict(state["autonomy_remediation_last"])
        if score < 60:
            self.set_acceleration_mode("standard")
            actions.append("set_acceleration_standard")
            self._run_weekly_cell_pruning_if_due(force=True)
            actions.append("force_weekly_pruning")
            self.run_self_scan(force=True)
            actions.append("force_self_scan")
            self._auto_recovery_check(force=True)
            actions.append("force_auto_recovery_check")
            backoff_until = now + timedelta(minutes=120)
            state["autonomy_suite_backoff_until"] = backoff_until.isoformat()
            actions.append("suite_backoff_120m")
            applied = True
        if score < 40:
            backoff_until = now + timedelta(minutes=240)
            state["autonomy_suite_backoff_until"] = backoff_until.isoformat()
            actions.append("suite_backoff_240m")
            applied = True
        state["autonomy_remediation_last"] = {
            "at": now.isoformat(),
            "score": score,
            "applied": applied,
            "actions": actions,
        }
        self._save_automation_state(state)
        if applied:
            self._notify("autonomy_auto_remediation_applied", state["autonomy_remediation_last"])
        return dict(state["autonomy_remediation_last"])

    def _current_trading_day_et(self) -> str:
        return datetime.now(tz=self._market_tz()).date().isoformat()

    def _rollover_daily_state_if_needed(self) -> bool:
        account = self._engine.account
        today_et = self._current_trading_day_et()
        if account.trading_day_et == today_et:
            return False
        account.trading_day_et = today_et
        account.todays_trade_count = 0
        account.daily_realized_pnl = 0.0
        account.closed_trades_today = []
        return True

    def _is_live_execution_mode(self) -> bool:
        if self._settings.execution_provider != "alpaca":
            return False
        endpoint = (self._settings.alpaca_trading_url or "").strip().lower()
        return "paper-api.alpaca.markets" not in endpoint

    def _save_runtime_config(self) -> None:
        accel = self._engine.acceleration_snapshot()
        self._persistence.save_runtime_config(
            RuntimeConfig(
                symbol=self._settings.symbol,
                symbols=tuple(self._settings.symbols),
                multi_symbol_enabled=bool(self._settings.multi_symbol_enabled),
                multi_symbol_shadow_enabled=bool(self._settings.multi_symbol_shadow_enabled),
                multi_symbol_active=bool(self._shadow_engines),
                multi_symbol_shadow_symbols=tuple(self._shadow_symbols),
                cycle_seconds=self._settings.cycle_seconds,
                llm_provider=self._settings.llm_provider,
                data_provider=self._settings.data_provider,
                execution_provider=self._settings.execution_provider,
                auto_retrain_enabled=bool(self._settings.auto_retrain_enabled),
                auto_retrain_check_seconds=int(self._settings.auto_retrain_check_seconds),
                auto_retrain_interval_hours=int(self._settings.auto_retrain_interval_hours),
                auto_retrain_min_new_trades=int(self._settings.auto_retrain_min_new_trades),
                auto_retrain_lookback=int(self._settings.auto_retrain_lookback),
                data_acceleration_mode=bool(accel.get("requested", self._settings.data_acceleration_mode)),
                acceleration_active=bool(accel.get("active", False)),
                entry_confluence_min=float(accel.get("entry_confluence_min", self._settings.entry_confluence_min)),
                ev_min_ticks=float(accel.get("ev_min_ticks", self._settings.ev_min_ticks)),
                max_spread_bps=float(accel.get("max_spread_bps", self._settings.max_spread_bps)),
                cost_model_enabled=bool(self._settings.cost_model_enabled),
                cost_slippage_bps_per_side=float(self._settings.cost_slippage_bps_per_side),
                cost_fee_per_share=float(self._settings.cost_fee_per_share),
                cost_min_fee_per_order=float(self._settings.cost_min_fee_per_order),
                max_position_size=self._settings.max_position_size,
                max_trades_per_day=self._settings.max_trades_per_day,
                max_daily_drawdown=self._settings.max_daily_drawdown,
                macro_context_enabled=bool(self._settings.macro_context_enabled),
                macro_provider=str(self._settings.macro_provider),
                macro_snapshot_path=str(self._settings.macro_snapshot_path),
                economic_calendar_enabled=bool(self._settings.economic_calendar_enabled),
                economic_calendar_provider=str(self._settings.economic_calendar_provider),
                economic_calendar_lookahead_days=int(self._settings.economic_calendar_lookahead_days),
                economic_calendar_high_impact_window_minutes=int(self._settings.economic_calendar_high_impact_window_minutes),
                economic_calendar_block_high_impact=bool(self._settings.economic_calendar_block_high_impact),
                finnhub_context_enabled=bool(self._settings.finnhub_context_enabled),
                finnhub_news_lookback_days=int(self._settings.finnhub_news_lookback_days),
                finnhub_news_max_headlines=int(self._settings.finnhub_news_max_headlines),
                finnhub_news_min_score=float(self._settings.finnhub_news_min_score),
                finnhub_earnings_lookahead_days=int(self._settings.finnhub_earnings_lookahead_days),
                earnings_risk_window_days=int(self._settings.earnings_risk_window_days),
                model_monitoring_enabled=bool(self._settings.model_monitoring_enabled),
                model_monitor_horizon_minutes=int(self._settings.model_monitor_horizon_minutes),
                model_monitor_short_window_days=int(self._settings.model_monitor_short_window_days),
                model_monitor_long_window_days=int(self._settings.model_monitor_long_window_days),
                model_monitor_min_labels=int(self._settings.model_monitor_min_labels),
                model_monitor_breach_streak=int(self._settings.model_monitor_breach_streak),
                model_monitor_safe_mode_enabled=bool(self._settings.model_monitor_safe_mode_enabled),
                max_position_age_minutes=int(self._settings.max_position_age_minutes),
                broker_sync_stale_seconds=int(self._settings.broker_sync_stale_seconds),
                stale_position_force_close=bool(self._settings.stale_position_force_close),
                require_broker_protection_orders=bool(self._settings.require_broker_protection_orders),
                protection_check_grace_seconds=int(self._settings.protection_check_grace_seconds),
            )
        )

    def _configured_collection_symbols(self) -> tuple[str, ...]:
        primary = self._settings.symbol.strip().upper()
        configured = self._settings.symbols or (primary,)
        values: list[str] = []
        seen: set[str] = set()
        for raw in (primary, *configured):
            symbol = str(raw or "").strip().upper()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            values.append(symbol)
        return tuple(values)

    def _build_shadow_engines(self) -> dict[str, TradingEngine]:
        if not self._settings.multi_symbol_enabled or not self._settings.multi_symbol_shadow_enabled:
            return {}
        if self._settings.multi_symbol_paper_only and self._is_live_execution_mode():
            return {}

        primary = self._settings.symbol.strip().upper()
        engines: dict[str, TradingEngine] = {}
        for symbol in self._configured_collection_symbols():
            if symbol == primary:
                continue
            shadow_settings = replace(
                self._settings,
                symbol=symbol,
                execution_provider="mock",
                auto_start_worker=False,
            )
            engines[symbol] = TradingEngine(shadow_settings)
        return engines

    def _decision_metadata(
        self,
        *,
        engine: TradingEngine | None = None,
        symbol: str | None = None,
        collection_role: str = "primary",
    ) -> dict[str, Any]:
        source = engine or self._engine
        accel = source.acceleration_snapshot()
        resolved_symbol = (symbol or source.settings.symbol).strip().upper()
        tier = str((self._load_automation_state().get("policy_tier_active") or "balanced")).strip().lower()
        return {
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "symbol": resolved_symbol,
            "collection_role": collection_role,
            "shadow_execution": collection_role != "primary",
            "mode": str(accel.get("mode", "standard")),
            "acceleration_active": bool(accel.get("active", False)),
            "thresholds": {
                "entry_confluence_min": float(accel.get("entry_confluence_min", self._settings.entry_confluence_min)),
                "ev_min_ticks": float(accel.get("ev_min_ticks", self._settings.ev_min_ticks)),
                "max_spread_bps": float(accel.get("max_spread_bps", self._settings.max_spread_bps)),
                "edge_min_win_rate": float(accel.get("edge_min_win_rate", self._settings.edge_min_win_rate)),
                "edge_min_expectancy": float(accel.get("edge_min_expectancy", self._settings.edge_min_expectancy)),
            },
            "costs": {
                "enabled": bool(self._settings.cost_model_enabled),
                "slippage_bps_per_side": float(self._settings.cost_slippage_bps_per_side),
                "fee_per_share": float(self._settings.cost_fee_per_share),
                "min_fee_per_order": float(self._settings.cost_min_fee_per_order),
            },
            "providers": {
                "data": source.settings.data_provider,
                "execution": source.settings.execution_provider,
                "llm": source.settings.llm_provider,
            },
            "policy_tier": tier if tier in {"strict", "balanced", "explore"} else "balanced",
        }

    def _notify(self, event_type: str, payload: dict[str, Any]) -> None:
        if not self._settings.notifications_enabled:
            return
        try:
            append_notification(self._settings.notifications_path, event_type, payload)
        except Exception:
            return

    def _persist_cycle_and_sample(self, cycle: CycleResult, *, metadata: dict[str, Any], symbol: str) -> None:
        decision = cycle.decision
        if decision.forecast_direction not in {"LONG", "SHORT"}:
            fallback_direction: str | None = None
            if decision.direction in {"LONG", "SHORT"}:
                fallback_direction = decision.direction
            else:
                dims = dict((cycle.metadata or {}).get("feature_dimensions") or {})
                try:
                    trend = float(dims.get("Trend", 0.0) or 0.0)
                except Exception:
                    trend = 0.0
                fallback_direction = "LONG" if trend >= 0.0 else "SHORT"
            decision.forecast_direction = fallback_direction
            if float(decision.forecast_confidence or 0.0) <= 0.0:
                decision.forecast_confidence = max(0.5, float(decision.confidence or 0.0))
            if int(decision.forecast_horizon_minutes or 0) <= 0:
                decision.forecast_horizon_minutes = 15

        # Reconcile sample-quality forecast flags after fallback normalization so
        # persisted flags reflect the final stored forecast fields.
        forecast_ok = (
            decision.forecast_direction in {"LONG", "SHORT"}
            and float(decision.forecast_confidence or 0.0) > 0.0
            and int(decision.forecast_horizon_minutes or 0) > 0
        )
        cycle.metadata = dict(cycle.metadata or {})
        sample_quality = dict(cycle.metadata.get("sample_quality") or {})
        quality_flags = dict(sample_quality.get("flags") or {})
        quote_last = float(((cycle.metadata or {}).get("quote") or {}).get("last", 0.0) or 0.0)
        quality_flags["no_quote"] = bool(quote_last <= 0.0)
        quality_flags["missing_forecast"] = not forecast_ok
        hard_fail_flags = dict(sample_quality.get("hard_fail_flags") or {})
        hard_fail_flags["no_quote"] = bool(quality_flags["no_quote"])
        hard_fail_flags["missing_forecast"] = bool(quality_flags["missing_forecast"])
        sample_quality["flags"] = quality_flags
        sample_quality["hard_fail_flags"] = hard_fail_flags
        sample_quality["good"] = not any(bool(v) for v in hard_fail_flags.values())
        cycle.metadata["sample_quality"] = sample_quality

        self._persistence.save_cycle(cycle, metadata=metadata)
        if bool(quality_flags.get("quote_stale", False)):
            cycle.metadata = dict(cycle.metadata or {})
            cycle.metadata["sample_balance_skipped"] = {"symbol": symbol, "reason": "quote_stale"}
            return
        if bool(quality_flags.get("spread_too_wide", False)):
            cycle.metadata = dict(cycle.metadata or {})
            cycle.metadata["sample_balance_skipped"] = {"symbol": symbol, "reason": "spread_too_wide"}
            return
        if bool(quality_flags.get("outside_session", False)):
            cycle.metadata = dict(cycle.metadata or {})
            cycle.metadata["sample_balance_skipped"] = {"symbol": symbol, "reason": "outside_session"}
            return
        if bool(quality_flags.get("no_quote", False)):
            cycle.metadata = dict(cycle.metadata or {})
            cycle.metadata["sample_balance_skipped"] = {"symbol": symbol, "reason": "no_quote"}
            return
        allow_sample, reason = self._should_store_sample(symbol)
        if allow_sample:
            self._persistence.save_data_sample(cycle, metadata=metadata)
        else:
            cycle.metadata = dict(cycle.metadata or {})
            cycle.metadata["sample_balance_skipped"] = {"symbol": symbol, "reason": reason}

    def _run_shadow_cycle_once(self) -> CycleResult | None:
        if not self._shadow_symbols:
            return None
        symbol = self._select_next_shadow_symbol()
        self._shadow_index += 1
        engine = self._shadow_engines[symbol]
        cycle = engine.run_cycle()
        metadata = self._decision_metadata(
            engine=engine,
            symbol=symbol,
            collection_role="shadow_multi_symbol",
        )
        self._persist_cycle_and_sample(cycle, metadata=metadata, symbol=str(metadata.get('symbol') or self._settings.symbol))
        self._last_shadow_symbol = symbol

        if "Closed position PnL=" in cycle.note and engine.account.closed_trades_today:
            closed = engine.account.closed_trades_today[-1]
            closed.metadata = metadata
            self._persistence.save_closed_trade(closed)
            self._notify(
                "shadow_position_closed",
                {
                    "symbol": closed.symbol,
                    "direction": closed.direction,
                    "pnl": closed.pnl,
                    "metadata": metadata,
                },
            )
        if cycle.decision.action == "trade":
            self._notify(
                "shadow_trade_placed",
                {
                    "symbol": symbol,
                    "direction": cycle.decision.direction,
                    "size": cycle.decision.size,
                    "note": cycle.note,
                    "metadata": metadata,
                },
            )
        return cycle

    def _select_next_shadow_symbol(self) -> str:
        symbols = list(self._shadow_symbols)
        if not symbols:
            return self._settings.symbol.strip().upper()
        if not bool(self._settings.sample_balance_enabled):
            return symbols[self._shadow_index % len(symbols)]

        rows = self._persistence.list_data_samples(5000)
        today_et = datetime.now(tz=timezone.utc).astimezone(self._market_tz()).date().isoformat()
        counts = {s: 0 for s in symbols}
        total = 0
        for r in rows:
            sym = str(r.get("symbol") or "").upper()
            if sym not in counts:
                continue
            ts = str(r.get("timestamp") or "")
            try:
                dt = datetime.fromisoformat(ts)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                day_et = dt.astimezone(self._market_tz()).date().isoformat()
            except Exception:
                continue
            if day_et != today_et:
                continue
            counts[sym] += 1
            total += 1

        min_daily = max(0, int(self._settings.sample_balance_min_per_symbol_per_day))
        under_min = [s for s in symbols if counts.get(s, 0) < min_daily]
        if under_min:
            return min(under_min, key=lambda s: counts.get(s, 0))

        # After minimum coverage is reached, keep balancing toward lower relative share.
        def share(sym: str) -> float:
            return counts.get(sym, 0) / float(max(1, total))

        return min(symbols, key=lambda s: (share(s), counts.get(s, 0), s))

    def start(self) -> bool:
        result = self.start_with_guard()
        return bool(result.get("started"))

    def start_with_guard(self) -> dict[str, Any]:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return {
                    "started": False,
                    "blocked": False,
                    "reason": "worker_already_running",
                    "status": {
                        "running": True,
                        "cycles_completed": self._cycles_completed,
                        "last_cycle_at": self._last_cycle_at,
                        "last_note": self._last_note,
                        "last_error": self._last_error,
                    },
                    "gate": self.go_live_gate_snapshot(),
                }
            gate = self.go_live_gate_snapshot()
            kill = self.kill_switch_snapshot()
            if bool(kill.get("enabled", False)):
                self._last_note = "worker start blocked by trading kill switch"
                return {
                    "started": False,
                    "blocked": True,
                    "reason": "trading_kill_switch_enabled",
                    "status": {
                        "running": False,
                        "cycles_completed": self._cycles_completed,
                        "last_cycle_at": self._last_cycle_at,
                        "last_note": self._last_note,
                        "last_error": self._last_error,
                    },
                    "gate": gate,
                    "kill_switch": kill,
                }
            if self._is_weekend_market_day():
                self._last_note = "worker start blocked on weekend"
                return {
                    "started": False,
                    "blocked": True,
                    "reason": "weekend_blocked",
                    "status": {
                        "running": False,
                        "cycles_completed": self._cycles_completed,
                        "last_cycle_at": self._last_cycle_at,
                        "last_note": self._last_note,
                        "last_error": self._last_error,
                    },
                    "gate": gate,
                    "weekend_blocked": True,
                }
            if gate.get("blocked_autonomous_live"):
                self._last_note = "worker start blocked by go-live gate"
                return {
                    "started": False,
                    "blocked": True,
                    "reason": "go_live_gate_failed",
                    "status": {
                        "running": False,
                        "cycles_completed": self._cycles_completed,
                        "last_cycle_at": self._last_cycle_at,
                        "last_note": self._last_note,
                        "last_error": self._last_error,
                    },
                    "gate": gate,
                }
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_loop, daemon=True, name="trading-worker")
            self._thread.start()
            self._last_note = "worker started"
            self._notify("worker_started", {"model_name": MODEL_NAME, "status": "running"})
            return {
                "started": True,
                "blocked": False,
                "reason": "started",
                "status": {
                    "running": True,
                    "cycles_completed": self._cycles_completed,
                    "last_cycle_at": self._last_cycle_at,
                    "last_note": self._last_note,
                    "last_error": self._last_error,
                },
                "gate": gate,
            }

    def stop(self) -> bool:
        with self._lock:
            thread = self._thread
            if not thread or not thread.is_alive():
                return False
            self._stop_event.set()
        thread.join(timeout=5)
        with self._lock:
            self._last_note = "worker stopped"
        self._notify("worker_stopped", {"model_name": MODEL_NAME, "status": "stopped"})
        return True

    def run_cycle_once(self) -> CycleResult:
        self._rollover_daily_state_if_needed()
        if self._is_weekend_market_day():
            cycle = self._build_guard_cycle(
                reason="Weekend block active; worker is disabled on Saturday/Sunday.",
                note="Guard hold: weekend block",
            )
            with self._lock:
                self._cycles_completed += 1
                self._last_cycle_at = cycle.timestamp.isoformat()
                self._last_note = cycle.note
                self._last_error = None
            return cycle
        kill = self.kill_switch_snapshot()
        if bool(kill.get("enabled", False)):
            cycle = self._build_guard_cycle(
                reason=f"Trading kill switch enabled: {kill.get('reason')}.",
                note="Guard hold: trading kill switch",
            )
            with self._lock:
                self._cycles_completed += 1
                self._last_cycle_at = cycle.timestamp.isoformat()
                self._last_note = cycle.note
                self._last_error = None
            metadata = self._decision_metadata()
            cycle.metadata = dict(cycle.metadata or {})
            cycle.metadata["kill_switch"] = kill
            self._persist_cycle_and_sample(cycle, metadata=metadata, symbol=str(metadata.get('symbol') or self._settings.symbol))
            self._persistence.save_account(self._engine.account)
            self._notify("trading_kill_switch_hold", {"kill_switch": kill, "metadata": metadata})
            return cycle
        if bool(self._settings.budget_lock_enabled) and self._engine.account.open_position is None:
            if bool(self._settings.budget_lock_in_session_only) and (not self._is_in_session_window()):
                cycle = self._build_guard_cycle(
                    reason="Budget lock active: outside configured session window; skipping expensive cycle.",
                    note="Guard hold: budget lock session window",
                )
                with self._lock:
                    self._cycles_completed += 1
                    self._last_cycle_at = cycle.timestamp.isoformat()
                    self._last_note = cycle.note
                    self._last_error = None
                metadata = self._decision_metadata()
                cycle.metadata = dict(cycle.metadata or {})
                cycle.metadata["budget_lock"] = {"enabled": True, "reason": "outside_session_window"}
                self._persist_cycle_and_sample(cycle, metadata=metadata, symbol=str(metadata.get('symbol') or self._settings.symbol))
                self._persistence.save_account(self._engine.account)
                return cycle
            cap = max(1, int(self._settings.budget_lock_max_llm_calls_per_hour))
            llm_calls = self._estimate_llm_calls_recent(window_minutes=60)
            if llm_calls >= cap:
                cycle = self._build_guard_cycle(
                    reason=f"Budget lock active: hourly LLM call cap reached ({llm_calls}/{cap}).",
                    note="Guard hold: budget lock llm cap",
                )
                with self._lock:
                    self._cycles_completed += 1
                    self._last_cycle_at = cycle.timestamp.isoformat()
                    self._last_note = cycle.note
                    self._last_error = None
                metadata = self._decision_metadata()
                cycle.metadata = dict(cycle.metadata or {})
                cycle.metadata["budget_lock"] = {"enabled": True, "reason": "llm_call_cap", "llm_calls_last_hour": llm_calls, "llm_cap_per_hour": cap}
                self._persist_cycle_and_sample(cycle, metadata=metadata, symbol=str(metadata.get('symbol') or self._settings.symbol))
                self._persistence.save_account(self._engine.account)
                return cycle
        synced = self._sync_account_from_broker(force_baseline=False, record_closures=True)
        pos = self._engine.account.open_position
        stale_sync_limit = max(10, int(self._settings.broker_sync_stale_seconds))
        broker_sync_required = self._settings.execution_provider == "alpaca"
        if broker_sync_required and pos is not None and (not synced):
            if self._last_broker_sync_at is None or (
                datetime.now(tz=timezone.utc) - self._last_broker_sync_at
            ).total_seconds() > float(stale_sync_limit):
                cycle = self._build_guard_cycle(
                    reason=f"Broker sync stale while position open (> {stale_sync_limit}s); pausing entries.",
                    note="Guard hold: broker sync stale",
                )
                with self._lock:
                    self._cycles_completed += 1
                    self._last_cycle_at = cycle.timestamp.isoformat()
                    self._last_note = cycle.note
                    self._last_error = "broker sync stale"
                metadata = self._decision_metadata()
                self._persist_cycle_and_sample(cycle, metadata=metadata, symbol=str(metadata.get('symbol') or self._settings.symbol))
                self._persistence.save_account(self._engine.account)
                self._notify(
                    "broker_sync_stale_hold",
                    {
                        "symbol": pos.symbol,
                        "stale_seconds_limit": stale_sync_limit,
                        "last_sync_at": self._last_broker_sync_at.isoformat() if self._last_broker_sync_at else None,
                        "metadata": metadata,
                    },
                )
                return cycle

        guarded, guard_reason = self._ensure_position_safety()
        if guarded:
            cycle = self._build_guard_cycle(reason=guard_reason, note=f"Guard hold: {guard_reason}")
            with self._lock:
                self._cycles_completed += 1
                self._last_cycle_at = cycle.timestamp.isoformat()
                self._last_note = cycle.note
                self._last_error = None
            metadata = self._decision_metadata()
            self._persist_cycle_and_sample(cycle, metadata=metadata, symbol=str(metadata.get('symbol') or self._settings.symbol))
            self._persistence.save_account(self._engine.account)
            return cycle

        monitor = self._evaluate_model_monitoring()
        if bool(monitor.get("safe_mode_active", False)):
            report = monitor.get("report") or {}
            reasons = ",".join(report.get("breach_reasons", []) or []) or "decay_detected"
            cycle = self._build_guard_cycle(
                reason=f"Model monitoring safe mode active: {reasons}.",
                note="Guard hold: model decay safe mode",
            )
            with self._lock:
                self._cycles_completed += 1
                self._last_cycle_at = cycle.timestamp.isoformat()
                self._last_note = cycle.note
                self._last_error = None
            metadata = self._decision_metadata()
            cycle.metadata = dict(cycle.metadata or {})
            cycle.metadata["model_monitoring"] = monitor
            self._persist_cycle_and_sample(cycle, metadata=metadata, symbol=str(metadata.get('symbol') or self._settings.symbol))
            self._persistence.save_account(self._engine.account)
            self._notify(
                "model_decay_safe_mode_hold",
                {"model_monitoring": monitor, "metadata": metadata},
            )
            return cycle

        quality_guard = self._evaluate_data_quality_guard()
        if bool(quality_guard.get("active", False)):
            reason_text = str(quality_guard.get("reason") or "")
            breaches = [s.strip() for s in reason_text.split(";") if s.strip()]
            low_volume_only = bool(breaches) and all(b.startswith("low_sample_volume:") for b in breaches)
            q_metrics = dict(quality_guard.get("metrics") or {})
            in_session = bool(q_metrics.get("in_session", False))
            if low_volume_only:
                if (
                    bool(self._settings.automation_cost_mode_enabled)
                    and bool(self._settings.automation_cost_mode_skip_low_volume_collect)
                    and (not in_session)
                ):
                    cycle = self._build_guard_cycle(
                        reason=f"Data quality guard active: {quality_guard.get('reason')}. Cost mode skipped low-volume collection cycle.",
                        note="Guard hold: data quality",
                    )
                else:
                    engine_settings = self._engine.settings
                    if bool(self._settings.automation_cost_mode_enabled):
                        engine_settings = replace(
                            self._engine.settings,
                            finnhub_context_enabled=(
                                False if bool(self._settings.automation_cost_mode_disable_finnhub_context) else self._engine.settings.finnhub_context_enabled
                            ),
                            economic_calendar_enabled=(
                                False if bool(self._settings.automation_cost_mode_disable_economic_calendar) else self._engine.settings.economic_calendar_enabled
                            ),
                        )
                    try:
                        self._engine.settings = engine_settings
                        cycle = self._run_engine_cycle_with_budget_lock(collect_only=True)
                    except Exception as exc:
                        cycle = self._build_guard_cycle(
                            reason=f"Data quality guard active: {quality_guard.get('reason')}.",
                            note="Guard hold: data quality",
                        )
                        cycle.metadata = dict(cycle.metadata or {})
                        cycle.metadata["quality_guard_collect_error"] = str(exc)
                    finally:
                        self._engine.settings = self._settings
                    cycle.note = "Guard hold: data quality"
                    cycle.decision.action = "hold"
                    cycle.decision.direction = None
                    cycle.decision.size = 1
                    cycle.decision.sl_ticks = 0
                    cycle.decision.tp_ticks = 0
                    cycle.decision.reasoning = f"Data quality guard active: {quality_guard.get('reason')}."
            else:
                cycle = self._build_guard_cycle(
                    reason=f"Data quality guard active: {quality_guard.get('reason')}.",
                    note="Guard hold: data quality",
                )
            with self._lock:
                self._cycles_completed += 1
                self._last_cycle_at = cycle.timestamp.isoformat()
                self._last_note = cycle.note
                self._last_error = None
            metadata = self._decision_metadata()
            cycle.metadata = dict(cycle.metadata or {})
            cycle.metadata["quality_guard"] = quality_guard
            self._persist_cycle_and_sample(cycle, metadata=metadata, symbol=str(metadata.get('symbol') or self._settings.symbol))
            self._persistence.save_account(self._engine.account)
            return cycle

        cost_guard = self._evaluate_cost_guard()
        if bool(cost_guard.get("active", False)):
            cycle = self._build_guard_cycle(
                reason=f"Cost guard active: estimated daily LLM cost hit budget ({cost_guard.get('metrics', {}).get('estimated_cost_usd', 0.0):.2f} USD).",
                note="Guard hold: cost budget",
            )
            with self._lock:
                self._cycles_completed += 1
                self._last_cycle_at = cycle.timestamp.isoformat()
                self._last_note = cycle.note
                self._last_error = None
            metadata = self._decision_metadata()
            cycle.metadata = dict(cycle.metadata or {})
            cycle.metadata["cost_guard"] = cost_guard
            self._persist_cycle_and_sample(cycle, metadata=metadata, symbol=str(metadata.get('symbol') or self._settings.symbol))
            self._persistence.save_account(self._engine.account)
            return cycle

        signed_bps_gate = self._evaluate_signed_bps_gate()
        if bool(signed_bps_gate.get("blocked", False)):
            cycle = self._build_guard_cycle(
                reason=f"Signed-bps kill switch active: {signed_bps_gate.get('reason')}.",
                note="Guard hold: signed-bps kill switch",
            )
            with self._lock:
                self._cycles_completed += 1
                self._last_cycle_at = cycle.timestamp.isoformat()
                self._last_note = cycle.note
                self._last_error = None
            metadata = self._decision_metadata()
            cycle.metadata = dict(cycle.metadata or {})
            cycle.metadata["signed_bps_gate"] = signed_bps_gate
            self._persist_cycle_and_sample(cycle, metadata=metadata, symbol=str(metadata.get('symbol') or self._settings.symbol))
            self._persistence.save_account(self._engine.account)
            return cycle

        symbol_gate = self._evaluate_symbol_gate()
        if bool(symbol_gate.get("blocked", False)):
            # Keep collecting market/quote samples while forcing a no-trade decision.
            # This preserves label generation throughput even when symbol gating is active.
            confidence_controls = self._evaluate_confidence_controls()
            regime_gate = self._evaluate_regime_gate()
            horizon_gate = self._evaluate_horizon_gate()
            symbol = str(self._settings.symbol or "").strip().upper()
            prior_horizon = dict((self._horizon_gate_state.get("controls") or {}))
            forced_horizon = {
                "enabled": bool(prior_horizon.get("enabled", False)),
                "symbol_best_horizon": dict(prior_horizon.get("symbol_best_horizon") or {}),
                "blocked_symbols": set(prior_horizon.get("blocked_symbols") or set()) | {symbol},
            }
            self._engine.set_horizon_controls(forced_horizon)
            try:
                cycle = self._run_engine_cycle_with_budget_lock()
            except Exception as exc:
                cycle = self._build_guard_cycle(
                    reason=f"Cycle execution failed during symbol gate hold: {exc}",
                    note="Guard hold: cycle execution failed",
                )
                with self._lock:
                    self._cycles_completed += 1
                    self._last_cycle_at = cycle.timestamp.isoformat()
                    self._last_note = cycle.note
                    self._last_error = str(exc)
                metadata = self._decision_metadata()
                cycle.metadata = dict(cycle.metadata or {})
                cycle.metadata["symbol_gate"] = symbol_gate
                self._persist_cycle_and_sample(cycle, metadata=metadata, symbol=str(metadata.get('symbol') or self._settings.symbol))
                self._persistence.save_account(self._engine.account)
                self._notify(
                    "worker_cycle_failed",
                    {"error": str(exc), "model_name": MODEL_NAME, "metadata": metadata},
                )
                return cycle
            finally:
                self._engine.set_horizon_controls(prior_horizon)

            cycle.note = "Guard hold: symbol gate"
            cycle.decision.action = "hold"
            cycle.decision.direction = None
            cycle.decision.size = 1
            cycle.decision.sl_ticks = 0
            cycle.decision.tp_ticks = 0
            cycle.decision.reasoning = f"Symbol gate active: {symbol_gate.get('reason')}."

            with self._lock:
                self._cycles_completed += 1
                self._last_cycle_at = cycle.timestamp.isoformat()
                self._last_note = cycle.note
                self._last_error = None
            metadata = self._decision_metadata()
            cycle.metadata = dict(cycle.metadata or {})
            cycle.metadata["confidence_controls"] = {
                "enabled": bool(confidence_controls.get("enabled", False)),
                "evaluated_at": confidence_controls.get("evaluated_at"),
                "global_min_confidence": (confidence_controls.get("controls") or {}).get("global_min_confidence"),
                "calibration_gate_enabled": bool((confidence_controls.get("controls") or {}).get("calibration_gate_enabled", False)),
                "calibration_gate_stress_bps": float((confidence_controls.get("controls") or {}).get("calibration_gate_stress_bps", 0.0) or 0.0),
            }
            cycle.metadata["regime_gate"] = {
                "enabled": bool(regime_gate.get("enabled", False)),
                "evaluated_at": regime_gate.get("evaluated_at"),
                "allowed_pairs": (regime_gate.get("controls") or {}).get("allowed_pairs", {}),
            }
            cycle.metadata["horizon_gate"] = {
                "enabled": bool(horizon_gate.get("enabled", False)),
                "evaluated_at": horizon_gate.get("evaluated_at"),
                "symbol_best_horizon": (horizon_gate.get("controls") or {}).get("symbol_best_horizon", {}),
            }
            cycle.metadata["symbol_gate"] = symbol_gate
            self._persist_cycle_and_sample(cycle, metadata=metadata, symbol=str(metadata.get('symbol') or self._settings.symbol))
            self._persistence.save_account(self._engine.account)
            return cycle

        confidence_controls = self._evaluate_confidence_controls()
        regime_gate = self._evaluate_regime_gate()
        horizon_gate = self._evaluate_horizon_gate()

        try:
            cycle = self._run_engine_cycle_with_budget_lock()
        except Exception as exc:
            cycle = self._build_guard_cycle(
                reason=f"Cycle execution failed: {exc}",
                note="Guard hold: cycle execution failed",
            )
            with self._lock:
                self._cycles_completed += 1
                self._last_cycle_at = cycle.timestamp.isoformat()
                self._last_note = cycle.note
                self._last_error = str(exc)
            metadata = self._decision_metadata()
            self._persist_cycle_and_sample(cycle, metadata=metadata, symbol=str(metadata.get('symbol') or self._settings.symbol))
            self._persistence.save_account(self._engine.account)
            self._notify(
                "worker_cycle_failed",
                {"error": str(exc), "model_name": MODEL_NAME, "metadata": metadata},
            )
            return cycle
        with self._lock:
            self._cycles_completed += 1
            self._last_cycle_at = cycle.timestamp.isoformat()
            self._last_note = cycle.note
            self._last_error = None

        metadata = self._decision_metadata()
        cycle.metadata = dict(cycle.metadata or {})
        cycle.metadata["model_monitoring"] = monitor
        cycle.metadata["confidence_controls"] = {
            "enabled": bool(confidence_controls.get("enabled", False)),
            "evaluated_at": confidence_controls.get("evaluated_at"),
            "global_min_confidence": (confidence_controls.get("controls") or {}).get("global_min_confidence"),
            "calibration_gate_enabled": bool((confidence_controls.get("controls") or {}).get("calibration_gate_enabled", False)),
            "calibration_gate_stress_bps": float((confidence_controls.get("controls") or {}).get("calibration_gate_stress_bps", 0.0) or 0.0),
        }
        cycle.metadata["regime_gate"] = {
            "enabled": bool(regime_gate.get("enabled", False)),
            "evaluated_at": regime_gate.get("evaluated_at"),
            "allowed_pairs": (regime_gate.get("controls") or {}).get("allowed_pairs", {}),
        }
        cycle.metadata["horizon_gate"] = {
            "enabled": bool(horizon_gate.get("enabled", False)),
            "evaluated_at": horizon_gate.get("evaluated_at"),
            "symbol_best_horizon": (horizon_gate.get("controls") or {}).get("symbol_best_horizon", {}),
        }
        self._persist_cycle_and_sample(cycle, metadata=metadata, symbol=str(metadata.get('symbol') or self._settings.symbol))
        self._persistence.save_account(self._engine.account)
        if "Closed position PnL=" in cycle.note and self._engine.account.closed_trades_today:
            closed = self._engine.account.closed_trades_today[-1]
            closed.metadata = metadata
            self._persistence.save_closed_trade(closed)
            self._notify(
                "position_closed",
                {
                    "symbol": closed.symbol,
                    "direction": closed.direction,
                    "pnl": closed.pnl,
                    "metadata": metadata,
                },
            )
        if cycle.decision.action == "trade":
            self._notify(
                "trade_placed",
                {
                    "symbol": self._settings.symbol,
                    "direction": cycle.decision.direction,
                    "size": cycle.decision.size,
                    "note": cycle.note,
                    "metadata": metadata,
                },
            )
        try:
            self._run_shadow_cycle_once()
        except Exception as exc:
            with self._lock:
                self._last_error = f"shadow cycle failed: {exc}"
            self._notify(
                "shadow_cycle_failed",
                {
                    "error": str(exc),
                    "model_name": MODEL_NAME,
                    "shadow_symbols": list(self._shadow_symbols),
                },
            )
        return cycle

    def status(self) -> WorkerStatus:
        with self._lock:
            running = bool(self._thread and self._thread.is_alive())
            monitor = self.model_monitoring_status()
            return WorkerStatus(
                running=running,
                cycles_completed=self._cycles_completed,
                last_cycle_at=self._last_cycle_at,
                last_note=self._last_note,
                last_error=self._last_error,
                model_monitoring=monitor,
                kill_switch=self.kill_switch_snapshot(),
                startup_warnings=self._engine.startup_warnings(),
            )

    def account(self) -> dict[str, Any]:
        return account_to_record(self._engine.account)

    def close_open_position_now(self) -> dict[str, Any]:
        with self._lock:
            symbol = self._settings.symbol
            closer = getattr(self._engine.execution, "close_symbol_position", None)
            if not callable(closer):
                raise RuntimeError("Execution provider does not support manual close endpoint.")

            result = closer(symbol)
            self._sync_account_from_broker(force_baseline=False, record_closures=True)
            self._persistence.save_account(self._engine.account)
            self._last_note = f"manual close requested for {symbol}"

            self._notify(
                "manual_close_requested",
                {
                    "symbol": symbol,
                    "result": result,
                    "metadata": self._decision_metadata(),
                },
            )
            return {
                "ok": True,
                "symbol": symbol,
                "result": result,
                "account": account_to_record(self._engine.account),
            }

    def decisions(self, limit: int) -> list[dict[str, Any]]:
        return self._persistence.list_decisions(limit)

    def data_samples(self, limit: int) -> list[dict[str, Any]]:
        return self._persistence.list_data_samples(limit)

    def data_samples_csv(self, limit: int) -> str:
        return self._persistence.export_data_samples_csv(limit)

    def prediction_quality_report(
        self,
        *,
        lookback: int = 10000,
        horizons_minutes: tuple[int, ...] = (5, 15, 30),
        min_confidence: float = 0.0,
        quality_mode: str = "all",
    ) -> dict[str, Any]:
        max_rows = max(1, min(100000, int(lookback)))
        rows = self._persistence.list_data_samples(max_rows)
        report = build_prediction_quality_report(
            list(reversed(rows)),
            horizons_minutes=horizons_minutes,
            min_confidence=max(0.0, min(1.0, float(min_confidence))),
            quality_mode=str(quality_mode or "all"),
            **self._research_label_filters(),
        )
        report["lookback"] = max_rows
        return report

    def symbol_performance_report(
        self,
        *,
        lookback: int = 100000,
        horizon_minutes: int = 15,
        quality_mode: str = "good_only",
    ) -> dict[str, Any]:
        report = self.prediction_quality_report(
            lookback=max(1, min(100000, int(lookback))),
            horizons_minutes=(max(1, min(390, int(horizon_minutes))),),
            min_confidence=0.0,
            quality_mode=quality_mode,
        )
        h = (report.get("horizons") or {}).get(str(max(1, min(390, int(horizon_minutes)))), {})
        by_symbol = h.get("by_symbol") or {}
        min_labels = int(self._settings.symbol_gate_min_labels)
        min_signed = float(self._settings.symbol_gate_min_signed_bps)
        min_acc = float(self._settings.symbol_gate_min_accuracy)

        rows: list[dict[str, Any]] = []
        for sym, m in by_symbol.items():
            n = int(m.get("count", 0) or 0)
            signed = float(m.get("avg_signed_return_bps", 0.0) or 0.0)
            acc = float(m.get("accuracy", 0.0) or 0.0)
            recommendation = "monitor"
            if n >= min_labels and signed >= min_signed and acc >= min_acc:
                recommendation = "allow"
            elif n >= min_labels and (signed < min_signed or acc < min_acc):
                recommendation = "block"
            rows.append(
                {
                    "symbol": str(sym).upper(),
                    "count": n,
                    "accuracy": acc,
                    "avg_signed_return_bps": signed,
                    "brier_score": float(m.get("brier_score", 0.0) or 0.0),
                    "recommendation": recommendation,
                }
            )
        rows.sort(key=lambda r: (r["recommendation"] != "allow", -float(r["avg_signed_return_bps"])))
        allowed = [r["symbol"] for r in rows if r["recommendation"] == "allow"]
        blocked = [r["symbol"] for r in rows if r["recommendation"] == "block"]
        return {
            "ok": True,
            "quality_mode": quality_mode,
            "horizon_minutes": int(horizon_minutes),
            "thresholds": {
                "min_labels": min_labels,
                "min_signed_bps": min_signed,
                "min_accuracy": min_acc,
            },
            "allowed_symbols": allowed,
            "blocked_symbols": blocked,
            "rows": rows,
        }

    def feature_ablation_report(
        self,
        *,
        lookback: int = 10000,
        horizon_minutes: int = 15,
        min_confidence: float = 0.0,
        min_count: int = 20,
        quality_mode: str = "all",
    ) -> dict[str, Any]:
        max_rows = max(1, min(100000, int(lookback)))
        rows = self._persistence.list_data_samples(max_rows)
        report = build_feature_ablation_report(
            list(reversed(rows)),
            horizon_minutes=max(1, min(390, int(horizon_minutes))),
            min_confidence=max(0.0, min(1.0, float(min_confidence))),
            min_count=max(1, int(min_count)),
            quality_mode=str(quality_mode or "all"),
            **self._research_label_filters(),
        )
        report["lookback"] = max_rows
        return report

    def sample_coverage_report(self, *, lookback: int = 10000) -> dict[str, Any]:
        max_rows = max(1, min(100000, int(lookback)))
        rows = self._persistence.list_data_samples(max_rows)
        report = build_sample_coverage_report(rows)
        report["lookback"] = max_rows
        return report

    def cell_leaderboard_report(
        self,
        *,
        lookback: int = 100000,
        horizons_minutes: tuple[int, ...] = (5, 15, 30),
        quality_mode: str = "good_only",
        min_labels: int = 20,
    ) -> dict[str, Any]:
        rows = self._persistence.list_data_samples(max(1, min(100000, int(lookback))))
        labels_report = build_prediction_labels(
            list(reversed(rows)),
            horizons_minutes=horizons_minutes,
            min_confidence=0.0,
            quality_mode=quality_mode,
            **self._research_label_filters(),
        )
        board: list[dict[str, Any]] = []
        for h in horizons_minutes:
            labels = labels_report.get("labels_by_horizon", {}).get(int(h), [])
            buckets: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
            for r in labels:
                key = (
                    str(r.get("symbol") or "").upper(),
                    str(r.get("indicator_regime") or "unknown").strip().lower(),
                    self._session_bucket_from_ts(str(r.get("timestamp") or "")),
                    int(h),
                )
                buckets.setdefault(key, []).append(r)
            for (sym, reg, sess, hz), bucket in buckets.items():
                n = len(bucket)
                if n < int(min_labels):
                    continue
                signed = sum(float(b.get("signed_return_bps", 0.0)) for b in bucket) / float(max(1, n))
                acc = sum(1.0 for b in bucket if float(b.get("signed_return_bps", 0.0)) > 0.0) / float(max(1, n))
                board.append(
                    {
                        "symbol": sym,
                        "regime": reg,
                        "session_bucket": sess,
                        "horizon_minutes": hz,
                        "count": n,
                        "accuracy": acc,
                        "avg_signed_return_bps": signed,
                    }
                )
        board.sort(key=lambda x: (float(x["avg_signed_return_bps"]), float(x["accuracy"]), int(x["count"])), reverse=True)
        return {
            "ok": True,
            "lookback": int(lookback),
            "quality_mode": str(quality_mode or "good_only"),
            "min_labels": int(min_labels),
            "rows": board,
            "top": board[:50],
        }

    def cell_leaderboard_bootstrap_report(
        self,
        *,
        lookback: int = 100000,
        horizons_minutes: tuple[int, ...] = (5, 15, 30),
        quality_mode: str = "good_only",
        min_labels: int = 12,
        n_bootstrap: int = 300,
        robust_only: bool = False,
    ) -> dict[str, Any]:
        rows = self._persistence.list_data_samples(max(1, min(100000, int(lookback))))
        labels_report = build_prediction_labels(
            list(reversed(rows)),
            horizons_minutes=horizons_minutes,
            min_confidence=0.0,
            quality_mode=quality_mode,
            **self._research_label_filters(),
        )
        rng = random.Random(42)
        out: list[dict[str, Any]] = []
        for h in horizons_minutes:
            labels = labels_report.get("labels_by_horizon", {}).get(int(h), [])
            buckets: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
            for r in labels:
                key = (
                    str(r.get("symbol") or "").upper(),
                    str(r.get("indicator_regime") or "unknown").strip().lower(),
                    self._session_bucket_from_ts(str(r.get("timestamp") or "")),
                    int(h),
                )
                buckets.setdefault(key, []).append(r)
            for (sym, reg, sess, hz), bucket in buckets.items():
                n = len(bucket)
                if n < int(min_labels):
                    continue
                signed_vals = [float(x.get("signed_return_bps", 0.0)) for x in bucket]
                acc_vals = [1.0 if s > 0.0 else 0.0 for s in signed_vals]
                base_signed = sum(signed_vals) / float(n)
                base_acc = sum(acc_vals) / float(n)
                boots: list[float] = []
                for _ in range(max(50, int(n_bootstrap))):
                    sample = [signed_vals[rng.randrange(0, n)] for _ in range(n)]
                    boots.append(sum(sample) / float(n))
                boots.sort()
                lo_idx = int(0.025 * (len(boots) - 1))
                hi_idx = int(0.975 * (len(boots) - 1))
                ci_low = float(boots[lo_idx])
                ci_high = float(boots[hi_idx])
                robust = ci_low > 0.0
                if robust_only and not robust:
                    continue
                out.append(
                    {
                        "symbol": sym,
                        "regime": reg,
                        "session_bucket": sess,
                        "horizon_minutes": hz,
                        "count": n,
                        "accuracy": base_acc,
                        "avg_signed_return_bps": base_signed,
                        "ci95_low_signed_bps": ci_low,
                        "ci95_high_signed_bps": ci_high,
                        "robust_positive": robust,
                    }
                )
        out.sort(
            key=lambda x: (
                bool(x.get("robust_positive", False)),
                float(x.get("ci95_low_signed_bps", 0.0)),
                float(x.get("avg_signed_return_bps", 0.0)),
                int(x.get("count", 0)),
            ),
            reverse=True,
        )
        return {
            "ok": True,
            "lookback": int(lookback),
            "quality_mode": str(quality_mode or "good_only"),
            "min_labels": int(min_labels),
            "n_bootstrap": int(n_bootstrap),
            "robust_only": bool(robust_only),
            "rows": out,
            "top": out[:50],
            "robust_count": int(sum(1 for r in out if bool(r.get("robust_positive", False)))),
        }

    def oos_daily_walkforward_report(
        self,
        *,
        lookback: int = 100000,
        horizon_minutes: int = 15,
        quality_mode: str = "good_only",
        min_train_labels: int = 150,
        min_cell_labels: int = 20,
    ) -> dict[str, Any]:
        rows = self._persistence.list_data_samples(max(1, min(100000, int(lookback))))
        labels = build_prediction_labels(
            list(reversed(rows)),
            horizons_minutes=(int(horizon_minutes),),
            min_confidence=0.0,
            quality_mode=quality_mode,
            **self._research_label_filters(),
        ).get("labels_by_horizon", {}).get(int(horizon_minutes), [])
        by_day: dict[str, list[dict[str, Any]]] = {}
        for r in labels:
            day = str(r.get("timestamp") or "")[:10]
            by_day.setdefault(day, []).append(r)
        days = sorted([d for d in by_day.keys() if len(d) == 10])
        out_days: list[dict[str, Any]] = []
        train_pool: list[dict[str, Any]] = []
        for day in days:
            test_rows = by_day.get(day, [])
            if len(train_pool) < int(min_train_labels):
                train_pool.extend(test_rows)
                continue
            cell_stats: dict[tuple[str, str, str], dict[str, float]] = {}
            cell_counts: dict[tuple[str, str, str], int] = {}
            for r in train_pool:
                key = (
                    str(r.get("symbol") or "").upper(),
                    str(r.get("indicator_regime") or "unknown").strip().lower(),
                    self._session_bucket_from_ts(str(r.get("timestamp") or "")),
                )
                cell_counts[key] = cell_counts.get(key, 0) + 1
                cell_stats.setdefault(key, {"signed": 0.0, "wins": 0.0})
                sb = float(r.get("signed_return_bps", 0.0))
                cell_stats[key]["signed"] += sb
                if sb > 0:
                    cell_stats[key]["wins"] += 1.0
            allow: set[tuple[str, str, str]] = set()
            for k, n in cell_counts.items():
                if n < int(min_cell_labels):
                    continue
                signed = float(cell_stats[k]["signed"]) / float(max(1, n))
                acc = float(cell_stats[k]["wins"]) / float(max(1, n))
                if signed > 0.0 and acc >= 0.5:
                    allow.add(k)
            selected = []
            for r in test_rows:
                key = (
                    str(r.get("symbol") or "").upper(),
                    str(r.get("indicator_regime") or "unknown").strip().lower(),
                    self._session_bucket_from_ts(str(r.get("timestamp") or "")),
                )
                if key in allow:
                    selected.append(r)
            n = len(selected)
            signed = sum(float(r.get("signed_return_bps", 0.0)) for r in selected) / float(max(1, n))
            acc = sum(1.0 for r in selected if float(r.get("signed_return_bps", 0.0)) > 0.0) / float(max(1, n))
            out_days.append(
                {
                    "day": day,
                    "selected_count": n,
                    "selected_accuracy": acc if n > 0 else 0.0,
                    "selected_signed_bps": signed if n > 0 else 0.0,
                    "allowed_cells": len(allow),
                    "train_pool_size": len(train_pool),
                }
            )
            train_pool.extend(test_rows)
        overall_n = sum(int(d["selected_count"]) for d in out_days)
        overall_signed = (
            sum(float(d["selected_signed_bps"]) * int(d["selected_count"]) for d in out_days) / float(max(1, overall_n))
        )
        return {
            "ok": True,
            "lookback": int(lookback),
            "horizon_minutes": int(horizon_minutes),
            "quality_mode": str(quality_mode or "good_only"),
            "min_train_labels": int(min_train_labels),
            "min_cell_labels": int(min_cell_labels),
            "days": out_days,
            "overall_selected_count": int(overall_n),
            "overall_selected_signed_bps": float(overall_signed),
        }

    def champion_challenger_daily_report(
        self,
        *,
        lookback: int = 100000,
        horizon_minutes: int = 15,
        quality_mode: str = "good_only",
        min_train_labels: int = 150,
        min_cell_labels: int = 20,
        challenger_min_confidence: float = 0.60,
        min_daily_selections: int = 10,
    ) -> dict[str, Any]:
        rows = self._persistence.list_data_samples(max(1, min(100000, int(lookback))))
        labels = build_prediction_labels(
            list(reversed(rows)),
            horizons_minutes=(int(horizon_minutes),),
            min_confidence=0.0,
            quality_mode=quality_mode,
            **self._research_label_filters(),
        ).get("labels_by_horizon", {}).get(int(horizon_minutes), [])
        by_day: dict[str, list[dict[str, Any]]] = {}
        for r in labels:
            day = str(r.get("timestamp") or "")[:10]
            by_day.setdefault(day, []).append(r)
        days = sorted([d for d in by_day.keys() if len(d) == 10])
        train_pool: list[dict[str, Any]] = []
        out_days: list[dict[str, Any]] = []

        for day in days:
            test_rows = by_day.get(day, [])
            if len(train_pool) < int(min_train_labels):
                train_pool.extend(test_rows)
                continue

            # Champion: cell-gated allow list from prior days
            cell_stats: dict[tuple[str, str, str], dict[str, float]] = {}
            cell_counts: dict[tuple[str, str, str], int] = {}
            for r in train_pool:
                key = (
                    str(r.get("symbol") or "").upper(),
                    str(r.get("indicator_regime") or "unknown").strip().lower(),
                    self._session_bucket_from_ts(str(r.get("timestamp") or "")),
                )
                cell_counts[key] = cell_counts.get(key, 0) + 1
                cell_stats.setdefault(key, {"signed": 0.0, "wins": 0.0})
                sb = float(r.get("signed_return_bps", 0.0))
                cell_stats[key]["signed"] += sb
                if sb > 0:
                    cell_stats[key]["wins"] += 1.0
            allow: set[tuple[str, str, str]] = set()
            for k, n in cell_counts.items():
                if n < int(min_cell_labels):
                    continue
                signed = float(cell_stats[k]["signed"]) / float(max(1, n))
                acc = float(cell_stats[k]["wins"]) / float(max(1, n))
                if signed > 0.0 and acc >= 0.5:
                    allow.add(k)

            champion_rows = []
            challenger_rows = []
            for r in test_rows:
                key = (
                    str(r.get("symbol") or "").upper(),
                    str(r.get("indicator_regime") or "unknown").strip().lower(),
                    self._session_bucket_from_ts(str(r.get("timestamp") or "")),
                )
                if key in allow:
                    champion_rows.append(r)
                if float(r.get("confidence", 0.0)) >= float(challenger_min_confidence):
                    challenger_rows.append(r)

            def _agg(selected: list[dict[str, Any]]) -> tuple[int, float, float]:
                n = len(selected)
                if n <= 0:
                    return 0, 0.0, 0.0
                signed = sum(float(x.get("signed_return_bps", 0.0)) for x in selected) / float(n)
                acc = sum(1.0 for x in selected if float(x.get("signed_return_bps", 0.0)) > 0.0) / float(n)
                return n, acc, signed

            c_n, c_acc, c_signed = _agg(champion_rows)
            ch_n, ch_acc, ch_signed = _agg(challenger_rows)
            out_days.append(
                {
                    "day": day,
                    "champion_count": c_n,
                    "champion_accuracy": c_acc,
                    "champion_signed_bps": c_signed,
                    "challenger_count": ch_n,
                    "challenger_accuracy": ch_acc,
                    "challenger_signed_bps": ch_signed,
                    "delta_signed_bps": c_signed - ch_signed,
                    "delta_accuracy": c_acc - ch_acc,
                    "allowed_cells": len(allow),
                    "train_pool_size": len(train_pool),
                }
            )
            train_pool.extend(test_rows)

        def _weighted(days_rows: list[dict[str, Any]], count_key: str, signed_key: str) -> tuple[int, float]:
            n = sum(int(d.get(count_key, 0) or 0) for d in days_rows)
            if n <= 0:
                return 0, 0.0
            s = sum(float(d.get(signed_key, 0.0)) * int(d.get(count_key, 0) or 0) for d in days_rows) / float(n)
            return n, s

        champ_n, champ_signed = _weighted(out_days, "champion_count", "champion_signed_bps")
        chall_n, chall_signed = _weighted(out_days, "challenger_count", "challenger_signed_bps")
        valid_days = [
            d for d in out_days
            if int(d.get("champion_count", 0) or 0) >= int(min_daily_selections)
            and int(d.get("challenger_count", 0) or 0) >= int(min_daily_selections)
        ]
        deltas = [float(d.get("delta_signed_bps", 0.0) or 0.0) for d in valid_days]
        mean_delta = sum(deltas) / float(max(1, len(deltas)))
        if len(deltas) > 1:
            variance = sum((x - mean_delta) ** 2 for x in deltas) / float(len(deltas) - 1)
            std = math.sqrt(max(0.0, variance))
            se = std / math.sqrt(float(len(deltas)))
            ci_low = mean_delta - 1.96 * se
            ci_high = mean_delta + 1.96 * se
        else:
            se = 0.0
            ci_low = mean_delta
            ci_high = mean_delta
        significant = len(valid_days) >= 5 and ci_low > 0.0
        return {
            "ok": True,
            "lookback": int(lookback),
            "horizon_minutes": int(horizon_minutes),
            "quality_mode": str(quality_mode or "good_only"),
            "min_train_labels": int(min_train_labels),
            "min_cell_labels": int(min_cell_labels),
            "challenger_min_confidence": float(challenger_min_confidence),
            "min_daily_selections": int(min_daily_selections),
            "days": out_days,
            "champion_overall": {"count": int(champ_n), "signed_bps": float(champ_signed)},
            "challenger_overall": {"count": int(chall_n), "signed_bps": float(chall_signed)},
            "overall_delta_signed_bps": float(champ_signed - chall_signed),
            "significance": {
                "valid_days": int(len(valid_days)),
                "mean_delta_signed_bps": float(mean_delta),
                "std_error": float(se),
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "significant_positive": bool(significant),
            },
        }

    def cost_stress_report(
        self,
        *,
        lookback: int = 100000,
        horizon_minutes: int = 15,
        quality_mode: str = "good_only",
        multipliers: tuple[float, ...] = (1.0, 1.5, 2.0),
    ) -> dict[str, Any]:
        rows = self._persistence.list_data_samples(max(1, min(100000, int(lookback))))
        labels = build_prediction_labels(
            list(reversed(rows)),
            horizons_minutes=(int(horizon_minutes),),
            min_confidence=0.0,
            quality_mode=quality_mode,
            **self._research_label_filters(),
        ).get("labels_by_horizon", {}).get(int(horizon_minutes), [])
        base_slip = float(self._settings.cost_slippage_bps_per_side)
        base_rtt = 2.0 * base_slip
        out: list[dict[str, Any]] = []
        for m in multipliers:
            mm = max(0.0, float(m))
            costs = base_rtt * mm
            if labels:
                stressed = [float(r.get("signed_return_bps", 0.0)) - costs for r in labels]
                avg = sum(stressed) / float(len(stressed))
                win = sum(1.0 for x in stressed if x > 0.0) / float(len(stressed))
            else:
                avg = 0.0
                win = 0.0
            out.append(
                {
                    "slippage_multiplier": mm,
                    "round_trip_slippage_bps": costs,
                    "label_count": int(len(labels)),
                    "avg_signed_bps_after_cost": float(avg),
                    "win_rate_after_cost": float(win),
                }
            )
        return {
            "ok": True,
            "lookback": int(lookback),
            "horizon_minutes": int(horizon_minutes),
            "quality_mode": str(quality_mode or "good_only"),
            "base_slippage_bps_per_side": float(base_slip),
            "rows": out,
        }

    def promotion_candidates_report(
        self,
        *,
        lookback: int = 100000,
        horizons_minutes: tuple[int, ...] = (5, 15, 30),
        quality_mode: str = "good_only",
        n_bootstrap: int = 300,
    ) -> dict[str, Any]:
        min_labels = max(1, int(self._settings.promotion_cell_min_labels))
        stress_mult = max(0.0, float(self._settings.promotion_cost_stress_multiplier))
        require_ci = bool(self._settings.promotion_require_ci95_positive)
        require_oos = bool(self._settings.promotion_require_oos_positive)

        cell_report = self.cell_leaderboard_bootstrap_report(
            lookback=lookback,
            horizons_minutes=horizons_minutes,
            quality_mode=quality_mode,
            min_labels=min_labels,
            n_bootstrap=n_bootstrap,
            robust_only=False,
        )
        oos = self.champion_challenger_daily_report(
            lookback=lookback,
            horizon_minutes=15,
            quality_mode=quality_mode,
            min_train_labels=150,
            min_cell_labels=max(12, min_labels // 2),
            challenger_min_confidence=0.55,
            min_daily_selections=10,
        )
        oos_positive = float(oos.get("overall_delta_signed_bps", 0.0) or 0.0) > 0.0
        oos_ok = bool(oos_positive) if require_oos else True
        round_trip_cost = 2.0 * float(self._settings.cost_slippage_bps_per_side) * stress_mult

        promoted: list[dict[str, Any]] = []
        for r in list(cell_report.get("rows") or []):
            n = int(r.get("count", 0) or 0)
            avg_signed = float(r.get("avg_signed_return_bps", 0.0) or 0.0)
            ci_low = float(r.get("ci95_low_signed_bps", 0.0) or 0.0)
            stressed_signed = avg_signed - round_trip_cost
            ci_ok = (ci_low > 0.0) if require_ci else True
            if n >= min_labels and stressed_signed > 0.0 and ci_ok and oos_ok:
                row = dict(r)
                row["stressed_signed_bps"] = stressed_signed
                row["round_trip_stress_cost_bps"] = round_trip_cost
                promoted.append(row)
        promoted.sort(
            key=lambda x: (float(x.get("stressed_signed_bps", 0.0)), float(x.get("ci95_low_signed_bps", 0.0)), int(x.get("count", 0))),
            reverse=True,
        )
        return {
            "ok": True,
            "lookback": int(lookback),
            "quality_mode": str(quality_mode or "good_only"),
            "thresholds": {
                "min_labels": int(min_labels),
                "cost_stress_multiplier": float(stress_mult),
                "round_trip_stress_cost_bps": float(round_trip_cost),
                "require_ci95_positive": bool(require_ci),
                "require_oos_positive": bool(require_oos),
            },
            "oos_gate": {
                "overall_delta_signed_bps": float(oos.get("overall_delta_signed_bps", 0.0) or 0.0),
                "passed": bool(oos_ok),
            },
            "candidate_count": int(len(promoted)),
            "candidates": promoted[:50],
        }

    def quality_controls_status(self, *, lookback: int = 2000) -> dict[str, Any]:
        self._evaluate_coverage_guard(force=True)
        self._evaluate_regime_gate(force=True)
        self._evaluate_horizon_gate(force=True)
        quarantine = self._update_symbol_quarantines(force=True)
        rows = self._persistence.list_data_samples(max(100, min(100000, int(lookback))))
        skips = 0
        for r in rows:
            md = dict(r.get("metadata") or {})
            if isinstance(md.get("sample_balance_skipped"), dict):
                skips += 1
        skip_rate = skips / float(max(1, len(rows)))
        regime_allowed_cells = 0
        for cells in ((self._regime_gate_state.get("controls") or {}).get("allowed_pairs") or {}).values():
            regime_allowed_cells += len(list(cells or []))
        horizon_map = dict((self._horizon_gate_state.get("controls") or {}).get("symbol_best_horizon") or {})
        auto_state = self._load_automation_state()
        return {
            "ok": True,
            "coverage_guard": dict(self._coverage_guard_state or {}),
            "confidence_controls": {
                "evaluated_at": self._confidence_control_state.get("evaluated_at"),
                "enabled": bool(self._confidence_control_state.get("enabled", False)),
                "controls": dict(self._confidence_control_state.get("controls") or {}),
            },
            "regime_allowed_cells": int(regime_allowed_cells),
            "best_horizon_symbols": int(len(horizon_map)),
            "best_horizon_map": horizon_map,
            "policy_tier": {
                "enabled": bool(self._settings.robust_policy_tiering_enabled),
                "active": str(auto_state.get("policy_tier_active") or "balanced"),
                "last_changed_at": auto_state.get("policy_tier_last_changed_at"),
            },
            "symbol_quarantine": quarantine,
            "sample_balance_skip_rate": float(skip_rate),
            "sample_balance_skipped": int(skips),
            "sample_rows": int(len(rows)),
        }

    def data_quality_counters(self, *, lookback: int = 2000, since_timestamp: str | None = None) -> dict[str, Any]:
        rows = self._persistence.list_data_samples(max(100, min(100000, int(lookback))))
        since_dt: datetime | None = None
        if since_timestamp:
            try:
                parsed = datetime.fromisoformat(str(since_timestamp))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                since_dt = parsed.astimezone(timezone.utc)
            except Exception:
                since_dt = None

        if since_dt is not None:
            filtered: list[dict[str, Any]] = []
            for r in rows:
                ts = str(r.get("timestamp") or "")
                try:
                    dt = datetime.fromisoformat(ts)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    dt = dt.astimezone(timezone.utc)
                except Exception:
                    continue
                if dt >= since_dt:
                    filtered.append(r)
            rows = filtered
        total = len(rows)
        stale = 0
        wide = 0
        outside = 0
        missing_forecast = 0
        fallback = 0
        malformed = 0
        no_quote = 0
        elig_sum = 0.0
        elig_high = 0
        elig_medium = 0
        elig_low = 0
        for r in rows:
            md = dict(r.get("metadata") or {})
            sq = dict(md.get("sample_quality") or {})
            flags = dict(sq.get("flags") or {})
            if bool(r.get("quality_quote_stale", flags.get("quote_stale", False))):
                stale += 1
            if bool(r.get("quality_spread_too_wide", flags.get("spread_too_wide", False))):
                wide += 1
            if bool(r.get("quality_outside_session", flags.get("outside_session", False))):
                outside += 1
            if bool(r.get("quality_missing_forecast", flags.get("missing_forecast", False))):
                missing_forecast += 1
            quote_last = float(r.get("quote_last", 0.0) or 0.0)
            if quote_last <= 0.0:
                no_quote += 1
            lr = dict(md.get("llm_routing") or {})
            reason = str(lr.get("state_gate_reason") or "").lower()
            sec_reason = str(lr.get("secondary_reason") or "").lower()
            if "fallback" in reason or "unavailable" in reason or "fallback" in sec_reason:
                fallback += 1
            if "no json object found" in reason or "parse_failed" in reason or "max_tokens" in sec_reason:
                malformed += 1
            elig = self._sample_eligibility_score(
                quote_stale=bool(r.get("quality_quote_stale", flags.get("quote_stale", False))),
                spread_too_wide=bool(r.get("quality_spread_too_wide", flags.get("spread_too_wide", False))),
                outside_session=bool(r.get("quality_outside_session", flags.get("outside_session", False))),
                missing_forecast=bool(r.get("quality_missing_forecast", flags.get("missing_forecast", False))),
                no_quote=bool(quote_last <= 0.0),
                fallback=bool("fallback" in reason or "unavailable" in reason or "fallback" in sec_reason),
                malformed=bool("no json object found" in reason or "parse_failed" in reason or "max_tokens" in sec_reason),
            )
            elig_sum += float(elig["score"])
            grade = str(elig["grade"])
            if grade == "high":
                elig_high += 1
            elif grade == "medium":
                elig_medium += 1
            else:
                elig_low += 1

        def _ratio(v: int) -> float:
            return float(v) / float(max(1, total))

        return {
            "ok": True,
            "lookback": int(lookback),
            "since_timestamp": since_dt.isoformat() if since_dt else None,
            "rows": int(total),
            "counts": {
                "quote_stale": int(stale),
                "spread_too_wide": int(wide),
                "outside_session": int(outside),
                "missing_forecast": int(missing_forecast),
                "fallback_decisions": int(fallback),
                "malformed_output": int(malformed),
                "no_quote": int(no_quote),
            },
            "rates": {
                "quote_stale_rate": _ratio(stale),
                "spread_too_wide_rate": _ratio(wide),
                "outside_session_rate": _ratio(outside),
                "missing_forecast_rate": _ratio(missing_forecast),
                "fallback_rate": _ratio(fallback),
                "malformed_output_rate": _ratio(malformed),
                "no_quote_rate": _ratio(no_quote),
            },
            "sample_eligibility": {
                "avg_score": (elig_sum / float(max(1, total))),
                "high_count": int(elig_high),
                "medium_count": int(elig_medium),
                "low_count": int(elig_low),
                "high_rate": _ratio(elig_high),
                "medium_rate": _ratio(elig_medium),
                "low_rate": _ratio(elig_low),
            },
        }

    def _sample_eligibility_score(
        self,
        *,
        quote_stale: bool,
        spread_too_wide: bool,
        outside_session: bool,
        missing_forecast: bool,
        no_quote: bool,
        fallback: bool,
        malformed: bool,
    ) -> dict[str, Any]:
        score = 1.0
        penalties = {
            "quote_stale": 0.35,
            "spread_too_wide": 0.25,
            "outside_session": 0.15,
            "missing_forecast": 0.15,
            "no_quote": 0.30,
            "fallback": 0.10,
            "malformed_output": 0.20,
        }
        applied: list[str] = []
        checks = {
            "quote_stale": bool(quote_stale),
            "spread_too_wide": bool(spread_too_wide),
            "outside_session": bool(outside_session),
            "missing_forecast": bool(missing_forecast),
            "no_quote": bool(no_quote),
            "fallback": bool(fallback),
            "malformed_output": bool(malformed),
        }
        for name, active in checks.items():
            if active:
                score -= float(penalties.get(name, 0.0))
                applied.append(name)
        score = max(0.0, min(1.0, score))
        grade = "high" if score >= 0.85 else ("medium" if score >= 0.65 else "low")
        return {"score": score, "grade": grade, "penalties": applied}

    def data_readiness_status(self, *, lookback: int = 2000) -> dict[str, Any]:
        dq = self.data_quality_counters(lookback=lookback)
        qc = self.quality_controls_status(lookback=lookback)
        rates = dict(dq.get("rates") or {})
        elig = dict(dq.get("sample_eligibility") or {})
        coverage = dict(qc.get("coverage_guard") or {})
        quality_score = max(
            0.0,
            min(
                1.0,
                1.0
                - (
                    0.20 * float(rates.get("quote_stale_rate", 0.0) or 0.0)
                    + 0.15 * float(rates.get("spread_too_wide_rate", 0.0) or 0.0)
                    + 0.10 * float(rates.get("outside_session_rate", 0.0) or 0.0)
                    + 0.20 * float(rates.get("no_quote_rate", 0.0) or 0.0)
                    + 0.20 * float(rates.get("missing_forecast_rate", 0.0) or 0.0)
                    + 0.10 * float(rates.get("fallback_rate", 0.0) or 0.0)
                    + 0.05 * float(rates.get("malformed_output_rate", 0.0) or 0.0)
                ),
            ),
        )
        target = max(1.0, float(coverage.get("target", 1.0) or 1.0))
        labels = max(0.0, float(coverage.get("daily_labels", 0.0) or 0.0))
        coverage_score = max(0.0, min(1.0, labels / target))
        eligibility_score = max(0.0, min(1.0, float(elig.get("avg_score", 0.0) or 0.0)))
        skip_score = max(0.0, min(1.0, 1.0 - float(qc.get("sample_balance_skip_rate", 0.0) or 0.0)))
        overall = (
            (0.40 * quality_score)
            + (0.25 * eligibility_score)
            + (0.25 * coverage_score)
            + (0.10 * skip_score)
        )
        score = int(round(max(0.0, min(1.0, overall)) * 100.0))
        grade = "ready" if score >= 80 else ("watch" if score >= 60 else "not_ready")
        return {
            "ok": True,
            "lookback": int(lookback),
            "score": score,
            "grade": grade,
            "components": {
                "quality": quality_score,
                "eligibility": eligibility_score,
                "coverage": coverage_score,
                "sample_balance": skip_score,
            },
            "inputs": {
                "rows": int(dq.get("rows", 0) or 0),
                "daily_labels": labels,
                "daily_label_target": target,
                "sample_balance_skip_rate": float(qc.get("sample_balance_skip_rate", 0.0) or 0.0),
                "rates": rates,
                "sample_eligibility": elig,
            },
        }

    def trades(self, limit: int) -> list[dict[str, Any]]:
        return self._persistence.list_closed_trades(limit)

    def runtime_config(self) -> dict[str, Any] | None:
        return self._persistence.get_runtime_config()

    def economic_calendar(self) -> dict[str, Any]:
        return load_economic_calendar(self._settings).to_record()

    def finnhub_context(self, symbol: str | None = None) -> dict[str, Any]:
        return load_finnhub_context(self._settings, symbol or self._settings.symbol).to_record()

    def symbol_collection_status(self) -> dict[str, Any]:
        return {
            "primary_symbol": self._settings.symbol,
            "configured_symbols": list(self._configured_collection_symbols()),
            "multi_symbol_enabled": bool(self._settings.multi_symbol_enabled),
            "shadow_enabled": bool(self._settings.multi_symbol_shadow_enabled),
            "paper_only": bool(self._settings.multi_symbol_paper_only),
            "shadow_symbols": list(self._shadow_symbols),
            "last_shadow_symbol": self._last_shadow_symbol,
            "shadow_execution_provider": "mock" if self._shadow_symbols else None,
        }

    def notifications(self, limit: int = 50) -> list[dict[str, Any]]:
        return list_notifications(self._settings.notifications_path, limit=limit)

    def adaptive_snapshot(self) -> dict[str, Any]:
        return self._engine.learner.snapshot()

    def acceleration_status(self) -> dict[str, Any]:
        return self._engine.acceleration_snapshot()

    def llm_provider_health_status(self) -> dict[str, Any]:
        return self._engine.llm_provider_health_snapshot()

    def _apply_llm_hard_error_guard(self, *, window_minutes: int = 60) -> dict[str, Any]:
        report = self.autonomy_error_causes_report(window_minutes=max(15, int(window_minutes)))
        hard = dict(report.get("top_hard_error_cause") or {})
        llm_rate = 0.0
        for c in list(report.get("hard_error_causes") or []):
            if str(c.get("cause") or "") == "llm_unavailable":
                llm_rate = float(c.get("share", 0.0) or 0.0)
                break
        triggered = llm_rate >= 0.10
        action = "none"
        if triggered:
            # Prefer forcing primary to the configured secondary provider during cooldown.
            target = str(self._settings.llm_two_tier_secondary_provider or "").strip().lower()
            cooldown = max(60, int(self._settings.llm_provider_cooldown_seconds))
            self._engine.force_primary_provider(target, cooldown_seconds=cooldown)
            action = f"force_primary_provider:{target}:{cooldown}s"
            state = self._load_automation_state()
            state["llm_guard_last"] = {
                "at": datetime.now(tz=timezone.utc).isoformat(),
                "triggered": True,
                "llm_unavailable_share": llm_rate,
                "top_hard_error": hard,
                "action": action,
                "baseline_llm_unavailable_share": llm_rate,
            }
            self._save_automation_state(state)
            self._notify("llm_hard_error_guard_triggered", state["llm_guard_last"])
        return {
            "ok": True,
            "window_minutes": int(window_minutes),
            "triggered": bool(triggered),
            "llm_unavailable_share": float(llm_rate),
            "top_hard_error": hard,
            "action": action,
        }

    def _update_llm_remediation_outcome(self) -> dict[str, Any]:
        state = self._load_automation_state()
        base = dict(state.get("llm_guard_last") or {})
        if not base:
            return {"ok": True, "updated": False, "reason": "no_llm_guard_baseline"}
        at_raw = str(base.get("at") or "")
        if not at_raw:
            return {"ok": True, "updated": False, "reason": "missing_baseline_timestamp"}
        try:
            at = datetime.fromisoformat(at_raw)
            if at.tzinfo is None:
                at = at.replace(tzinfo=timezone.utc)
        except Exception:
            return {"ok": True, "updated": False, "reason": "invalid_baseline_timestamp"}
        now = datetime.now(tz=timezone.utc)
        mins = (now - at).total_seconds() / 60.0
        report = self.autonomy_error_causes_report(window_minutes=60)
        current = 0.0
        for c in list(report.get("hard_error_causes") or []):
            if str(c.get("cause") or "") == "llm_unavailable":
                current = float(c.get("share", 0.0) or 0.0)
                break
        baseline = float(base.get("baseline_llm_unavailable_share", 0.0) or 0.0)
        out = dict(state.get("llm_guard_outcome") or {})
        changed = False
        if mins >= 30 and out.get("delta_30m") is None:
            out["delta_30m"] = float(current - baseline)
            out["rate_30m"] = float(current)
            changed = True
        if mins >= 60 and out.get("delta_60m") is None:
            out["delta_60m"] = float(current - baseline)
            out["rate_60m"] = float(current)
            changed = True
        out["updated_at"] = now.isoformat()
        state["llm_guard_outcome"] = out
        if changed:
            self._save_automation_state(state)
            self._notify("llm_guard_outcome_updated", {"baseline": baseline, "current": current, **out})
        return {"ok": True, "updated": bool(changed), "baseline": baseline, "current": current, "outcome": out}

    def set_acceleration_mode(self, mode: str) -> dict[str, Any]:
        normalized = str(mode or "").strip().lower()
        enabled = normalized == "accelerated"
        if normalized not in {"standard", "accelerated"}:
            raise ValueError("mode must be one of: standard, accelerated")
        with self._lock:
            accel = self._engine.set_acceleration_mode(enabled)
            self._save_runtime_config()
            self._last_note = f"acceleration mode set to {accel.get('mode', normalized)}"
        return {
            "ok": True,
            "requested_mode": normalized,
            "applied": accel,
        }

    def metrics_snapshot(self, limit: int = 500, *, include_shadow: bool = True) -> dict[str, Any]:
        trades = self._persistence.list_closed_trades(limit)
        if not include_shadow:
            trades = [
                t
                for t in trades
                if str(((t.get("metadata") or {}).get("collection_role") or "primary")).lower() == "primary"
            ]
        decisions = self._persistence.list_decisions(limit)

        pnls = [float(t.get("pnl", 0.0)) for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total = float(sum(pnls))
        trade_count = len(pnls)
        win_rate = (len(wins) / trade_count) if trade_count else 0.0
        avg_win = (sum(wins) / len(wins)) if wins else 0.0
        avg_loss = (sum(losses) / len(losses)) if losses else 0.0
        expectancy = (win_rate * avg_win) + ((1.0 - win_rate) * avg_loss)

        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else math.inf

        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnls:
            equity += p
            peak = max(peak, equity)
            max_dd = min(max_dd, equity - peak)

        mean = (sum(pnls) / trade_count) if trade_count else 0.0
        var = (sum((p - mean) ** 2 for p in pnls) / max(1, trade_count - 1)) if trade_count > 1 else 0.0
        std = math.sqrt(var)
        sharpe_like = (mean / std * math.sqrt(trade_count)) if std > 0 else 0.0

        trade_actions = sum(1 for d in decisions if str((d.get("decision") or {}).get("action", "")).lower() == "trade")
        hold_actions = sum(1 for d in decisions if str((d.get("decision") or {}).get("action", "")).lower() == "hold")
        action_total = trade_actions + hold_actions
        hold_rate = (hold_actions / action_total) if action_total else 0.0

        adaptive = self._engine.learner.snapshot()
        regime_direction = adaptive.get("regime_direction", {}) if isinstance(adaptive, dict) else {}

        return {
            "include_shadow": include_shadow,
            "trade_count": trade_count,
            "net_pnl": total,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "expectancy": expectancy,
            "profit_factor": profit_factor,
            "max_drawdown": max_dd,
            "sharpe_like": sharpe_like,
            "decision_count": action_total,
            "trade_action_count": trade_actions,
            "hold_action_count": hold_actions,
            "hold_rate": hold_rate,
            "regime_direction_stats": regime_direction,
            "anti_decay": self._engine.edge_monitor.snapshot(),
            "model_monitoring": self.model_monitoring_status(),
        }

    def model_monitoring_status(self) -> dict[str, Any]:
        state = self._load_model_monitor_state()
        out = {
            "enabled": bool(self._settings.model_monitoring_enabled),
            "state_path": str(self._model_monitor_state_path),
            "required_breach_streak": int(self._settings.model_monitor_breach_streak),
            "breach_streak": int(state.get("breach_streak", 0)),
            "safe_mode_active": bool(state.get("safe_mode_active", False)),
            "last_evaluated_at": state.get("last_evaluated_at"),
            "last_breached_at": state.get("last_breached_at"),
            "safe_mode_triggered_at": state.get("safe_mode_triggered_at"),
            "last_report": state.get("last_report", {}),
        }
        return out

    def portfolio_optimisation(
        self,
        *,
        lookback: int = 5000,
        min_trades: int = 5,
        max_weight: float = 0.35,
        cash_floor: float = 0.25,
        include_shadow: bool = True,
    ) -> dict[str, Any]:
        max_rows = max(1, min(50000, int(lookback)))
        rows = self._persistence.list_closed_trades(max_rows)
        if not include_shadow:
            rows = [
                r
                for r in rows
                if str(((r.get("metadata") or {}).get("collection_role") or "primary")).lower() == "primary"
            ]
        trades = [_restore_trade(row) for row in reversed(rows)]
        report = optimise_portfolio(
            trades,
            min_trades=max(1, int(min_trades)),
            max_weight=max(0.01, min(1.0, float(max_weight))),
            cash_floor=max(0.0, min(1.0, float(cash_floor))),
        )
        report["lookback"] = max_rows
        report["include_shadow"] = bool(include_shadow)
        return report

    def retrain_adaptive_from_history(self, limit: int = 2000) -> dict[str, Any]:
        max_rows = max(1, min(50000, int(limit)))
        rows = self._persistence.list_closed_trades(max_rows)
        trades: list[ClosedTrade] = []
        regime_backfilled = 0
        confidence_backfilled = 0
        for row in reversed(rows):
            trade = _restore_trade(row)
            trade, rb, cb = _enrich_trade_for_learning(trade)
            regime_backfilled += int(rb)
            confidence_backfilled += int(cb)
            trades.append(trade)

        with self._lock:
            result = self._engine.learner.retrain_from_trades(trades)
            self._last_note = f"adaptive retrain complete (used={result['used_trades']}, skipped={result['skipped_trades']})"
        return {
            "retrained": True,
            "lookback": max_rows,
            "regime_backfilled": regime_backfilled,
            "confidence_backfilled": confidence_backfilled,
            **result,
        }

    def research_walk_forward_report(
        self,
        *,
        lookback: int = 10000,
        folds: int = 4,
        min_train: int = 40,
        min_test: int = 20,
        bins: int = 10,
    ) -> dict[str, Any]:
        max_rows = max(1, min(50000, int(lookback)))
        rows = self._persistence.list_closed_trades(max_rows)
        trades: list[ClosedTrade] = []
        regime_backfilled = 0
        confidence_backfilled = 0
        for row in reversed(rows):
            trade = _restore_trade(row)
            trade, rb, cb = _enrich_trade_for_learning(trade)
            regime_backfilled += int(rb)
            confidence_backfilled += int(cb)
            trades.append(trade)

        samples = build_trade_dataset(trades)
        report = run_walk_forward(
            samples,
            folds=folds,
            min_train=min_train,
            min_test=min_test,
            bins=bins,
        )
        out_path = save_walk_forward_report(report)
        with self._lock:
            self._last_note = f"walk-forward research report saved ({out_path.name})"
        return {
            "ok": bool(report.get("ok", False)),
            "report_path": str(out_path),
            "lookback": max_rows,
            "sample_count": len(samples),
            "regime_backfilled": regime_backfilled,
            "confidence_backfilled": confidence_backfilled,
            "aggregate": report.get("aggregate", {}),
            "report": report,
        }

    def research_predictive_model_report(
        self,
        *,
        lookback: int = 10000,
        folds: int = 4,
        min_train: int = 40,
        min_test: int = 20,
        n_estimators: int = 80,
        learning_rate: float = 0.1,
        max_bins: int = 16,
    ) -> dict[str, Any]:
        max_rows = max(1, min(50000, int(lookback)))
        rows = self._persistence.list_closed_trades(max_rows)
        trades: list[ClosedTrade] = []
        regime_backfilled = 0
        confidence_backfilled = 0
        for row in reversed(rows):
            trade = _restore_trade(row)
            trade, rb, cb = _enrich_trade_for_learning(trade)
            regime_backfilled += int(rb)
            confidence_backfilled += int(cb)
            trades.append(trade)

        samples = build_predictive_dataset(trades)
        report = run_predictive_walk_forward(
            samples,
            folds=folds,
            min_train=min_train,
            min_test=min_test,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_bins=max_bins,
        )
        out_path = save_predictive_report(report)
        with self._lock:
            self._last_note = f"predictive research report saved ({out_path.name})"
        return {
            "ok": bool(report.get("ok", False)),
            "report_path": str(out_path),
            "lookback": max_rows,
            "sample_count": len(samples),
            "regime_backfilled": regime_backfilled,
            "confidence_backfilled": confidence_backfilled,
            "aggregate": report.get("aggregate", {}),
            "report": report,
        }

    def policy_tier_performance_report(self, *, lookback: int = 100000, horizon_minutes: int = 15) -> dict[str, Any]:
        horizon = max(1, min(390, int(horizon_minutes)))
        rows = self._persistence.list_data_samples(max(200, min(100000, int(lookback))))
        labels = build_prediction_labels(
            list(reversed(rows)),
            horizons_minutes=(horizon,),
            min_confidence=0.0,
            quality_mode="good_only",
            **self._research_label_filters(),
        ).get("labels_by_horizon", {}).get(horizon, [])
        by_tier: dict[str, list[dict[str, Any]]] = {"strict": [], "balanced": [], "explore": [], "unknown": []}
        for row in labels:
            md = dict(row.get("metadata") or {})
            tier = str(md.get("policy_tier") or "unknown").strip().lower()
            if tier not in by_tier:
                tier = "unknown"
            by_tier[tier].append(row)
        tier_rows: list[dict[str, Any]] = []
        for tier in ("strict", "balanced", "explore", "unknown"):
            items = by_tier.get(tier, [])
            n = len(items)
            if n <= 0:
                tier_rows.append({"policy_tier": tier, "count": 0, "accuracy": 0.0, "avg_signed_return_bps": 0.0, "brier": 0.0})
                continue
            wins = sum(1.0 for b in items if float(b.get("signed_return_bps", 0.0) or 0.0) > 0.0)
            signed = sum(float(b.get("signed_return_bps", 0.0) or 0.0) for b in items) / float(n)
            brier = sum(float(b.get("brier_score", 0.0) or 0.0) for b in items) / float(n)
            tier_rows.append(
                {
                    "policy_tier": tier,
                    "count": int(n),
                    "accuracy": float(wins / float(n)),
                    "avg_signed_return_bps": float(signed),
                    "brier": float(brier),
                }
            )
        active = str((self._load_automation_state().get("policy_tier_active") or "balanced")).strip().lower()
        return {
            "ok": True,
            "horizon_minutes": horizon,
            "active_tier": active if active in {"strict", "balanced", "explore"} else "balanced",
            "rows": tier_rows,
        }

    def automation_status(self) -> dict[str, Any]:
        state = self._load_automation_state()
        self._repair_daily_research_summary_if_inconsistent(state)
        state = self._load_automation_state()
        quarantine = self._update_symbol_quarantines(force=True)
        return {
            "enabled": bool(self._settings.auto_retrain_enabled),
            "check_seconds": int(self._settings.auto_retrain_check_seconds),
            "interval_hours": int(self._settings.auto_retrain_interval_hours),
            "min_new_trades": int(self._settings.auto_retrain_min_new_trades),
            "neg_expectancy_lookback": int(self._settings.auto_retrain_neg_expectancy_lookback),
            "retrain_lookback": int(self._settings.auto_retrain_lookback),
            "auto_research_enabled": bool(self._settings.auto_research_enabled),
            "research_lookback": int(self._settings.auto_research_lookback),
            "research_folds": int(self._settings.auto_research_folds),
            "research_min_train": int(self._settings.auto_research_min_train),
            "research_min_test": int(self._settings.auto_research_min_test),
            "research_bins": int(self._settings.auto_research_bins),
            "daily_research_last_day": state.get("daily_research_last_day"),
            "daily_research_last_at": state.get("daily_research_last_at"),
            "daily_research_last_error": state.get("daily_research_last_error"),
            "daily_research_last": state.get("daily_research_last"),
            "quality_guard": self._evaluate_data_quality_guard(),
            "cost_guard": self._evaluate_cost_guard(),
            "auto_recovery": {
                "enabled": bool(self._settings.automation_auto_recovery_enabled),
                "last_at": state.get("auto_recovery_last_at"),
                "last_reason": state.get("auto_recovery_last_reason"),
            },
            "autonomy": self.autonomous_research_status(),
            "policy_tier": {
                "enabled": bool(self._settings.robust_policy_tiering_enabled),
                "active": str(state.get("policy_tier_active") or "balanced"),
                "last_changed_at": state.get("policy_tier_last_changed_at"),
                "performance_15m": self.policy_tier_performance_report(lookback=100000, horizon_minutes=15),
            },
            "symbol_quarantine": quarantine,
            "weekly_experiments": {
                "enabled": bool(self._settings.weekly_experiments_enabled),
                "freeze_enabled": bool(self._settings.research_freeze_enabled),
                "max_per_week": int(self._settings.weekly_experiments_max_per_week),
                "last_week": state.get("weekly_experiments_last_week"),
                "last": state.get("weekly_experiments_last"),
                "research_policy": state.get("research_policy"),
            },
            "state_path": str(self._automation_state_path),
            "state": state,
        }

    def automation_guard_status(self) -> dict[str, Any]:
        state = self._load_automation_state()
        return {
            "quality_guard": self._evaluate_data_quality_guard(force=True),
            "sample_flow_guard": self._evaluate_sample_flow_guard(force=True),
            "cost_guard": self._evaluate_cost_guard(force=True),
            "auto_recovery": {
                "enabled": bool(self._settings.automation_auto_recovery_enabled),
                "last_at": state.get("auto_recovery_last_at"),
                "last_reason": state.get("auto_recovery_last_reason"),
                "cooldown_seconds": int(self._settings.automation_auto_recovery_cooldown_seconds),
            },
            "sample_flow_recovery": {
                "last_at": state.get("sample_flow_recovery_last_at"),
                "last": state.get("sample_flow_recovery_last"),
            },
        }

    def autonomy_health_score(self) -> dict[str, Any]:
        score = 100
        reasons: list[str] = []
        status = self.autonomous_research_status()
        guards = self.automation_guard_status()
        qg = dict(guards.get("quality_guard") or {})
        sfg = dict(guards.get("sample_flow_guard") or {})
        cg = dict(guards.get("cost_guard") or {})
        ss = dict(status.get("self_scan_last") or {})

        if not bool(status.get("running", False)):
            score -= 25
            reasons.append("autonomy daemon not running")
        if bool(qg.get("active", False)):
            score -= 25
            reasons.append(f"quality guard active: {qg.get('reason', 'unknown')}")
        if bool(sfg.get("active", False)):
            score -= 20
            reasons.append(f"sample flow stall: {sfg.get('reason', 'unknown')}")
        if bool(cg.get("active", False)):
            score -= 30
            reasons.append("cost guard blocking")
        elif bool(cg.get("warning", False)):
            score -= 10
            reasons.append("cost near budget")

        sev = str(ss.get("severity") or "").lower()
        if sev == "critical":
            score -= 30
            reasons.append("self-scan critical")
        elif sev == "warning":
            score -= 15
            reasons.append("self-scan warning")
        elif sev == "":
            score -= 10
            reasons.append("self-scan not yet available")

        score = max(0, min(100, int(score)))
        grade = "A"
        if score < 90:
            grade = "B"
        if score < 75:
            grade = "C"
        if score < 60:
            grade = "D"
        if score < 40:
            grade = "F"
        return {
            "ok": True,
            "score": score,
            "grade": grade,
            "health": "healthy" if score >= 75 else ("watch" if score >= 50 else "risk"),
            "reasons": reasons[:5],
        }

    def _latest_predictive_report_path(self) -> Path | None:
        p = Path("data/research")
        if not p.exists():
            return None
        files = sorted(p.glob("predictive_walk_forward_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        return files[0] if files else None

    def _policy_from_settings(
        self,
        *,
        min_folds: int | None = None,
        min_model_selected_trades: int | None = None,
        min_expectancy: float | None = None,
        min_net_pnl_edge: float | None = None,
        require_recommendation_promote: bool | None = None,
    ) -> PromotionPolicy:
        return PromotionPolicy(
            min_folds=int(self._settings.promotion_min_folds if min_folds is None else min_folds),
            min_model_selected_trades=int(
                self._settings.promotion_min_model_selected_trades
                if min_model_selected_trades is None
                else min_model_selected_trades
            ),
            min_expectancy=float(self._settings.promotion_min_expectancy if min_expectancy is None else min_expectancy),
            min_net_pnl_edge=float(
                self._settings.promotion_min_net_pnl_edge if min_net_pnl_edge is None else min_net_pnl_edge
            ),
            require_recommendation_promote=bool(
                self._settings.promotion_require_recommendation_promote
                if require_recommendation_promote is None
                else require_recommendation_promote
            ),
        )

    def evaluate_promotion_policy(
        self,
        *,
        report_path: str | None = None,
        min_folds: int | None = None,
        min_model_selected_trades: int | None = None,
        min_expectancy: float | None = None,
        min_net_pnl_edge: float | None = None,
        require_recommendation_promote: bool | None = None,
    ) -> dict[str, Any]:
        rp = Path(report_path) if report_path else self._latest_predictive_report_path()
        if rp is None or not rp.exists():
            return {
                "ok": False,
                "reason": "no_predictive_report_found",
                "report_path": str(rp) if rp else None,
            }
        report = load_json(rp)
        policy = self._policy_from_settings(
            min_folds=min_folds,
            min_model_selected_trades=min_model_selected_trades,
            min_expectancy=min_expectancy,
            min_net_pnl_edge=min_net_pnl_edge,
            require_recommendation_promote=require_recommendation_promote,
        )
        evaluation = evaluate_predictive_report(report, policy)
        return {
            "ok": True,
            "report_path": str(rp),
            "policy": {
                "min_folds": policy.min_folds,
                "min_model_selected_trades": policy.min_model_selected_trades,
                "min_expectancy": policy.min_expectancy,
                "min_net_pnl_edge": policy.min_net_pnl_edge,
                "require_recommendation_promote": policy.require_recommendation_promote,
            },
            "evaluation": evaluation,
        }

    def promote_predictive_candidate(
        self,
        *,
        report_path: str | None = None,
        min_folds: int | None = None,
        min_model_selected_trades: int | None = None,
        min_expectancy: float | None = None,
        min_net_pnl_edge: float | None = None,
        require_recommendation_promote: bool | None = None,
        note: str = "",
    ) -> dict[str, Any]:
        evaluated = self.evaluate_promotion_policy(
            report_path=report_path,
            min_folds=min_folds,
            min_model_selected_trades=min_model_selected_trades,
            min_expectancy=min_expectancy,
            min_net_pnl_edge=min_net_pnl_edge,
            require_recommendation_promote=require_recommendation_promote,
        )
        if not evaluated.get("ok"):
            return {"promoted": False, "evaluated": evaluated}

        evaluation = evaluated["evaluation"]
        passed = bool((evaluation or {}).get("passed", False))
        if not passed:
            return {"promoted": False, "evaluated": evaluated}

        state = load_json(self._promotion_state_path)
        state.update(
            {
                "last_promoted_at": utc_now_iso(),
                "last_promoted_report_path": evaluated.get("report_path"),
                "last_promotion_note": str(note or ""),
                "last_evaluation": evaluation,
            }
        )
        save_json(self._promotion_state_path, state)
        with self._lock:
            self._last_note = f"promotion approved ({Path(str(evaluated.get('report_path'))).name})"
        return {"promoted": True, "promotion_state_path": str(self._promotion_state_path), "evaluated": evaluated}

    def promotion_status(self) -> dict[str, Any]:
        return {
            "state_path": str(self._promotion_state_path),
            "state": load_json(self._promotion_state_path),
            "latest_predictive_report_path": str(self._latest_predictive_report_path() or ""),
            "default_policy": {
                "min_folds": int(self._settings.promotion_min_folds),
                "min_model_selected_trades": int(self._settings.promotion_min_model_selected_trades),
                "min_expectancy": float(self._settings.promotion_min_expectancy),
                "min_net_pnl_edge": float(self._settings.promotion_min_net_pnl_edge),
                "require_recommendation_promote": bool(self._settings.promotion_require_recommendation_promote),
            },
        }

    def go_live_gate_snapshot(self) -> dict[str, Any]:
        lookback = max(50, int(self._settings.go_live_metrics_lookback))
        metrics = self.metrics_snapshot(limit=lookback, include_shadow=False)
        starting_balance = max(1e-9, float(self._engine.account.starting_balance))
        max_dd = abs(min(0.0, float(metrics.get("max_drawdown", 0.0))))
        max_dd_pct = max_dd / starting_balance

        checks = [
            GoLiveCheck(
                name="closed_trades",
                actual=float(metrics.get("trade_count", 0)),
                threshold=float(self._settings.go_live_min_closed_trades),
                comparator=">=",
                passed=float(metrics.get("trade_count", 0)) >= float(self._settings.go_live_min_closed_trades),
            ),
            GoLiveCheck(
                name="win_rate",
                actual=float(metrics.get("win_rate", 0.0)),
                threshold=float(self._settings.go_live_min_win_rate),
                comparator=">=",
                passed=float(metrics.get("win_rate", 0.0)) >= float(self._settings.go_live_min_win_rate),
            ),
            GoLiveCheck(
                name="profit_factor",
                actual=float(metrics.get("profit_factor", 0.0)),
                threshold=float(self._settings.go_live_min_profit_factor),
                comparator=">=",
                passed=float(metrics.get("profit_factor", 0.0)) >= float(self._settings.go_live_min_profit_factor),
            ),
            GoLiveCheck(
                name="expectancy",
                actual=float(metrics.get("expectancy", 0.0)),
                threshold=float(self._settings.go_live_min_expectancy),
                comparator=">=",
                passed=float(metrics.get("expectancy", 0.0)) >= float(self._settings.go_live_min_expectancy),
            ),
            GoLiveCheck(
                name="max_drawdown_pct",
                actual=float(max_dd_pct),
                threshold=float(self._settings.go_live_max_drawdown_pct),
                comparator="<=",
                passed=float(max_dd_pct) <= float(self._settings.go_live_max_drawdown_pct),
            ),
            GoLiveCheck(
                name="hold_rate",
                actual=float(metrics.get("hold_rate", 0.0)),
                threshold=float(self._settings.go_live_max_hold_rate),
                comparator="<=",
                passed=float(metrics.get("hold_rate", 0.0)) <= float(self._settings.go_live_max_hold_rate),
            ),
        ]
        all_passed = all(c.passed for c in checks)
        live_mode = self._is_live_execution_mode()
        live_enabled = bool(self._settings.autonomous_live_enabled)
        blocked_autonomous_live = bool(live_mode and (not live_enabled or not all_passed))
        reason = "ok"
        if live_mode and not live_enabled:
            reason = "autonomous_live_disabled"
        elif live_mode and not all_passed:
            reason = "metrics_below_threshold"

        return {
            "live_mode": live_mode,
            "autonomous_live_enabled": live_enabled,
            "metrics_lookback": lookback,
            "metrics_scope": "primary_only",
            "passed": all_passed,
            "blocked_autonomous_live": blocked_autonomous_live,
            "reason": reason,
            "checks": [c.__dict__ for c in checks],
        }

    def audit_events(self, limit: int = 100) -> list[dict[str, Any]]:
        p = Path(self._settings.audit_log_path)
        if not p.exists():
            return []
        lines = p.read_text(encoding="utf-8").splitlines()
        out: list[dict[str, Any]] = []
        for raw in lines[-max(1, limit):]:
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        return out

