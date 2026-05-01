from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_trading_engine.models import AiDecision, ClosedTrade


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class ThrottleResult:
    decision: AiDecision
    note: str


class EdgeMonitor:
    """Tracks rolling edge and applies anti-decay throttling."""

    def __init__(
        self,
        state_path: str,
        window_trades: int,
        min_win_rate: float,
        min_expectancy: float,
        throttle_size_mult: float,
        shadow_enabled: bool,
    ) -> None:
        self._state_path = Path(state_path)
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._window = max(5, int(window_trades))
        self._min_win_rate = float(min_win_rate)
        self._min_expectancy = float(min_expectancy)
        self._throttle_mult = _clip(float(throttle_size_mult), 0.1, 1.0)
        self._shadow_enabled = bool(shadow_enabled)
        self._state = self._load_state()

    def _default_state(self) -> dict[str, Any]:
        return {
            "rolling_trades": [],
            "shadow": {
                "observations": 0,
                "champion_trade_count": 0,
                "challenger_trade_count": 0,
                "disagreement_count": 0,
            },
        }

    def _load_state(self) -> dict[str, Any]:
        if not self._state_path.exists():
            return self._default_state()
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return self._default_state()
            payload.setdefault("rolling_trades", [])
            payload.setdefault("shadow", self._default_state()["shadow"])
            return payload
        except Exception:
            return self._default_state()

    def _save(self) -> None:
        self._state_path.write_text(json.dumps(self._state, ensure_ascii=True, indent=2), encoding="utf-8")

    def update_trade(self, trade: ClosedTrade) -> None:
        row = {
            "pnl": float(trade.pnl),
            "regime": str(trade.regime or ""),
            "direction": str(trade.direction or ""),
        }
        buf = self._state.setdefault("rolling_trades", [])
        buf.append(row)
        if len(buf) > self._window:
            del buf[: len(buf) - self._window]
        self._save()

    def rolling_metrics(self) -> dict[str, float]:
        buf = self._state.get("rolling_trades", [])
        n = len(buf)
        if n == 0:
            return {
                "count": 0,
                "net_pnl": 0.0,
                "win_rate": 0.0,
                "expectancy": 0.0,
            }
        pnls = [float(x.get("pnl", 0.0)) for x in buf]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / n
        avg_win = (sum(wins) / len(wins)) if wins else 0.0
        avg_loss = (sum(losses) / len(losses)) if losses else 0.0
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        return {
            "count": float(n),
            "net_pnl": float(sum(pnls)),
            "win_rate": float(win_rate),
            "expectancy": float(expectancy),
        }

    def throttle(self, decision: AiDecision) -> ThrottleResult:
        if decision.action != "trade":
            return ThrottleResult(decision=decision, note="edge_bypass")

        m = self.rolling_metrics()
        n = int(m["count"])
        if n < 10:
            return ThrottleResult(decision=decision, note="edge_warmup")

        win_rate = float(m["win_rate"])
        expectancy = float(m["expectancy"])

        severe = win_rate < (self._min_win_rate - 0.12) or expectancy < (self._min_expectancy - 40.0)
        weak = win_rate < self._min_win_rate or expectancy < self._min_expectancy

        if severe:
            return ThrottleResult(
                decision=AiDecision(
                    action="hold",
                    direction=None,
                    confidence=decision.confidence,
                    size=1,
                    sl_ticks=decision.sl_ticks,
                    tp_ticks=decision.tp_ticks,
                    reasoning=(
                        f"Edge throttle hold: rolling win_rate={win_rate:.2f}, "
                        f"expectancy={expectancy:.2f}, n={n}."
                    ),
                ),
                note="edge_hold",
            )

        if weak:
            reduced = AiDecision(
                action=decision.action,
                direction=decision.direction,
                confidence=_clip(decision.confidence * 0.9, 0.0, 1.0),
                size=max(1, int(round(decision.size * self._throttle_mult))),
                sl_ticks=decision.sl_ticks,
                tp_ticks=decision.tp_ticks,
                reasoning=decision.reasoning,
            )
            return ThrottleResult(
                decision=reduced,
                note=(
                    f"edge_reduce win_rate={win_rate:.2f} expectancy={expectancy:.2f} n={n}"
                ),
            )

        return ThrottleResult(decision=decision, note="edge_ok")

    def update_shadow(self, champion_trade: bool, challenger_trade: bool) -> None:
        if not self._shadow_enabled:
            return
        shadow = self._state.setdefault("shadow", self._default_state()["shadow"])
        shadow["observations"] = int(shadow.get("observations", 0)) + 1
        shadow["champion_trade_count"] = int(shadow.get("champion_trade_count", 0)) + (1 if champion_trade else 0)
        shadow["challenger_trade_count"] = int(shadow.get("challenger_trade_count", 0)) + (1 if challenger_trade else 0)
        if champion_trade != challenger_trade:
            shadow["disagreement_count"] = int(shadow.get("disagreement_count", 0)) + 1
        self._save()

    def configure_thresholds(self, min_win_rate: float, min_expectancy: float) -> None:
        self._min_win_rate = float(min_win_rate)
        self._min_expectancy = float(min_expectancy)
        self._save()

    def snapshot(self) -> dict[str, Any]:
        return {
            "rolling": self.rolling_metrics(),
            "thresholds": {
                "window_trades": self._window,
                "min_win_rate": self._min_win_rate,
                "min_expectancy": self._min_expectancy,
                "throttle_size_mult": self._throttle_mult,
            },
            "shadow": self._state.get("shadow", {}),
        }
