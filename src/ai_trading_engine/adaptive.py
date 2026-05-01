from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_trading_engine.models import AiDecision, ClosedTrade


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _bin_label(confidence: float) -> str:
    c = _clip(confidence, 0.0, 0.9999)
    low = int(c * 10) / 10
    high = low + 0.1
    return f"{low:.1f}-{high:.1f}"


@dataclass
class AdaptiveAction:
    decision: AiDecision
    note: str


class AdaptiveLearner:
    """Online performance learner for regime/direction sizing and confidence calibration."""

    def __init__(self, state_path: str, enabled: bool = True) -> None:
        self._enabled = enabled
        self._state_path = Path(state_path)
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state = self._load_state()

    def _default_state(self) -> dict[str, Any]:
        return {
            "total_trades": 0,
            "regime_direction": {},
            "confidence_bins": {},
        }

    def _load_state(self) -> dict[str, Any]:
        if not self._state_path.exists():
            return self._default_state()
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return self._default_state()
            return payload
        except Exception:
            return self._default_state()

    def _save_state(self) -> None:
        self._state_path.write_text(json.dumps(self._state, ensure_ascii=True, indent=2), encoding="utf-8")

    @staticmethod
    def _rd_key(regime: str, direction: str) -> str:
        return f"{regime}|{direction}"

    @staticmethod
    def _update_bucket(bucket: dict[str, float], pnl: float) -> None:
        count = int(bucket.get("count", 0)) + 1
        wins = int(bucket.get("wins", 0)) + (1 if pnl > 0 else 0)
        total = float(bucket.get("total_pnl", 0.0)) + pnl
        bucket["count"] = count
        bucket["wins"] = wins
        bucket["total_pnl"] = total
        bucket["avg_pnl"] = total / max(1, count)
        bucket["win_rate"] = wins / max(1, count)

    def _ingest_trade(self, state: dict[str, Any], trade: ClosedTrade) -> bool:
        direction = str(trade.direction or "")
        regime = str(trade.regime or "")
        confidence = float(trade.confidence or 0.0)
        if not direction or not regime:
            return False

        state["total_trades"] = int(state.get("total_trades", 0)) + 1

        rd = state.setdefault("regime_direction", {})
        rd_key = self._rd_key(regime, direction)
        rd_bucket = rd.setdefault(rd_key, {})
        self._update_bucket(rd_bucket, float(trade.pnl))

        bins = state.setdefault("confidence_bins", {})
        conf_key = _bin_label(confidence)
        conf_bucket = bins.setdefault(conf_key, {})
        self._update_bucket(conf_bucket, float(trade.pnl))
        return True

    def update_from_trade(self, trade: ClosedTrade) -> None:
        if not self._enabled:
            return
        if not self._ingest_trade(self._state, trade):
            return
        self._save_state()

    def retrain_from_trades(self, trades: list[ClosedTrade]) -> dict[str, Any]:
        rebuilt = self._default_state()
        used = 0
        skipped = 0
        for trade in trades:
            if self._ingest_trade(rebuilt, trade):
                used += 1
            else:
                skipped += 1

        self._state = rebuilt
        self._save_state()
        return {
            "requested_trades": len(trades),
            "used_trades": used,
            "skipped_trades": skipped,
            "state_path": str(self._state_path),
            "snapshot": self._state,
        }

    def adapt(self, decision: AiDecision, regime: str) -> AdaptiveAction:
        if not self._enabled or decision.action != "trade" or decision.direction is None:
            return AdaptiveAction(decision=decision, note="adaptive_bypass")

        rd = self._state.get("regime_direction", {})
        rd_key = self._rd_key(str(regime), str(decision.direction))
        bucket = rd.get(rd_key, {})

        count = int(bucket.get("count", 0))
        win_rate = float(bucket.get("win_rate", 0.5))
        avg_pnl = float(bucket.get("avg_pnl", 0.0))

        if count >= 8 and win_rate < 0.35:
            return AdaptiveAction(
                decision=AiDecision(
                    action="hold",
                    direction=None,
                    confidence=decision.confidence,
                    size=1,
                    sl_ticks=decision.sl_ticks,
                    tp_ticks=decision.tp_ticks,
                    reasoning=(
                        f"Adaptive hold: {regime}/{decision.direction} underperforming "
                        f"(win_rate={win_rate:.2f}, n={count}). {decision.reasoning}"
                    ),
                ),
                note="adaptive_hold_weak_regime",
            )

        conf_mult = 1.0
        size_mult = 1.0
        if count >= 5:
            conf_mult *= _clip(0.8 + (win_rate * 0.6), 0.75, 1.25)
            size_mult *= _clip(0.75 + (win_rate * 0.9), 0.5, 1.5)
            if avg_pnl < 0:
                size_mult *= 0.9

        conf_bins = self._state.get("confidence_bins", {})
        ckey = _bin_label(decision.confidence)
        cb = conf_bins.get(ckey, {})
        cb_count = int(cb.get("count", 0))
        cb_avg = float(cb.get("avg_pnl", 0.0))
        if cb_count >= 6 and cb_avg < 0:
            conf_mult *= 0.9
            size_mult *= 0.9

        adapted = AiDecision(
            action=decision.action,
            direction=decision.direction,
            confidence=_clip(decision.confidence * conf_mult, 0.0, 1.0),
            size=max(1, int(round(decision.size * size_mult))),
            sl_ticks=decision.sl_ticks,
            tp_ticks=decision.tp_ticks,
            reasoning=decision.reasoning,
        )
        note = (
            f"adaptive_adjust conf_mult={conf_mult:.2f} size_mult={size_mult:.2f} "
            f"sample={count}"
        )
        return AdaptiveAction(decision=adapted, note=note)

    def snapshot(self) -> dict[str, Any]:
        return self._state
