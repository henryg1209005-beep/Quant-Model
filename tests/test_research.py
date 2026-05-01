from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from ai_trading_engine.models import ClosedTrade
from ai_trading_engine.research import build_trade_dataset, run_walk_forward


def _sample_trade(i: int, pnl: float, confidence: float, direction: str = "LONG", regime: str = "trend") -> ClosedTrade:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ts = base + timedelta(minutes=i * 5)
    return ClosedTrade(
        symbol="SPY",
        direction=direction,  # type: ignore[arg-type]
        size=1,
        entry_price=100.0,
        exit_price=100.0 + pnl,
        pnl=pnl,
        opened_at=ts - timedelta(minutes=5),
        closed_at=ts,
        thesis=f"[REGIME:{regime}][CONF:{confidence:.2f}] synthetic",
        regime=regime,
        confidence=confidence,
    )


class TestResearch(unittest.TestCase):
    def test_walk_forward_runs(self) -> None:
        trades: list[ClosedTrade] = []
        for i in range(80):
            # Higher confidence is better in this synthetic sample.
            conf = 0.2 + ((i % 10) / 10.0) * 0.7
            pnl = 2.0 if conf >= 0.6 else -1.0
            trades.append(_sample_trade(i, pnl=pnl, confidence=conf, direction="LONG", regime="trend_up"))

        samples = build_trade_dataset(trades)
        report = run_walk_forward(samples, folds=3, min_train=30, min_test=10, bins=10)
        self.assertTrue(report.get("ok"))
        self.assertGreater(len(report.get("folds", [])), 0)


if __name__ == "__main__":
    unittest.main()
