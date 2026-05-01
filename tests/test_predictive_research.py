from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from ai_trading_engine.models import ClosedTrade
from ai_trading_engine.predictive_research import (
    build_predictive_dataset,
    run_predictive_walk_forward,
)


def _mk_trade(i: int, conf: float, pnl: float, regime: str = "trend_up", direction: str = "LONG") -> ClosedTrade:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    closed = base + timedelta(minutes=i * 5)
    opened = closed - timedelta(minutes=5)
    return ClosedTrade(
        symbol="SPY",
        direction=direction,  # type: ignore[arg-type]
        size=1,
        entry_price=100.0,
        exit_price=100.0 + pnl,
        pnl=pnl,
        opened_at=opened,
        closed_at=closed,
        thesis=f"[REGIME:{regime}][CONF:{conf:.2f}] synthetic",
        regime=regime,
        confidence=conf,
    )


class TestPredictiveResearch(unittest.TestCase):
    def test_predictive_walk_forward_executes(self) -> None:
        trades: list[ClosedTrade] = []
        for i in range(120):
            conf = 0.2 + 0.7 * ((i % 10) / 9.0)
            pnl = 2.0 if conf >= 0.6 else -1.0
            trades.append(_mk_trade(i=i, conf=conf, pnl=pnl, direction="LONG", regime="trend_up"))

        samples = build_predictive_dataset(trades)
        report = run_predictive_walk_forward(
            samples,
            folds=4,
            min_train=40,
            min_test=15,
            n_estimators=30,
            learning_rate=0.1,
            max_bins=12,
        )
        self.assertTrue(report.get("ok"))
        self.assertGreater(len(report.get("folds", [])), 0)
        self.assertIn("aggregate", report)


if __name__ == "__main__":
    unittest.main()
