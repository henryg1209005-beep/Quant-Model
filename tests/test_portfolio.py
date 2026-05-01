from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from ai_trading_engine.models import ClosedTrade
from ai_trading_engine.portfolio import optimise_portfolio


def _trade(i: int, symbol: str, pnl: float) -> ClosedTrade:
    closed = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=i)
    return ClosedTrade(
        symbol=symbol,
        direction="LONG",
        size=1,
        entry_price=100.0,
        exit_price=100.0 + pnl,
        pnl=pnl,
        opened_at=closed - timedelta(minutes=1),
        closed_at=closed,
        thesis="synthetic",
        confidence=0.7,
    )


class TestPortfolioOptimisation(unittest.TestCase):
    def test_positive_symbols_receive_capped_weights(self) -> None:
        trades: list[ClosedTrade] = []
        for i in range(20):
            trades.append(_trade(i, "SPY", 2.0 if i % 4 else -1.0))
            trades.append(_trade(i + 100, "QQQ", 1.0 if i % 3 else -1.0))
            trades.append(_trade(i + 200, "IWM", -1.0))

        report = optimise_portfolio(trades, min_trades=5, max_weight=0.5, cash_floor=0.2)

        self.assertTrue(report["ok"])
        self.assertIn("SPY", report["weights"])
        self.assertIn("QQQ", report["weights"])
        self.assertNotIn("IWM", report["weights"])
        self.assertGreaterEqual(report["cash_weight"], 0.2)
        self.assertLessEqual(max(report["weights"].values()), 0.5)

    def test_no_positive_symbols_returns_cash(self) -> None:
        report = optimise_portfolio([_trade(i, "SPY", -1.0) for i in range(10)])

        self.assertFalse(report["ok"])
        self.assertEqual(report["cash_weight"], 1.0)
        self.assertEqual(report["weights"], {})


if __name__ == "__main__":
    unittest.main()

