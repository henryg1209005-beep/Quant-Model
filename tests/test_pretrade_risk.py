from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from ai_trading_engine.models import AccountState, AiDecision, Quote
from ai_trading_engine.pre_trade_risk import evaluate_pre_trade_risk


class TestPreTradeRisk(unittest.TestCase):
    def test_blocks_order_notional_limit(self) -> None:
        account = AccountState(balance=100000.0, starting_balance=100000.0)
        decision = AiDecision(
            action="trade",
            direction="LONG",
            confidence=0.8,
            size=100,
            sl_ticks=8,
            tp_ticks=12,
            reasoning="x",
        )
        quote = Quote(
            timestamp=datetime.now(tz=timezone.utc),
            bid=100.0,
            ask=100.05,
            last=100.02,
            size=10.0,
        )
        result = evaluate_pre_trade_risk(
            account=account,
            decision=decision,
            quote=quote,
            max_order_notional_usd=5000.0,
            max_total_exposure_usd=100000.0,
            max_quote_age_seconds=10,
            max_pretrade_spread_bps=20.0,
            min_buying_power_usd=100.0,
            duplicate_cooldown_seconds=15,
            last_order_key="",
            last_order_submitted_at=None,
            current_order_key="A",
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "order_notional_limit_exceeded")

    def test_blocks_duplicate_cooldown(self) -> None:
        account = AccountState(balance=100000.0, starting_balance=100000.0)
        decision = AiDecision(
            action="trade",
            direction="LONG",
            confidence=0.8,
            size=1,
            sl_ticks=8,
            tp_ticks=12,
            reasoning="x",
        )
        quote = Quote(
            timestamp=datetime.now(tz=timezone.utc) - timedelta(seconds=1),
            bid=100.0,
            ask=100.01,
            last=100.005,
            size=10.0,
        )
        result = evaluate_pre_trade_risk(
            account=account,
            decision=decision,
            quote=quote,
            max_order_notional_usd=5000.0,
            max_total_exposure_usd=100000.0,
            max_quote_age_seconds=10,
            max_pretrade_spread_bps=20.0,
            min_buying_power_usd=100.0,
            duplicate_cooldown_seconds=30,
            last_order_key="K1",
            last_order_submitted_at=datetime.now(tz=timezone.utc),
            current_order_key="K1",
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "duplicate_order_cooldown")


if __name__ == "__main__":
    unittest.main()
