from __future__ import annotations

import unittest
from dataclasses import replace

from ai_trading_engine.config import DEFAULT_SETTINGS
from ai_trading_engine.trade_costs import estimate_round_trip_cost, should_apply_cost_model


class TestTradeCosts(unittest.TestCase):
    def test_estimate_round_trip_cost_positive(self) -> None:
        cost = estimate_round_trip_cost(
            entry_price=100.0,
            exit_price=101.0,
            size=10,
            slippage_bps_per_side=2.0,
            fee_per_share=0.0035,
            min_fee_per_order=0.35,
        )
        self.assertGreater(cost, 0.0)

    def test_should_apply_cost_model_paper(self) -> None:
        settings = replace(
            DEFAULT_SETTINGS,
            cost_model_enabled=True,
            cost_apply_in_paper=True,
            cost_apply_in_mock=True,
        )
        self.assertTrue(
            should_apply_cost_model(settings, execution_provider="alpaca", is_live=False)
        )
        self.assertFalse(
            should_apply_cost_model(settings, execution_provider="alpaca", is_live=True)
        )
        self.assertTrue(
            should_apply_cost_model(settings, execution_provider="mock", is_live=False)
        )


if __name__ == "__main__":
    unittest.main()
