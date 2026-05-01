from __future__ import annotations

import unittest

from ai_trading_engine.promotion import PromotionPolicy, evaluate_predictive_report


class TestPromotionPolicy(unittest.TestCase):
    def test_policy_passes_and_fails(self) -> None:
        report_ok = {
            "ok": True,
            "folds": [{}, {}, {}],
            "aggregate": {
                "model_selected_trades_total": 60,
                "model_selected_expectancy_weighted": 12.0,
                "model_selected_net_pnl_total": 1200.0,
                "baseline_selected_net_pnl_total": 900.0,
                "promotion_recommendation": "promote",
            },
        }
        policy = PromotionPolicy(
            min_folds=3,
            min_model_selected_trades=40,
            min_expectancy=0.0,
            min_net_pnl_edge=0.0,
            require_recommendation_promote=True,
        )
        out1 = evaluate_predictive_report(report_ok, policy)
        self.assertTrue(out1["passed"])

        report_bad = {
            "ok": True,
            "folds": [{}],
            "aggregate": {
                "model_selected_trades_total": 10,
                "model_selected_expectancy_weighted": -2.0,
                "model_selected_net_pnl_total": -100.0,
                "baseline_selected_net_pnl_total": 50.0,
                "promotion_recommendation": "hold_current",
            },
        }
        out2 = evaluate_predictive_report(report_bad, policy)
        self.assertFalse(out2["passed"])


if __name__ == "__main__":
    unittest.main()
