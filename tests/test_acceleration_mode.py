from __future__ import annotations

import unittest
from dataclasses import replace

from ai_trading_engine.config import DEFAULT_SETTINGS
from ai_trading_engine.engine import TradingEngine


class TestAccelerationMode(unittest.TestCase):
    def test_acceleration_profile_active_in_paper_mode(self) -> None:
        settings = replace(
            DEFAULT_SETTINGS,
            data_provider="mock",
            execution_provider="mock",
            llm_provider="mock",
            data_acceleration_mode=True,
            entry_confluence_min=60.0,
            ev_min_ticks=0.2,
            max_spread_bps=6.0,
            edge_min_win_rate=0.42,
            edge_min_expectancy=0.0,
            accel_relax_confluence=8.0,
            accel_relax_ev_ticks=0.1,
            accel_spread_mult=1.5,
            accel_edge_min_win_rate_delta=0.08,
            accel_edge_min_expectancy_delta=20.0,
        )
        engine = TradingEngine(settings)
        profile = engine.acceleration_snapshot()

        self.assertTrue(profile["requested"])
        self.assertTrue(profile["active"])
        self.assertFalse(profile["blocked_live"])
        self.assertEqual(profile["mode"], "accelerated")
        self.assertEqual(float(profile["entry_confluence_min"]), 52.0)
        self.assertEqual(float(profile["ev_min_ticks"]), 0.1)
        self.assertEqual(float(profile["max_spread_bps"]), 9.0)
        self.assertAlmostEqual(float(profile["edge_min_win_rate"]), 0.34, places=6)
        self.assertEqual(float(profile["edge_min_expectancy"]), -20.0)

    def test_acceleration_blocked_for_live_endpoint(self) -> None:
        settings = replace(
            DEFAULT_SETTINGS,
            data_provider="mock",
            execution_provider="alpaca",
            llm_provider="mock",
            data_acceleration_mode=True,
            alpaca_trading_url="https://api.alpaca.markets",
            entry_confluence_min=58.0,
            ev_min_ticks=0.2,
            max_spread_bps=6.0,
        )
        engine = TradingEngine(settings)
        profile = engine.acceleration_snapshot()

        self.assertTrue(profile["requested"])
        self.assertFalse(profile["active"])
        self.assertTrue(profile["blocked_live"])
        self.assertEqual(profile["mode"], "standard_live_guard")
        self.assertEqual(float(profile["entry_confluence_min"]), 58.0)
        self.assertEqual(float(profile["ev_min_ticks"]), 0.2)
        self.assertEqual(float(profile["max_spread_bps"]), 6.0)

    def test_runtime_toggle_between_modes(self) -> None:
        settings = replace(
            DEFAULT_SETTINGS,
            data_provider="mock",
            execution_provider="mock",
            llm_provider="mock",
            data_acceleration_mode=False,
            entry_confluence_min=58.0,
            ev_min_ticks=0.2,
            max_spread_bps=6.0,
        )
        engine = TradingEngine(settings)

        before = engine.acceleration_snapshot()
        self.assertFalse(before["requested"])
        self.assertFalse(before["active"])
        self.assertEqual(float(before["entry_confluence_min"]), 58.0)

        engine.set_acceleration_mode(True)
        after_on = engine.acceleration_snapshot()
        self.assertTrue(after_on["requested"])
        self.assertTrue(after_on["active"])
        self.assertLess(float(after_on["entry_confluence_min"]), 58.0)

        engine.set_acceleration_mode(False)
        after_off = engine.acceleration_snapshot()
        self.assertFalse(after_off["requested"])
        self.assertFalse(after_off["active"])
        self.assertEqual(float(after_off["entry_confluence_min"]), 58.0)


if __name__ == "__main__":
    unittest.main()
