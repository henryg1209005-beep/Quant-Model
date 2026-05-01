from __future__ import annotations

import unittest
from dataclasses import replace

from ai_trading_engine.config import load_settings
from ai_trading_engine.economic_calendar import load_economic_calendar, render_economic_calendar


class TestEconomicCalendar(unittest.TestCase):
    def test_disabled_calendar_snapshot(self) -> None:
        settings = replace(load_settings(), economic_calendar_enabled=False)
        snapshot = load_economic_calendar(settings)

        self.assertFalse(snapshot.enabled)
        self.assertEqual(snapshot.event_risk, "disabled")
        self.assertIn("ECONOMIC CALENDAR", render_economic_calendar(snapshot))

    def test_missing_key_is_unavailable(self) -> None:
        settings = replace(
            load_settings(),
            economic_calendar_enabled=True,
            economic_calendar_provider="finnhub",
            finnhub_api_key="",
        )
        snapshot = load_economic_calendar(settings)

        self.assertTrue(snapshot.enabled)
        self.assertEqual(snapshot.event_risk, "unavailable")


if __name__ == "__main__":
    unittest.main()
