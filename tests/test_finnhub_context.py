from __future__ import annotations

import unittest
from dataclasses import replace

from ai_trading_engine.config import load_settings
from ai_trading_engine.finnhub_context import load_finnhub_context, render_finnhub_context


class TestFinnhubContext(unittest.TestCase):
    def test_disabled_context(self) -> None:
        settings = replace(load_settings(), finnhub_context_enabled=False)
        snapshot = load_finnhub_context(settings, "AAPL")

        self.assertFalse(snapshot.enabled)
        self.assertEqual(snapshot.news_risk, "disabled")
        self.assertEqual(snapshot.news_sentiment_label, "neutral")
        self.assertIn("FINNHUB NEWS/EARNINGS CONTEXT", render_finnhub_context(snapshot))

    def test_missing_key_is_unavailable(self) -> None:
        settings = replace(load_settings(), finnhub_context_enabled=True, finnhub_api_key="")
        snapshot = load_finnhub_context(settings, "AAPL")

        self.assertTrue(snapshot.enabled)
        self.assertEqual(snapshot.news_risk, "unavailable")
        self.assertEqual(snapshot.earnings_risk, "unavailable")
        self.assertEqual(snapshot.news_sentiment_score, 0.0)


if __name__ == "__main__":
    unittest.main()
