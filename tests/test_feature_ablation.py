from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from ai_trading_engine.prediction_quality import build_feature_ablation_report


def _sample(ts: datetime, price: float, forecast: str, news_risk: str) -> dict:
    return {
        "timestamp": ts.isoformat(),
        "symbol": "SPY",
        "quote_last": price,
        "forecast_direction": forecast,
        "forecast_confidence": 0.7,
        "finnhub_news_risk": news_risk,
        "economic_calendar_event_risk": "clear",
        "finnhub_earnings_risk": "clear",
        "finnhub_news_sentiment_label": "neutral",
        "macro_provider": "fred",
        "macro_risk_regime": "mixed",
    }


class TestFeatureAblation(unittest.TestCase):
    def test_ablation_reports_feature_deltas(self) -> None:
        start = datetime(2026, 4, 29, 13, 0, tzinfo=timezone.utc)
        samples = [
            _sample(start, 100.0, "LONG", "symbol_news_active"),
            _sample(start + timedelta(minutes=5), 101.0, "LONG", "symbol_news_active"),
            _sample(start + timedelta(minutes=10), 102.0, "SHORT", "clear"),
            _sample(start + timedelta(minutes=15), 103.0, "SHORT", "clear"),
        ]

        report = build_feature_ablation_report(samples, horizon_minutes=5, min_count=1)
        news = report["ablations"]["news_active"]

        self.assertEqual(report["label_count"], 3)
        self.assertTrue(news["sufficient_sample"])
        self.assertGreater(news["delta_signed_return_bps"], 0.0)


if __name__ == "__main__":
    unittest.main()
