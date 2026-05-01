from __future__ import annotations

import unittest

from ai_trading_engine.sample_coverage import build_sample_coverage_report


class TestSampleCoverage(unittest.TestCase):
    def test_coverage_counts_quality_and_groups(self) -> None:
        rows = [
            {
                "symbol": "SPY",
                "collection_role": "primary",
                "session_bucket": "open",
                "indicator_regime": "trending_up",
                "macro_provider": "fred",
                "economic_calendar_event_risk": "clear",
                "finnhub_news_risk": "low",
                "finnhub_earnings_risk": "clear",
                "sample_quality_good": True,
                "quote_last": 100.0,
                "forecast_direction": "LONG",
                "quality_quote_stale": False,
            },
            {
                "symbol": "QQQ",
                "collection_role": "shadow_multi_symbol",
                "session_bucket": "outside",
                "sample_quality_good": False,
                "quote_last": 0.0,
                "forecast_direction": None,
                "quality_quote_stale": True,
            },
        ]

        report = build_sample_coverage_report(rows)

        self.assertEqual(report["sample_count"], 2)
        self.assertEqual(report["good_sample_count"], 1)
        self.assertEqual(report["quote_label_ready_count"], 1)
        self.assertEqual(report["forecast_ready_count"], 1)
        self.assertEqual(report["by_symbol"]["SPY"], 1)
        self.assertEqual(report["quality_flag_counts"]["quality_quote_stale"], 1)


if __name__ == "__main__":
    unittest.main()
