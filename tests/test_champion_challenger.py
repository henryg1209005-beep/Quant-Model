from __future__ import annotations

import shutil
import unittest
import uuid
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from ai_trading_engine.app_service import TradingAppService
from ai_trading_engine.config import DEFAULT_SETTINGS
from ai_trading_engine.persistence import Persistence


def _service_settings(**overrides):
    return replace(
        DEFAULT_SETTINGS,
        llm_provider="mock",
        llm_two_tier_enabled=False,
        data_provider="mock",
        execution_provider="mock",
        auto_retrain_enabled=False,
        autonomous_research_enabled=False,
        research_allowed_session_buckets="open,midday,close,unknown",
        **overrides,
    )


class TestChampionChallengerDirection(unittest.TestCase):
    def test_direction_short_filters_report(self) -> None:
        tmp = Path("tests") / f"_tmp_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            settings = _service_settings(app_db_path=str(tmp / "app.db"))
            service = TradingAppService(settings, Persistence(settings.app_db_path))

            rows = [
                # Day 1 (train pool)
                {"timestamp": "2026-05-01T14:00:00+00:00", "symbol": "SPY", "quote_last": 100.0, "forecast_direction": "LONG", "forecast_confidence": 0.65, "session_bucket": "midday", "sample_quality_good": True},
                {"timestamp": "2026-05-01T14:15:00+00:00", "symbol": "SPY", "quote_last": 101.0, "forecast_direction": "LONG", "forecast_confidence": 0.65, "session_bucket": "midday", "sample_quality_good": True},
                {"timestamp": "2026-05-01T14:30:00+00:00", "symbol": "SPY", "quote_last": 100.0, "forecast_direction": "SHORT", "forecast_confidence": 0.65, "session_bucket": "midday", "sample_quality_good": True},
                {"timestamp": "2026-05-01T14:45:00+00:00", "symbol": "SPY", "quote_last": 101.0, "forecast_direction": "SHORT", "forecast_confidence": 0.65, "session_bucket": "midday", "sample_quality_good": True},
                # Day 2 (test pool)
                {"timestamp": "2026-05-02T14:00:00+00:00", "symbol": "SPY", "quote_last": 100.0, "forecast_direction": "LONG", "forecast_confidence": 0.65, "session_bucket": "midday", "sample_quality_good": True},
                {"timestamp": "2026-05-02T14:15:00+00:00", "symbol": "SPY", "quote_last": 101.0, "forecast_direction": "LONG", "forecast_confidence": 0.65, "session_bucket": "midday", "sample_quality_good": True},
                {"timestamp": "2026-05-02T14:30:00+00:00", "symbol": "SPY", "quote_last": 100.0, "forecast_direction": "SHORT", "forecast_confidence": 0.65, "session_bucket": "midday", "sample_quality_good": True},
                {"timestamp": "2026-05-02T14:45:00+00:00", "symbol": "SPY", "quote_last": 101.0, "forecast_direction": "SHORT", "forecast_confidence": 0.65, "session_bucket": "midday", "sample_quality_good": True},
            ]
            with patch.object(service._persistence, "list_data_samples", return_value=rows):
                all_report = service.champion_challenger_daily_report(
                    lookback=5000,
                    horizon_minutes=15,
                    quality_mode="good_only",
                    min_train_labels=1,
                    min_cell_labels=1,
                    challenger_min_confidence=0.0,
                    challenger_max_confidence=1.0,
                    min_daily_selections=1,
                    direction="ALL",
                )
                short_report = service.champion_challenger_daily_report(
                    lookback=5000,
                    horizon_minutes=15,
                    quality_mode="good_only",
                    min_train_labels=1,
                    min_cell_labels=1,
                    challenger_min_confidence=0.0,
                    challenger_max_confidence=1.0,
                    min_daily_selections=1,
                    direction="SHORT",
                )

            self.assertEqual(short_report["direction"], "SHORT")
            self.assertLess(
                int(short_report["challenger_overall"]["count"]),
                int(all_report["challenger_overall"]["count"]),
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
