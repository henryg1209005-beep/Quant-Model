from __future__ import annotations

import shutil
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import uuid

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
        **overrides,
    )


class TestDataReadiness(unittest.TestCase):
    def test_data_quality_counters_include_sample_eligibility(self) -> None:
        tmp = Path("tests") / f"_tmp_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            settings = _service_settings(app_db_path=str(tmp / "app.db"))
            service = TradingAppService(settings, Persistence(settings.app_db_path))
            now = datetime.now(tz=timezone.utc).isoformat()
            good_row = {
                "timestamp": now,
                "quote_last": 100.0,
                "quality_quote_stale": False,
                "quality_spread_too_wide": False,
                "quality_outside_session": False,
                "quality_missing_forecast": False,
                "metadata": {"llm_routing": {"state_gate_reason": "", "secondary_reason": ""}},
            }
            bad_row = {
                "timestamp": now,
                "quote_last": 0.0,
                "quality_quote_stale": True,
                "quality_spread_too_wide": True,
                "quality_outside_session": True,
                "quality_missing_forecast": True,
                "metadata": {"llm_routing": {"state_gate_reason": "fallback", "secondary_reason": "parse_failed"}},
            }
            with patch.object(service._persistence, "list_data_samples", return_value=[good_row, bad_row]):
                out = service.data_quality_counters(lookback=2000)
            elig = out.get("sample_eligibility") or {}
            self.assertIn("avg_score", elig)
            self.assertEqual(int(elig.get("high_count", 0)) + int(elig.get("medium_count", 0)) + int(elig.get("low_count", 0)), 2)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_data_readiness_score_degrades_with_bad_quality(self) -> None:
        tmp = Path("tests") / f"_tmp_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            settings = _service_settings(app_db_path=str(tmp / "app.db"))
            service = TradingAppService(settings, Persistence(settings.app_db_path))
            now = datetime.now(tz=timezone.utc).isoformat()
            rows_good = [
                {
                    "timestamp": now,
                    "quote_last": 100.0,
                    "quality_quote_stale": False,
                    "quality_spread_too_wide": False,
                    "quality_outside_session": False,
                    "quality_missing_forecast": False,
                    "metadata": {"llm_routing": {"state_gate_reason": "", "secondary_reason": ""}},
                }
                for _ in range(5)
            ]
            rows_bad = [
                {
                    "timestamp": now,
                    "quote_last": 0.0,
                    "quality_quote_stale": True,
                    "quality_spread_too_wide": True,
                    "quality_outside_session": True,
                    "quality_missing_forecast": True,
                    "metadata": {"llm_routing": {"state_gate_reason": "fallback", "secondary_reason": "parse_failed"}},
                }
                for _ in range(5)
            ]
            with patch.object(service._persistence, "list_data_samples", return_value=rows_good):
                good = service.data_readiness_status(lookback=2000)
            with patch.object(service._persistence, "list_data_samples", return_value=rows_bad):
                bad = service.data_readiness_status(lookback=2000)
            self.assertGreaterEqual(int(good.get("score", 0)), int(bad.get("score", 0)))
            self.assertIn("components", good)
            self.assertIn("inputs", good)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
