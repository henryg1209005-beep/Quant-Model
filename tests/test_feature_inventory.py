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


def _settings(**overrides):
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


class TestFeatureInventory(unittest.TestCase):
    def test_inventory_reports_all_current_feature_groups(self) -> None:
        tmp = Path("tests") / f"_tmp_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            settings = _settings(app_db_path=str(tmp / "app.db"))
            service = TradingAppService(settings, Persistence(settings.app_db_path))
            rows = [
                {
                    "feature_trend_score": 22.5,
                    "feature_momentum_score": 11.0,
                    "feature_mean_reversion_score": 4.0,
                    "feature_sr_distance_atr": 0.8,
                    "feature_volatility_label": "normal",
                    "feature_volume_label": "surge",
                    "feature_sr_label": "vwap",
                    "feature_pattern_bias": "bullish",
                    "feature_pattern_score": 2.0,
                    "feature_pattern_summary": "x",
                    "feature_patterns": {
                        "vwap_state": "reclaim",
                        "trend_pullback": "bull_pullback_hold",
                        "recent_range_break": "inside",
                        "failed_break": "none",
                        "range_compression": "compressed",
                    },
                    "feature_structure_bias": "bullish",
                    "feature_structure_score": 2.5,
                    "feature_structure_summary": "y",
                    "feature_structure": {
                        "session_segment": "open_drive",
                        "swing_structure": "hh_hl",
                        "breakout_state": "accepted_breakout_up",
                        "close_location": "near_high",
                        "impulse_state": "bull_impulse",
                    },
                }
            ]
            with patch.object(service._persistence, "list_data_samples", return_value=rows):
                report = service.feature_inventory_report(lookback=100)

            self.assertEqual(report["feature_count"], 23)
            self.assertEqual(report["by_category"]["core"], 7)
            self.assertEqual(report["by_category"]["pattern"], 8)
            self.assertEqual(report["by_category"]["structure"], 8)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
