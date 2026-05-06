from __future__ import annotations

import shutil
import unittest
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from ai_trading_engine.app_service import TradingAppService
from ai_trading_engine.config import DEFAULT_SETTINGS
from ai_trading_engine.persistence import Persistence
from ai_trading_engine.prediction_quality import build_prediction_labels


def _settings(**overrides):
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


def _sample(ts: datetime, price: float, *, direction: str | None, segment: str, bias: str = "bullish") -> dict:
    out = {
        "timestamp": ts.isoformat(),
        "symbol": "SPY",
        "quote_last": price,
        "sample_quality_good": True,
        "session_bucket": "midday",
        "indicator_regime": "trending_up" if bias == "bullish" else "trending_down",
        "feature_volatility_label": "normal",
        "feature_volume_label": "above_avg" if bias == "bullish" else "dead",
        "feature_sr_label": "vwap",
        "feature_structure_bias": bias,
        "feature_structure_score": 2.0 if bias == "bullish" else -2.0,
        "feature_structure_summary": f"segment={segment}",
        "feature_structure": {
            "session_segment": segment,
            "swing_structure": "hh_hl" if bias == "bullish" else "lh_ll",
            "breakout_state": "accepted_breakout_up" if bias == "bullish" else "accepted_breakout_down",
            "close_location": "near_high" if bias == "bullish" else "near_low",
            "impulse_state": "bull_impulse" if bias == "bullish" else "bear_impulse",
        },
    }
    if direction is not None:
        out["forecast_direction"] = direction
        out["forecast_confidence"] = 0.58
    return out


class TestContextLeaderboard(unittest.TestCase):
    def test_prediction_labels_preserve_structure_features(self) -> None:
        start = datetime(2026, 5, 6, 14, 0, tzinfo=timezone.utc)
        samples = [
            _sample(start, 100.0, direction="LONG", segment="open_drive"),
            _sample(start + timedelta(minutes=15), 101.0, direction=None, segment="open_drive"),
        ]

        labels = build_prediction_labels(samples, horizons_minutes=(15,), quality_mode="good_only")
        row = labels["labels_by_horizon"][15][0]

        self.assertEqual(row["session_segment"], "open_drive")
        self.assertEqual(row["feature_structure_bias"], "bullish")
        self.assertEqual(row["feature_structure"]["swing_structure"], "hh_hl")

    def test_context_leaderboard_allows_best_session_segment(self) -> None:
        tmp = Path("tests") / f"_tmp_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            settings = _settings(app_db_path=str(tmp / "app.db"))
            service = TradingAppService(settings, Persistence(settings.app_db_path))
            start = datetime(2026, 5, 6, 14, 0, tzinfo=timezone.utc)
            rows = []
            for i in range(8):
                ts = start + timedelta(minutes=i * 30)
                rows.append(_sample(ts, 100.0, direction="LONG", segment="open_drive", bias="bullish"))
                rows.append(_sample(ts + timedelta(minutes=15), 101.0, direction=None, segment="open_drive", bias="bullish"))
            for i in range(8):
                ts = start + timedelta(hours=5, minutes=i * 30)
                rows.append(_sample(ts, 100.0, direction="LONG", segment="lunch", bias="bearish"))
                rows.append(_sample(ts + timedelta(minutes=15), 99.0, direction=None, segment="lunch", bias="bearish"))

            with patch.object(service._persistence, "list_data_samples", return_value=rows):
                report = service.context_leaderboard_report(
                    lookback=5000,
                    horizons_minutes=(15,),
                    quality_mode="good_only",
                    min_labels=4,
                    stress_bps=3.0,
                    min_accuracy=0.5,
                )

            allowed = report["allowed"]
            self.assertGreater(report["allowed_count"], 0)
            self.assertTrue(
                any(
                    r["feature"] == "session_segment"
                    and r["value"] == "open_drive"
                    and r["status"] == "allow"
                    for r in allowed
                )
            )
            blocked = [
                r for r in report["rows"]
                if r["feature"] == "session_segment" and r["value"] == "lunch"
            ]
            self.assertTrue(blocked)
            self.assertEqual(blocked[0]["status"], "block")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
