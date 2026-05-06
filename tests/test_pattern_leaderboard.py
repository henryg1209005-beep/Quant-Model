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


def _sample(ts: datetime, price: float, *, direction: str | None, pullback: str, bias: str = "bullish") -> dict:
    out = {
        "timestamp": ts.isoformat(),
        "symbol": "SPY",
        "quote_last": price,
        "sample_quality_good": True,
        "session_bucket": "midday",
        "feature_pattern_bias": bias,
        "feature_pattern_score": 2.0 if bias == "bullish" else -2.0,
        "feature_patterns": {
            "vwap_state": "above" if bias == "bullish" else "below",
            "trend_pullback": pullback,
            "recent_range_break": "inside",
            "failed_break": "none",
            "range_compression": "normal",
        },
    }
    if direction is not None:
        out["forecast_direction"] = direction
        out["forecast_confidence"] = 0.55
    return out


class TestPatternLeaderboard(unittest.TestCase):
    def test_prediction_labels_preserve_pattern_features(self) -> None:
        start = datetime(2026, 5, 6, 14, 0, tzinfo=timezone.utc)
        samples = [
            _sample(start, 100.0, direction="LONG", pullback="bull_pullback_hold"),
            _sample(start + timedelta(minutes=15), 101.0, direction=None, pullback="bull_pullback_hold"),
        ]

        labels = build_prediction_labels(samples, horizons_minutes=(15,), quality_mode="good_only")
        row = labels["labels_by_horizon"][15][0]

        self.assertEqual(row["feature_pattern_bias"], "bullish")
        self.assertEqual(row["feature_patterns"]["trend_pullback"], "bull_pullback_hold")

    def test_pattern_leaderboard_allows_best_stressed_pattern(self) -> None:
        tmp = Path("tests") / f"_tmp_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            settings = _settings(app_db_path=str(tmp / "app.db"))
            service = TradingAppService(settings, Persistence(settings.app_db_path))
            start = datetime(2026, 5, 6, 14, 0, tzinfo=timezone.utc)
            rows = []
            for i in range(8):
                ts = start + timedelta(minutes=i * 30)
                rows.append(_sample(ts, 100.0, direction="LONG", pullback="bull_pullback_hold"))
                rows.append(_sample(ts + timedelta(minutes=15), 101.0, direction=None, pullback="bull_pullback_hold"))
            for i in range(8):
                ts = start + timedelta(hours=5, minutes=i * 30)
                rows.append(_sample(ts, 100.0, direction="LONG", pullback="bull_pullback_failed", bias="bearish"))
                rows.append(_sample(ts + timedelta(minutes=15), 99.0, direction=None, pullback="bull_pullback_failed", bias="bearish"))

            with patch.object(service._persistence, "list_data_samples", return_value=rows):
                report = service.pattern_leaderboard_report(
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
                    r["feature"] == "trend_pullback"
                    and r["value"] == "bull_pullback_hold"
                    and r["status"] == "allow"
                    for r in allowed
                )
            )
            failed_rows = [
                r for r in report["rows"]
                if r["feature"] == "trend_pullback" and r["value"] == "bull_pullback_failed"
            ]
            self.assertTrue(failed_rows)
            self.assertEqual(failed_rows[0]["status"], "block")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
