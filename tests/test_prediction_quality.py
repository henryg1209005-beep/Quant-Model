from __future__ import annotations

import unittest

from ai_trading_engine.prediction_quality import build_confidence_control_report, build_prediction_quality_report


class TestPredictionQuality(unittest.TestCase):
    def test_scores_forward_directional_predictions(self) -> None:
        samples = [
            {
                "timestamp": "2026-04-29T13:00:00+00:00",
                "symbol": "SPY",
                "quote_last": 100.0,
                "llm_raw": '{"action":"trade","direction":"LONG","confidence":0.8,"size":1,"sl_ticks":8,"tp_ticks":12,"reasoning":"x"}',
                "collection_role": "primary",
            },
            {
                "timestamp": "2026-04-29T13:05:00+00:00",
                "symbol": "SPY",
                "quote_last": 101.0,
                "llm_raw": '{"action":"hold","confidence":0.4,"size":1,"sl_ticks":0,"tp_ticks":0,"reasoning":"x"}',
                "collection_role": "primary",
            },
            {
                "timestamp": "2026-04-29T13:00:00+00:00",
                "symbol": "QQQ",
                "quote_last": 200.0,
                "llm_raw": '{"action":"trade","direction":"SHORT","confidence":0.7,"size":1,"sl_ticks":8,"tp_ticks":12,"reasoning":"x"}',
                "collection_role": "shadow_multi_symbol",
            },
            {
                "timestamp": "2026-04-29T13:05:00+00:00",
                "symbol": "QQQ",
                "quote_last": 198.0,
                "llm_raw": '{"action":"hold","confidence":0.4,"size":1,"sl_ticks":0,"tp_ticks":0,"reasoning":"x"}',
                "collection_role": "shadow_multi_symbol",
            },
        ]

        report = build_prediction_quality_report(samples, horizons_minutes=(5,))
        five = report["horizons"]["5"]

        self.assertEqual(report["eligible_trade_predictions"], 2)
        self.assertEqual(five["label_count"], 2)
        self.assertEqual(five["overall"]["accuracy"], 1.0)
        self.assertGreater(five["overall"]["avg_signed_return_bps"], 0.0)

    def test_scores_hold_forecasts(self) -> None:
        samples = [
            {
                "timestamp": "2026-04-29T13:00:00+00:00",
                "symbol": "SPY",
                "quote_last": 100.0,
                "llm_raw": '{"action":"hold","confidence":0.3,"size":1,"sl_ticks":0,"tp_ticks":0,"forecast_direction":"SHORT","forecast_confidence":0.66,"forecast_horizon_minutes":5,"reasoning":"x"}',
            },
            {
                "timestamp": "2026-04-29T13:05:00+00:00",
                "symbol": "SPY",
                "quote_last": 99.0,
                "llm_raw": '{"action":"hold","confidence":0.4,"size":1,"sl_ticks":0,"tp_ticks":0,"forecast_direction":"LONG","forecast_confidence":0.51,"forecast_horizon_minutes":5,"reasoning":"x"}',
            },
        ]

        report = build_prediction_quality_report(samples, horizons_minutes=(5,))

        self.assertEqual(report["eligible_trade_predictions"], 2)
        self.assertEqual(report["horizons"]["5"]["overall"]["accuracy"], 1.0)

    def test_reports_expectancy_decomposition_and_calibration(self) -> None:
        samples = [
            {
                "timestamp": "2026-04-29T13:00:00+00:00",
                "symbol": "SPY",
                "quote_last": 100.0,
                "forecast_direction": "LONG",
                "forecast_confidence": 0.75,
            },
            {
                "timestamp": "2026-04-29T13:05:00+00:00",
                "symbol": "SPY",
                "quote_last": 101.0,
                "forecast_direction": "LONG",
                "forecast_confidence": 0.75,
            },
            {
                "timestamp": "2026-04-29T13:10:00+00:00",
                "symbol": "SPY",
                "quote_last": 100.0,
                "forecast_direction": "SHORT",
                "forecast_confidence": 0.65,
            },
        ]

        report = build_prediction_quality_report(samples, horizons_minutes=(5,))
        five = report["horizons"]["5"]
        overall = five["overall"]

        self.assertEqual(five["label_count"], 2)
        self.assertEqual(overall["win_count"], 1)
        self.assertEqual(overall["loss_count"], 1)
        self.assertIn("expectancy_bps", overall)
        self.assertIn("trimmed_mean_signed_return_bps", overall)
        self.assertLess(overall["tail_loss_bps_p5"], 0.0)
        self.assertGreater(overall["tail_win_bps_p95"], 0.0)

        calibration = five["calibration"]
        self.assertEqual(calibration["count"], 2)
        self.assertIn("0.7-0.8", calibration["by_confidence_bin"])
        bucket = calibration["by_confidence_bin"]["0.7-0.8"]
        self.assertEqual(bucket["count"], 2)
        self.assertAlmostEqual(bucket["realized_accuracy"], 0.5)
        self.assertAlmostEqual(bucket["mean_confidence"], 0.75)
        self.assertAlmostEqual(bucket["calibration_error"], -0.25)

    def test_confidence_control_report_classifies_direction_policy(self) -> None:
        samples = []
        for i in range(8):
            ts = f"2026-04-29T13:{i:02d}:00+00:00"
            future = f"2026-04-29T13:{i + 15:02d}:00+00:00"
            samples.append(
                {
                    "timestamp": ts,
                    "symbol": "SPY",
                    "quote_last": 100.0,
                    "forecast_direction": "LONG",
                    "forecast_confidence": 0.55,
                    "sample_quality_good": True,
                }
            )
            samples.append(
                {
                    "timestamp": future,
                    "symbol": "SPY",
                    "quote_last": 101.0,
                    "forecast_direction": "LONG",
                    "forecast_confidence": 0.55,
                    "sample_quality_good": True,
                }
            )
        for i in range(8):
            ts = f"2026-04-29T14:{i:02d}:00+00:00"
            future = f"2026-04-29T14:{i + 15:02d}:00+00:00"
            samples.append(
                {
                    "timestamp": ts,
                    "symbol": "SPY",
                    "quote_last": 100.0,
                    "forecast_direction": "SHORT",
                    "forecast_confidence": 0.55,
                    "sample_quality_good": True,
                }
            )
            samples.append(
                {
                    "timestamp": future,
                    "symbol": "SPY",
                    "quote_last": 101.0,
                    "forecast_direction": "SHORT",
                    "forecast_confidence": 0.55,
                    "sample_quality_good": True,
                }
            )

        report = build_confidence_control_report(
            samples,
            horizon_minutes=15,
            min_bin_count=4,
            min_symbol_labels=4,
            min_threshold_labels=4,
            policy_min_accuracy=0.50,
            policy_stress_bps=0.0,
        )
        by_direction = report["policy"]["by_symbol"]["SPY"]["by_direction"]

        self.assertEqual(by_direction["LONG"]["status"], "allow")
        self.assertEqual(by_direction["SHORT"]["status"], "block")

    def test_confidence_control_report_excludes_fallback_rows_and_tracks_readiness(self) -> None:
        samples = []
        for idx, conf in enumerate((0.55, 0.65, 0.75)):
            base_hour = 13 + idx
            for minute in (0, 1):
                ts = f"2026-04-29T{base_hour:02d}:{minute:02d}:00+00:00"
                future = f"2026-04-29T{base_hour:02d}:{minute + 15:02d}:00+00:00"
                samples.append(
                    {
                        "timestamp": ts,
                        "symbol": "SPY",
                        "quote_last": 100.0,
                        "forecast_direction": "LONG",
                        "forecast_confidence": conf,
                        "forecast_confidence_source": "model",
                        "sample_quality_good": True,
                    }
                )
                samples.append(
                    {
                        "timestamp": future,
                        "symbol": "SPY",
                        "quote_last": 101.0,
                        "forecast_direction": "LONG",
                        "forecast_confidence": conf,
                        "forecast_confidence_source": "model",
                        "sample_quality_good": True,
                    }
                )
        for minute in (0, 1):
            ts = f"2026-04-29T16:{minute:02d}:00+00:00"
            future = f"2026-04-29T16:{minute + 15:02d}:00+00:00"
            samples.append(
                {
                    "timestamp": ts,
                    "symbol": "SPY",
                    "quote_last": 100.0,
                    "forecast_direction": "LONG",
                    "forecast_confidence": 0.95,
                    "forecast_confidence_source": "fallback",
                    "sample_quality_good": True,
                }
            )
            samples.append(
                {
                    "timestamp": future,
                    "symbol": "SPY",
                    "quote_last": 101.0,
                    "forecast_direction": "LONG",
                    "forecast_confidence": 0.95,
                    "forecast_confidence_source": "fallback",
                    "sample_quality_good": True,
                }
            )

        report = build_confidence_control_report(
            samples,
            horizon_minutes=15,
            min_bin_count=1,
            min_symbol_labels=1,
            min_threshold_labels=1,
            included_confidence_sources=("model",),
            readiness_min_populated_bins=3,
            readiness_min_bin_labels=2,
        )

        readiness = report["calibration_readiness"]
        calibration = report["symbol_controls"]["SPY"]["calibration"]
        self.assertTrue(readiness["ready"])
        self.assertEqual(readiness["populated_bin_count"], 3)
        self.assertEqual(sorted(readiness["populated_bins"].keys()), ["0.5-0.6", "0.6-0.7", "0.7-0.8"])
        self.assertNotIn("0.9-1.0", calibration)


if __name__ == "__main__":
    unittest.main()
