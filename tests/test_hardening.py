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
from ai_trading_engine.engine import CycleResult, TradingEngine
from ai_trading_engine.models import AiDecision
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


class TestHardening(unittest.TestCase):
    def test_missing_secondary_llm_key_disables_secondary_without_crashing(self) -> None:
        settings = replace(
            DEFAULT_SETTINGS,
            llm_provider="mock",
            llm_two_tier_enabled=True,
            llm_two_tier_secondary_provider="gemini",
            data_provider="mock",
            execution_provider="mock",
        )

        with patch.dict("os.environ", {"GEMINI_API_KEY": ""}, clear=False):
            engine = TradingEngine(settings)

        self.assertIsNone(engine.llm_secondary)
        self.assertTrue(any("secondary_llm_disabled:gemini" in w for w in engine.startup_warnings()))

    def test_trading_kill_switch_blocks_start_and_run_once(self) -> None:
        tmp = Path("tests") / f"_tmp_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            settings = _service_settings(
                trading_kill_switch_enabled=True,
                trading_kill_switch_reason="maintenance",
                app_db_path=str(tmp / "app.db"),
            )
            service = TradingAppService(settings, Persistence(settings.app_db_path))
            monday_utc = datetime(2026, 5, 4, 14, 0, tzinfo=timezone.utc)
            with patch.object(service, "_now_utc", return_value=monday_utc):
                started = service.start_with_guard()
                self.assertFalse(started["started"])
                self.assertTrue(started["blocked"])
                self.assertEqual(started["reason"], "trading_kill_switch_enabled")

                cycle = service.run_cycle_once()
                self.assertEqual(cycle.decision.action, "hold")
                self.assertIn("Trading kill switch enabled", cycle.decision.reasoning)
                self.assertEqual((cycle.metadata.get("kill_switch") or {}).get("reason"), "maintenance")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_cost_guard_tolerates_non_numeric_call_counts(self) -> None:
        tmp = Path("tests") / f"_tmp_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            settings = _service_settings(
                automation_cost_guard_enabled=True,
                app_db_path=str(tmp / "app.db"),
            )
            service = TradingAppService(settings, Persistence(settings.app_db_path))
            now = datetime.now(tz=timezone.utc).isoformat()
            bad_row = {
                "timestamp": now,
                "metadata": {
                    "llm_routing": {
                        "primary_call_count": "not-a-number",
                        "secondary_call_count": "1.0",
                    }
                },
            }
            with patch.object(service._persistence, "list_data_samples", return_value=[bad_row]):
                guard = service._evaluate_cost_guard(force=True)
            metrics = dict(guard.get("metrics") or {})
            self.assertGreaterEqual(int(metrics.get("estimated_primary_calls", 0) or 0), 0)
            self.assertGreaterEqual(int(metrics.get("estimated_secondary_calls", 0) or 0), 0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_weekend_block_prevents_start_and_cycle(self) -> None:
        tmp = Path("tests") / f"_tmp_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            settings = _service_settings(app_db_path=str(tmp / "app.db"))
            service = TradingAppService(settings, Persistence(settings.app_db_path))
            saturday_utc = datetime(2026, 5, 2, 15, 0, tzinfo=timezone.utc)

            with patch.object(service, "_now_utc", return_value=saturday_utc):
                started = service.start_with_guard()
                self.assertFalse(started["started"])
                self.assertTrue(started["blocked"])
                self.assertEqual(started["reason"], "weekend_blocked")

                cycle = service.run_cycle_once()
                self.assertEqual(cycle.decision.action, "hold")
                self.assertIn("Weekend block active", cycle.decision.reasoning)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_no_quote_samples_are_not_persisted_as_data_samples(self) -> None:
        tmp = Path("tests") / f"_tmp_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            settings = _service_settings(app_db_path=str(tmp / "app.db"))
            service = TradingAppService(settings, Persistence(settings.app_db_path))
            cycle = service._build_guard_cycle(reason="guard", note="guard")
            meta = {"symbol": settings.symbol}
            with patch.object(service._persistence, "save_data_sample") as save_sample:
                service._persist_cycle_and_sample(cycle, metadata=meta, symbol=settings.symbol)
            save_sample.assert_not_called()
            flags = dict((cycle.metadata or {}).get("sample_quality", {}).get("flags", {}))
            self.assertTrue(bool(flags.get("no_quote", False)))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_low_volume_quality_guard_collects_cycle_in_collection_mode(self) -> None:
        tmp = Path("tests") / f"_tmp_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            settings = _service_settings(
                app_db_path=str(tmp / "app.db"),
                automation_cost_mode_enabled=False,
                budget_lock_enabled=False,
            )
            service = TradingAppService(settings, Persistence(settings.app_db_path))
            cycle = CycleResult(
                timestamp=datetime.now(tz=timezone.utc),
                dashboard="DASH",
                llm_raw="",
                decision=AiDecision(
                    action="hold",
                    direction=None,
                    confidence=0.0,
                    size=1,
                    sl_ticks=0,
                    tp_ticks=0,
                    reasoning="base",
                    forecast_direction="LONG",
                    forecast_confidence=0.5,
                    forecast_horizon_minutes=15,
                ),
                note="base",
                metadata={"quote": {"last": 100.0}},
            )
            with patch.object(service, "_evaluate_data_quality_guard", return_value={"active": True, "reason": "low_sample_volume:0<20"}):
                with patch.object(service._engine, "run_cycle", return_value=cycle) as run_cycle:
                    out = service.run_cycle_once()
            run_cycle.assert_called_once_with(collect_only=True)
            self.assertEqual(out.note, "Guard hold: data quality")
            self.assertEqual(out.decision.action, "hold")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_low_volume_quality_guard_skips_collection_when_cost_mode_enabled(self) -> None:
        tmp = Path("tests") / f"_tmp_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            settings = _service_settings(
                app_db_path=str(tmp / "app.db"),
                automation_cost_mode_enabled=True,
                automation_cost_mode_skip_low_volume_collect=True,
                budget_lock_enabled=False,
            )
            service = TradingAppService(settings, Persistence(settings.app_db_path))
            with patch.object(
                service,
                "_evaluate_data_quality_guard",
                return_value={"active": True, "reason": "low_sample_volume:0<20", "metrics": {"in_session": False}},
            ):
                with patch.object(service._engine, "run_cycle") as run_cycle:
                    out = service.run_cycle_once()
            run_cycle.assert_not_called()
            self.assertEqual(out.decision.action, "hold")
            self.assertIn("Cost mode skipped low-volume collection cycle", out.decision.reasoning)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_low_volume_quality_guard_collects_in_session_even_when_cost_mode_skip_enabled(self) -> None:
        tmp = Path("tests") / f"_tmp_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            settings = _service_settings(
                app_db_path=str(tmp / "app.db"),
                automation_cost_mode_enabled=True,
                automation_cost_mode_skip_low_volume_collect=True,
                budget_lock_enabled=False,
            )
            service = TradingAppService(settings, Persistence(settings.app_db_path))
            cycle = CycleResult(
                timestamp=datetime.now(tz=timezone.utc),
                dashboard="DASH",
                llm_raw="",
                decision=AiDecision(
                    action="hold",
                    direction=None,
                    confidence=0.0,
                    size=1,
                    sl_ticks=0,
                    tp_ticks=0,
                    reasoning="base",
                    forecast_direction="LONG",
                    forecast_confidence=0.5,
                    forecast_horizon_minutes=15,
                ),
                note="base",
                metadata={"quote": {"last": 100.0}},
            )
            with patch.object(
                service,
                "_evaluate_data_quality_guard",
                return_value={"active": True, "reason": "low_sample_volume:0<20", "metrics": {"in_session": True}},
            ):
                with patch.object(service._engine, "run_cycle", return_value=cycle) as run_cycle:
                    out = service.run_cycle_once()
            run_cycle.assert_called_once_with(collect_only=True)
            self.assertEqual(out.decision.action, "hold")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_cost_mode_increases_worker_wait_when_quality_guard_active(self) -> None:
        tmp = Path("tests") / f"_tmp_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            settings = _service_settings(
                app_db_path=str(tmp / "app.db"),
                cycle_seconds=30,
                automation_cost_mode_enabled=True,
                automation_quality_throttle_cycle_seconds=120,
            )
            service = TradingAppService(settings, Persistence(settings.app_db_path, database_url=settings.database_url))
            service._quality_guard_state = {"active": True}
            wait_seconds = service._next_cycle_wait_seconds()
            self.assertEqual(wait_seconds, 120)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_budget_lock_blocks_outside_session_when_no_open_position(self) -> None:
        tmp = Path("tests") / f"_tmp_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            settings = _service_settings(
                app_db_path=str(tmp / "app.db"),
                budget_lock_enabled=True,
                budget_lock_in_session_only=True,
            )
            service = TradingAppService(settings, Persistence(settings.app_db_path))
            with patch.object(service, "_is_in_session_window", return_value=False):
                cycle = service.run_cycle_once()
            self.assertEqual(cycle.decision.action, "hold")
            self.assertIn("Budget lock active: outside configured session window", cycle.decision.reasoning)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_budget_lock_blocks_when_hourly_llm_cap_reached(self) -> None:
        tmp = Path("tests") / f"_tmp_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            settings = _service_settings(
                app_db_path=str(tmp / "app.db"),
                budget_lock_enabled=True,
                budget_lock_in_session_only=False,
                budget_lock_max_llm_calls_per_hour=2,
            )
            service = TradingAppService(settings, Persistence(settings.app_db_path))
            with patch.object(service, "_estimate_llm_calls_recent", return_value=2):
                cycle = service.run_cycle_once()
            self.assertEqual(cycle.decision.action, "hold")
            self.assertIn("hourly LLM call cap reached", cycle.decision.reasoning)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
