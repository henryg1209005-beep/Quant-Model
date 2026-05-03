from __future__ import annotations

import shutil
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import uuid

from ai_trading_engine.app_service import TradingAppService
from ai_trading_engine.config import DEFAULT_SETTINGS
from ai_trading_engine.engine import TradingEngine
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


if __name__ == "__main__":
    unittest.main()
