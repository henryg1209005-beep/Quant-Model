from __future__ import annotations

import shutil
import unittest
import uuid
from dataclasses import replace
from pathlib import Path

from ai_trading_engine.app_service import TradingAppService
from ai_trading_engine.config import DEFAULT_SETTINGS
from ai_trading_engine.persistence import Persistence


class TestMultiSymbolCollection(unittest.TestCase):
    def test_shadow_symbol_cycle_is_paper_only_and_logged(self) -> None:
        tmp = Path("tests") / f"_tmp_multisymbol_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            settings = replace(
                DEFAULT_SETTINGS,
                symbol="SPY",
                symbols=("SPY", "QQQ"),
                multi_symbol_enabled=True,
                multi_symbol_shadow_enabled=True,
                multi_symbol_paper_only=True,
                data_provider="mock",
                execution_provider="mock",
                llm_provider="mock",
                enable_session_filter=False,
                auto_start_worker=False,
                auto_retrain_enabled=False,
                app_db_path=str(tmp / "app.db"),
                database_url="",
                notifications_path=str(tmp / "notifications.jsonl"),
                adaptive_state_path=str(tmp / "adaptive.json"),
                edge_monitor_state_path=str(tmp / "edge.json"),
                auto_retrain_state_path=str(tmp / "automation.json"),
                promotion_state_path=str(tmp / "promotion.json"),
            )
            persistence = Persistence(settings.app_db_path, database_url="")
            service = TradingAppService(settings, persistence)

            service.run_cycle_once()

            rows = service.decisions(10)
            metadata = [row["decision"].get("metadata", {}) for row in rows]
            shadow = [m for m in metadata if m.get("collection_role") == "shadow_multi_symbol"]

            self.assertGreaterEqual(len(rows), 2)
            self.assertEqual(shadow[0]["symbol"], "QQQ")
            self.assertEqual(shadow[0]["providers"]["execution"], "mock")
            self.assertTrue(shadow[0]["shadow_execution"])
            self.assertEqual(service.symbol_collection_status()["shadow_symbols"], ["QQQ"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
