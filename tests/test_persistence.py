from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path
import shutil
import uuid

from ai_trading_engine.app_service import restore_account
from ai_trading_engine.engine import CycleResult
from ai_trading_engine.models import AccountState, AiDecision
from ai_trading_engine.persistence import Persistence


class TestPersistence(unittest.TestCase):
    def test_save_and_read_cycle_and_account(self) -> None:
        tmp = Path("tests") / f"_tmp_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            db_path = tmp / "app.db"
            store = Persistence(str(db_path))

            account = AccountState(balance=50010.0, starting_balance=50000.0)
            store.save_account(account)
            snap = store.latest_account_snapshot()
            self.assertIsNotNone(snap)

            restored = restore_account(snap)
            self.assertEqual(restored.balance, 50010.0)

            cycle = CycleResult(
                timestamp=datetime.now(timezone.utc),
                dashboard="x",
                llm_raw='{"ok": true}',
                decision=AiDecision(action="hold", direction=None, confidence=0.2, size=1, sl_ticks=8, tp_ticks=10, reasoning="n"),
                note="hold",
            )
            store.save_cycle(cycle)
            decisions = store.list_decisions(limit=5)
            self.assertEqual(len(decisions), 1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
