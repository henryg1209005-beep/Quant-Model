from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from ai_trading_engine.notifications import append_notification, list_notifications


class TestNotifications(unittest.TestCase):
    def test_append_and_list_notifications(self) -> None:
        tmp = Path("tests") / f"_tmp_notifications_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            path = tmp / "events.jsonl"
            append_notification(str(path), "trade_placed", {"symbol": "SPY"})
            rows = list_notifications(str(path), limit=10)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["event_type"], "trade_placed")
            self.assertEqual(rows[0]["payload"]["symbol"], "SPY")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
