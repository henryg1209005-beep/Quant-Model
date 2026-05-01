from __future__ import annotations

import json
import unittest
from pathlib import Path

from ai_trading_engine.audit_log import append_audit_event


class TestAuditLog(unittest.TestCase):
    def test_hash_chain_is_written(self) -> None:
        Path("data").mkdir(parents=True, exist_ok=True)
        log_path = str(Path("data/test_audit_log.jsonl"))
        for p in [Path(log_path), Path(log_path + ".state")]:
            if p.exists():
                p.unlink()

        r1 = append_audit_event(log_path, "evt1", {"a": 1})
        r2 = append_audit_event(log_path, "evt2", {"b": 2})
        self.assertTrue(r1.get("hash"))
        self.assertEqual(r2.get("prev_hash"), r1.get("hash"))

        lines = Path(log_path).read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        parsed = [json.loads(x) for x in lines]
        self.assertEqual(parsed[1]["prev_hash"], parsed[0]["hash"])


if __name__ == "__main__":
    unittest.main()
