from __future__ import annotations

import base64
import unittest

from ai_trading_engine.auth import verify_basic_auth


def _basic(user: str, pw: str) -> str:
    token = base64.b64encode(f"{user}:{pw}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


class TestAuth(unittest.TestCase):
    def test_verify_basic_auth_success(self) -> None:
        ok = verify_basic_auth(
            _basic("alice", "secret123"),
            username="alice",
            password="secret123",
        )
        self.assertTrue(ok)

    def test_verify_basic_auth_failure(self) -> None:
        bad = verify_basic_auth(
            _basic("alice", "wrong"),
            username="alice",
            password="secret123",
        )
        self.assertFalse(bad)


if __name__ == "__main__":
    unittest.main()
