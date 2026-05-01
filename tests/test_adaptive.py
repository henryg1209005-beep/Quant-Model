from __future__ import annotations

import shutil
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ai_trading_engine.adaptive import AdaptiveLearner
from ai_trading_engine.models import AiDecision, ClosedTrade


class TestAdaptiveLearner(unittest.TestCase):
    def test_learns_and_adjusts(self) -> None:
        tmp = Path("tests") / f"_tmp_adapt_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            learner = AdaptiveLearner(str(tmp / "state.json"), enabled=True)
            for _ in range(8):
                learner.update_from_trade(
                    ClosedTrade(
                        symbol="SPY",
                        direction="LONG",
                        size=1,
                        entry_price=100,
                        exit_price=99,
                        pnl=-1,
                        opened_at=datetime.now(timezone.utc),
                        closed_at=datetime.now(timezone.utc),
                        thesis="",
                        regime="trending_down",
                        confidence=0.7,
                    )
                )

            decision = AiDecision(
                action="trade",
                direction="LONG",
                confidence=0.7,
                size=3,
                sl_ticks=8,
                tp_ticks=14,
                reasoning="test",
            )
            action = learner.adapt(decision, "trending_down")
            self.assertEqual(action.decision.action, "hold")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
