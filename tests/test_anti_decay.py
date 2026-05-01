from __future__ import annotations

import shutil
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ai_trading_engine.anti_decay import EdgeMonitor
from ai_trading_engine.models import AiDecision, ClosedTrade


class TestAntiDecay(unittest.TestCase):
    def test_throttle_to_hold_when_edge_degrades(self) -> None:
        tmp = Path("tests") / f"_tmp_edge_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            mon = EdgeMonitor(
                state_path=str(tmp / "edge.json"),
                window_trades=20,
                min_win_rate=0.5,
                min_expectancy=0.0,
                throttle_size_mult=0.6,
                shadow_enabled=True,
            )
            for _ in range(20):
                mon.update_trade(
                    ClosedTrade(
                        symbol="SPY",
                        direction="LONG",
                        size=1,
                        entry_price=100,
                        exit_price=99,
                        pnl=-10,
                        opened_at=datetime.now(timezone.utc),
                        closed_at=datetime.now(timezone.utc),
                        thesis="",
                        regime="ranging",
                        confidence=0.7,
                    )
                )

            d = AiDecision(action="trade", direction="LONG", confidence=0.7, size=3, sl_ticks=8, tp_ticks=12, reasoning="x")
            res = mon.throttle(d)
            self.assertEqual(res.decision.action, "hold")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
