from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from ai_trading_engine.indicators import compute_indicator_set, format_dt
from ai_trading_engine.context_builder import build_llm_context
from ai_trading_engine.models import AccountState, Bar, TimeframeBars
from ai_trading_engine.scorer import build_market_context, render_context_dashboard


def _sample_bars(n: int = 30) -> list[Bar]:
    now = datetime.now(tz=timezone.utc)
    out: list[Bar] = []
    price = 100.0
    for i in range(n):
        ts = now - timedelta(minutes=(n - i) * 5)
        price += 0.2
        out.append(Bar(timestamp=ts, open=price - 0.3, high=price + 0.8, low=price - 0.9, close=price, volume=100 + i))
    return out


class TestContext(unittest.TestCase):
    def test_context_dashboard_contains_dimensions(self) -> None:
        bars = _sample_bars(30)
        ind = compute_indicator_set(bars)
        ctx = build_market_context(ind, bars)
        text = render_context_dashboard(ctx)

        self.assertIn("MARKET CONTEXT DASHBOARD", text)
        self.assertIn("Trend", text)
        self.assertIn("Momentum", text)
        self.assertIn("Key Levels", text)

    def test_llm_context_compacts_bars_to_recent_window(self) -> None:
        bars = _sample_bars(8)
        context = build_llm_context(
            dashboard_text="MARKET CONTEXT DASHBOARD",
            tf_bars=TimeframeBars(primary=bars, short=bars, long=bars),
            account=AccountState(balance=1000.0, starting_balance=1000.0),
            recent_bars=3,
        )

        self.assertIn("Primary bars summary (8 total)", context)
        self.assertIn("Primary recent bars (latest 3)", context)
        self.assertIn("first_close=", context)
        self.assertNotIn(format_dt(bars[0].timestamp), context)
        self.assertIn(format_dt(bars[-1].timestamp), context)


if __name__ == "__main__":
    unittest.main()
