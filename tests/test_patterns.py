from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from ai_trading_engine.indicators import compute_indicator_set
from ai_trading_engine.market_structure import classify_session_segment, detect_market_structure
from ai_trading_engine.models import Bar
from ai_trading_engine.patterns import detect_price_action_patterns
from ai_trading_engine.scorer import build_market_context, render_context_dashboard


def _bars(closes: list[float]) -> list[Bar]:
    start = datetime(2026, 5, 6, 13, 30, tzinfo=timezone.utc)
    out: list[Bar] = []
    for i, close in enumerate(closes):
        open_ = closes[i - 1] if i else close
        high = max(open_, close) + 0.2
        low = min(open_, close) - 0.2
        out.append(
            Bar(
                timestamp=start + timedelta(minutes=i),
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=1000.0 + i,
            )
        )
    return out


class TestPatterns(unittest.TestCase):
    def test_detects_bullish_vwap_reclaim_context(self) -> None:
        bars = _bars([100 + (i * 0.05) for i in range(24)] + [100.7, 101.4])
        indicators = compute_indicator_set(bars)
        patterns = detect_price_action_patterns(bars, indicators)

        self.assertIn(patterns["bias"], {"bullish", "neutral"})
        self.assertIn("vwap_state", patterns["patterns"])
        self.assertIn("summary", patterns)

    def test_detects_failed_breakout_as_bearish_pressure(self) -> None:
        bars = _bars([100.0 + (i * 0.02) for i in range(25)])
        prior_high = max(b.high for b in bars[:-1])
        bars[-1] = Bar(
            timestamp=bars[-1].timestamp,
            open=prior_high - 0.1,
            high=prior_high + 1.0,
            low=prior_high - 0.5,
            close=prior_high - 0.05,
            volume=2000.0,
        )
        indicators = compute_indicator_set(bars)
        patterns = detect_price_action_patterns(bars, indicators)

        self.assertEqual(patterns["patterns"]["failed_break"], "failed_breakout")
        self.assertLess(patterns["score"], 1.0)

    def test_context_dashboard_renders_price_action_patterns(self) -> None:
        bars = _bars([100 + (i * 0.1) for i in range(30)])
        indicators = compute_indicator_set(bars)
        context = build_market_context(indicators, bars)
        rendered = render_context_dashboard(context)

        self.assertIn("PRICE ACTION PATTERNS", rendered)
        self.assertIn("vwap_state", rendered)

    def test_detects_session_segment_and_structure(self) -> None:
        bars = _bars([100.0, 100.2, 100.4, 100.7, 101.0, 101.3, 101.6, 101.9, 102.2, 102.5])
        indicators = compute_indicator_set(bars)
        structure = detect_market_structure(bars, indicators)

        self.assertEqual(classify_session_segment(bars[-1].timestamp), "open_drive")
        self.assertEqual(structure["structure"]["session_segment"], "open_drive")
        self.assertIn(structure["structure"]["swing_structure"], {"hh_hl", "mixed"})
        self.assertIn(structure["bias"], {"bullish", "neutral"})

    def test_context_dashboard_renders_market_structure(self) -> None:
        bars = _bars([100 + (i * 0.08) for i in range(30)])
        indicators = compute_indicator_set(bars)
        context = build_market_context(indicators, bars)
        rendered = render_context_dashboard(context)

        self.assertIn("MARKET STRUCTURE", rendered)
        self.assertIn("session_segment", rendered)


if __name__ == "__main__":
    unittest.main()
