from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any

from ai_trading_engine.models import Bar, IndicatorSet


_NY_TZ = ZoneInfo("America/New_York")


def classify_session_segment(ts: datetime) -> str:
    et = ts.astimezone(_NY_TZ)
    minute = (et.hour * 60) + et.minute
    if minute < (9 * 60 + 30) or minute >= (16 * 60):
        return "overnight"
    if minute < (10 * 60 + 15):
        return "open_drive"
    if minute < (11 * 60 + 30):
        return "morning"
    if minute < (13 * 60 + 30):
        return "lunch"
    if minute < (15 * 60):
        return "afternoon"
    return "power_hour"


def _strict_monotonic(values: list[float], *, direction: str) -> bool:
    if len(values) < 3:
        return False
    pairs = zip(values[:-1], values[1:])
    if direction == "up":
        return all(b > a for a, b in pairs)
    return all(b < a for a, b in pairs)


def detect_market_structure(bars: list[Bar], indicators: IndicatorSet) -> dict[str, Any]:
    if not bars:
        return {
            "bias": "neutral",
            "score": 0.0,
            "summary": "no_bars",
            "structure": {},
        }

    current = bars[-1]
    recent = bars[-4:] if len(bars) >= 4 else bars
    highs = [float(b.high) for b in recent]
    lows = [float(b.low) for b in recent]
    atr = max(1e-9, float(indicators.atr14 or 0.0))
    score = 0.0

    if _strict_monotonic(highs, direction="up") and _strict_monotonic(lows, direction="up"):
        swing_structure = "hh_hl"
        score += 2.0
    elif _strict_monotonic(highs, direction="down") and _strict_monotonic(lows, direction="down"):
        swing_structure = "lh_ll"
        score -= 2.0
    else:
        span = max(highs) - min(lows)
        swing_structure = "range" if span <= (atr * 2.2) else "mixed"

    prior = bars[-9:-1] if len(bars) >= 9 else bars[:-1]
    if prior:
        prior_high = max(float(b.high) for b in prior)
        prior_low = min(float(b.low) for b in prior)
        if current.close > prior_high:
            breakout_state = "accepted_breakout_up"
            score += 1.5
        elif current.close < prior_low:
            breakout_state = "accepted_breakout_down"
            score -= 1.5
        elif current.high > prior_high and current.close <= prior_high:
            breakout_state = "failed_breakout_up"
            score -= 1.0
        elif current.low < prior_low and current.close >= prior_low:
            breakout_state = "failed_breakout_down"
            score += 1.0
        else:
            breakout_state = "inside"
    else:
        breakout_state = "unknown"

    bar_range = max(1e-9, float(current.high - current.low))
    close_location = (float(current.close) - float(current.low)) / bar_range
    if close_location >= 0.7:
        close_location_label = "near_high"
        score += 0.5
    elif close_location <= 0.3:
        close_location_label = "near_low"
        score -= 0.5
    else:
        close_location_label = "mid"

    if indicators.regime == "trending_up" and current.close >= indicators.ema9:
        impulse_state = "bull_impulse"
        score += 1.0
    elif indicators.regime == "trending_down" and current.close <= indicators.ema9:
        impulse_state = "bear_impulse"
        score -= 1.0
    else:
        impulse_state = "two_sided"

    session_segment = classify_session_segment(current.timestamp)
    summary_parts = [
        f"segment={session_segment}",
        f"structure={swing_structure}",
        f"breakout={breakout_state}",
        f"close={close_location_label}",
        f"impulse={impulse_state}",
    ]
    if score >= 2.0:
        bias = "bullish"
    elif score <= -2.0:
        bias = "bearish"
    else:
        bias = "neutral"
    return {
        "bias": bias,
        "score": float(round(score, 2)),
        "summary": f"bias={bias} score={score:.1f} " + " ".join(summary_parts),
        "structure": {
            "session_segment": session_segment,
            "swing_structure": swing_structure,
            "breakout_state": breakout_state,
            "close_location": close_location_label,
            "impulse_state": impulse_state,
        },
    }
