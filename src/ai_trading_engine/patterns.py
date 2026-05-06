from __future__ import annotations

from typing import Any

from ai_trading_engine.models import Bar, IndicatorSet


def _avg(values: list[float]) -> float:
    return sum(values) / float(max(1, len(values)))


def _near(a: float, b: float, tolerance: float) -> bool:
    return abs(a - b) <= max(1e-9, tolerance)


def detect_price_action_patterns(bars: list[Bar], indicators: IndicatorSet) -> dict[str, Any]:
    """Convert recent bars into explicit setup labels for downstream gating/research."""
    if not bars:
        return {
            "bias": "neutral",
            "score": 0.0,
            "summary": "no_bars",
            "patterns": {},
        }

    current = bars[-1]
    previous = bars[-2] if len(bars) >= 2 else current
    atr = max(1e-9, float(indicators.atr14 or 0.0))
    score = 0.0
    patterns: dict[str, Any] = {}

    prev_below_vwap = previous.close < indicators.vwap
    prev_above_vwap = previous.close > indicators.vwap
    close_above_vwap = current.close > indicators.vwap
    close_below_vwap = current.close < indicators.vwap
    touched_vwap = current.low <= indicators.vwap <= current.high
    if prev_below_vwap and close_above_vwap:
        vwap_state = "reclaim"
        score += 2.0
    elif prev_above_vwap and close_below_vwap:
        vwap_state = "lost"
        score -= 2.0
    elif touched_vwap and close_above_vwap:
        vwap_state = "support_rejection"
        score += 1.0
    elif touched_vwap and close_below_vwap:
        vwap_state = "resistance_rejection"
        score -= 1.0
    elif close_above_vwap:
        vwap_state = "above"
        score += 0.5
    elif close_below_vwap:
        vwap_state = "below"
        score -= 0.5
    else:
        vwap_state = "at"
    patterns["vwap_state"] = vwap_state

    near_ema9 = _near(current.low, indicators.ema9, atr * 0.25) or _near(current.close, indicators.ema9, atr * 0.25)
    near_ema21 = _near(current.low, indicators.ema21, atr * 0.30) or _near(current.close, indicators.ema21, atr * 0.30)
    if indicators.regime == "trending_up" and current.close >= indicators.ema21 and (near_ema9 or near_ema21):
        trend_pullback = "bull_pullback_hold"
        score += 2.0
    elif indicators.regime == "trending_down" and current.close <= indicators.ema21 and (near_ema9 or near_ema21):
        trend_pullback = "bear_pullback_hold"
        score -= 2.0
    elif indicators.regime == "trending_up" and current.close < indicators.ema21:
        trend_pullback = "bull_pullback_failed"
        score -= 1.5
    elif indicators.regime == "trending_down" and current.close > indicators.ema21:
        trend_pullback = "bear_pullback_failed"
        score += 1.5
    else:
        trend_pullback = "none"
    patterns["trend_pullback"] = trend_pullback

    lookback = bars[-21:-1] if len(bars) >= 21 else bars[:-1]
    if lookback:
        range_high = max(b.high for b in lookback)
        range_low = min(b.low for b in lookback)
        if current.close > range_high:
            range_break = "breakout_up"
            score += 1.5
        elif current.close < range_low:
            range_break = "breakout_down"
            score -= 1.5
        else:
            range_break = "inside"
        patterns["recent_range_break"] = range_break

        prior_high = max(b.high for b in lookback)
        prior_low = min(b.low for b in lookback)
        if current.high > prior_high and current.close <= prior_high:
            failed_break = "failed_breakout"
            score -= 2.0
        elif current.low < prior_low and current.close >= prior_low:
            failed_break = "failed_breakdown"
            score += 2.0
        else:
            failed_break = "none"
        patterns["failed_break"] = failed_break
    else:
        patterns["recent_range_break"] = "unknown"
        patterns["failed_break"] = "unknown"

    recent = bars[-5:] if len(bars) >= 5 else bars
    baseline = bars[-25:-5] if len(bars) >= 25 else bars[:-5]
    recent_range = _avg([b.high - b.low for b in recent])
    baseline_range = _avg([b.high - b.low for b in baseline]) if baseline else recent_range
    compression_ratio = recent_range / max(1e-9, baseline_range)
    if compression_ratio <= 0.65:
        compression = "compressed"
    elif compression_ratio >= 1.35:
        compression = "expanding"
    else:
        compression = "normal"
    patterns["range_compression"] = compression
    patterns["range_compression_ratio"] = round(compression_ratio, 3)

    if score >= 2.0:
        bias = "bullish"
    elif score <= -2.0:
        bias = "bearish"
    else:
        bias = "neutral"
    summary_parts = [
        f"bias={bias}",
        f"score={score:.1f}",
        f"vwap={patterns['vwap_state']}",
        f"pullback={patterns['trend_pullback']}",
        f"range={patterns['recent_range_break']}",
        f"fail={patterns['failed_break']}",
        f"compression={compression}",
    ]
    return {
        "bias": bias,
        "score": float(round(score, 2)),
        "summary": " ".join(summary_parts),
        "patterns": patterns,
    }
