from __future__ import annotations

from ai_trading_engine.indicators import nearest_level_distance_atr
from ai_trading_engine.market_structure import detect_market_structure
from ai_trading_engine.models import ContextDimension, IndicatorSet, MarketContext
from ai_trading_engine.patterns import detect_price_action_patterns
from ai_trading_engine.volume_profile import volume_ratio_and_label


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def trend_label(score: float) -> str:
    if score >= 65:
        return "STRONG_BULL"
    if score >= 20:
        return "BULL"
    if score <= -65:
        return "STRONG_BEAR"
    if score <= -20:
        return "BEAR"
    return "NEUTRAL"


def volatility_label(atr_ratio: float) -> str:
    if atr_ratio < 0.0015:
        return "LOW"
    if atr_ratio < 0.0035:
        return "NORMAL"
    if atr_ratio < 0.006:
        return "HIGH"
    return "EXTREME"


def build_market_context(
    indicators: IndicatorSet,
    primary_bars,
    gex_overlay: dict[str, float | str] | None = None,
) -> MarketContext:
    price = primary_bars[-1].close

    ema_component = 40.0 if indicators.ema9 > indicators.ema21 else -40.0
    macd_component = _clamp(indicators.macd_hist * 60.0, -25.0, 25.0)
    vwap_component = _clamp(((price - indicators.vwap) / max(1e-9, indicators.atr14)) * 12.0, -20.0, 20.0)
    rsi_component = _clamp((indicators.rsi14 - 50.0) * 0.8, -15.0, 15.0)
    trend_score = _clamp(ema_component + macd_component + vwap_component + rsi_component, -100.0, 100.0)

    momentum_raw = (indicators.macd_hist * 70.0) + ((indicators.rsi14 - 50.0) * 1.2)
    momentum_score = _clamp(momentum_raw, -100.0, 100.0)

    mean_reversion = _clamp(abs(price - indicators.bb_mid) / max(1e-9, indicators.atr14) * 18.0, 0.0, 100.0)
    snap = "DOWN" if price > indicators.bb_mid else "UP"

    atr_ratio = indicators.atr14 / max(1e-9, price)
    vol_label = volatility_label(atr_ratio)

    volume_ratio, volume_label, bucket = volume_ratio_and_label(primary_bars)

    levels = {
        "VWAP": indicators.vwap,
        "EMA9": indicators.ema9,
        "EMA21": indicators.ema21,
        "SMA50": indicators.sma50,
        "BB_UPPER": indicators.bb_upper,
        "BB_LOWER": indicators.bb_lower,
        "PIVOT": indicators.pivot,
        "SUPPORT_1": indicators.support_1,
        "RESISTANCE_1": indicators.resistance_1,
    }
    nearest_name, nearest_atr = nearest_level_distance_atr(price, indicators.atr14, levels)

    dims = [
        ContextDimension(
            name="Trend",
            value=round(trend_score, 1),
            label=trend_label(trend_score),
            detail="Composite of EMA cross, MACD histogram, VWAP distance, RSI bias.",
        ),
        ContextDimension(
            name="Momentum",
            value=round(momentum_score, 1),
            label="ACCELERATING" if momentum_score > 20 else "FADING" if momentum_score < -20 else "STABLE",
            detail="Rate of change and acceleration from MACD histogram and RSI drift.",
        ),
        ContextDimension(
            name="Mean Reversion",
            value=round(mean_reversion, 1),
            label=f"SNAP_{snap}",
            detail="Distance from mean in ATR units; higher values imply stronger reversion pressure.",
        ),
        ContextDimension(
            name="Volatility",
            value=vol_label,
            label=vol_label,
            detail=f"ATR ratio={atr_ratio:.4f}; regime={indicators.regime}",
        ),
        ContextDimension(
            name="Volume",
            value=round(volume_ratio, 2),
            label=volume_label,
            detail=f"Session bucket={bucket}; compared vs same-bucket baseline.",
        ),
        ContextDimension(
            name="S/R Proximity",
            value=round(nearest_atr, 2),
            label=nearest_name,
            detail="Distance to nearest structural level in ATR units.",
        ),
    ]

    pattern_features = detect_price_action_patterns(primary_bars, indicators)
    structure_features = detect_market_structure(primary_bars, indicators)
    return MarketContext(
        dimensions=dims,
        key_levels=levels,
        gex_overlay=gex_overlay,
        pattern_features=pattern_features,
        structure_features=structure_features,
    )


def render_context_dashboard(context: MarketContext) -> str:
    lines: list[str] = []
    lines.append("MARKET CONTEXT DASHBOARD")
    lines.append("----------------------------------------")
    for idx, dim in enumerate(context.dimensions, start=1):
        lines.append(f"{idx}. {dim.name}: {dim.label} ({dim.value})")
        lines.append(f"   {dim.detail}")

    lines.append("7. Key Levels")
    for k, v in context.key_levels.items():
        lines.append(f"   - {k}: {v:.2f}")

    if context.gex_overlay:
        lines.append("GEX Overlay (fresh)")
        for k, v in context.gex_overlay.items():
            if isinstance(v, (int, float)):
                lines.append(f"   - {k}: {v:.2f}")
            else:
                lines.append(f"   - {k}: {v}")

    if context.pattern_features:
        patterns = dict(context.pattern_features.get("patterns") or {})
        lines.append("PRICE ACTION PATTERNS")
        lines.append(f"   - Summary: {context.pattern_features.get('summary', 'none')}")
        for key in ("vwap_state", "trend_pullback", "recent_range_break", "failed_break", "range_compression"):
            lines.append(f"   - {key}: {patterns.get(key, 'unknown')}")

    if context.structure_features:
        structure = dict(context.structure_features.get("structure") or {})
        lines.append("MARKET STRUCTURE")
        lines.append(f"   - Summary: {context.structure_features.get('summary', 'none')}")
        for key in ("session_segment", "swing_structure", "breakout_state", "close_location", "impulse_state"):
            lines.append(f"   - {key}: {structure.get(key, 'unknown')}")

    return "\n".join(lines)
