from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime

from ai_trading_engine.models import Bar, IndicatorSet


def _closes(bars: list[Bar]) -> list[float]:
    return [b.close for b in bars]


def _highs(bars: list[Bar]) -> list[float]:
    return [b.high for b in bars]


def _lows(bars: list[Bar]) -> list[float]:
    return [b.low for b in bars]


def sma(values: list[float], period: int) -> float:
    if not values:
        return 0.0
    period = max(1, min(period, len(values)))
    return sum(values[-period:]) / period


def ema(values: list[float], period: int) -> float:
    if not values:
        return 0.0
    k = 2 / (period + 1)
    current = values[0]
    for v in values[1:]:
        current = (v * k) + (current * (1 - k))
    return current


def rsi(values: list[float], period: int = 14) -> float:
    if len(values) < 2:
        return 50.0
    period = max(1, min(period, len(values) - 1))
    gains: list[float] = []
    losses: list[float] = []
    for i in range(-period, 0):
        change = values[i] - values[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(values: list[float], fast: int = 12, slow: int = 26, signal_period: int = 9) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    m_fast = ema(values, min(fast, len(values)))
    m_slow = ema(values, min(slow, len(values)))
    line = m_fast - m_slow

    # Build a lightweight signal approximation series from recent macd lines.
    macd_series: list[float] = []
    for i in range(2, min(len(values), 60) + 1):
        window = values[-i:]
        macd_series.append(ema(window, min(fast, len(window))) - ema(window, min(slow, len(window))))
    sig = ema(macd_series, min(signal_period, len(macd_series))) if macd_series else 0.0
    hist = line - sig
    return line, sig, hist


def true_range(curr: Bar, prev: Bar | None) -> float:
    if prev is None:
        return curr.high - curr.low
    return max(curr.high - curr.low, abs(curr.high - prev.close), abs(curr.low - prev.close))


def atr(bars: list[Bar], period: int = 14) -> float:
    if not bars:
        return 0.0
    trs: list[float] = []
    prev: Bar | None = None
    for bar in bars:
        trs.append(true_range(bar, prev))
        prev = bar
    period = max(1, min(period, len(trs)))
    return sum(trs[-period:]) / period


def bollinger(values: list[float], period: int = 20, std_mult: float = 2.0) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    period = max(2, min(period, len(values)))
    window = values[-period:]
    mid = sum(window) / period
    sd = statistics.pstdev(window) if len(window) > 1 else 0.0
    upper = mid + (sd * std_mult)
    lower = mid - (sd * std_mult)
    return mid, upper, lower


def vwap(bars: list[Bar]) -> tuple[float, float]:
    if not bars:
        return 0.0, 0.0
    typical_prices = [((b.high + b.low + b.close) / 3.0) for b in bars]
    total_vol = sum(max(1e-9, b.volume) for b in bars)
    vw = sum(tp * b.volume for tp, b in zip(typical_prices, bars)) / total_vol
    sd = statistics.pstdev(typical_prices) if len(typical_prices) > 1 else 0.0
    return vw, sd


def pivot_sr(latest: Bar) -> tuple[float, float, float]:
    p = (latest.high + latest.low + latest.close) / 3.0
    r1 = (2 * p) - latest.low
    s1 = (2 * p) - latest.high
    return p, s1, r1


def classify_regime(ema9: float, ema21: float, atr_val: float, closes: list[float]) -> str:
    if len(closes) < 2:
        return "ranging"
    change = closes[-1] - closes[0]
    pct_move = abs(change) / max(1e-9, closes[0])
    volatile = atr_val / max(1e-9, closes[-1]) > 0.004

    if volatile and pct_move < 0.002:
        return "volatile"
    if ema9 > ema21 and change > 0:
        return "trending_up"
    if ema9 < ema21 and change < 0:
        return "trending_down"
    return "ranging"


def compute_indicator_set(primary_bars: list[Bar]) -> IndicatorSet:
    closes = _closes(primary_bars)
    highs = _highs(primary_bars)
    lows = _lows(primary_bars)

    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)
    sma50 = sma(closes, 50)
    rsi14 = rsi(closes, 14)
    macd_line, macd_signal, macd_hist = macd(closes)
    atr14 = atr(primary_bars, 14)
    bb_mid, bb_upper, bb_lower = bollinger(closes, 20, 2.0)
    vwap_value, vwap_std = vwap(primary_bars)

    if primary_bars:
        pivot, s1, r1 = pivot_sr(primary_bars[-1])
    else:
        pivot, s1, r1 = 0.0, 0.0, 0.0

    regime = classify_regime(ema9, ema21, atr14, closes)

    return IndicatorSet(
        ema9=ema9,
        ema21=ema21,
        sma50=sma50,
        rsi14=rsi14,
        macd=macd_line,
        macd_signal=macd_signal,
        macd_hist=macd_hist,
        atr14=atr14,
        bb_mid=bb_mid,
        bb_upper=bb_upper,
        bb_lower=bb_lower,
        vwap=vwap_value,
        vwap_std=vwap_std,
        pivot=pivot,
        support_1=s1,
        resistance_1=r1,
        regime=regime,
    )


def nearest_level_distance_atr(price: float, atr_value: float, levels: dict[str, float]) -> tuple[str, float]:
    if not levels:
        return "none", math.inf
    nearest_name = "none"
    nearest_dist = math.inf
    for name, level in levels.items():
        dist = abs(price - level)
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_name = name
    atr_units = nearest_dist / max(1e-9, atr_value)
    return nearest_name, atr_units


def format_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")
