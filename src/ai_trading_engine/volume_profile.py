from __future__ import annotations

from datetime import datetime

from ai_trading_engine.models import Bar


BUCKET_OPEN = "OPEN"
BUCKET_MIDDAY = "MIDDAY"
BUCKET_AFTERNOON = "AFTERNOON"
BUCKET_OVERNIGHT = "OVERNIGHT"


def session_bucket(ts: datetime) -> str:
    hour = ts.hour
    # Generic US futures-style buckets, adaptable per market.
    if 13 <= hour < 15:
        return BUCKET_OPEN
    if 15 <= hour < 18:
        return BUCKET_MIDDAY
    if 18 <= hour < 21:
        return BUCKET_AFTERNOON
    return BUCKET_OVERNIGHT


def _iqr_filtered(values: list[float]) -> list[float]:
    if len(values) < 4:
        return values
    ordered = sorted(values)
    n = len(ordered)
    q1 = ordered[n // 4]
    q3 = ordered[(3 * n) // 4]
    iqr = q3 - q1
    lo = q1 - (1.5 * iqr)
    hi = q3 + (1.5 * iqr)
    filtered = [v for v in ordered if lo <= v <= hi]
    return filtered or ordered


def volume_ratio_and_label(bars: list[Bar]) -> tuple[float, str, str]:
    if len(bars) < 2:
        return 1.0, "NORMAL", BUCKET_OVERNIGHT

    current = bars[-1]
    bucket = session_bucket(current.timestamp)
    historical = [b.volume for b in bars[:-1] if session_bucket(b.timestamp) == bucket]

    if not historical:
        baseline = sum(b.volume for b in bars[:-1]) / max(1, len(bars[:-1]))
    else:
        filtered = _iqr_filtered(historical)
        baseline = sum(filtered) / max(1, len(filtered))

    ratio = current.volume / max(1e-9, baseline)

    if ratio < 0.5:
        label = "DEAD"
    elif ratio < 0.8:
        label = "BELOW_AVG"
    elif ratio < 1.2:
        label = "NORMAL"
    elif ratio < 1.8:
        label = "ABOVE_AVG"
    else:
        label = "SURGE"

    return ratio, label, bucket
