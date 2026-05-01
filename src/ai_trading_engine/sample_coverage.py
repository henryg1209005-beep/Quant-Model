from __future__ import annotations

from collections import Counter
from typing import Any


def _top_counts(rows: list[dict[str, Any]], key: str, limit: int = 50) -> dict[str, int]:
    counts = Counter(str(r.get(key) or "unknown") for r in rows)
    return dict(counts.most_common(max(1, int(limit))))


def _flag_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    keys = [
        "quality_outside_session",
        "quality_quote_stale",
        "quality_spread_too_wide",
        "quality_macro_fallback",
        "quality_economic_calendar_error",
        "quality_finnhub_context_error",
        "quality_missing_forecast",
    ]
    return {k: sum(1 for r in rows if bool(r.get(k))) for k in keys}


def build_sample_coverage_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    good = sum(1 for r in rows if bool(r.get("sample_quality_good")))
    with_quote = sum(1 for r in rows if float(r.get("quote_last") or 0.0) > 0.0)
    with_forecast = sum(1 for r in rows if str(r.get("forecast_direction") or "").upper() in {"LONG", "SHORT"})
    return {
        "ok": True,
        "sample_count": total,
        "good_sample_count": good,
        "good_sample_rate": (good / total) if total else 0.0,
        "quote_label_ready_count": with_quote,
        "forecast_ready_count": with_forecast,
        "forecast_ready_rate": (with_forecast / total) if total else 0.0,
        "by_symbol": _top_counts(rows, "symbol"),
        "by_collection_role": _top_counts(rows, "collection_role"),
        "by_session_bucket": _top_counts(rows, "session_bucket"),
        "by_indicator_regime": _top_counts(rows, "indicator_regime"),
        "by_macro_provider": _top_counts(rows, "macro_provider"),
        "by_economic_calendar_risk": _top_counts(rows, "economic_calendar_event_risk"),
        "by_news_risk": _top_counts(rows, "finnhub_news_risk"),
        "by_earnings_risk": _top_counts(rows, "finnhub_earnings_risk"),
        "quality_flag_counts": _flag_counts(rows),
    }
