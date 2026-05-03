from __future__ import annotations

import json
import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

from ai_trading_engine.config import Settings


MARKET_MOVING_KEYWORDS = {
    "fed",
    "fomc",
    "powell",
    "rates",
    "rate",
    "inflation",
    "cpi",
    "pce",
    "jobs",
    "payroll",
    "unemployment",
    "gdp",
    "treasury",
    "yields",
    "yield",
    "oil",
    "crude",
    "hormuz",
    "sanction",
    "tariff",
    "war",
    "missile",
    "earnings",
    "guidance",
    "revenue",
    "profit",
    "outlook",
    "downgrade",
    "upgrade",
    "antitrust",
    "sec",
    "lawsuit",
    "recall",
    "layoffs",
    "bankruptcy",
    "merger",
    "acquisition",
}

LOW_SIGNAL_TERMS = {
    "cramer",
    "watchlist",
    "things to watch",
    "opinion",
    "why shares",
    "could be",
    "might be",
}

POSITIVE_SENTIMENT_TERMS = {
    "beat",
    "beats",
    "raises",
    "raised",
    "upgrade",
    "upgrades",
    "outperform",
    "buy rating",
    "record",
    "surge",
    "jumps",
    "rallies",
    "growth",
    "profit rises",
    "revenue rises",
    "forecast",
    "strong demand",
    "contract win",
    "partnership",
    "approval",
}

NEGATIVE_SENTIMENT_TERMS = {
    "miss",
    "misses",
    "cuts",
    "cut",
    "downgrade",
    "downgrades",
    "underperform",
    "sell rating",
    "falls",
    "plunges",
    "drops",
    "slumps",
    "lawsuit",
    "probe",
    "investigation",
    "sanction",
    "tariff",
    "recall",
    "delay",
    "weak demand",
    "profit warning",
    "revenue falls",
    "bankruptcy",
    "layoffs",
}

ETF_SYMBOLS = {"SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLI", "XLP", "XLU", "SMH"}

SYMBOL_KEYWORDS = {
    "AAPL": {"apple", "iphone", "ipad", "mac", "ios", "app store"},
    "MSFT": {"microsoft", "azure", "windows", "copilot", "openai"},
    "NVDA": {"nvidia", "gpu", "gpus", "blackwell", "cuda", "ai chip", "semiconductor", "data center"},
    "TSLA": {"tesla", "ev", "robotaxi", "musk", "model y", "model 3"},
    "AMD": {"amd", "radeon", "epyc", "instinct", "ai chip", "semiconductor"},
    "META": {"meta", "facebook", "instagram", "whatsapp", "reels", "metaverse"},
}

HIGH_QUALITY_SOURCES = {"reuters", "bloomberg", "cnbc", "financial times", "wsj", "wall street journal"}
LOW_QUALITY_SOURCES = {"benzinga", "seeking alpha"}

EVENT_BUCKET_KEYWORDS: dict[str, set[str]] = {
    "macro_policy": {"fed", "fomc", "powell", "rates", "inflation", "cpi", "pce", "treasury", "yield", "yields"},
    "macro_growth": {"jobs", "payroll", "unemployment", "gdp"},
    "geopolitics": {"war", "missile", "sanction", "tariff", "hormuz"},
    "earnings": {"earnings", "guidance", "revenue", "profit", "outlook"},
    "corporate_action": {"merger", "acquisition", "lawsuit", "antitrust", "sec", "recall", "layoffs", "bankruptcy"},
}


@dataclass(frozen=True)
class FinnhubContextSnapshot:
    timestamp: str
    provider: str
    enabled: bool
    symbol: str
    news_risk: str
    earnings_risk: str
    news_sentiment_label: str
    news_sentiment_score: float
    news_confidence: float
    cleaned_summary: dict[str, Any] = field(default_factory=dict)
    market_headlines: list[dict[str, Any]] = field(default_factory=list)
    company_headlines: list[dict[str, Any]] = field(default_factory=list)
    earnings: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _get_json(path: str, params: dict[str, Any], timeout: int) -> Any:
    query = urlencode({k: v for k, v in params.items() if v is not None and v != ""})
    url = f"https://finnhub.io/api/v1/{path}?{query}"
    with urlopen(url, timeout=max(1, int(timeout))) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _norm_text(value: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", value.lower()).strip()


def _canonical_headline(value: str) -> str:
    text = _norm_text(value)
    text = re.sub(r"\b(reuters|yahoo|cnbc|seekingalpha|marketwatch|bloomberg)\b", "", text)
    return re.sub(r"\s+", " ", text).strip()[:120]


def _score_headline(headline: str, summary: str, symbol: str, related: str, source: str, *, market: bool) -> float:
    text = _norm_text(f"{headline} {summary}")
    words = set(text.split())
    score = 0.0
    hits = words.intersection(MARKET_MOVING_KEYWORDS)
    score += min(3.0, float(len(hits)))
    if symbol and symbol.lower() in _norm_text(related).split():
        score += 1.0
    if not market and symbol and symbol.lower() in words:
        score += 0.5
    symbol_terms = SYMBOL_KEYWORDS.get(symbol.upper(), set())
    if not market and symbol_terms:
        score += 2.0 * sum(1 for term in symbol_terms if term in text)
    if any(term in text for term in LOW_SIGNAL_TERMS):
        score -= 1.0
    if str(source or "").lower() in {"reuters", "ap", "associated press", "bloomberg"}:
        score += 0.5
    return max(0.0, score)


def _headline_sentiment(headline: str, summary: str) -> tuple[float, list[str]]:
    text = _norm_text(f"{headline} {summary}")
    pos_hits = [term for term in POSITIVE_SENTIMENT_TERMS if term in text]
    neg_hits = [term for term in NEGATIVE_SENTIMENT_TERMS if term in text]
    score = (0.35 * len(pos_hits)) - (0.35 * len(neg_hits))
    return max(-1.0, min(1.0, score)), [*pos_hits[:3], *neg_hits[:3]]


def _sentiment_label(score: float) -> str:
    if score >= 0.2:
        return "bullish"
    if score <= -0.2:
        return "bearish"
    return "neutral"


def _aggregate_sentiment(headlines: list[dict[str, Any]]) -> tuple[str, float]:
    if not headlines:
        return "neutral", 0.0
    weighted = 0.0
    weight_sum = 0.0
    for h in headlines:
        score = float(h.get("sentiment_score") or 0.0)
        weight = max(1.0, float(h.get("relevance_score") or 1.0))
        weighted += score * weight
        weight_sum += weight
    out = weighted / weight_sum if weight_sum > 0 else 0.0
    return _sentiment_label(out), round(max(-1.0, min(1.0, out)), 3)


def _headline(item: dict[str, Any]) -> dict[str, Any]:
    dt = item.get("datetime")
    published = ""
    try:
        published = datetime.fromtimestamp(float(dt), tz=timezone.utc).isoformat() if dt else ""
    except (TypeError, ValueError, OSError):
        published = ""
    return {
        "id": str(item.get("id") or ""),
        "published_at": published,
        "source": str(item.get("source") or ""),
        "headline": str(item.get("headline") or "")[:240],
        "summary": str(item.get("summary") or "")[:360],
        "url": str(item.get("url") or ""),
        "related": str(item.get("related") or ""),
    }


def _published_dt(value: str) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _headline_key(h: dict[str, Any]) -> str:
    parts = [
        str(h.get("id") or "").strip().lower(),
        str(h.get("url") or "").strip().lower(),
        _canonical_headline(str(h.get("headline") or "")),
    ]
    payload = "|".join(parts).encode("utf-8", errors="ignore")
    return hashlib.sha1(payload).hexdigest()


def _event_bucket(headline: str, summary: str) -> str:
    text = _norm_text(f"{headline} {summary}")
    for bucket, keys in EVENT_BUCKET_KEYWORDS.items():
        if any(k in text for k in keys):
            return bucket
    return "general"


def _filter_headlines(
    items: list[dict[str, Any]],
    *,
    symbol: str,
    max_headlines: int,
    min_score: float,
    market: bool,
    max_age_hours: float,
) -> list[dict[str, Any]]:
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    seen: set[str] = set()
    now = _now()
    for raw in items:
        headline = _headline(raw)
        title = str(headline.get("headline") or "")
        if not title:
            continue
        key = _headline_key(headline)
        if key in seen:
            continue
        seen.add(key)
        published_dt = _published_dt(str(headline.get("published_at") or ""))
        if published_dt is not None:
            age_h = max(0.0, (now - published_dt).total_seconds() / 3600.0)
            if age_h > float(max_age_hours):
                continue
        source = str(headline.get("source") or "").strip().lower()
        if source in LOW_QUALITY_SOURCES:
            continue
        if not market and symbol in ETF_SYMBOLS:
            related = _norm_text(str(headline.get("related") or ""))
            title_text = _norm_text(title)
            if symbol.lower() not in related.split() and symbol.lower() not in title_text.split():
                continue
        if not market and symbol not in ETF_SYMBOLS and symbol in SYMBOL_KEYWORDS:
            text = _norm_text(f"{title} {headline.get('summary') or ''}")
            if not any(term in text for term in SYMBOL_KEYWORDS[symbol]):
                continue
        score = _score_headline(
            title,
            str(headline.get("summary") or ""),
            symbol,
            str(headline.get("related") or ""),
            str(headline.get("source") or ""),
            market=market,
        )
        if score < min_score:
            continue
        if source in HIGH_QUALITY_SOURCES:
            score += 0.4
        headline["relevance_score"] = round(score, 2)
        sentiment_score, sentiment_terms = _headline_sentiment(
            title,
            str(headline.get("summary") or ""),
        )
        headline["sentiment_score"] = round(sentiment_score, 3)
        headline["sentiment_label"] = _sentiment_label(sentiment_score)
        headline["sentiment_terms"] = sentiment_terms
        headline["event_bucket"] = _event_bucket(title, str(headline.get("summary") or ""))
        headline["dedup_key"] = key
        ranked.append((score, str(headline.get("published_at") or ""), headline))
    ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return [row[2] for row in ranked[:max_headlines]]


def _earnings_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "date": str(item.get("date") or ""),
        "hour": str(item.get("hour") or ""),
        "symbol": str(item.get("symbol") or ""),
        "eps_estimate": item.get("epsEstimate"),
        "revenue_estimate": item.get("revenueEstimate"),
        "year": item.get("year"),
        "quarter": item.get("quarter"),
    }


def _days_until(value: str, today: date) -> int | None:
    try:
        dt = date.fromisoformat(str(value))
    except ValueError:
        return None
    return (dt - today).days


def load_finnhub_context(settings: Settings, symbol: str) -> FinnhubContextSnapshot:
    now = _now()
    symbol = str(symbol or settings.symbol).strip().upper()
    if not settings.finnhub_context_enabled:
        return FinnhubContextSnapshot(
            timestamp=now.isoformat(),
            provider="disabled",
            enabled=False,
            symbol=symbol,
            news_risk="disabled",
            earnings_risk="disabled",
            news_sentiment_label="neutral",
            news_sentiment_score=0.0,
            news_confidence=0.0,
        )
    if not settings.finnhub_api_key:
        return FinnhubContextSnapshot(
            timestamp=now.isoformat(),
            provider="finnhub",
            enabled=True,
            symbol=symbol,
            news_risk="unavailable",
            earnings_risk="unavailable",
            news_sentiment_label="neutral",
            news_sentiment_score=0.0,
            news_confidence=0.0,
            notes="FINNHUB_API_KEY is not configured.",
        )

    max_headlines = max(1, min(10, int(settings.finnhub_news_max_headlines)))
    min_score = max(0.0, float(settings.finnhub_news_min_score))
    timeout = max(1, int(settings.finnhub_request_timeout_seconds))
    max_age_hours = max(2.0, float(max(1, int(settings.finnhub_news_lookback_days))) * 24.0)
    today = now.date()
    news_from = today - timedelta(days=max(0, min(14, int(settings.finnhub_news_lookback_days))))
    earnings_to = today + timedelta(days=max(1, min(60, int(settings.finnhub_earnings_lookahead_days))))
    notes: list[str] = []

    market_headlines: list[dict[str, Any]] = []
    company_headlines: list[dict[str, Any]] = []
    earnings: list[dict[str, Any]] = []
    try:
        market_raw = _get_json(
            "news",
            {"category": "general", "token": settings.finnhub_api_key},
            timeout,
        )
        if isinstance(market_raw, list):
            market_headlines = _filter_headlines(
                [x for x in market_raw if isinstance(x, dict)],
                symbol=symbol,
                max_headlines=max_headlines,
                min_score=min_score,
                market=True,
                max_age_hours=max_age_hours,
            )
    except Exception as exc:
        notes.append(f"market_news_error={exc}")

    try:
        company_raw = _get_json(
            "company-news",
            {
                "symbol": symbol,
                "from": news_from.isoformat(),
                "to": today.isoformat(),
                "token": settings.finnhub_api_key,
            },
            timeout,
        )
        if isinstance(company_raw, list):
            company_headlines = _filter_headlines(
                [x for x in company_raw if isinstance(x, dict)],
                symbol=symbol,
                max_headlines=max_headlines,
                min_score=min_score,
                market=False,
                max_age_hours=max_age_hours,
            )
    except Exception as exc:
        notes.append(f"company_news_error={exc}")

    try:
        earnings_raw = _get_json(
            "calendar/earnings",
            {
                "from": today.isoformat(),
                "to": earnings_to.isoformat(),
                "symbol": symbol,
                "international": "false",
                "token": settings.finnhub_api_key,
            },
            timeout,
        )
        rows = earnings_raw.get("earningsCalendar") if isinstance(earnings_raw, dict) else []
        if isinstance(rows, list):
            earnings = [_earnings_item(x) for x in rows if isinstance(x, dict)]
    except Exception as exc:
        notes.append(f"earnings_error={exc}")

    company_score = sum(float(h.get("relevance_score") or 0.0) for h in company_headlines)
    market_score = sum(float(h.get("relevance_score") or 0.0) for h in market_headlines)
    if company_score >= 5.0 or len(company_headlines) >= 3:
        news_risk = "symbol_news_active"
    elif market_score >= 5.0 or len(market_headlines) >= max_headlines:
        news_risk = "market_news_active"
    elif company_headlines or market_headlines:
        news_risk = "low"
    else:
        news_risk = "clear"

    window = max(0, int(settings.earnings_risk_window_days))
    upcoming_earnings = [
        e
        for e in earnings
        if (days := _days_until(str(e.get("date") or ""), today)) is not None and 0 <= days <= window
    ]
    if upcoming_earnings:
        earnings_risk = "earnings_window"
    elif earnings:
        earnings_risk = "earnings_upcoming"
    else:
        earnings_risk = "clear"
    sentiment_label, sentiment_score = _aggregate_sentiment([*company_headlines, *market_headlines])
    kept = [*company_headlines, *market_headlines]
    if kept:
        conf = sum(float(h.get("relevance_score") or 0.0) for h in kept) / float(max(1, len(kept)))
        conf = min(1.0, conf / 4.0)
    else:
        conf = 0.0
    bucket_counts: dict[str, int] = {}
    for h in kept:
        b = str(h.get("event_bucket") or "general")
        bucket_counts[b] = int(bucket_counts.get(b, 0)) + 1
    cleaned_summary = {
        "kept_headlines": int(len(kept)),
        "company_kept": int(len(company_headlines)),
        "market_kept": int(len(market_headlines)),
        "max_age_hours": float(max_age_hours),
        "event_buckets": bucket_counts,
        "high_quality_sources_used": sorted({str(h.get("source") or "").lower() for h in kept if str(h.get("source") or "").lower() in HIGH_QUALITY_SOURCES}),
    }

    return FinnhubContextSnapshot(
        timestamp=now.isoformat(),
        provider="finnhub",
        enabled=True,
        symbol=symbol,
        news_risk=news_risk,
        earnings_risk=earnings_risk,
        news_sentiment_label=sentiment_label,
        news_sentiment_score=sentiment_score,
        news_confidence=round(conf, 3),
        cleaned_summary=cleaned_summary,
        market_headlines=market_headlines,
        company_headlines=company_headlines,
        earnings=earnings[:10],
        notes="; ".join([*notes, f"news_filter_min_score={min_score}"]).strip("; "),
    )


def render_finnhub_context(snapshot: FinnhubContextSnapshot) -> str:
    if not snapshot.enabled:
        return "FINNHUB NEWS/EARNINGS CONTEXT\nProvider: disabled"
    market = "\n".join(
        f"- [{h.get('relevance_score')}] {h.get('headline')}" for h in snapshot.market_headlines[:3]
    ) or "- none"
    company = "\n".join(
        f"- [{h.get('relevance_score')}] {h.get('headline')}" for h in snapshot.company_headlines[:3]
    ) or "- none"
    earnings = "\n".join(
        f"- {e.get('date')} {e.get('hour')}: {e.get('symbol')} EPS est={e.get('eps_estimate')}"
        for e in snapshot.earnings[:4]
    ) or "- none"
    return (
        "FINNHUB NEWS/EARNINGS CONTEXT\n"
        f"Provider: {snapshot.provider}\n"
        f"Symbol: {snapshot.symbol}\n"
        f"News risk: {snapshot.news_risk}\n"
        f"Earnings risk: {snapshot.earnings_risk}\n"
        f"Headline sentiment: {snapshot.news_sentiment_label} ({snapshot.news_sentiment_score:.3f})\n"
        f"News confidence: {snapshot.news_confidence:.3f}\n"
        f"Market headlines:\n{market}\n"
        f"Company headlines:\n{company}\n"
        f"Earnings:\n{earnings}\n"
        f"Notes: {snapshot.notes or 'none'}"
    )
