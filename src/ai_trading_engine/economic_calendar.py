from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

from ai_trading_engine.config import Settings


@dataclass(frozen=True)
class EconomicEvent:
    time: str
    country: str
    event: str
    impact: str
    actual: float | str | None = None
    estimate: float | str | None = None
    previous: float | str | None = None
    minutes_until: float | None = None


@dataclass(frozen=True)
class EconomicCalendarSnapshot:
    timestamp: str
    provider: str
    enabled: bool
    event_risk: str
    event_count: int
    high_impact_count: int
    next_event: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _parse_event_time(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _coerce_value(value: Any) -> float | str | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _event_from_payload(payload: dict[str, Any], now: datetime) -> EconomicEvent:
    event_time = _parse_event_time(payload.get("time"))
    minutes_until = None
    if event_time is not None:
        minutes_until = (event_time - now).total_seconds() / 60.0
    return EconomicEvent(
        time=str(payload.get("time") or ""),
        country=str(payload.get("country") or ""),
        event=str(payload.get("event") or ""),
        impact=str(payload.get("impact") or "").lower(),
        actual=_coerce_value(payload.get("actual")),
        estimate=_coerce_value(payload.get("estimate")),
        previous=_coerce_value(payload.get("prev")),
        minutes_until=minutes_until,
    )


def _fetch_finnhub_events(settings: Settings, start: date, end: date) -> list[dict[str, Any]]:
    query = urlencode(
        {
            "from": start.isoformat(),
            "to": end.isoformat(),
            "token": settings.finnhub_api_key,
        }
    )
    url = f"https://finnhub.io/api/v1/calendar/economic?{query}"
    with urlopen(url, timeout=max(1, int(settings.economic_calendar_request_timeout_seconds))) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    events = payload.get("economicCalendar") or []
    return [e for e in events if isinstance(e, dict)]


def load_economic_calendar(settings: Settings) -> EconomicCalendarSnapshot:
    now = _now()
    if not settings.economic_calendar_enabled:
        return EconomicCalendarSnapshot(
            timestamp=now.isoformat(),
            provider="disabled",
            enabled=False,
            event_risk="disabled",
            event_count=0,
            high_impact_count=0,
        )
    if settings.economic_calendar_provider != "finnhub" or not settings.finnhub_api_key:
        return EconomicCalendarSnapshot(
            timestamp=now.isoformat(),
            provider=str(settings.economic_calendar_provider or "unknown"),
            enabled=True,
            event_risk="unavailable",
            event_count=0,
            high_impact_count=0,
            notes="Economic calendar provider/key not configured.",
        )

    try:
        lookahead = max(0, min(30, int(settings.economic_calendar_lookahead_days)))
        raw_events = _fetch_finnhub_events(settings, now.date(), now.date() + timedelta(days=lookahead))
        events = [_event_from_payload(e, now) for e in raw_events]
        events = [e for e in events if not e.country or e.country.upper() == "US"]
        events.sort(key=lambda e: _parse_event_time(e.time) or datetime.max.replace(tzinfo=timezone.utc))
        high = [e for e in events if e.impact == "high"]

        upcoming = [e for e in events if e.minutes_until is None or e.minutes_until >= -15]
        next_event = asdict(upcoming[0]) if upcoming else {}
        window = max(1, int(settings.economic_calendar_high_impact_window_minutes))
        high_in_window = [
            e
            for e in high
            if e.minutes_until is not None and (-15.0 <= e.minutes_until <= float(window))
        ]
        if high_in_window:
            risk = "high_impact_window"
        elif high:
            risk = "high_impact_upcoming"
        elif events:
            risk = "scheduled_events"
        else:
            risk = "clear"

        return EconomicCalendarSnapshot(
            timestamp=now.isoformat(),
            provider="finnhub",
            enabled=True,
            event_risk=risk,
            event_count=len(events),
            high_impact_count=len(high),
            next_event=next_event,
            events=[asdict(e) for e in events[:20]],
            notes=f"Lookahead {lookahead}d; high-impact risk window {window}m.",
        )
    except Exception as exc:
        return EconomicCalendarSnapshot(
            timestamp=now.isoformat(),
            provider="finnhub",
            enabled=True,
            event_risk="error",
            event_count=0,
            high_impact_count=0,
            notes=f"economic_calendar_error={exc}",
        )


def render_economic_calendar(snapshot: EconomicCalendarSnapshot) -> str:
    if not snapshot.enabled:
        return "ECONOMIC CALENDAR\nProvider: disabled\nEvent risk: disabled"
    next_event = snapshot.next_event or {}
    event_lines = []
    for event in snapshot.events[:6]:
        mins = event.get("minutes_until")
        mins_text = "n/a" if mins is None else f"{float(mins):.0f}m"
        event_lines.append(
            f"- {event.get('time') or '-'} | {event.get('impact') or '-'} | "
            f"{event.get('event') or '-'} | in {mins_text}"
        )
    events_text = "\n".join(event_lines) if event_lines else "- none"
    return (
        "ECONOMIC CALENDAR\n"
        f"Provider: {snapshot.provider}\n"
        f"Event risk: {snapshot.event_risk}\n"
        f"Events: {snapshot.event_count}; high impact: {snapshot.high_impact_count}\n"
        f"Next event: {next_event.get('event') or 'none'} "
        f"({next_event.get('impact') or '-'}) at {next_event.get('time') or '-'}\n"
        f"Upcoming:\n{events_text}\n"
        f"Notes: {snapshot.notes or 'none'}"
    )
