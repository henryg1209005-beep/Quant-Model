from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

from ai_trading_engine.config import Settings


@dataclass(frozen=True)
class MacroSnapshot:
    timestamp: str
    provider: str
    risk_regime: str
    rate_regime: str
    inflation_regime: str
    growth_regime: str
    policy_bias: str
    notes: str = ""
    values: dict[str, float | str] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _normalise(value: Any, default: str = "unknown") -> str:
    out = str(value or "").strip().lower().replace(" ", "_")
    return out or default


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _manual_snapshot(settings: Settings, *, provider: str = "manual") -> MacroSnapshot:
    path = Path(settings.macro_snapshot_path)
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return MacroSnapshot(
                timestamp=str(payload.get("timestamp") or _now_iso()),
                provider=str(payload.get("provider") or provider),
                risk_regime=_normalise(payload.get("risk_regime")),
                rate_regime=_normalise(payload.get("rate_regime")),
                inflation_regime=_normalise(payload.get("inflation_regime")),
                growth_regime=_normalise(payload.get("growth_regime")),
                policy_bias=_normalise(payload.get("policy_bias")),
                notes=str(payload.get("notes") or ""),
                values=dict(payload.get("values") or {}),
            )
        except Exception:
            pass

    return MacroSnapshot(
        timestamp=_now_iso(),
        provider=provider,
        risk_regime=_normalise(settings.macro_risk_regime),
        rate_regime=_normalise(settings.macro_rate_regime),
        inflation_regime=_normalise(settings.macro_inflation_regime),
        growth_regime=_normalise(settings.macro_growth_regime),
        policy_bias=_normalise(settings.macro_policy_bias),
        notes=str(settings.macro_notes or ""),
        values={},
    )


def _fred_latest(series_id: str, api_key: str, timeout: int) -> float | None:
    query = urlencode(
        {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": 1,
        }
    )
    with urlopen(f"https://api.stlouisfed.org/fred/series/observations?{query}", timeout=max(1, int(timeout))) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    obs = payload.get("observations") or []
    if not obs:
        return None
    return _float_or_none(obs[0].get("value"))


def _regimes_from_fred(values: dict[str, float]) -> tuple[str, str, str, str, str]:
    ten_year = values.get("DGS10")
    two_year = values.get("DGS2")
    curve = values.get("T10Y2Y")
    fed_funds = values.get("FEDFUNDS")
    unemployment = values.get("UNRATE")

    rate_regime = "unknown"
    if ten_year is not None and two_year is not None:
        if ten_year >= 4.5 and two_year >= 4.5:
            rate_regime = "restrictive"
        elif ten_year <= 3.0 and two_year <= 3.0:
            rate_regime = "easy"
        else:
            rate_regime = "neutral"

    policy_bias = "unknown"
    if fed_funds is not None and two_year is not None:
        if two_year > fed_funds + 0.25:
            policy_bias = "market_pricing_hikes"
        elif two_year < fed_funds - 0.25:
            policy_bias = "market_pricing_cuts"
        else:
            policy_bias = "steady"

    growth_regime = "unknown"
    if unemployment is not None:
        if unemployment >= 5.0:
            growth_regime = "labor_weakening"
        elif unemployment <= 4.0:
            growth_regime = "labor_tight"
        else:
            growth_regime = "mixed"

    risk_regime = "unknown"
    if curve is not None:
        if curve < -0.25:
            risk_regime = "late_cycle_inversion"
        elif curve > 0.75:
            risk_regime = "risk_on_steepening"
        else:
            risk_regime = "mixed"

    inflation_regime = "unknown"
    return risk_regime, rate_regime, inflation_regime, growth_regime, policy_bias


def load_macro_snapshot(settings: Settings) -> MacroSnapshot:
    if not settings.macro_context_enabled:
        return MacroSnapshot(
            timestamp=_now_iso(),
            provider="disabled",
            risk_regime="disabled",
            rate_regime="disabled",
            inflation_regime="disabled",
            growth_regime="disabled",
            policy_bias="disabled",
        )

    if settings.macro_provider != "fred" or not settings.fred_api_key:
        return _manual_snapshot(settings)

    try:
        values: dict[str, float] = {}
        for series_id in ("DGS10", "DGS2", "T10Y2Y", "FEDFUNDS", "UNRATE"):
            latest = _fred_latest(series_id, settings.fred_api_key, settings.macro_request_timeout_seconds)
            if latest is not None:
                values[series_id] = latest
        risk, rates, inflation, growth, policy = _regimes_from_fred(values)
        return MacroSnapshot(
            timestamp=_now_iso(),
            provider="fred",
            risk_regime=risk,
            rate_regime=rates,
            inflation_regime=inflation,
            growth_regime=growth,
            policy_bias=policy,
            notes="Derived from latest FRED DGS10/DGS2/T10Y2Y/FEDFUNDS/UNRATE observations.",
            values=values,
        )
    except Exception as exc:
        fallback = _manual_snapshot(settings, provider="manual_fallback")
        return MacroSnapshot(
            timestamp=fallback.timestamp,
            provider=fallback.provider,
            risk_regime=fallback.risk_regime,
            rate_regime=fallback.rate_regime,
            inflation_regime=fallback.inflation_regime,
            growth_regime=fallback.growth_regime,
            policy_bias=fallback.policy_bias,
            notes=f"{fallback.notes} macro_provider_error={exc}".strip(),
            values=fallback.values,
        )


def render_macro_context(snapshot: MacroSnapshot) -> str:
    values = ", ".join(f"{k}={v}" for k, v in sorted(snapshot.values.items())) or "none"
    return (
        "MACRO CONTEXT\n"
        f"Provider: {snapshot.provider}\n"
        f"Risk regime: {snapshot.risk_regime}\n"
        f"Rate regime: {snapshot.rate_regime}\n"
        f"Inflation regime: {snapshot.inflation_regime}\n"
        f"Growth regime: {snapshot.growth_regime}\n"
        f"Policy bias: {snapshot.policy_bias}\n"
        f"Values: {values}\n"
        f"Notes: {snapshot.notes or 'none'}"
    )

