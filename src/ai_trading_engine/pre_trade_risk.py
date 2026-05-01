from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from ai_trading_engine.models import AccountState, AiDecision, Quote


@dataclass(frozen=True)
class RiskCheckResult:
    allowed: bool
    reason: str
    checks: dict[str, float | int | str | bool]


def _notional(size: int, price: float) -> float:
    return max(0.0, float(size)) * max(0.0, float(price))


def _quote_age_seconds(ts: datetime) -> float:
    now = datetime.now(tz=timezone.utc)
    base = ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
    return max(0.0, (now - base.astimezone(timezone.utc)).total_seconds())


def _spread_bps(quote: Quote) -> float:
    if quote.last <= 0:
        return 0.0
    return max(0.0, (quote.ask - quote.bid) / quote.last) * 10000.0


def evaluate_pre_trade_risk(
    *,
    account: AccountState,
    decision: AiDecision,
    quote: Quote,
    max_order_notional_usd: float,
    max_total_exposure_usd: float,
    max_quote_age_seconds: int,
    max_pretrade_spread_bps: float,
    min_buying_power_usd: float,
    duplicate_cooldown_seconds: int,
    last_order_key: str,
    last_order_submitted_at: datetime | None,
    current_order_key: str,
) -> RiskCheckResult:
    order_notional = _notional(decision.size, quote.last)
    open_notional = (
        _notional(account.open_position.size, account.open_position.entry_price) if account.open_position else 0.0
    )
    total_exposure_after = open_notional + order_notional
    quote_age = _quote_age_seconds(quote.timestamp)
    spread_bps = _spread_bps(quote)
    cooldown_active = False
    if last_order_submitted_at is not None and last_order_key and current_order_key == last_order_key:
        elapsed = (datetime.now(tz=timezone.utc) - last_order_submitted_at).total_seconds()
        cooldown_active = elapsed < max(0, duplicate_cooldown_seconds)

    checks: dict[str, float | int | str | bool] = {
        "order_notional_usd": round(order_notional, 4),
        "open_notional_usd": round(open_notional, 4),
        "total_exposure_after_usd": round(total_exposure_after, 4),
        "quote_age_seconds": round(quote_age, 3),
        "spread_bps": round(spread_bps, 4),
        "duplicate_cooldown_active": cooldown_active,
    }

    if cooldown_active:
        return RiskCheckResult(False, "duplicate_order_cooldown", checks)
    if order_notional > max(0.0, max_order_notional_usd):
        return RiskCheckResult(False, "order_notional_limit_exceeded", checks)
    if total_exposure_after > max(0.0, max_total_exposure_usd):
        return RiskCheckResult(False, "total_exposure_limit_exceeded", checks)
    if quote_age > max(0, max_quote_age_seconds):
        return RiskCheckResult(False, "stale_quote", checks)
    if spread_bps > max(0.0, max_pretrade_spread_bps):
        return RiskCheckResult(False, "pretrade_spread_limit_exceeded", checks)
    if account.balance < max(0.0, min_buying_power_usd):
        return RiskCheckResult(False, "min_buying_power_failed", checks)

    return RiskCheckResult(True, "ok", checks)
