from __future__ import annotations

from ai_trading_engine.config import Settings


def estimate_round_trip_cost(
    *,
    entry_price: float,
    exit_price: float,
    size: int,
    slippage_bps_per_side: float,
    fee_per_share: float,
    min_fee_per_order: float,
) -> float:
    qty = max(0, int(size))
    if qty <= 0:
        return 0.0
    entry = max(0.0, float(entry_price))
    exit_ = max(0.0, float(exit_price))
    bps = max(0.0, float(slippage_bps_per_side))
    fee_ps = max(0.0, float(fee_per_share))
    min_fee = max(0.0, float(min_fee_per_order))

    entry_slip = entry * (bps / 10_000.0) * qty
    exit_slip = exit_ * (bps / 10_000.0) * qty
    entry_fee = max(min_fee, fee_ps * qty)
    exit_fee = max(min_fee, fee_ps * qty)
    return float(entry_slip + exit_slip + entry_fee + exit_fee)


def should_apply_cost_model(settings: Settings, *, execution_provider: str, is_live: bool) -> bool:
    if not bool(settings.cost_model_enabled):
        return False
    provider = str(execution_provider or "").strip().lower()
    if provider == "mock":
        return bool(settings.cost_apply_in_mock)
    if provider == "alpaca":
        if is_live:
            return False
        return bool(settings.cost_apply_in_paper)
    return False
