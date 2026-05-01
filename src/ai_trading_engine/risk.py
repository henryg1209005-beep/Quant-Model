from __future__ import annotations

from ai_trading_engine.config import Settings
from ai_trading_engine.models import AccountState, AiDecision


def apply_risk_rules(decision: AiDecision, account: AccountState, settings: Settings) -> AiDecision:
    decision.confidence = max(0.0, min(1.0, decision.confidence))
    decision.size = max(1, min(settings.max_position_size, decision.size))
    decision.sl_ticks = max(settings.min_sl_ticks, decision.sl_ticks)
    decision.tp_ticks = max(settings.min_tp_ticks, decision.tp_ticks)

    if decision.action == "hold":
        decision.direction = None
        decision.size = 1
        return decision

    if decision.direction not in {"LONG", "SHORT"}:
        decision.action = "hold"
        decision.direction = None
        decision.reasoning = f"Risk override: invalid direction. {decision.reasoning}"
        return decision

    if account.open_position is not None:
        decision.action = "hold"
        decision.direction = None
        decision.reasoning = f"Risk override: existing open position. {decision.reasoning}"
        return decision

    if account.todays_trade_count >= settings.max_trades_per_day:
        decision.action = "hold"
        decision.direction = None
        decision.reasoning = f"Risk override: max trades reached. {decision.reasoning}"
        return decision

    if abs(account.drawdown) >= settings.max_daily_drawdown:
        decision.action = "hold"
        decision.direction = None
        decision.reasoning = f"Risk override: daily drawdown exceeded. {decision.reasoning}"
        return decision

    return decision
