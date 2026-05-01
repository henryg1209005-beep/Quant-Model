from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any

from ai_trading_engine.models import AccountState, AiDecision, ClosedTrade, Position


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.isoformat()


def decision_to_record(decision: AiDecision) -> dict[str, Any]:
    out = asdict(decision)
    return out


def position_to_record(position: Position | None) -> dict[str, Any] | None:
    if position is None:
        return None
    return {
        "symbol": position.symbol,
        "direction": position.direction,
        "size": position.size,
        "entry_price": position.entry_price,
        "sl_ticks": position.sl_ticks,
        "tp_ticks": position.tp_ticks,
        "opened_at": _iso(position.opened_at),
        "thesis": position.thesis,
    }


def trade_to_record(trade: ClosedTrade) -> dict[str, Any]:
    return {
        "symbol": trade.symbol,
        "direction": trade.direction,
        "size": trade.size,
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
        "pnl": trade.pnl,
        "opened_at": _iso(trade.opened_at),
        "closed_at": _iso(trade.closed_at),
        "thesis": trade.thesis,
        "regime": trade.regime,
        "confidence": trade.confidence,
        "metadata": dict(trade.metadata or {}),
    }


def account_to_record(account: AccountState) -> dict[str, Any]:
    return {
        "balance": account.balance,
        "starting_balance": account.starting_balance,
        "trading_day_et": account.trading_day_et,
        "todays_trade_count": account.todays_trade_count,
        "daily_realized_pnl": account.daily_realized_pnl,
        "drawdown": account.drawdown,
        "risk_budget_remaining": account.risk_budget_remaining,
        "open_position": position_to_record(account.open_position),
        "closed_trades_today": [trade_to_record(t) for t in account.closed_trades_today],
    }
