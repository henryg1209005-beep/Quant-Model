from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ai_trading_engine.models import ClosedTrade


@dataclass(frozen=True)
class SymbolPortfolioStats:
    symbol: str
    trade_count: int
    net_pnl: float
    expectancy: float
    win_rate: float
    volatility: float
    max_drawdown: float
    raw_score: float
    shrunk_score: float


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / float(len(values))
    var = sum((v - mean) ** 2 for v in values) / float(len(values) - 1)
    return math.sqrt(max(0.0, var))


def _max_drawdown(pnls: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in pnls:
        equity += float(pnl)
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
    return float(max_dd)


def _symbol_from_trade(trade: ClosedTrade) -> str:
    return str(trade.symbol or "").strip().upper()


def _symbol_stats(symbol: str, trades: list[ClosedTrade], min_trades: int, shrinkage_trades: int) -> SymbolPortfolioStats:
    pnls = [float(t.pnl) for t in trades]
    wins = [p for p in pnls if p > 0]
    count = len(pnls)
    net = float(sum(pnls))
    expectancy = net / float(count) if count else 0.0
    win_rate = len(wins) / float(count) if count else 0.0
    volatility = _std(pnls)
    max_dd = _max_drawdown(pnls)

    # Penalize noisy symbols and tiny samples; negative expectancy receives no allocation.
    risk = max(volatility, abs(max_dd) / max(1.0, math.sqrt(max(1, count))), 1e-9)
    raw_score = max(0.0, expectancy) / risk
    sample_weight = count / float(count + max(1, int(shrinkage_trades)))
    if count < max(1, int(min_trades)):
        sample_weight *= count / float(max(1, int(min_trades)))
    shrunk_score = raw_score * sample_weight

    return SymbolPortfolioStats(
        symbol=symbol,
        trade_count=count,
        net_pnl=net,
        expectancy=float(expectancy),
        win_rate=float(win_rate),
        volatility=float(volatility),
        max_drawdown=float(max_dd),
        raw_score=float(raw_score),
        shrunk_score=float(shrunk_score),
    )


def _cap_and_redistribute(weights: dict[str, float], max_weight: float, target_weight: float) -> dict[str, float]:
    cap = max(0.01, min(1.0, float(max_weight)))
    target = max(0.0, min(1.0, float(target_weight)))
    capped = {k: min(float(v), cap) for k, v in weights.items()}
    for _ in range(20):
        total = sum(capped.values())
        leftover = max(0.0, target - total)
        if leftover <= 1e-9:
            break
        eligible = [k for k, v in capped.items() if v < cap - 1e-9]
        if not eligible:
            break
        base = sum(weights[k] for k in eligible)
        if base <= 0:
            break
        changed = False
        for k in eligible:
            add = leftover * (weights[k] / base)
            new_value = min(cap, capped[k] + add)
            changed = changed or abs(new_value - capped[k]) > 1e-12
            capped[k] = new_value
        if not changed:
            break
    return capped


def optimise_portfolio(
    trades: list[ClosedTrade],
    *,
    min_trades: int = 5,
    max_weight: float = 0.35,
    cash_floor: float = 0.25,
    shrinkage_trades: int = 25,
) -> dict[str, Any]:
    ordered = sorted(
        trades,
        key=lambda t: t.closed_at if isinstance(t.closed_at, datetime) else datetime.min,
    )
    by_symbol: dict[str, list[ClosedTrade]] = {}
    for trade in ordered:
        symbol = _symbol_from_trade(trade)
        if not symbol:
            continue
        by_symbol.setdefault(symbol, []).append(trade)

    stats = [
        _symbol_stats(symbol, symbol_trades, min_trades=min_trades, shrinkage_trades=shrinkage_trades)
        for symbol, symbol_trades in sorted(by_symbol.items())
    ]
    score_sum = sum(s.shrunk_score for s in stats if s.shrunk_score > 0)
    cash = max(0.0, min(1.0, float(cash_floor)))

    if score_sum <= 0:
        return {
            "ok": False,
            "reason": "no_positive_risk_adjusted_symbols",
            "trade_count": len(ordered),
            "cash_weight": 1.0,
            "weights": {},
            "diagnostics": [s.__dict__ for s in stats],
            "params": {
                "min_trades": min_trades,
                "max_weight": max_weight,
                "cash_floor": cash_floor,
                "shrinkage_trades": shrinkage_trades,
            },
        }

    investable = 1.0 - cash
    raw_weights = {s.symbol: investable * (s.shrunk_score / score_sum) for s in stats if s.shrunk_score > 0}
    capped = _cap_and_redistribute(raw_weights, max_weight=max_weight, target_weight=investable)
    allocated = sum(capped.values())
    cash_weight = max(cash, 1.0 - allocated)
    total = allocated + cash_weight
    weights = {k: float(v / total) for k, v in sorted(capped.items())}
    cash_weight = float(cash_weight / total)

    return {
        "ok": True,
        "reason": "ok",
        "trade_count": len(ordered),
        "cash_weight": cash_weight,
        "weights": weights,
        "diagnostics": [s.__dict__ for s in stats],
        "params": {
            "min_trades": min_trades,
            "max_weight": max_weight,
            "cash_floor": cash_floor,
            "shrinkage_trades": shrinkage_trades,
        },
    }
