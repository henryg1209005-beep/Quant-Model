from __future__ import annotations

from collections import Counter

from ai_trading_engine.models import ClosedTrade


def summarize_trade_feedback(closed_trades_today: list[ClosedTrade]) -> str:
    if not closed_trades_today:
        return "No closed trades yet today."

    pnl_total = sum(t.pnl for t in closed_trades_today)
    wins = [t for t in closed_trades_today if t.pnl > 0]
    losses = [t for t in closed_trades_today if t.pnl <= 0]

    streak = 0
    for trade in reversed(closed_trades_today):
        if trade.pnl <= 0:
            streak += 1
        else:
            break

    direction_counts = Counter(t.direction for t in closed_trades_today)
    losing_reasons = [t.thesis for t in losses if t.thesis]

    lines = [
        f"Closed trades: {len(closed_trades_today)}",
        f"Win rate: {len(wins)}/{len(closed_trades_today)}",
        f"Realized PnL: {pnl_total:.2f}",
        f"Current losing streak: {streak}",
        f"Directional bias: LONG={direction_counts.get('LONG', 0)}, SHORT={direction_counts.get('SHORT', 0)}",
    ]

    if losing_reasons:
        sample = "; ".join(losing_reasons[-3:])
        lines.append(f"Recent failed theses: {sample}")

    if streak >= 3:
        lines.append("Warning: pause aggressive sizing; recent pattern shows repeated thesis failure.")

    return "\n".join(lines)
