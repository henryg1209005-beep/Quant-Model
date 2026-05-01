from __future__ import annotations

from ai_trading_engine.feedback import summarize_trade_feedback
from ai_trading_engine.indicators import format_dt
from ai_trading_engine.models import AccountState, TimeframeBars
from ai_trading_engine.scorer import render_context_dashboard


SYSTEM_PROMPT = """You are a disciplined systematic trading decision model for liquid US equities and ETFs.
Rules:
- Respect risk limits and account state.
- Use the market context, multi-timeframe bars, trade feedback, macro context, economic calendar, and news/earnings context.
- Treat macro context as a regime prior, not as a reason to force trades.
- Treat high-impact economic calendar windows as execution risk; prefer holding unless the edge is exceptional.
- Treat fresh company news and earnings windows as symbol-specific execution risk.
- Use confidence continuously; do not use binary heuristics.
- If edge is weak, conflicting, or execution conditions are poor, return hold.
- Prefer fewer high-quality trades over constant activity.
- Do not invent data that is not present in the prompt.
- Only include direction when action is trade; omit direction when action is hold.
- Always include a prediction-only forecast, even when action is hold:
  forecast_direction must be LONG or SHORT, forecast_confidence must be 0.0-1.0.
  This forecast is for research labelling and does not imply permission to trade.
- Output JSON only with schema:
  {
    \"action\": \"trade\" | \"hold\",
    \"direction\": \"LONG\" | \"SHORT\",
    \"confidence\": 0.0-1.0,
    \"size\": 1-10,
    \"sl_ticks\": int,
    \"tp_ticks\": int,
    \"forecast_direction\": \"LONG\" | \"SHORT\",
    \"forecast_confidence\": 0.0-1.0,
    \"forecast_horizon_minutes\": 5 | 15 | 30,
    \"reasoning\": string
  }
"""


def _format_bars(name: str, bars) -> str:
    lines = [f"{name} bars ({len(bars)}):"]
    for b in bars:
        lines.append(
            f"- {format_dt(b.timestamp)} O:{b.open:.2f} H:{b.high:.2f} L:{b.low:.2f} C:{b.close:.2f} V:{b.volume:.0f}"
        )
    return "\n".join(lines)


def build_llm_context(
    dashboard_text: str,
    tf_bars: TimeframeBars,
    account: AccountState,
) -> str:
    account_text = (
        "ACCOUNT STATE\n"
        f"Balance: {account.balance:.2f}\n"
        f"Starting balance: {account.starting_balance:.2f}\n"
        f"Daily PnL: {account.daily_realized_pnl:.2f}\n"
        f"Trade count today: {account.todays_trade_count}\n"
        f"Risk budget used: {account.risk_budget_remaining:.2f}\n"
        f"Open position: {account.open_position}\n"
    )

    feedback_text = "INTRA-SESSION TRADE FEEDBACK\n" + summarize_trade_feedback(account.closed_trades_today)

    bars_text = "\n\n".join(
        [
            _format_bars("Primary", tf_bars.primary),
            _format_bars("Short", tf_bars.short),
            _format_bars("Long", tf_bars.long),
        ]
    )

    return "\n\n".join([dashboard_text, bars_text, account_text, feedback_text])
