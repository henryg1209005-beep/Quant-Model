from __future__ import annotations

import argparse
import json
import math
from bisect import bisect_right
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ai_trading_engine.config import load_settings
from ai_trading_engine.context_builder import SYSTEM_PROMPT, build_llm_context
from ai_trading_engine.engine import TradingEngine, _hold_decision
from ai_trading_engine.indicators import compute_indicator_set
from ai_trading_engine.llm.parser import parse_decision
from ai_trading_engine.models import Bar, ClosedTrade, Position, Quote, TimeframeBars
from ai_trading_engine.risk import apply_risk_rules
from ai_trading_engine.scorer import build_market_context, render_context_dashboard
from ai_trading_engine.trade_costs import estimate_round_trip_cost, should_apply_cost_model


def _parse_dt(value: str) -> datetime:
    text = value.strip()
    if "T" not in text:
        text = f"{text}T00:00:00"
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _tf_label(minutes: int) -> str:
    if minutes == 1:
        return "1Min"
    if minutes == 5:
        return "5Min"
    if minutes == 15:
        return "15Min"
    if minutes == 60:
        return "1Hour"
    return f"{minutes}Min"


def _fetch_alpaca_bars(
    *,
    key_id: str,
    secret_key: str,
    data_url: str,
    symbol: str,
    timeframe_minutes: int,
    start: datetime,
    end: datetime,
) -> list[Bar]:
    bars: list[Bar] = []
    page_token: str | None = None
    timeframe = _tf_label(timeframe_minutes)
    base_url = data_url.rstrip("/")

    while True:
        params = {
            "timeframe": timeframe,
            "start": _to_iso(start),
            "end": _to_iso(end),
            "adjustment": "raw",
            "feed": "iex",
            "sort": "asc",
            "limit": "10000",
        }
        if page_token:
            params["page_token"] = page_token
        req = Request(f"{base_url}/v2/stocks/{symbol}/bars?{urlencode(params)}")
        req.add_header("APCA-API-KEY-ID", key_id)
        req.add_header("APCA-API-SECRET-KEY", secret_key)
        with urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        rows = payload.get("bars") or []
        for row in rows:
            bars.append(
                Bar(
                    timestamp=datetime.fromisoformat(str(row["t"]).replace("Z", "+00:00")),
                    open=float(row["o"]),
                    high=float(row["h"]),
                    low=float(row["l"]),
                    close=float(row["c"]),
                    volume=float(row.get("v", 0.0)),
                )
            )

        page_token = payload.get("next_page_token")
        if not page_token:
            break

    return bars


def _window_at_or_before(bars: list[Bar], timestamps: list[datetime], ts: datetime, count: int) -> list[Bar]:
    if not bars:
        return []
    idx = bisect_right(timestamps, ts)
    if idx <= 0:
        return []
    lo = max(0, idx - max(1, count))
    return bars[lo:idx]


def _quote_from_bar(bar: Bar) -> Quote:
    spread = max(0.01, bar.close * 0.0002)
    bid = bar.close - spread / 2.0
    ask = bar.close + spread / 2.0
    return Quote(timestamp=bar.timestamp, bid=bid, ask=ask, last=bar.close, size=max(1.0, bar.volume / 5000.0))


def _close_position_on_bar(engine: TradingEngine, bar: Bar) -> ClosedTrade | None:
    pos = engine.account.open_position
    if pos is None:
        return None

    tick = max(1e-9, float(engine.settings.price_tick_size))
    sl_move = pos.sl_ticks * tick
    tp_move = pos.tp_ticks * tick

    if pos.direction == "LONG":
        sl_price = pos.entry_price - sl_move
        tp_price = pos.entry_price + tp_move
        hit_sl = bar.low <= sl_price
        hit_tp = bar.high >= tp_price
        if not hit_sl and not hit_tp:
            return None
        exit_price = sl_price if (hit_sl and hit_tp) or hit_sl else tp_price
        pnl = (exit_price - pos.entry_price) * pos.size
    else:
        sl_price = pos.entry_price + sl_move
        tp_price = pos.entry_price - tp_move
        hit_sl = bar.high >= sl_price
        hit_tp = bar.low <= tp_price
        if not hit_sl and not hit_tp:
            return None
        exit_price = sl_price if (hit_sl and hit_tp) or hit_sl else tp_price
        pnl = (pos.entry_price - exit_price) * pos.size

    if should_apply_cost_model(
        engine.settings,
        execution_provider=engine.settings.execution_provider,
        is_live=False,
    ):
        pnl -= estimate_round_trip_cost(
            entry_price=pos.entry_price,
            exit_price=float(exit_price),
            size=pos.size,
            slippage_bps_per_side=engine.settings.cost_slippage_bps_per_side,
            fee_per_share=engine.settings.cost_fee_per_share,
            min_fee_per_order=engine.settings.cost_min_fee_per_order,
        )

    closed = ClosedTrade(
        symbol=pos.symbol,
        direction=pos.direction,
        size=pos.size,
        entry_price=pos.entry_price,
        exit_price=float(exit_price),
        pnl=float(pnl),
        opened_at=pos.opened_at,
        closed_at=bar.timestamp,
        thesis=pos.thesis,
    )
    regime, conf = engine._extract_regime_and_confidence(closed.thesis)
    closed.regime = regime
    closed.confidence = conf

    engine.account.balance += closed.pnl
    engine.account.daily_realized_pnl += closed.pnl
    engine.account.closed_trades_today.append(closed)
    engine.account.open_position = None
    engine.learner.update_from_trade(closed)
    engine.edge_monitor.update_trade(closed)
    return closed


def _open_trade_from_decision(engine: TradingEngine, decision, quote: Quote) -> None:
    if decision.action != "trade" or decision.direction is None:
        return
    spread_half = max(0.0, quote.ask - quote.bid) / 2.0
    entry = quote.last + spread_half if decision.direction == "LONG" else quote.last - spread_half
    engine.account.open_position = Position(
        symbol=engine.settings.symbol,
        direction=decision.direction,
        size=decision.size,
        entry_price=float(entry),
        sl_ticks=decision.sl_ticks,
        tp_ticks=decision.tp_ticks,
        opened_at=quote.timestamp,
        thesis=decision.reasoning,
    )
    engine.account.todays_trade_count += 1


def run_historical_backtest(
    *,
    symbol: str,
    start: datetime,
    end: datetime,
    initial_balance: float,
) -> dict:
    settings = load_settings()
    if not settings.alpaca_key_id or not settings.alpaca_secret_key:
        raise RuntimeError("Missing ALPACA_KEY_ID/ALPACA_SECRET_KEY in environment.")

    safe_settings = replace(
        settings,
        symbol=symbol,
        data_provider="mock",
        execution_provider="mock",
        llm_provider="mock",
        adaptive_state_path="data/backtest_adaptive_state.json",
        edge_monitor_state_path="data/backtest_edge_monitor_state.json",
    )
    engine = TradingEngine(safe_settings, initial_balance=initial_balance)

    primary = _fetch_alpaca_bars(
        key_id=settings.alpaca_key_id,
        secret_key=settings.alpaca_secret_key,
        data_url=settings.alpaca_data_url,
        symbol=symbol,
        timeframe_minutes=safe_settings.primary_timeframe_min,
        start=start,
        end=end,
    )
    short = _fetch_alpaca_bars(
        key_id=settings.alpaca_key_id,
        secret_key=settings.alpaca_secret_key,
        data_url=settings.alpaca_data_url,
        symbol=symbol,
        timeframe_minutes=safe_settings.short_timeframe_min,
        start=start,
        end=end,
    )
    long = _fetch_alpaca_bars(
        key_id=settings.alpaca_key_id,
        secret_key=settings.alpaca_secret_key,
        data_url=settings.alpaca_data_url,
        symbol=symbol,
        timeframe_minutes=safe_settings.long_timeframe_min,
        start=start,
        end=end,
    )

    if not primary:
        raise RuntimeError("No primary bars returned for selected range.")

    cycle_notes: list[str] = []
    trade_actions = 0
    hold_actions = 0
    primary_ts = [b.timestamp for b in primary]
    short_ts = [b.timestamp for b in short]
    long_ts = [b.timestamp for b in long]

    for bar in primary:
        closed = _close_position_on_bar(engine, bar)
        closed_note = f"Closed position PnL={closed.pnl:.2f}" if closed else "No bracket exit this cycle"

        p_win = _window_at_or_before(primary, primary_ts, bar.timestamp, safe_settings.primary_bars)
        s_win = _window_at_or_before(short, short_ts, bar.timestamp, safe_settings.short_bars)
        l_win = _window_at_or_before(long, long_ts, bar.timestamp, safe_settings.long_bars)
        if not p_win or not s_win or not l_win:
            cycle_notes.append(f"{bar.timestamp.isoformat()} | {closed_note}; hold (insufficient bars)")
            hold_actions += 1
            continue

        indicators = compute_indicator_set(p_win)
        context = build_market_context(indicators, p_win)
        dashboard = render_context_dashboard(context)
        llm_context = build_llm_context(
            dashboard_text=dashboard,
            tf_bars=TimeframeBars(primary=p_win, short=s_win, long=l_win),
            account=engine.account,
        )
        quote = _quote_from_bar(bar)

        raw = engine.llm.decide(SYSTEM_PROMPT, llm_context)
        decision = parse_decision(raw)
        candidate_trade = decision.action == "trade" and decision.direction is not None
        challenger_trade = False

        adaptive = engine.learner.adapt(decision, indicators.regime)
        decision = adaptive.decision
        if adaptive.note not in {"adaptive_bypass", ""}:
            decision.reasoning = f"{decision.reasoning} [{adaptive.note}]".strip()

        if decision.action == "trade" and decision.direction is not None:
            in_session, session_meta = engine._passes_session_filter(bar.timestamp)
            if not in_session:
                decision = _hold_decision(f"Session gate: outside trading window ({session_meta}).")
            else:
                spread_bps = engine._spread_bps(quote)
                if spread_bps > safe_settings.max_spread_bps:
                    decision = _hold_decision(
                        f"Spread gate: {spread_bps:.2f} bps exceeds max {safe_settings.max_spread_bps:.2f}."
                    )
                else:
                    score, detail = engine._confluence_score(decision, context)
                    if score < safe_settings.entry_confluence_min:
                        decision = _hold_decision(
                            f"Confluence gate: score {score:.1f} < min {safe_settings.entry_confluence_min:.1f}. {detail}"
                        )
                    else:
                        ev_ticks = engine._ev_ticks(decision)
                        if ev_ticks < safe_settings.ev_min_ticks:
                            decision = _hold_decision(
                                f"EV gate: ev_ticks {ev_ticks:.2f} < min {safe_settings.ev_min_ticks:.2f}."
                            )
                        else:
                            sized = engine._atr_target_size(decision, indicators.atr14)
                            decision.size = max(1, min(safe_settings.max_position_size, sized))
                            decision.reasoning = (
                                f"{decision.reasoning} "
                                f"[confluence={score:.1f}/{safe_settings.entry_confluence_min:.1f}] "
                                f"[ev_ticks={ev_ticks:.2f}/{safe_settings.ev_min_ticks:.2f}] "
                                f"[atr={indicators.atr14:.4f} size={decision.size}]"
                            )
                            if candidate_trade and safe_settings.shadow_challenger_enabled:
                                challenger_trade = (
                                    score >= (safe_settings.entry_confluence_min + safe_settings.shadow_confluence_bump)
                                    and ev_ticks >= (safe_settings.ev_min_ticks + safe_settings.shadow_ev_bump)
                                )

        edge_action = engine.edge_monitor.throttle(decision)
        decision = edge_action.decision
        if edge_action.note not in {"edge_bypass", "edge_warmup", "edge_ok", ""}:
            decision.reasoning = f"{decision.reasoning} [{edge_action.note}]".strip()

        decision = apply_risk_rules(decision, engine.account, safe_settings)
        engine.edge_monitor.update_shadow(
            champion_trade=(decision.action == "trade"),
            challenger_trade=challenger_trade,
        )

        if decision.action == "trade":
            decision.reasoning = f"[REGIME:{indicators.regime}][CONF:{decision.confidence:.2f}] {decision.reasoning}"
            _open_trade_from_decision(engine, decision, quote)
            cycle_notes.append(f"{bar.timestamp.isoformat()} | {closed_note}; trade {decision.direction} x{decision.size}")
            trade_actions += 1
        else:
            cycle_notes.append(f"{bar.timestamp.isoformat()} | {closed_note}; hold")
            hold_actions += 1

    pnls = [float(t.pnl) for t in engine.account.closed_trades_today]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    trade_count = len(pnls)
    win_rate = (len(wins) / trade_count) if trade_count else 0.0
    avg_win = (sum(wins) / len(wins)) if wins else 0.0
    avg_loss = (sum(losses) / len(losses)) if losses else 0.0
    net_pnl = float(sum(pnls))
    expectancy = (win_rate * avg_win) + ((1.0 - win_rate) * avg_loss)
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else math.inf

    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity - peak)
    mean = (sum(pnls) / trade_count) if trade_count else 0.0
    var = (sum((p - mean) ** 2 for p in pnls) / max(1, trade_count - 1)) if trade_count > 1 else 0.0
    std = math.sqrt(var)
    sharpe_like = (mean / std * math.sqrt(trade_count)) if std > 0 else 0.0

    return {
        "run_at": datetime.now(tz=timezone.utc).isoformat(),
        "symbol": symbol,
        "start": _to_iso(start),
        "end": _to_iso(end),
        "bar_counts": {
            "primary": len(primary),
            "short": len(short),
            "long": len(long),
        },
        "starting_balance": initial_balance,
        "ending_balance": float(engine.account.balance),
        "trade_count": trade_count,
        "net_pnl": net_pnl,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "max_drawdown": float(max_drawdown),
        "sharpe_like": sharpe_like,
        "decision_count": trade_actions + hold_actions,
        "trade_action_count": trade_actions,
        "hold_action_count": hold_actions,
        "hold_rate": (hold_actions / max(1, trade_actions + hold_actions)),
        "open_position": asdict(engine.account.open_position) if engine.account.open_position else None,
        "adaptive": engine.learner.snapshot(),
        "anti_decay": engine.edge_monitor.snapshot(),
        "last_notes": cycle_notes[-20:],
        "closed_trades_tail": [asdict(t) for t in engine.account.closed_trades_today[-30:]],
    }


def _save_report(report: dict) -> Path:
    out_dir = Path("data/backtests")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"historical_backtest_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run historical Alpaca replay backtest.")
    parser.add_argument("--symbol", type=str, default=None, help="Ticker, e.g. SPY")
    parser.add_argument("--start", type=str, required=True, help="UTC date/time, e.g. 2025-01-01 or 2025-01-01T00:00:00Z")
    parser.add_argument("--end", type=str, required=True, help="UTC date/time, e.g. 2025-12-31 or 2025-12-31T23:59:59Z")
    parser.add_argument("--starting-balance", type=float, default=100000.0, help="Backtest starting balance.")
    args = parser.parse_args()

    settings = load_settings()
    symbol = (args.symbol or settings.symbol).upper()
    report = run_historical_backtest(
        symbol=symbol,
        start=_parse_dt(args.start),
        end=_parse_dt(args.end),
        initial_balance=float(args.starting_balance),
    )
    out = _save_report(report)
    print(f"Historical backtest complete: {out}")
    print(
        json.dumps(
            {
                "symbol": report["symbol"],
                "start": report["start"],
                "end": report["end"],
                "bar_counts": report["bar_counts"],
                "starting_balance": report["starting_balance"],
                "ending_balance": report["ending_balance"],
                "trade_count": report["trade_count"],
                "net_pnl": report["net_pnl"],
                "win_rate": report["win_rate"],
                "profit_factor": report["profit_factor"],
                "max_drawdown": report["max_drawdown"],
                "sharpe_like": report["sharpe_like"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
