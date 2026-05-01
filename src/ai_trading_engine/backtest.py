from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

from ai_trading_engine.config import load_settings
from ai_trading_engine.engine import TradingEngine


def _build_backtest_engine(starting_balance: float) -> TradingEngine:
    settings = load_settings()
    safe_settings = replace(
        settings,
        data_provider="mock",
        execution_provider="mock",
        llm_provider="mock",
        enable_session_filter=False,
        max_trades_per_day=10_000_000,
        max_daily_drawdown=1e12,
        adaptive_state_path="data/backtest_adaptive_state.json",
        edge_monitor_state_path="data/backtest_edge_monitor_state.json",
    )
    return TradingEngine(safe_settings, initial_balance=starting_balance)


def run_backtest(cycles: int, starting_balance: float) -> dict:
    engine = _build_backtest_engine(starting_balance=starting_balance)

    notes: list[str] = []
    for _ in range(max(1, cycles)):
        result = engine.run_cycle()
        notes.append(result.note)

    trades = engine.account.closed_trades_today
    pnls = [float(t.pnl) for t in trades]
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
        "run_at": datetime.now().isoformat(),
        "cycles": cycles,
        "starting_balance": starting_balance,
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
        "open_position": asdict(engine.account.open_position) if engine.account.open_position else None,
        "last_notes": notes[-10:],
        "adaptive": engine.learner.snapshot(),
        "anti_decay": engine.edge_monitor.snapshot(),
        "closed_trades_tail": [asdict(t) for t in trades[-20:]],
    }


def _save_report(report: dict) -> Path:
    out_dir = Path("data/backtests")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"backtest_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local synthetic backtest for the trading engine.")
    parser.add_argument("--cycles", type=int, default=2000, help="Number of engine cycles to simulate.")
    parser.add_argument("--starting-balance", type=float, default=100000.0, help="Backtest starting balance.")
    args = parser.parse_args()

    report = run_backtest(cycles=args.cycles, starting_balance=args.starting_balance)
    out = _save_report(report)

    print(f"Backtest complete: {out}")
    print(
        json.dumps(
            {
                "cycles": report["cycles"],
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
