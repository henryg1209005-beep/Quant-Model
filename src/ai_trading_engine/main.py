from __future__ import annotations

import argparse
import time

from ai_trading_engine.config import load_settings
from ai_trading_engine.engine import TradingEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AI Trading Engine cycles")
    parser.add_argument("--cycles", type=int, default=1, help="Number of cycles to run")
    parser.add_argument("--sleep", type=int, default=None, help="Override sleep seconds between cycles")
    parser.add_argument("--show-dashboard", action="store_true", help="Print full market context dashboard")
    args = parser.parse_args()

    settings = load_settings()
    engine = TradingEngine(settings=settings)

    sleep_seconds = args.sleep if args.sleep is not None else settings.cycle_seconds
    for idx in range(args.cycles):
        result = engine.run_cycle()
        print(f"[{result.timestamp.isoformat()}] cycle={idx + 1}")
        print(f"decision={result.decision}")
        print(f"note={result.note}")
        if args.show_dashboard:
            print(result.dashboard)
            print("-" * 70)

        if idx + 1 < args.cycles:
            time.sleep(max(0, sleep_seconds))


if __name__ == "__main__":
    main()
