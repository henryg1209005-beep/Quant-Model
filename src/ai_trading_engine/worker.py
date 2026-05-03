from __future__ import annotations

import time

from ai_trading_engine.app_service import TradingAppService
from ai_trading_engine.config import load_settings
from ai_trading_engine.persistence import Persistence


def main() -> None:
    settings = load_settings()
    persistence = Persistence(settings.app_db_path, database_url=settings.database_url)
    service = TradingAppService(settings, persistence)
    kill = service.kill_switch_snapshot()
    if kill.get("enabled"):
        print(f"worker start blocked: trading kill switch enabled ({kill.get('reason')})")
        return
    gate = service.go_live_gate_snapshot()
    if gate.get("blocked_autonomous_live"):
        print(f"worker start blocked: {gate.get('reason', 'go_live_gate_failed')}")
        return

    while True:
        service.run_cycle_once()
        time.sleep(max(1, settings.cycle_seconds))


if __name__ == "__main__":
    main()
