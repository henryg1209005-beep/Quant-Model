from __future__ import annotations

import uvicorn

from ai_trading_engine.config import load_settings


def main() -> None:
    settings = load_settings()
    uvicorn.run("ai_trading_engine.api:app", host=settings.api_host, port=settings.api_port, reload=False)


if __name__ == "__main__":
    main()
