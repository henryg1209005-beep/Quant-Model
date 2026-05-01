from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from ai_trading_engine.models import Bar, Quote


class MarketDataAdapter(ABC):
    @abstractmethod
    def get_historical_bars(
        self,
        symbol: str,
        timeframe_minutes: int,
        count: int,
        as_of: datetime,
    ) -> list[Bar]:
        raise NotImplementedError

    @abstractmethod
    def get_latest_quote(self, symbol: str) -> Quote:
        raise NotImplementedError
