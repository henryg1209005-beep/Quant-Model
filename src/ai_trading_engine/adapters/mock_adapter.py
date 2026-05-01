from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

from ai_trading_engine.adapters.base import MarketDataAdapter
from ai_trading_engine.models import Bar, Quote


class MockMarketDataAdapter(MarketDataAdapter):
    """Deterministic-ish synthetic bars for architecture testing."""

    def __init__(self, seed: int = 42, base_price: float = 5300.0) -> None:
        self._rng = random.Random(seed)
        self._base_price = base_price

    def get_historical_bars(
        self,
        symbol: str,
        timeframe_minutes: int,
        count: int,
        as_of: datetime,
    ) -> list[Bar]:
        del symbol
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)

        bars: list[Bar] = []
        total = max(count, 2)
        phase = as_of.timestamp() / 3600.0
        for i in range(total):
            idx = total - i
            ts = as_of - timedelta(minutes=timeframe_minutes * idx)
            trend = math.sin((phase - idx) / 5.0) * 4.0
            noise = self._rng.uniform(-1.5, 1.5)
            mid = self._base_price + trend + noise + (i * 0.15)
            spread = self._rng.uniform(0.75, 3.5)
            o = mid - self._rng.uniform(0.3, 1.2)
            c = mid + self._rng.uniform(-1.0, 1.4)
            h = max(o, c) + spread
            l = min(o, c) - spread
            vol = 100 + abs(math.sin((phase + i) / 3.0)) * 450 + self._rng.uniform(0, 50)
            bars.append(
                Bar(
                    timestamp=ts,
                    open=float(round(o, 2)),
                    high=float(round(h, 2)),
                    low=float(round(l, 2)),
                    close=float(round(c, 2)),
                    volume=float(round(vol, 2)),
                )
            )

        return bars

    def get_latest_quote(self, symbol: str) -> Quote:
        del symbol
        now = datetime.now(tz=timezone.utc)
        last = self._base_price + self._rng.uniform(-5.0, 5.0)
        spread = self._rng.uniform(0.25, 1.0)
        bid = last - spread / 2
        ask = last + spread / 2
        return Quote(timestamp=now, bid=bid, ask=ask, last=last, size=self._rng.uniform(1, 20))
