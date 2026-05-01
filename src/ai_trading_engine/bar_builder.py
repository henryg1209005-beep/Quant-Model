from __future__ import annotations

from datetime import datetime, timezone
from threading import Lock

from ai_trading_engine.adapters.base import MarketDataAdapter
from ai_trading_engine.models import Bar, TimeframeBars


class BarBuilder:
    """Rebuilds completed bars each cycle from adapter REST calls."""

    def __init__(self, adapter: MarketDataAdapter) -> None:
        self._adapter = adapter
        self._lock = Lock()

    @staticmethod
    def _drop_last_in_progress(bars: list[Bar]) -> list[Bar]:
        if len(bars) <= 1:
            return bars
        return bars[:-1]

    @staticmethod
    def _dedup_completed_bars(bars: list[Bar]) -> list[Bar]:
        seen: set[datetime] = set()
        out: list[Bar] = []
        for bar in sorted(bars, key=lambda b: b.timestamp):
            if bar.timestamp in seen:
                continue
            seen.add(bar.timestamp)
            out.append(bar)
        return out

    def _fetch_completed(
        self,
        symbol: str,
        timeframe_minutes: int,
        count: int,
        as_of: datetime,
    ) -> list[Bar]:
        bars = self._adapter.get_historical_bars(symbol, timeframe_minutes, count + 2, as_of)
        bars = self._drop_last_in_progress(bars)
        bars = self._dedup_completed_bars(bars)
        return bars[-count:]

    def build_multi_timeframe(
        self,
        symbol: str,
        primary_min: int,
        short_min: int,
        long_min: int,
        primary_count: int,
        short_count: int,
        long_count: int,
    ) -> TimeframeBars:
        with self._lock:
            now = datetime.now(tz=timezone.utc)
            primary = self._fetch_completed(symbol, primary_min, primary_count, now)
            short = self._fetch_completed(symbol, short_min, short_count, now)
            long = self._fetch_completed(symbol, long_min, long_count, now)
            return TimeframeBars(primary=primary, short=short, long=long)
