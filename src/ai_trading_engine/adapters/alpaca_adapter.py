from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ai_trading_engine.adapters.base import MarketDataAdapter
from ai_trading_engine.models import Bar, Quote


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


def _parse_utc_timestamp(value: Any) -> datetime | None:
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalise_bar(row: dict[str, Any]) -> Bar | None:
    ts = _parse_utc_timestamp(row.get("t"))
    open_ = _float_or_none(row.get("o"))
    high = _float_or_none(row.get("h"))
    low = _float_or_none(row.get("l"))
    close = _float_or_none(row.get("c"))
    volume = _float_or_none(row.get("v")) or 0.0
    if ts is None or open_ is None or high is None or low is None or close is None:
        return None
    if min(open_, high, low, close) <= 0.0 or volume < 0.0:
        return None
    if high < max(open_, close) or low > min(open_, close) or high < low:
        return None
    return Bar(timestamp=ts, open=open_, high=high, low=low, close=close, volume=volume)


class AlpacaMarketDataAdapter(MarketDataAdapter):
    """Alpaca equities market data adapter (paper/live key compatible)."""

    def __init__(self, key_id: str, secret_key: str, data_url: str = "https://data.alpaca.markets") -> None:
        if not key_id or not secret_key:
            raise RuntimeError("Alpaca credentials missing: set ALPACA_KEY_ID and ALPACA_SECRET_KEY")
        self._key = key_id
        self._secret = secret_key
        self._data_url = data_url.rstrip("/")

    def _request_json(self, path: str, params: dict[str, str]) -> dict:
        qs = urlencode(params)
        url = f"{self._data_url}{path}?{qs}"
        req = Request(url)
        req.add_header("APCA-API-KEY-ID", self._key)
        req.add_header("APCA-API-SECRET-KEY", self._secret)
        with urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def get_historical_bars(
        self,
        symbol: str,
        timeframe_minutes: int,
        count: int,
        as_of: datetime,
    ) -> list[Bar]:
        symbol = str(symbol or "").strip().upper()
        if not symbol:
            raise RuntimeError("Alpaca historical bars request requires a symbol")
        tf = _tf_label(timeframe_minutes)
        lookback_minutes = max(60 * 24 * 7, timeframe_minutes * max(1, count) * 32)
        start = as_of - timedelta(minutes=lookback_minutes)
        payload = self._request_json(
            f"/v2/stocks/{symbol}/bars",
            {
                "timeframe": tf,
                "limit": str(max(1, count)),
                "adjustment": "raw",
                "feed": "iex",
                "start": _to_iso(start),
                "end": _to_iso(as_of),
            },
        )
        rows = payload.get("bars") or []
        bars: list[Bar] = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            bar = _normalise_bar(r)
            if bar is not None:
                bars.append(bar)
        bars.sort(key=lambda b: b.timestamp)
        return bars[-max(1, int(count)):]

    def get_latest_quote(self, symbol: str) -> Quote:
        symbol = str(symbol or "").strip().upper()
        if not symbol:
            raise RuntimeError("Alpaca latest quote request requires a symbol")
        payload = self._request_json(
            "/v2/stocks/quotes/latest",
            {
                "symbols": symbol,
                "feed": "iex",
            },
        )
        quotes = payload.get("quotes", {})
        q = quotes.get(symbol)
        if not q:
            raise RuntimeError(f"No latest quote returned for symbol {symbol}")
        ts = _parse_utc_timestamp(q.get("t"))
        if ts is None:
            raise RuntimeError(f"Latest quote for {symbol} has invalid timestamp")
        bid_raw = _float_or_none(q.get("bp"))
        ask_raw = _float_or_none(q.get("ap"))
        bid = bid_raw if bid_raw is not None and bid_raw > 0.0 else ask_raw
        ask = ask_raw if ask_raw is not None and ask_raw > 0.0 else bid_raw
        if bid is None or ask is None or bid <= 0.0 or ask <= 0.0:
            raise RuntimeError(f"Latest quote for {symbol} has no usable bid/ask")
        if bid > ask:
            raise RuntimeError(f"Latest quote for {symbol} is crossed: bid={bid} ask={ask}")
        last = (bid + ask) / 2 if bid and ask else max(bid, ask)
        size = _float_or_none(q.get("bs")) or _float_or_none(q.get("as")) or 0.0
        return Quote(
            timestamp=ts,
            bid=bid,
            ask=ask,
            last=last,
            size=max(0.0, size),
        )
