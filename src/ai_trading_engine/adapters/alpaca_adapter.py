from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
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
            bars.append(
                Bar(
                    timestamp=datetime.fromisoformat(str(r["t"]).replace("Z", "+00:00")),
                    open=float(r["o"]),
                    high=float(r["h"]),
                    low=float(r["l"]),
                    close=float(r["c"]),
                    volume=float(r.get("v", 0.0)),
                )
            )
        return bars

    def get_latest_quote(self, symbol: str) -> Quote:
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
        bid = float(q.get("bp") or q.get("ap") or 0.0)
        ask = float(q.get("ap") or q.get("bp") or bid)
        last = (bid + ask) / 2 if bid and ask else max(bid, ask)
        return Quote(
            timestamp=datetime.fromisoformat(str(q["t"]).replace("Z", "+00:00")),
            bid=bid,
            ask=ask,
            last=last,
            size=float(q.get("bs") or q.get("as") or 0.0),
        )
