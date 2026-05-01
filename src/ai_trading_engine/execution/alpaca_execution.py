from __future__ import annotations

import json
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ai_trading_engine.models import AccountState, AiDecision, Position, Quote


class AlpacaPaperExecutionEngine:
    """Submits paper orders to Alpaca and reads broker account/position state."""

    def __init__(
        self,
        key_id: str,
        secret_key: str,
        trading_url: str = "https://paper-api.alpaca.markets",
        tick_size: float = 0.01,
    ) -> None:
        if not key_id or not secret_key:
            raise RuntimeError("Alpaca credentials missing: set ALPACA_KEY_ID and ALPACA_SECRET_KEY")
        self._key = key_id
        self._secret = secret_key
        self._base = trading_url.rstrip("/")
        self._tick_size = max(1e-6, float(tick_size))

    def _post(self, path: str, payload: dict) -> dict:
        req = Request(
            f"{self._base}{path}",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
        )
        req.add_header("Content-Type", "application/json")
        req.add_header("APCA-API-KEY-ID", self._key)
        req.add_header("APCA-API-SECRET-KEY", self._secret)
        with urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get(self, path: str) -> dict:
        req = Request(f"{self._base}{path}", method="GET")
        req.add_header("accept", "application/json")
        req.add_header("APCA-API-KEY-ID", self._key)
        req.add_header("APCA-API-SECRET-KEY", self._secret)
        with urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _delete(self, path: str) -> dict:
        req = Request(f"{self._base}{path}", method="DELETE")
        req.add_header("accept", "application/json")
        req.add_header("APCA-API-KEY-ID", self._key)
        req.add_header("APCA-API-SECRET-KEY", self._secret)
        with urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8").strip()
            return json.loads(body) if body else {}

    def fetch_account_summary(self) -> dict:
        payload = self._get("/v2/account")
        equity = float(payload.get("equity") or payload.get("portfolio_value") or 0.0)
        last_equity = float(payload.get("last_equity") or equity or 0.0)
        return {
            "account_number": str(payload.get("account_number", "")),
            "status": str(payload.get("status", "")),
            "equity": equity,
            "last_equity": last_equity,
            "portfolio_value": float(payload.get("portfolio_value") or equity or 0.0),
            "cash": float(payload.get("cash") or 0.0),
            "buying_power": float(payload.get("buying_power") or 0.0),
        }

    def fetch_symbol_position(self, symbol: str) -> Position | None:
        try:
            payload = self._get(f"/v2/positions/{symbol}")
        except HTTPError as exc:
            if exc.code == 404:
                return None
            raise

        side = str(payload.get("side", "")).lower()
        if side == "long":
            direction = "LONG"
        elif side == "short":
            direction = "SHORT"
        else:
            return None

        qty = int(float(payload.get("qty") or 0))
        if qty <= 0:
            return None

        entry = float(payload.get("avg_entry_price") or 0.0)
        if entry <= 0:
            return None

        return Position(
            symbol=str(payload.get("symbol") or symbol),
            direction=direction,
            size=qty,
            entry_price=entry,
            sl_ticks=0,
            tp_ticks=0,
            opened_at=datetime.now(tz=timezone.utc),
        )

    @staticmethod
    def _round_price(price: float) -> str:
        if price >= 1:
            return f"{price:.2f}"
        return f"{price:.4f}"

    def _build_bracket_payload(self, symbol: str, decision: AiDecision, quote: Quote) -> dict:
        if decision.direction == "LONG":
            side = "buy"
            tp = quote.last + (decision.tp_ticks * self._tick_size)
            sl = max(0.01, quote.last - (decision.sl_ticks * self._tick_size))
        else:
            side = "sell"
            tp = max(0.01, quote.last - (decision.tp_ticks * self._tick_size))
            sl = quote.last + (decision.sl_ticks * self._tick_size)

        return {
            "symbol": symbol,
            "qty": str(max(1, int(decision.size))),
            "side": side,
            "type": "market",
            "time_in_force": "day",
            "order_class": "bracket",
            "take_profit": {"limit_price": self._round_price(tp)},
            "stop_loss": {"stop_price": self._round_price(sl)},
        }

    def process_open_position(self, quote: Quote, account: AccountState, thesis: str = ""):
        del quote, account, thesis
        return None

    def close_symbol_position(self, symbol: str) -> dict:
        try:
            payload = self._delete(f"/v2/positions/{symbol}")
            return {"closed": True, "symbol": symbol, "broker_response": payload}
        except HTTPError as exc:
            if exc.code == 404:
                return {"closed": False, "symbol": symbol, "reason": "no_open_position"}
            raise

    def list_open_orders(self, symbol: str) -> list[dict]:
        payload = self._get("/v2/orders?status=open&nested=true&limit=200")
        if not isinstance(payload, list):
            return []
        target = str(symbol or "").upper()
        return [o for o in payload if str(o.get("symbol", "")).upper() == target]

    def has_protection_orders(self, symbol: str) -> bool:
        orders = self.list_open_orders(symbol)
        for order in orders:
            order_class = str(order.get("order_class", "")).strip().lower()
            order_type = str(order.get("order_type") or order.get("type") or "").strip().lower()
            if order_class == "bracket":
                return True
            if order_type in {"stop", "stop_limit", "limit"}:
                return True
            for leg in order.get("legs") or []:
                leg_type = str(leg.get("order_type") or leg.get("type") or "").strip().lower()
                if leg_type in {"stop", "stop_limit", "limit"}:
                    return True
        return False

    def submit_bracket(self, symbol: str, decision: AiDecision, quote: Quote, account: AccountState) -> None:
        if decision.action != "trade" or decision.direction is None:
            return
        payload = self._build_bracket_payload(symbol, decision, quote)
        self._post("/v2/orders", payload)
        account.open_position = Position(
            symbol=symbol,
            direction=decision.direction,
            size=max(1, int(decision.size)),
            entry_price=float(quote.last),
            sl_ticks=int(decision.sl_ticks),
            tp_ticks=int(decision.tp_ticks),
            opened_at=datetime.now(tz=timezone.utc),
            thesis=str(decision.reasoning or ""),
        )
