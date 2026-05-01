from __future__ import annotations

from datetime import datetime, timezone

from ai_trading_engine.trade_costs import estimate_round_trip_cost
from ai_trading_engine.models import AccountState, AiDecision, ClosedTrade, Position, Quote


class MockExecutionEngine:
    """Simulates bracket order entry + exits via quote checks."""

    def __init__(
        self,
        tick_size: float = 0.25,
        dollars_per_tick: float = 12.5,
        cost_model_enabled: bool = False,
        cost_slippage_bps_per_side: float = 1.5,
        cost_fee_per_share: float = 0.0035,
        cost_min_fee_per_order: float = 0.35,
    ) -> None:
        self._tick_size = tick_size
        self._dollars_per_tick = dollars_per_tick
        self._cost_model_enabled = bool(cost_model_enabled)
        self._cost_slippage_bps_per_side = float(cost_slippage_bps_per_side)
        self._cost_fee_per_share = float(cost_fee_per_share)
        self._cost_min_fee_per_order = float(cost_min_fee_per_order)

    def fetch_account_summary(self) -> dict | None:
        return None

    def submit_bracket(self, symbol: str, decision: AiDecision, quote: Quote, account: AccountState) -> None:
        if decision.action != "trade" or decision.direction is None:
            return
        entry = quote.last
        account.open_position = Position(
            symbol=symbol,
            direction=decision.direction,
            size=decision.size,
            entry_price=entry,
            sl_ticks=decision.sl_ticks,
            tp_ticks=decision.tp_ticks,
            opened_at=datetime.now(tz=timezone.utc),
            thesis=decision.reasoning,
        )

    def _tp_price(self, pos: Position) -> float:
        move = pos.tp_ticks * self._tick_size
        return pos.entry_price + move if pos.direction == "LONG" else pos.entry_price - move

    def _sl_price(self, pos: Position) -> float:
        move = pos.sl_ticks * self._tick_size
        return pos.entry_price - move if pos.direction == "LONG" else pos.entry_price + move

    def _pnl(self, pos: Position, exit_price: float) -> float:
        ticks = (exit_price - pos.entry_price) / self._tick_size
        if pos.direction == "SHORT":
            ticks *= -1
        return ticks * self._dollars_per_tick * pos.size

    def process_open_position(self, quote: Quote, account: AccountState, thesis: str = "") -> ClosedTrade | None:
        pos = account.open_position
        if pos is None:
            return None

        tp = self._tp_price(pos)
        sl = self._sl_price(pos)
        last = quote.last

        hit_tp = last >= tp if pos.direction == "LONG" else last <= tp
        hit_sl = last <= sl if pos.direction == "LONG" else last >= sl

        if not (hit_tp or hit_sl):
            return None

        exit_price = tp if hit_tp else sl
        pnl = self._pnl(pos, exit_price)
        if self._cost_model_enabled:
            pnl -= estimate_round_trip_cost(
                entry_price=pos.entry_price,
                exit_price=exit_price,
                size=pos.size,
                slippage_bps_per_side=self._cost_slippage_bps_per_side,
                fee_per_share=self._cost_fee_per_share,
                min_fee_per_order=self._cost_min_fee_per_order,
            )
        closed = ClosedTrade(
            symbol=pos.symbol,
            direction=pos.direction,
            size=pos.size,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            pnl=pnl,
            opened_at=pos.opened_at,
            closed_at=datetime.now(tz=timezone.utc),
            thesis=pos.thesis or thesis,
        )

        account.balance += pnl
        account.daily_realized_pnl += pnl
        account.closed_trades_today.append(closed)
        account.open_position = None

        return closed
