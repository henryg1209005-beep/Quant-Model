from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


Direction = Literal["LONG", "SHORT"]
Action = Literal["trade", "hold"]


@dataclass(frozen=True)
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Quote:
    timestamp: datetime
    bid: float
    ask: float
    last: float
    size: float


@dataclass
class Position:
    symbol: str
    direction: Direction
    size: int
    entry_price: float
    sl_ticks: int
    tp_ticks: int
    opened_at: datetime
    thesis: str = ""


@dataclass
class ClosedTrade:
    symbol: str
    direction: Direction
    size: int
    entry_price: float
    exit_price: float
    pnl: float
    opened_at: datetime
    closed_at: datetime
    thesis: str
    regime: str = ""
    confidence: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class AccountState:
    balance: float
    starting_balance: float
    trading_day_et: str | None = None
    todays_trade_count: int = 0
    daily_realized_pnl: float = 0.0
    open_position: Position | None = None
    closed_trades_today: list[ClosedTrade] = field(default_factory=list)

    @property
    def drawdown(self) -> float:
        return min(0.0, self.balance - self.starting_balance)

    @property
    def risk_budget_remaining(self) -> float:
        return max(0.0, self.starting_balance - self.balance)


@dataclass
class AiDecision:
    action: Action
    direction: Direction | None
    confidence: float
    size: int
    sl_ticks: int
    tp_ticks: int
    reasoning: str
    forecast_direction: Direction | None = None
    forecast_confidence: float = 0.0
    forecast_horizon_minutes: int = 15


@dataclass
class ContextDimension:
    name: str
    value: float | str
    label: str
    detail: str


@dataclass
class MarketContext:
    dimensions: list[ContextDimension]
    key_levels: dict[str, float]
    gex_overlay: dict[str, float | str] | None = None
    pattern_features: dict | None = None
    structure_features: dict | None = None


@dataclass
class IndicatorSet:
    ema9: float
    ema21: float
    sma50: float
    rsi14: float
    macd: float
    macd_signal: float
    macd_hist: float
    atr14: float
    bb_mid: float
    bb_upper: float
    bb_lower: float
    vwap: float
    vwap_std: float
    pivot: float
    support_1: float
    resistance_1: float
    regime: str


@dataclass
class TimeframeBars:
    primary: list[Bar]
    short: list[Bar]
    long: list[Bar]
