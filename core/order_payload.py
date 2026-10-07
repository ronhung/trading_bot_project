"""
Bracket Order Protocol — data contract for Python → C++ orders.

Python computes entry + bracket exit parameters. C++ manages the full
order lifecycle: entry fill, stop-loss monitoring, trailing exit updates,
and take-profit. Python is NOT in the hot loop for exit detection.

Once an OrderPayload is sent, Python's StrategyWrapper enters WAITING_CLOSE
state and pauses entry detection until C++ reports POSITION_CLOSED.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time

from core.execution_spec import (
    ExecutionSpec,
    DEFAULT_ENTRY_EXECUTION,
    DEFAULT_EXIT_EXECUTION,
)


class Action(str, Enum):
    """Order action: BUY (open long) or SELL (open short)."""
    BUY = "BUY"
    SELL = "SELL"


class TrailingExitIndicator(str, Enum):
    """
    Dynamic exit rule managed by the C++ engine.

    The engine updates the exit price each bar based on the selected indicator.
    """
    DONCHIAN_LOW = "donchian_low"        # exit when price < N-period low
    DONCHIAN_HIGH = "donchian_high"      # exit when price > N-period high
    MOVING_AVERAGE = "moving_average"    # exit when price crosses MA


@dataclass(frozen=True)
class OrderPayload:
    """
    Immutable order sent from Python StrategyWrapper → C++ execution engine.

    Contains everything the C++ engine needs to execute the entry AND manage
    the full bracket-order exit lifecycle without further Python involvement.

    Attributes:
        action: BUY (open long) or SELL (open short).
        symbol: Trading pair, e.g. "BTCUSDT".
        quantity: Position size in base currency units (e.g., BTC).
        entry_price: Limit price for the entry order.
        hard_stop_loss: Absolute stop-loss price. C++ monitors every bar.
            For long: price below entry. For short: price above entry.
            E.g., entry - 2*ATR for a long.
        trailing_exit_indicator: Which dynamic exit rule C++ should apply.
        trailing_exit_period: Lookback period (bars) for the trailing indicator.
            E.g., 14400 for a 10-day Donchian low exit at 1m bars.
        take_profit: Optional take-profit price. C++ places a standing LIMIT.
        signal_timestamp: Unix timestamp when the signal was generated.
    """
    action: Action
    symbol: str
    quantity: float
    entry_price: float
    hard_stop_loss: float
    trailing_exit_indicator: TrailingExitIndicator
    trailing_exit_period: int
    take_profit: Optional[float] = None
    signal_timestamp: float = field(default_factory=time.time)
    entry_execution: ExecutionSpec = field(default_factory=lambda: DEFAULT_ENTRY_EXECUTION)
    exit_execution: ExecutionSpec = field(default_factory=lambda: DEFAULT_EXIT_EXECUTION)

    def to_dict(self) -> dict:
        """Serialize to dict for ZMQ JSON transmission."""
        d = {
            "action": self.action.value,
            "symbol": self.symbol,
            "quantity": self.quantity,
            "price": self.entry_price,
            "stop_price": self.hard_stop_loss,
            "trailing_exit": {
                "indicator": self.trailing_exit_indicator.value,
                "period": self.trailing_exit_period,
            },
            "execution": {
                "entry": self.entry_execution.to_dict(),
                "exit": self.exit_execution.to_dict(),
            },
            "timestamp": self.signal_timestamp,
        }
        if self.take_profit is not None:
            d["take_profit"] = self.take_profit
        return d


@dataclass(frozen=True)
class BracketExit:
    """
    Exit spec — how the execution engine manages the exit lifecycle.

    This is the pluggable "exit strategy" counterpart to BaseEventTrigger /
    BasePositionSizer / BaseRiskManager. It captures the bracket-order
    parameters that are serialized into the OrderPayload.

    hard_stop_mode selects how the initial (hard) stop is computed:
      "donchian" — the trailing Donchian channel's current level
                   (pure trailing exit: no separate fixed stop)
      "atr"      — entry ± atr_mult * ATR (fixed ATR stop)
      "none"     — no hard stop (trailing only; sizing must use another source)
    """

    trailing_exit_indicator: TrailingExitIndicator = TrailingExitIndicator.DONCHIAN_LOW
    trailing_exit_period: int = 14400
    hard_stop_mode: str = "donchian"
    atr_mult: float = 2.0

    def initial_stop(
        self,
        side: int,
        entry_price: float,
        atr: float,
        trailing_low: Optional[float] = None,
        trailing_high: Optional[float] = None,
    ) -> Optional[float]:
        """
        Compute the initial (hard) stop price for a side (+1 long / -1 short).

        Returns None when hard_stop_mode="none".
        """
        if self.hard_stop_mode == "atr":
            dist = self.atr_mult * max(atr, 1e-9)
            return entry_price - dist if side > 0 else entry_price + dist
        if self.hard_stop_mode == "donchian":
            if side > 0:
                return trailing_low
            return trailing_high
        return None

    def stop_distance(
        self,
        side: int,
        entry_price: float,
        atr: float,
        trailing_low: Optional[float] = None,
        trailing_high: Optional[float] = None,
    ) -> float:
        """Risk distance (|entry - stop|) used for position sizing."""
        stop = self.initial_stop(side, entry_price, atr, trailing_low, trailing_high)
        if stop is not None:
            return abs(entry_price - stop)
        return self.atr_mult * max(atr, 1e-9)
