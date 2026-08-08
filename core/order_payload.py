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
            "timestamp": self.signal_timestamp,
        }
        if self.take_profit is not None:
            d["take_profit"] = self.take_profit
        return d
