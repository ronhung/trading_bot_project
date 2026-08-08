"""Phase 4: Position Sizer — convert signal context → contract size."""

from abc import ABC, abstractmethod
from typing import Optional


class BasePositionSizer(ABC):
    """
    Abstract base for position sizing.

    Isolated from order execution — the sizer computes the target size
    in base currency units; the execution layer places the order.

    Subclasses must implement calculate_size().
    """

    @abstractmethod
    def calculate_size(
        self,
        signal_strength: float,
        current_atr: float,
        account_equity: float,
        entry_price: Optional[float] = None,
    ) -> float:
        """
        Compute position size in base currency units.

        Args:
            signal_strength: Interpreted per sizer — e.g., ATR multiplier
                             for stop distance in Turtle sizing.
            current_atr: Current ATR value (absolute, in price units).
            account_equity: Available account equity (quote currency).
            entry_price: Entry price (optional). Required for leverage cap.

        Returns:
            Position size in base units (e.g., BTC). Returns 0.0 to skip.
        """
        ...
