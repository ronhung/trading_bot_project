"""Phase 1: Event Trigger — filter market noise, return event timestamps + directions."""

from abc import ABC, abstractmethod

import pandas as pd


class BaseEventTrigger(ABC):
    """
    Abstract base for event triggers.

    An event trigger scans price data and returns a Series marking timestamps
    where economically-meaningful events occur (e.g., Donchian breakouts).

    Subclasses must implement generate_signals().
    """

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Scan data and return event signals.

        Args:
            data: OHLCV DataFrame sorted chronologically (oldest first).
                  Must contain at minimum: open, high, low, close.

        Returns:
            pd.Series with the same index as `data`.
            Values: 1 = long entry, -1 = short entry, 0 = no event.
        """
        ...
