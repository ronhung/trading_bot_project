"""Phase 5-6: LiveDataFeeder ABC — abstracts market data source."""

from abc import ABC, abstractmethod
from typing import Callable, Any, Dict


class LiveDataFeeder(ABC):
    """
    Abstract base for live market data feeds.

    Decouples data source (WebSocket, CSV replay, REST polling) from strategy
    logic. The feeder owns the buffer/warmup logic and calls back into the
    strategy on each new bar.

    Subclasses must implement connect() and start().
    """

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the data source."""
        ...

    @abstractmethod
    def start(self, on_bar_callback: Callable[[Dict[str, Any], Dict[str, Any]], None]) -> None:
        """
        Begin feeding bars.

        Args:
            on_bar_callback: Called as on_bar_callback(bar_data, portfolio_state)
                for each closed bar. bar_data is a dict with OHLCV + timestamp.
                portfolio_state has position, balance, stop_price from C++.
        """
        ...

    def warmup(self) -> None:
        """
        Optional: pre-fill the strategy buffer with historical bars.
        Default is no-op. Override to implement REST warmup.
        """
        pass
