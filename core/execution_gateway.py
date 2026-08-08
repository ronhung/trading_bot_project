"""Phase 5-6: LiveExecutionGateway ABC — abstracts order submission."""

from abc import ABC, abstractmethod
from typing import Callable, Any, Dict, Optional

from core.order_payload import OrderPayload


class LiveExecutionGateway(ABC):
    """
    Abstract base for order execution gateways.

    Decouples order transmission (ZMQ, REST, mock) from strategy logic.
    The gateway sends OrderPayload objects and receives position-closed
    callbacks from the execution engine.

    Subclasses must implement send_order().
    """

    @abstractmethod
    def send_order(self, order: OrderPayload) -> None:
        """
        Transmit an order to the execution engine.

        Args:
            order: Immutable OrderPayload with entry + bracket exit parameters.
        """
        ...

    def set_position_closed_callback(
        self, callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """
        Register a callback for POSITION_CLOSED notifications.

        The execution engine calls this when a position is fully closed
        (stop-loss, trailing exit, or take-profit).

        Args:
            callback: Called as callback(close_info) where close_info has
                      keys: symbol, reason, entry_price, exit_price, pnl.
        """
        self._position_closed_callback: Optional[Callable] = callback

    def _on_position_closed(self, close_info: Dict[str, Any]) -> None:
        """Subclasses call this when engine reports position closed."""
        cb = getattr(self, "_position_closed_callback", None)
        if cb is not None:
            cb(close_info)
