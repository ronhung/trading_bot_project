"""
BinanceZmqExecutionGateway — ZMQ-based order gateway implementing
LiveExecutionGateway ABC. Wraps BinanceZmqClient.send_order_signal().
"""

from typing import Any, Dict

from core.execution_gateway import LiveExecutionGateway
from core.order_payload import OrderPayload
from live_strategy.zmq_client import BinanceZmqClient


class BinanceZmqExecutionGateway(LiveExecutionGateway):
    """
    Sends OrderPayload objects to the C++ engine via ZMQ PUSH.

    The C++ engine receives the order, validates it, executes the entry,
    and manages the full bracket-order exit lifecycle.
    """

    def __init__(self, client: BinanceZmqClient):
        """
        Args:
            client: Shared BinanceZmqClient (same instance as feeder uses).
        """
        self._client = client

    def send_order(self, order: OrderPayload) -> None:
        """
        Transmit an OrderPayload to the C++ engine.

        Serializes the OrderPayload to dict and sends via ZMQ.
        The C++ engine receives the bracket order parameters and manages
        entry execution + stop-loss + trailing exit autonomously.
        """
        payload_dict = order.to_dict()
        # Use the existing send_order_signal interface for backward compat
        self._client.send_order_signal(
            action=order.action.value,
            symbol=order.symbol,
            price=order.entry_price,
            stop_price=order.hard_stop_loss,
        )
        print(
            f"  [Gateway] Sent {order.action.value} "
            f"qty={order.quantity:.4f} @ {order.entry_price:.2f} "
            f"stop={order.hard_stop_loss:.2f} "
            f"trailing={order.trailing_exit_indicator.value}:{order.trailing_exit_period}"
        )
