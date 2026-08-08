"""Tests for OrderPayload dataclass — bracket order protocol."""

import os
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.order_payload import (
    OrderPayload,
    Action,
    TrailingExitIndicator,
)


def test_order_payload_creation():
    """OrderPayload can be created with all required fields."""
    order = OrderPayload(
        action=Action.BUY,
        symbol="BTCUSDT",
        quantity=0.123,
        entry_price=42100.0,
        hard_stop_loss=41800.0,
        trailing_exit_indicator=TrailingExitIndicator.DONCHIAN_LOW,
        trailing_exit_period=14400,
    )
    assert order.action == Action.BUY
    assert order.quantity == 0.123
    assert order.take_profit is None


def test_order_payload_immutable():
    """OrderPayload is frozen (immutable)."""
    order = OrderPayload(
        action=Action.SELL,
        symbol="BTCUSDT",
        quantity=0.1,
        entry_price=42000.0,
        hard_stop_loss=42300.0,
        trailing_exit_indicator=TrailingExitIndicator.DONCHIAN_HIGH,
        trailing_exit_period=100,
    )
    try:
        order.quantity = 0.2  # type: ignore
        assert False, "Should have raised FrozenInstanceError"
    except Exception:
        pass  # expected


def test_order_payload_to_dict():
    """to_dict() serializes correctly for ZMQ JSON."""
    order = OrderPayload(
        action=Action.BUY,
        symbol="BTCUSDT",
        quantity=0.5,
        entry_price=40000.0,
        hard_stop_loss=39800.0,
        trailing_exit_indicator=TrailingExitIndicator.DONCHIAN_LOW,
        trailing_exit_period=14400,
        take_profit=41000.0,
    )
    d = order.to_dict()
    assert d["action"] == "BUY"
    assert d["symbol"] == "BTCUSDT"
    assert d["price"] == 40000.0
    assert d["stop_price"] == 39800.0
    assert d["take_profit"] == 41000.0
    assert d["trailing_exit"]["indicator"] == "donchian_low"
    assert d["trailing_exit"]["period"] == 14400


def test_order_payload_with_take_profit():
    """Optional take_profit is supported."""
    order = OrderPayload(
        action=Action.BUY,
        symbol="BTCUSDT",
        quantity=1.0,
        entry_price=50000.0,
        hard_stop_loss=49000.0,
        trailing_exit_indicator=TrailingExitIndicator.MOVING_AVERAGE,
        trailing_exit_period=200,
        take_profit=52000.0,
    )
    assert order.take_profit == 52000.0


if __name__ == "__main__":
    test_order_payload_creation()
    test_order_payload_immutable()
    test_order_payload_to_dict()
    test_order_payload_with_take_profit()
    print("All OrderPayload tests passed!")
