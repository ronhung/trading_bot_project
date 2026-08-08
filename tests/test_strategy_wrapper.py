"""Tests for StrategyWrapper state machine and OrderPayload assembly."""

import os
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd

from core.order_payload import TrailingExitIndicator
from core.strategy_wrapper import StrategyWrapper, StrategyState
from execution.sizers import VolatilityTargetingSizer
from execution.risk_managers import AllowAllRiskManager, MaxDrawdownRiskManager
from research.features import ATRFeature
from research.triggers.turtle_breakout import TurtleBreakoutTrigger


def _make_synthetic_bar(close=42000.0, high=42100.0, low=41900.0):
    """Helper: create a synthetic bar dict matching C++ KLineData struct."""
    return {
        "symbol": "BTCUSDT",
        "open_time": 1700000000000,
        "close_time": 1700000059999,
        "open": close - 50,
        "high": high,
        "low": low,
        "close": close,
        "volume": 10.0,
        "quote_volume": 420000.0,
        "taker_buy_base": 5.0,
        "taker_buy_quote": 210000.0,
        "trades_count": 100,
        "is_closed": True,
    }


def _make_portfolio_state(equity=10000.0, position=0.0, drawdown=0.0):
    """Helper: synthetic portfolio state."""
    return {
        "current_position": position,
        "available_balance": equity,
        "stop_price": 0.0,
        "equity": equity,
        "peak_equity": equity * 1.05,
        "current_drawdown": drawdown,
        "open_positions": 0,
    }


def test_initial_state_is_idle():
    """StrategyWrapper starts in IDLE state."""
    wrapper = StrategyWrapper(
        trigger=TurtleBreakoutTrigger(),
        feature=ATRFeature(),
        model=None,
        feature_names=[],
        sizer=VolatilityTargetingSizer(),
        risk_manager=AllowAllRiskManager(),
        signal_threshold=0.0,
        symbol="BTCUSDT",
        trailing_exit_indicator=TrailingExitIndicator.DONCHIAN_LOW,
        trailing_exit_period=10,
    )
    assert wrapper.state == StrategyState.IDLE


def test_on_position_closed_resets_to_idle():
    """on_position_closed() transitions WAITING_CLOSE → IDLE."""
    wrapper = StrategyWrapper(
        trigger=TurtleBreakoutTrigger(),
        feature=ATRFeature(),
        model=None,
        feature_names=[],
        sizer=VolatilityTargetingSizer(),
        risk_manager=AllowAllRiskManager(),
        signal_threshold=0.0,
        symbol="BTCUSDT",
        trailing_exit_indicator=TrailingExitIndicator.DONCHIAN_LOW,
        trailing_exit_period=10,
    )
    # Manually set to WAITING_CLOSE
    wrapper._state = StrategyState.WAITING_CLOSE
    wrapper.on_position_closed({"reason": "stop_loss", "pnl": -50.0})
    assert wrapper.state == StrategyState.IDLE


def test_risk_manager_blocks_entry():
    """When risk manager returns False, on_bar returns None."""
    wrapper = StrategyWrapper(
        trigger=TurtleBreakoutTrigger(),
        feature=ATRFeature(),
        model=None,
        feature_names=[],
        sizer=VolatilityTargetingSizer(),
        risk_manager=MaxDrawdownRiskManager(max_dd_pct=0.05),
        signal_threshold=0.0,
        symbol="BTCUSDT",
        trailing_exit_indicator=TrailingExitIndicator.DONCHIAN_LOW,
        trailing_exit_period=10,
    )
    # Drawdown > 5% → blocked
    state = _make_portfolio_state(drawdown=0.10)
    bar = _make_synthetic_bar()
    order = wrapper.on_bar(bar, state)
    assert order is None


def test_waits_in_waiting_close():
    """When WAITING_CLOSE, on_bar always returns None."""
    wrapper = StrategyWrapper(
        trigger=TurtleBreakoutTrigger(),
        feature=ATRFeature(),
        model=None,
        feature_names=[],
        sizer=VolatilityTargetingSizer(),
        risk_manager=AllowAllRiskManager(),
        signal_threshold=0.0,
        symbol="BTCUSDT",
        trailing_exit_indicator=TrailingExitIndicator.DONCHIAN_LOW,
        trailing_exit_period=10,
    )
    wrapper._state = StrategyState.WAITING_CLOSE
    bar = _make_synthetic_bar(close=50000.0, high=51000.0)  # would be breakout
    state = _make_portfolio_state()
    order = wrapper.on_bar(bar, state)
    assert order is None  # blocked because we're waiting


def test_reset_clears_state():
    """reset() returns to IDLE and clears buffer."""
    wrapper = StrategyWrapper(
        trigger=TurtleBreakoutTrigger(),
        feature=ATRFeature(),
        model=None,
        feature_names=[],
        sizer=VolatilityTargetingSizer(),
        risk_manager=AllowAllRiskManager(),
        signal_threshold=0.0,
        symbol="BTCUSDT",
        trailing_exit_indicator=TrailingExitIndicator.DONCHIAN_LOW,
        trailing_exit_period=10,
    )
    wrapper._state = StrategyState.WAITING_CLOSE
    wrapper.reset()
    assert wrapper.state == StrategyState.IDLE
    assert len(wrapper._kline_buffer) == 0


if __name__ == "__main__":
    test_initial_state_is_idle()
    test_on_position_closed_resets_to_idle()
    test_risk_manager_blocks_entry()
    test_waits_in_waiting_close()
    test_reset_clears_state()
    print("All StrategyWrapper tests passed!")
