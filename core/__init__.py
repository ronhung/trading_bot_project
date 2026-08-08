"""
core — Abstract Base Classes for the unified quant trading system.

Phase 1: BaseEventTrigger
Phase 2: BaseFeature, BaseLabeler
Phase 4: BasePositionSizer, BaseRiskManager
Phase 5-6: StrategyWrapper, OrderPayload, LiveDataFeeder, LiveExecutionGateway
"""

from core.trigger import BaseEventTrigger
from core.feature import BaseFeature
from core.labeler import BaseLabeler
from core.position_sizer import BasePositionSizer
from core.risk_manager import BaseRiskManager
from core.order_payload import (
    OrderPayload,
    Action,
    TrailingExitIndicator,
)
from core.data_feeder import LiveDataFeeder
from core.execution_gateway import LiveExecutionGateway
from core.strategy_wrapper import StrategyWrapper, StrategyState

__all__ = [
    "BaseEventTrigger",
    "BaseFeature",
    "BaseLabeler",
    "BasePositionSizer",
    "BaseRiskManager",
    "OrderPayload",
    "Action",
    "TrailingExitIndicator",
    "LiveDataFeeder",
    "LiveExecutionGateway",
    "StrategyWrapper",
    "StrategyState",
]
