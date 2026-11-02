import os
import pytest
from core.strategy_factory import (
    register_trigger,
    register_sizer,
    register_risk,
    build_strategy_from_config,
    TRIGGER_REGISTRY,
)
from core.trigger import BaseEventTrigger


def test_registry_builtins():
    # Verify built-in triggers and components are registered
    wrapper, meta = build_strategy_from_config("config/trend_breakout_highfreq.yaml")
    assert wrapper is not None
    assert meta["symbol"] == "BTCUSDT"
    assert "trend_breakout" in TRIGGER_REGISTRY


def test_custom_trigger_registration():
    @register_trigger("dummy_test_trigger")
    class DummyTrigger(BaseEventTrigger):
        def __init__(self, foo=1):
            self.foo = foo

        def generate_signals(self, df):
            return df["close"] * 0

    assert "dummy_test_trigger" in TRIGGER_REGISTRY
    cfg = {
        "strategy": {"symbol": "ETHUSDT"},
        "trigger": "dummy_test_trigger",
        "trigger_params": {"foo": 42},
    }
    wrapper, meta = build_strategy_from_config(cfg)
    assert meta["symbol"] == "ETHUSDT"
    assert wrapper._trigger.foo == 42
