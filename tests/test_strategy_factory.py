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


def test_half_kelly_sizer():
    from execution.sizers import HalfKellySizer
    sizer = HalfKellySizer(win_rate=0.5, profit_factor=2.0, max_risk_cap=0.03, max_leverage=10.0)

    # Positive edge: risk_pct should be capped at max_risk_cap 0.03
    # Equity = 100,000, stop_distance = 1,000 -> size = (100000 * 0.03) / 1000 = 3.0 BTC
    sz = sizer.calculate_size(signal_strength=0.5, current_atr=100.0, account_equity=100000.0, entry_price=30000.0, stop_distance=1000.0)
    assert sz == 3.0

    # Negative edge: p=0.2, b=1.0 -> Kelly < 0 -> size = 0.0
    bad_sizer = HalfKellySizer(win_rate=0.2, profit_factor=1.0)
    sz_bad = bad_sizer.calculate_size(signal_strength=0.2, current_atr=100.0, account_equity=100000.0, entry_price=30000.0, stop_distance=1000.0)
    assert sz_bad == 0.0

    # Config-driven creation
    cfg = {
        "strategy": {"symbol": "BTCUSDT"},
        "trigger": "trend_breakout",
        "sizing": {
            "type": "half_kelly",
            "win_rate": 0.48,
            "profit_factor": 2.2,
            "max_risk_cap": 0.035,
            "max_leverage": 8.0,
        }
    }
    wrapper, meta = build_strategy_from_config(cfg)
    assert isinstance(wrapper._sizer, HalfKellySizer)
    assert wrapper._sizer.win_rate == 0.48
    assert wrapper._sizer.max_risk_cap == 0.035


def test_microstructure_triggers_registered():
    from core.strategy_factory import _lazy_register_builtins, TRIGGER_REGISTRY
    _lazy_register_builtins()
    assert "ofi_momentum" in TRIGGER_REGISTRY
    assert "absorption_divergence" in TRIGGER_REGISTRY
    assert "liquidation_climax" in TRIGGER_REGISTRY
    assert "dynamic_vwap_band" in TRIGGER_REGISTRY


