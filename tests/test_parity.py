"""Parity tests: new ABC implementations == existing code."""

import os
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd

from core.trigger import BaseEventTrigger
from core.feature import BaseFeature
from core.labeler import BaseLabeler
from core.position_sizer import BasePositionSizer
from core.risk_manager import BaseRiskManager
from research.triggers.turtle_breakout import TurtleBreakoutTrigger
from research.features import (
    add_indicators,
    feature_volume_ratio,
    feature_breakout_intensity,
    default_feature_pipeline,
    VolumeRatioFeature,
    BreakoutIntensityFeature,
    CompositeFeature,
)
from research.labeling import (
    apply_triple_barrier,
    fixed_horizon_label,
    TripleBarrierLabeler,
    FixedHorizonLabeler,
)
from execution.sizers import VolatilityTargetingSizer
from execution.risk_managers import MaxDrawdownRiskManager, AllowAllRiskManager


def _make_test_data(n: int = 500):
    """Generate synthetic OHLCV data for parity testing."""
    np.random.seed(42)
    close = 40000.0 + np.cumsum(np.random.randn(n) * 50)
    close = np.maximum(close, 100.0)
    high = close * (1.0 + np.random.uniform(0.001, 0.005, n))
    low = close * (1.0 - np.random.uniform(0.001, 0.005, n))
    open_p = close * (1.0 + np.random.uniform(-0.002, 0.002, n))
    volume = np.random.lognormal(10, 0.5, n)
    taker_buy = volume * np.random.uniform(0.3, 0.7, n)

    dates = pd.date_range("2023-01-01", periods=n, freq="1min")
    df = pd.DataFrame({
        "datetime": dates,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "quote_volume": close * volume,
        "trades_count": np.random.randint(50, 500, n),
        "taker_buy_base": taker_buy,
        "taker_buy_quote": taker_buy * close,
    }, index=dates)
    return df


def test_trigger_is_abc():
    """TurtleBreakoutTrigger is a BaseEventTrigger."""
    trigger = TurtleBreakoutTrigger()
    assert isinstance(trigger, BaseEventTrigger)


def test_features_are_abc():
    """Feature classes are BaseFeature instances."""
    assert isinstance(VolumeRatioFeature(), BaseFeature)
    assert isinstance(BreakoutIntensityFeature(), BaseFeature)
    assert isinstance(CompositeFeature([VolumeRatioFeature()]), BaseFeature)


def test_labelers_are_abc():
    """Labeler classes are BaseLabeler instances."""
    assert isinstance(TripleBarrierLabeler(), BaseLabeler)
    assert isinstance(FixedHorizonLabeler(), BaseLabeler)


def test_sizer_is_abc():
    """VolatilityTargetingSizer is a BasePositionSizer."""
    assert isinstance(VolatilityTargetingSizer(), BasePositionSizer)


def test_risk_managers_are_abc():
    """Risk managers are BaseRiskManager instances."""
    assert isinstance(MaxDrawdownRiskManager(), BaseRiskManager)
    assert isinstance(AllowAllRiskManager(), BaseRiskManager)


def test_feature_parity_volume_ratio():
    """VolumeRatioFeature.compute_one() == feature_volume_ratio()."""
    df = _make_test_data(200)
    df = add_indicators(df)
    idx = 150

    feat_class = VolumeRatioFeature(vol_period=20)
    result_class = feat_class.compute_one(df, idx)
    result_func = feature_volume_ratio(df, idx, 20)

    for key in result_func:
        assert abs(result_class[key] - result_func[key]) < 1e-10, \
            f"Mismatch on {key}: {result_class[key]} vs {result_func[key]}"


def test_feature_parity_breakout_intensity():
    """BreakoutIntensityFeature.compute_one() == feature_breakout_intensity()."""
    df = _make_test_data(200)
    df = add_indicators(df)
    idx = 150

    feat_class = BreakoutIntensityFeature(entry_period=20, atr_period=20)
    result_class = feat_class.compute_one(df, idx)
    result_func = feature_breakout_intensity(df, idx, 20, 20)

    for key in result_func:
        assert abs(result_class[key] - result_func[key]) < 1e-10, \
            f"Mismatch on {key}"


def test_sizer_matches_inline_formula():
    """VolatilityTargetingSizer matches the inline sizing formula."""
    sizer = VolatilityTargetingSizer(risk_pct=0.02, max_leverage=20.0)
    import math

    # Test case: same as inline formula
    equity = 10000.0
    atr = 500.0
    atr_mult = 2.0
    entry = 40000.0

    size = sizer.calculate_size(
        signal_strength=atr_mult,
        current_atr=atr,
        account_equity=equity,
        entry_price=entry,
    )

    # Manual formula
    stop_distance = atr_mult * atr
    risk_amount = equity * 0.02
    expected_risk = risk_amount / stop_distance
    expected_leverage = (equity * 20.0) / entry
    expected = math.floor(min(expected_risk, expected_leverage) * 1000) / 1000.0

    assert abs(size - expected) < 1e-10, f"{size} != {expected}"


def test_risk_manager_allow_all():
    """AllowAllRiskManager always returns True."""
    rm = AllowAllRiskManager()
    assert rm.check_risk_limits({}) is True


def test_risk_manager_max_drawdown():
    """MaxDrawdownRiskManager blocks when DD > threshold."""
    rm = MaxDrawdownRiskManager(max_dd_pct=0.05)
    assert rm.check_risk_limits({"current_drawdown": 0.03}) is True
    assert rm.check_risk_limits({"current_drawdown": 0.07}) is False
    assert rm.check_risk_limits({"current_drawdown": 0.05}) is False


if __name__ == "__main__":
    test_trigger_is_abc()
    test_features_are_abc()
    test_labelers_are_abc()
    test_sizer_is_abc()
    test_risk_managers_are_abc()
    test_feature_parity_volume_ratio()
    test_feature_parity_breakout_intensity()
    test_sizer_matches_inline_formula()
    test_risk_manager_allow_all()
    test_risk_manager_max_drawdown()
    print("All parity tests passed!")
