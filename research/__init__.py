"""
Research toolkit for quantitative trading strategy development.

Pure pandas/numpy — no Backtrader dependency.

Modules
-------
labeling        — Triple-barrier + fixed-horizon labelers + BaseLabeler classes
features        — Indicator precomputation + feature callables + BaseFeature classes
dataset_builder — build_ml_dataset(): raw klines → (X, y) ML dataset
backtest        — lightweight_backtest(): vectorized backtester
param_sweep     — run_parameter_sweep(): multiprocessing grid search
evaluator       — ModelEvaluator: time-series CV, IC, decile, train/save
pipeline_runner — Config-driven DI runner for research pipeline
triggers/       — Concrete event trigger implementations
"""

from research.labeling import (
    apply_triple_barrier,
    apply_triple_barrier_loop,
    fixed_horizon_label,
    TripleBarrierLabeler,
    FixedHorizonLabeler,
)
from research.features import (
    add_indicators,
    default_feature_pipeline,
    default_feature_set,
    make_feature_pipeline,
    CompositeFeature,
)
from research.dataset_builder import (
    build_ml_dataset,
    make_turtle_breakout_trigger,
    turtle_breakout_trigger,
    make_synthetic_ohlcv,
)
from research.backtest import lightweight_backtest
from research.param_sweep import run_parameter_sweep, _backtest_target, _dataset_target
from research.evaluator import ModelEvaluator
from research.triggers.turtle_breakout import TurtleBreakoutTrigger

__all__ = [
    # labeling
    "apply_triple_barrier",
    "apply_triple_barrier_loop",
    "fixed_horizon_label",
    "TripleBarrierLabeler",
    "FixedHorizonLabeler",
    # features
    "add_indicators",
    "default_feature_pipeline",
    "default_feature_set",
    "make_feature_pipeline",
    "CompositeFeature",
    # dataset
    "build_ml_dataset",
    "make_turtle_breakout_trigger",
    "turtle_breakout_trigger",
    "make_synthetic_ohlcv",
    # backtest
    "lightweight_backtest",
    # param sweep
    "run_parameter_sweep",
    "_backtest_target",
    "_dataset_target",
    # evaluator
    "ModelEvaluator",
    # triggers
    "TurtleBreakoutTrigger",
]
