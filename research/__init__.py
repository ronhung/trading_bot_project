"""
Research toolkit for quantitative trading strategy development.

Pure pandas/numpy — no Backtrader dependency at import time.

Modules
-------
labeling       — Triple-barrier labeler for supervised learning
features       — Indicator precomputation + feature callables
dataset_builder— build_ml_dataset(): raw klines → (X, y) ML dataset
backtest       — lightweight_backtest(): vectorized backtester (no Backtrader)
param_sweep    — run_parameter_sweep(): multiprocessing grid search
"""

__all__ = [
    "apply_triple_barrier",
    "add_indicators",
    "build_ml_dataset",
    "lightweight_backtest",
    "run_parameter_sweep",
]
