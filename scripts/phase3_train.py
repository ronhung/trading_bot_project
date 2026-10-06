"""Phase 3: build dataset, train XGBoost, check IC + decile gates.

Usage:
  python scripts/phase3_train.py
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd

from research.features import add_indicators, default_feature_set, mean_reversion_feature_set
from research.evaluator import ModelEvaluator
from research.labeling import MeanReversionLabeler, TrailingExitLabeler
from research.triggers.bollinger_mean_reversion import BollingerMeanReversionTrigger
from research.triggers.trend_breakout import TrendFilteredBreakoutTrigger


def load_train():
    path = os.path.join(_PROJECT_ROOT, "data", "historical_data", "BTCUSDT_1m_full.parquet")
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df[df["datetime"] < "2024-01-01"].reset_index(drop=True)


def run_strategy(name, df, trigger, labeler, indicator_params, target="actual_return", feature_set=None):
    ind = add_indicators(df, **indicator_params)
    events = trigger.generate_signals(ind)
    feature_df = (feature_set or default_feature_set()).compute(ind, events)
    labels_df = labeler.compute_labels(ind, events)
    X = feature_df.join(labels_df, how="left")

    evaluator = ModelEvaluator(target=target)
    X_np, y_np, feature_names = evaluator.prepare_features(X)
    n = len(X_np)
    split = int(n * 0.8)
    if split < 100 or n - split < 50:
        print(f"[{name}] insufficient data (n={n}); skipping ML gate")
        return None

    model = evaluator.train_model(X_np[:split], y_np[:split])
    y_pred = model.predict(X_np[split:])
    ic = evaluator.evaluate_rank_ic(y_np[split:], y_pred)
    decile = evaluator.evaluate_decile_spread(y_np[split:], y_pred)

    print(f"[{name}] n={n}  IC={ic['ic']:+.4f} (p={ic['p_value']:.4f}, pass={ic['pass']})"
          f"  | decile spread={decile['spread']:+.5f} monotonic={decile['monotonic']}")
    return {"ic": ic, "decile": decile, "n": n}


def main():
    print("loading train (2020-2023)...")
    df = load_train()
    print(f"bars: {len(df):,}\n")

    # Strategy A — high-frequency: Bollinger mean reversion
    run_strategy(
        "A: Bollinger MR (high-freq)",
        df,
        BollingerMeanReversionTrigger(period=120, num_std=2.0),
        MeanReversionLabeler(period=120, num_std=2.0, max_hold=240),
        {"entry_period": 120, "exit_period": 60, "atr_period": 120, "ma_period": 120},
        feature_set=mean_reversion_feature_set(),
    )

    # Strategy B — low-frequency: trend-filtered breakout
    run_strategy(
        "B: Trend breakout (low-freq)",
        df,
        TrendFilteredBreakoutTrigger(entry_period=7200, trend_period=28800),
        TrailingExitLabeler(trail_period=2880),
        {"entry_period": 7200, "exit_period": 2880, "atr_period": 7200, "ma_period": 28800},
    )


if __name__ == "__main__":
    main()
