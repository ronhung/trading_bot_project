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


import argparse
import yaml
from core.strategy_factory import build_strategy_from_config
from core.labeler import BaseLabeler
import importlib


def _resolve_labeler(labeler_cfg: dict) -> BaseLabeler:
    l_type = labeler_cfg.get("type", "research.labeling.TrailingExitLabeler")
    l_params = labeler_cfg.get("params", {})
    if "." in l_type:
        mod_name, cls_name = l_type.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        cls = getattr(mod, cls_name)
    elif "trail" in l_type.lower():
        cls = TrailingExitLabeler
    else:
        cls = MeanReversionLabeler
    return cls(**l_params)


def run_from_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_source = cfg.get("data", {}).get("source", "data/historical_data/BTCUSDT_1m_full.parquet")
    if not os.path.isabs(data_source):
        data_source = os.path.join(_PROJECT_ROOT, data_source)

    df = pd.read_parquet(data_source)
    df["datetime"] = pd.to_datetime(df.get("datetime", pd.to_datetime(df.get("open_time", 0), unit="ms")))

    train_range = cfg.get("data", {}).get("train_range", None)
    if train_range and len(train_range) == 2:
        df = df[(df["datetime"] >= train_range[0]) & (df["datetime"] < train_range[1])].reset_index(drop=True)
        print(f"Loaded train data: {len(df):,} bars ({train_range[0]} to {train_range[1]})")
    else:
        print(f"Loaded train data: {len(df):,} bars")

    wrapper, meta = build_strategy_from_config(cfg)
    trigger = wrapper._trigger
    features = wrapper._feature
    ind_params = meta["indicator_params"]

    labeler_cfg = cfg.get("labeler", {"type": "research.labeling.TrailingExitLabeler", "params": {"trail_period": ind_params["exit_period"]}})
    labeler = _resolve_labeler(labeler_cfg)

    ind = add_indicators(df, **ind_params)
    events = trigger.generate_signals(ind)
    feature_df = features.compute(ind, events)
    labels_df = labeler.compute_labels(ind, events)
    X = feature_df.join(labels_df, how="left")

    model_cfg = cfg.get("model", {})
    target = model_cfg.get("target", "r_multiple")
    model_type = model_cfg.get("type", "xgboost_classifier")
    output_prefix = model_cfg.get("output_prefix", cfg.get("strategy", {}).get("name", "strategy"))

    evaluator = ModelEvaluator(target=target)
    X_np, y_np, feature_names = evaluator.prepare_features(X)
    n = len(X_np)
    split = int(n * 0.8)

    print(f"\n[Phase 3 Training] {output_prefix} | Total Events: {n} (Train: {split}, Val: {n - split})")
    if model_type == "xgboost_classifier":
        y_bin = (y_np > 0).astype(int)
        val_model = evaluator.train_classifier(X_np[:split], y_bin[:split])
        prob = val_model.predict_proba(X_np[split:])[:, 1]
        auc = evaluator.evaluate_auc_roc(y_bin[split:], prob)
        print(f"  AUC-ROC: {auc['auc']:.3f} (pass={auc.get('pass')})")

        # Threshold trade-offs
        for thr in (0.50, 0.52, 0.55, 0.60):
            keep = prob >= thr
            pct = keep.mean() * 100
            m_r = y_np[split:][keep].mean() if keep.sum() else 0.0
            print(f"    threshold={thr:.2f} -> keep={pct:5.1f}% events, avg r_multiple={m_r:+.3f}")

        # Train full model for deployment
        full_model = evaluator.train_classifier(X_np, y_bin)
    else:
        val_model = evaluator.train_model(X_np[:split], y_np[:split])
        y_pred = val_model.predict(X_np[split:])
        ic = evaluator.evaluate_rank_ic(y_np[split:], y_pred)
        decile = evaluator.evaluate_decile_spread(y_np[split:], y_pred)
        print(f"  Rank IC: {ic['ic']:+.4f} (p-value: {ic['p_value']:.4f}, pass={ic['pass']})")
        print(f"  Decile Spread: {decile['spread']:+.5f} (monotonic={decile['monotonic']})")
        full_model = evaluator.train_model(X_np, y_np)

    evaluator._model = full_model
    evaluator._feature_names = feature_names
    out_dir = os.path.join(_PROJECT_ROOT, "research", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    saved = evaluator.save(out_dir, prefix=output_prefix)
    print(f"  [Artifacts Saved] Model & features saved to: {saved}\n")
    return evaluator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="Path to strategy YAML config")
    args = ap.parse_args()

    if args.config:
        run_from_config(args.config)
        return

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
