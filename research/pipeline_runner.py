#!/usr/bin/env python
"""
Config-driven research pipeline runner.

Loads a YAML config, assembles trigger + features + labeler via
dependency injection, builds an ML dataset, trains/evaluates a model,
and saves outputs. The runner code is NEVER modified to add a new alpha
— only the YAML config and registered components change.

Usage:
    python research/pipeline_runner.py config/research_pipeline.yaml
    python research/pipeline_runner.py config/example_turtle_vol.yaml
"""

import argparse
import importlib
import os
import sys
import time
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

# Project root for cross-package imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def instantiate(spec: Dict[str, Any]) -> Any:
    """
    Instantiate a component from a config spec.

    spec format:
        {"type": "module.path.ClassName", "params": {"key": value, ...}}
    """
    type_path = spec["type"]
    params = spec.get("params", {})

    module_path, class_name = type_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**params)


def instantiate_features(feature_specs: list) -> Any:
    """Instantiate a list of feature specs → CompositeFeature if multiple."""
    from research.features import CompositeFeature

    instances = [instantiate(s) for s in feature_specs]
    if len(instances) == 1:
        return instances[0]
    return CompositeFeature(instances)


def load_data(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Load and filter data per config."""
    data_cfg = cfg["data"]
    source = data_cfg["source"]

    if source.endswith(".parquet"):
        df = pd.read_parquet(source)
    elif source.endswith(".csv"):
        df = pd.read_csv(source)
    else:
        raise ValueError(f"Unsupported data source: {source}")

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])

    # Date range filter
    if "date_range" in data_cfg:
        start, end = data_cfg["date_range"]
        if "datetime" in df.columns:
            mask = (df["datetime"] >= start) & (df["datetime"] < end)
            df = df.loc[mask].copy()

    return df


def build_dataset(
    df: pd.DataFrame,
    trigger: Any,
    features: Any,
    labeler: Any,
    verbose: bool = True,
) -> tuple:
    """Build ML dataset from assembled components."""
    from research.dataset_builder import build_ml_dataset
    from research.features import add_indicators

    # Precompute indicators (required for lookahead-free features)
    df = add_indicators(df)

    # Build event series from trigger
    events = trigger.generate_signals(df)

    # Build feature matrix from features at event positions
    feature_df = features.compute(df, events)

    # Build labels from labeler
    labels_df = labeler.compute_labels(df, events)

    # Combine
    X = feature_df.join(labels_df, how="left")
    meta = {
        "n_events": len(X),
        "feature_names": list(feature_df.columns),
    }

    return X, meta


def run_evaluation(
    X: pd.DataFrame,
    cfg: Dict[str, Any],
    out_dir: str,
) -> Dict[str, Any]:
    """Train model and evaluate."""
    from research.evaluator import ModelEvaluator

    model_cfg = cfg.get("model", {})
    evaluator = ModelEvaluator(
        model_params=model_cfg.get("params"),
        target=model_cfg.get("target", "y_norm"),
    )

    X_np, y_np, feature_names = evaluator.prepare_features(X)

    cv_folds = model_cfg.get("cv_folds", 1)
    train_split = model_cfg.get("train_split", 0.8)
    n = len(X_np)

    # --- Purged time-series cross-validation ---
    if cv_folds > 1:
        # Compute event-level gap from label horizon (bars) ÷ avg bars/event.
        # Prevents label-window overlap across train/val boundary.
        event_indices = X.index.values  # bar positions in original kline data
        avg_bars_per_event = float(np.mean(np.diff(event_indices))) if len(event_indices) > 1 else 1.0

        labeler_cfg = cfg.get("labeler", {})
        label_horizon = labeler_cfg.get("params", {}).get("horizon", 14400)
        gap = max(1, int(np.ceil(label_horizon / max(avg_bars_per_event, 1.0))))

        folds = evaluator.time_series_split(X_np, y_np, n_splits=cv_folds, gap=gap)

        print(f"\n  Purged Time-Series CV: {cv_folds} folds")
        print(f"    avg bars/event={avg_bars_per_event:.0f}  "
              f"label horizon={label_horizon} bars  "
              f"→ event-level gap={gap}")
        ic_values = []
        for i, (X_tr, X_val, y_tr, y_val) in enumerate(folds):
            model = evaluator.train_model(X_tr, y_tr)
            y_pred = model.predict(X_val)
            ic = evaluator.evaluate_rank_ic(y_val, y_pred)["ic"]
            ic_values.append(ic)
            print(f"    Fold {i+1}: Train={len(X_tr):,}  Val={len(X_val):,}  IC={ic:.4f}")

        mean_ic = float(np.mean(ic_values))
        std_ic = float(np.std(ic_values, ddof=1)) if len(ic_values) > 1 else 0.0
        print(f"\n  Mean IC: {mean_ic:.4f} ± {std_ic:.4f}  "
              f"[{'PASS' if abs(mean_ic) > evaluator.ic_threshold else 'FAIL'}]")

        # Train final model on all data for saving
        model = evaluator.train_model(X_np, y_np)

        ic_result = {"ic": mean_ic, "p_value": 0.0, "pass": abs(mean_ic) > evaluator.ic_threshold}
        decile_result = {"spread": 0.0, "monotonic": False}

    else:
        # --- Single chronological split (no shuffle, preserves time order) ---
        split_idx = int(n * train_split)
        X_train, y_train = X_np[:split_idx], y_np[:split_idx]
        X_test, y_test = X_np[split_idx:], y_np[split_idx:]

        print(f"\n  Chronological split: Train={len(X_train):,}  Test={len(X_test):,}")

        model = evaluator.train_model(X_train, y_train)
        y_pred = model.predict(X_test)

        ic_result = evaluator.evaluate_rank_ic(y_test, y_pred)
        decile_result = evaluator.evaluate_decile_spread(y_test, y_pred)

        print(f"\n  Spearman Rank IC: {ic_result['ic']:.4f} (p={ic_result['p_value']:.4f})  "
              f"[{'PASS' if ic_result['pass'] else 'FAIL'}]")
        print(f"  Decile spread: {decile_result['spread']:+.4f}  "
              f"monotonic={decile_result['monotonic']}  "
              f"[{'PASS' if decile_result['spread'] > 0 and decile_result['monotonic'] else 'FAIL'}]")

    # Save
    prefix = model_cfg.get("output_prefix", "xgb")
    evaluator.save(out_dir, prefix=prefix, feature_names=feature_names)
    print(f"\n  Model + features saved to {out_dir}/")

    return {
        "ic": ic_result,
        "decile": decile_result,
        "feature_names": feature_names,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Config-driven research pipeline runner"
    )
    parser.add_argument(
        "config", help="Path to YAML config file"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override output directory (default: research/outputs)",
    )
    args = parser.parse_args()

    print("=" * 64)
    print("Research Pipeline Runner")
    print("=" * 64)

    # Load config
    cfg = load_yaml(args.config)
    strategy_name = cfg.get("strategy", {}).get("name", "unnamed")
    print(f"\nStrategy: {strategy_name}")

    # Determine output directory
    out_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "outputs"
    )

    t0 = time.perf_counter()

    # 1. Load data
    print("\n[1/4] Loading data...")
    df = load_data(cfg)

    # 2. Instantiate components (DI — no hardcoded types)
    print("[2/4] Assembling components from config...")
    trigger = instantiate(cfg["trigger"])
    features = instantiate_features(cfg["features"])
    labeler = instantiate(cfg["labeler"])

    print(f"  Trigger : {type(trigger).__name__}")
    print(f"  Features: {type(features).__name__}")
    print(f"  Labeler : {type(labeler).__name__}")

    # 3. Build dataset
    print("[3/4] Building dataset...")
    X, meta = build_dataset(df, trigger, features, labeler)
    print(f"  Events: {meta['n_events']:,}")
    print(f"  Features: {meta['feature_names']}")

    # Save dataset
    os.makedirs(out_dir, exist_ok=True)
    dataset_path = os.path.join(out_dir, f"X_{strategy_name}.parquet")
    X.to_parquet(dataset_path, index=True)
    print(f"  Dataset saved: {dataset_path}")

    # 4. Train + evaluate
    if "model" in cfg:
        print("[4/4] Training + evaluating model...")
        results = run_evaluation(X, cfg, out_dir)
    else:
        print("[4/4] No model config — skipping training.")

    elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 64}")
    print(f"Done in {elapsed:.1f}s")
    print(f"{'=' * 64}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
