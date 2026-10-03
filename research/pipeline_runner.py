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


def load_data(cfg: Dict[str, Any], range_key: str = "date_range") -> pd.DataFrame:
    """Load and filter data per config.

    Args:
        cfg: Full config dict.
        range_key: Which data.<key> holds the [start, end] date filter
            ("date_range", "train_range", or "test_range").
    """
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
    if range_key in data_cfg:
        start, end = data_cfg[range_key]
        if "datetime" in df.columns:
            mask = (df["datetime"] >= start) & (df["datetime"] < end)
            df = df.loc[mask].copy()

    return df


def compute_period_stats(X: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Per-year breakdown of a dataset (events, win rate, avg profit).

    Args:
        X: Dataset DataFrame whose index = event bar positions in `df`.
        df: Original kline DataFrame (has a 'datetime' column).

    Returns:
        DataFrame with one row per year: events, win_rate, avg_y_norm,
        median_y_norm, avg_raw_return.
    """
    if len(X) == 0 or "datetime" not in df.columns:
        return pd.DataFrame()

    event_positions = X.index.values
    years = df["datetime"].iloc[event_positions].dt.year.values

    stats = []
    for year in sorted(np.unique(years)):
        mask = years == year
        sub = X.iloc[mask]
        n = int(mask.sum())
        row = {"year": int(year), "events": n}
        if "label" in sub.columns:      # classification: hit take-profit
            row["win_rate"] = float((sub["label"] == 1).mean()) if n else 0.0
        if "y_norm" in sub.columns:     # regression
            row["win_rate"] = float((sub["y_norm"] > 0).mean()) if n else 0.0
            row["avg_y_norm"] = float(sub["y_norm"].mean()) if n else 0.0
            row["median_y_norm"] = float(sub["y_norm"].median()) if n else 0.0
        if "actual_return" in sub.columns:
            row["avg_actual_return"] = float(sub["actual_return"].mean()) if n else 0.0
        if "raw_return" in sub.columns:
            row["avg_raw_return"] = float(sub["raw_return"].mean()) if n else 0.0
        stats.append(row)

    return pd.DataFrame(stats)


def build_dataset(
    df: pd.DataFrame,
    trigger: Any,
    features: Any,
    labeler: Any,
    indicator_params: Dict[str, Any] = None,
    verbose: bool = True,
) -> tuple:
    """Build ML dataset from assembled components.

    Args:
        indicator_params: dict of periods passed to add_indicators()
            (e.g. entry_period, atr_period, vol_period, ma_period).
            This is the single source of truth for indicator periods —
            features are pure readers and never recompute them.
    """
    from research.dataset_builder import build_ml_dataset
    from research.features import add_indicators

    # Precompute indicators with config-driven periods (required for
    # lookahead-free features; periods must match the strategy).
    df = add_indicators(df, **(indicator_params or {}))

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
    X_train: pd.DataFrame,
    cfg: Dict[str, Any],
    out_dir: str,
    X_test: pd.DataFrame = None,
) -> Dict[str, Any]:
    """Train model and evaluate.

    Args:
        X_train: Training dataset (index = event bar positions).
        cfg: Full config dict.
        out_dir: Output directory.
        X_test: Optional out-of-sample test dataset. If provided, trains on
            X_train and evaluates on X_test. Otherwise falls back to
            CV / chronological split within X_train.
    """
    from research.evaluator import ModelEvaluator

    model_cfg = cfg.get("model", {})
    target = model_cfg.get("target", "y_norm")
    is_clf = target == "label"

    evaluator = ModelEvaluator(
        model_params=model_cfg.get("params"),
        target=target,
    )

    def _binarize(y: np.ndarray) -> np.ndarray:
        # TripleBarrier label: 1=take-profit, -1=stop-loss, 0=timeout.
        return (y == 1.0).astype(np.float32)

    def _fit_predict(X_tr, y_tr, X_va):
        if is_clf:
            model = evaluator.train_classifier(X_tr, y_tr)
            return model, model.predict_proba(X_va)[:, 1]
        model = evaluator.train_model(X_tr, y_tr)
        return model, model.predict(X_va)

    X_np, y_np, feature_names = evaluator.prepare_features(X_train)
    if is_clf:
        y_np = _binarize(y_np)

    # --- Out-of-sample evaluation (explicit train/test datasets) ---
    if X_test is not None:
        X_test_np, y_test, _ = evaluator.prepare_features(X_test)
        if is_clf:
            y_test = _binarize(y_test)

        print(f"\n  Out-of-sample split: Train={len(X_np):,}  Test={len(X_test_np):,}")

        model, y_pred = _fit_predict(X_np, y_np, X_test_np)

        if is_clf:
            auc = evaluator.evaluate_auc_roc(y_test, y_pred)
            print(f"\n  AUC-ROC (test): {auc['auc']:.4f}  "
                  f"[{'PASS' if auc.get('pass') else 'FAIL'}]")
            result = {"auc": auc}
        else:
            ic_result = evaluator.evaluate_rank_ic(y_test, y_pred)
            decile_result = evaluator.evaluate_decile_spread(y_test, y_pred)
            print(f"\n  Spearman Rank IC (test): {ic_result['ic']:.4f} (p={ic_result['p_value']:.4f})  "
                  f"[{'PASS' if ic_result['pass'] else 'FAIL'}]")
            print(f"  Decile spread (test): {decile_result['spread']:+.4f}  "
                  f"monotonic={decile_result['monotonic']}  "
                  f"[{'PASS' if decile_result['spread'] > 0 and decile_result['monotonic'] else 'FAIL'}]")
            if decile_result.get("quantile_means"):
                print(f"    quantile means: {[f'{m:+.2f}' for m in decile_result['quantile_means']]}")
            result = {"ic": ic_result, "decile": decile_result}

    else:
        cv_folds = model_cfg.get("cv_folds", 1)
        train_split = model_cfg.get("train_split", 0.8)
        n = len(X_np)

        # --- Purged time-series cross-validation ---
        if cv_folds > 1:
            event_indices = X_train.index.values
            avg_bars_per_event = float(np.mean(np.diff(event_indices))) if len(event_indices) > 1 else 1.0
            labeler_cfg = cfg.get("labeler", {})
            label_horizon = labeler_cfg.get("params", {}).get("horizon", 14400)
            gap = max(1, int(np.ceil(label_horizon / max(avg_bars_per_event, 1.0))))

            folds = evaluator.time_series_split(X_np, y_np, n_splits=cv_folds, gap=gap)

            print(f"\n  Purged Time-Series CV: {cv_folds} folds")
            print(f"    avg bars/event={avg_bars_per_event:.0f}  "
                  f"label horizon={label_horizon} bars  "
                  f"→ event-level gap={gap}")
            scores = []
            for i, (X_tr, X_val, y_tr, y_val) in enumerate(folds):
                model, y_pred = _fit_predict(X_tr, y_tr, X_val)
                s = (evaluator.evaluate_auc_roc(y_val, y_pred)["auc"] if is_clf
                     else evaluator.evaluate_rank_ic(y_val, y_pred)["ic"])
                scores.append(s)
                print(f"    Fold {i+1}: Train={len(X_tr):,}  Val={len(X_val):,}  "
                      f"{'AUC' if is_clf else 'IC'}={s:.4f}")

            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
            if is_clf:
                passed = mean_score > 0.55
                print(f"\n  Mean AUC: {mean_score:.4f} ± {std_score:.4f}  [{'PASS' if passed else 'FAIL'}]")
                result = {"auc": {"auc": mean_score, "pass": passed}}
            else:
                passed = abs(mean_score) > evaluator.ic_threshold
                print(f"\n  Mean IC: {mean_score:.4f} ± {std_score:.4f}  [{'PASS' if passed else 'FAIL'}]")
                result = {"ic": {"ic": mean_score, "p_value": 0.0, "pass": passed},
                          "decile": {"spread": 0.0, "monotonic": False}}

            model = evaluator.train_classifier(X_np, y_np) if is_clf else evaluator.train_model(X_np, y_np)

        else:
            # --- Single chronological split ---
            split_idx = int(n * train_split)
            X_tr, y_tr = X_np[:split_idx], y_np[:split_idx]
            X_te, y_te = X_np[split_idx:], y_np[split_idx:]

            print(f"\n  Chronological split: Train={len(X_tr):,}  Test={len(X_te):,}")

            model, y_pred = _fit_predict(X_tr, y_tr, X_te)

            if is_clf:
                auc = evaluator.evaluate_auc_roc(y_te, y_pred)
                print(f"\n  AUC-ROC: {auc['auc']:.4f}  [{'PASS' if auc.get('pass') else 'FAIL'}]")
                result = {"auc": auc}
            else:
                ic_result = evaluator.evaluate_rank_ic(y_te, y_pred)
                decile_result = evaluator.evaluate_decile_spread(y_te, y_pred)
                print(f"\n  Spearman Rank IC: {ic_result['ic']:.4f} (p={ic_result['p_value']:.4f})  "
                      f"[{'PASS' if ic_result['pass'] else 'FAIL'}]")
                print(f"  Decile spread: {decile_result['spread']:+.4f}  "
                      f"monotonic={decile_result['monotonic']}  "
                      f"[{'PASS' if decile_result['spread'] > 0 and decile_result['monotonic'] else 'FAIL'}]")
                result = {"ic": ic_result, "decile": decile_result}

    # Save
    prefix = model_cfg.get("output_prefix", "xgb")
    evaluator.save(out_dir, prefix=prefix, feature_names=feature_names)
    print(f"\n  Model + features saved to {out_dir}/")

    return {
        "ic": result.get("ic", {"ic": 0.0, "p_value": 1.0, "pass": False}),
        "decile": result.get("decile", {"spread": 0.0, "monotonic": False}),
        "auc": result.get("auc", {"auc": 0.5, "pass": False}),
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
    print("\n[1/5] Loading data...")
    indicator_params = cfg.get("indicators", {})
    data_cfg = cfg["data"]

    # 2. Instantiate components (DI — no hardcoded types)
    print("[2/5] Assembling components from config...")
    trigger = instantiate(cfg["trigger"])
    features = instantiate_features(cfg["features"])
    labeler = instantiate(cfg["labeler"])

    print(f"  Trigger : {type(trigger).__name__}")
    print(f"  Features: {type(features).__name__}")
    print(f"  Labeler : {type(labeler).__name__}")

    # 3. Build dataset(s)
    print("[3/5] Building dataset...")
    os.makedirs(out_dir, exist_ok=True)

    if "test_range" in data_cfg:
        # --- Out-of-sample: explicit train/test date ranges ---
        df_train = load_data(cfg, "train_range")
        df_test = load_data(cfg, "test_range")

        X_train, meta_tr = build_dataset(df_train, trigger, features, labeler, indicator_params)
        X_test, meta_te = build_dataset(df_test, trigger, features, labeler, indicator_params)

        print(f"  Train events: {meta_tr['n_events']:,}   Test events: {meta_te['n_events']:,}")
        print(f"  Features: {meta_tr['feature_names']}")

        X_train.to_parquet(os.path.join(out_dir, f"X_{strategy_name}_train.parquet"), index=True)
        X_test.to_parquet(os.path.join(out_dir, f"X_{strategy_name}_test.parquet"), index=True)

        # Per-year breakdown
        is_clf = cfg.get("model", {}).get("target", "y_norm") == "label"
        print("\n  Per-year breakdown:")
        for _, row in compute_period_stats(X_train, df_train).iterrows():
            line = f"  {row['year']:<6} {row['events']:>8} {row['win_rate']*100:>8.1f}%"
            if is_clf:
                line += f"  AvgRet {row.get('avg_actual_return', 0):+.4f}"
            else:
                line += f"  Avg_y {row['avg_y_norm']:>9.3f}  Med_y {row['median_y_norm']:>9.3f}  AvgRawRet {row.get('avg_raw_return', 0):+.4f}"
            print(line)
        for _, row in compute_period_stats(X_test, df_test).iterrows():
            line = f"  {row['year']:<6} {row['events']:>8} {row['win_rate']*100:>8.1f}%"
            if is_clf:
                line += f"  AvgRet {row.get('avg_actual_return', 0):+.4f}"
            else:
                line += f"  Avg_y {row['avg_y_norm']:>9.3f}  Med_y {row['median_y_norm']:>9.3f}  AvgRawRet {row.get('avg_raw_return', 0):+.4f}"
            print(line)

    else:
        # --- Single dataset (CV / chronological split) ---
        df = load_data(cfg)
        X_train, meta = build_dataset(df, trigger, features, labeler, indicator_params)
        X_test = None
        print(f"  Events: {meta['n_events']:,}")
        print(f"  Features: {meta['feature_names']}")
        X_train.to_parquet(os.path.join(out_dir, f"X_{strategy_name}.parquet"), index=True)

    # 4. Train + evaluate
    if "model" in cfg:
        print("\n[4/5] Training + evaluating model...")
        results = run_evaluation(X_train, cfg, out_dir, X_test)
    else:
        print("\n[4/5] No model config — skipping training.")

    elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 64}")
    print(f"Done in {elapsed:.1f}s")
    print(f"{'=' * 64}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
