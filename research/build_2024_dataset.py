"""
Build 2024 ML pretrain dataset (X, y) from BTCUSDT 1-minute K-lines.

Uses the research modules to:
  1. Load and filter 2024 data
  2. Precompute indicators
  3. Detect Turtle breakout entry events
  4. Compute features at each event
  5. Apply triple-barrier labeling
  6. Save X.parquet + metadata.json

Usage:
    python research/build_2024_dataset.py
"""

import os
import sys
import json
import time
import pandas as pd

# --- path setup (mirrors project convention) ---
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from research.features import add_indicators, default_feature_pipeline
from research.dataset_builder import (
    build_ml_dataset,
    make_turtle_breakout_trigger,
)


def main():
    t_start = time.perf_counter()

    # ---- 1. Load & filter 2024 data ----
    parquet_path = os.path.join(
        _PROJECT_ROOT, "data", "historical_data", "BTCUSDT_1m_full.parquet"
    )
    print(f"[1/6] Loading {parquet_path} ...")
    df = pd.read_parquet(parquet_path)
    print(f"      Full dataset: {len(df):,} bars ({df['datetime'].min()} to {df['datetime'].max()})")

    # Filter to 2024
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        mask = (df["datetime"] >= "2024-01-01") & (df["datetime"] < "2025-01-01")
        df = df.loc[mask].copy()
    print(f"      2024 subset: {len(df):,} bars")

    # ---- 2. Precompute indicators ----
    print(f"\n[2/6] Precomputing indicators (entry=20, exit=10, atr=20) ...")
    t0 = time.perf_counter()
    df = add_indicators(df, entry_period=20, exit_period=10, atr_period=20, vol_period=20)
    print(f"      Done in {time.perf_counter() - t0:.1f}s. Columns: {list(df.columns[-10:])}")

    # ---- 3. Build event trigger ----
    print(f"\n[3/6] Building turtle breakout trigger ...")
    trigger = make_turtle_breakout_trigger(
        entry_period=20,
        atr_period=20,
        atr_mult=2.0,
        intensity_threshold=0.0,
        signed=True,
    )

    # ---- 4. Build feature pipeline ----
    print(f"[4/6] Building feature pipeline (12 features) ...")
    pipeline = default_feature_pipeline()

    # ---- 5. Build ML dataset ----
    labeling_config = {
        "upper_barrier": 0.02,   # +2% take-profit
        "lower_barrier": -0.01,  # -1% stop-loss
        "horizon": 288,          # max ~4.8 hours at 1m
    }
    print(f"[5/6] Building ML dataset with {labeling_config} ...")
    t0 = time.perf_counter()
    X, meta = build_ml_dataset(df, trigger, pipeline, labeling_config, verbose=True)
    elapsed = time.perf_counter() - t0
    print(f"      Dataset built in {elapsed:.1f}s ({len(X):,} events)")

    # ---- 6. Export ----
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # Save X (features + labels) as parquet
    x_path = os.path.join(out_dir, "X_2024.parquet")
    print(f"\n[6/6] Exporting ...")
    X.to_parquet(x_path, index=True)
    print(f"      -> {x_path} ({os.path.getsize(x_path) / 1e6:.1f} MB)")

    # Save metadata as JSON
    meta_path = os.path.join(out_dir, "meta_2024.json")
    # Convert numpy types for JSON serialization
    meta_clean = {}
    for k, v in meta.items():
        if isinstance(v, dict):
            meta_clean[k] = {str(k2): float(v2) if hasattr(v2, "item") else v2 for k2, v2 in v.items()}
        elif hasattr(v, "item"):
            meta_clean[k] = float(v)
        else:
            meta_clean[k] = v
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_clean, f, indent=2, default=str)
    print(f"      -> {meta_path}")

    # ---- Summary ----
    total_time = time.perf_counter() - t_start
    print(f"\n{'=' * 60}")
    print(f"2024 ML Dataset Complete")
    print(f"{'=' * 60}")
    print(f"  Rows (events):   {len(X):,}")
    print(f"  Features:        {len(meta['feature_names'])}")
    print(f"  Labels:")
    for label, rate in sorted(meta["label_rates"].items()):
        name = {1: "Take-Profit (+2%)", -1: "Stop-Loss (-1%)", 0: "Timeout (288 bars)"}[label]
        print(f"    {name:25s}: {rate*100:5.1f}% ({meta['label_counts'][label]:,} events)")
    print(f"  Mean return:     {meta['mean_actual_return']:.4f}")
    print(f"  Mean bars held:  {meta['mean_n_bars_held']:.1f}")
    print(f"  Total time:      {total_time:.1f}s")
    print(f"  Output:")
    print(f"    {x_path}")
    print(f"    {meta_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
