"""
Build 2023 (train) and 2024 (test) ML datasets with ATR-based barriers.

Uses the research modules to:
  - Load and filter data by year
  - Precompute indicators (including 200MA trend filter)
  - Detect Turtle breakout entry events
  - Compute 13 features at each event
  - Apply ATR-based triple-barrier labeling (TP = 2*ATR, SL = 1*ATR)
  - Save X_train.parquet + X_test.parquet

Usage:
    python research/build_train_test_dataset.py
"""

import os
import sys
import json
import time
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from research.features import add_indicators, default_feature_pipeline
from research.dataset_builder import (
    build_ml_dataset,
    make_turtle_breakout_trigger,
)


def build_dataset_for_year(year: int):
    """Load, filter, and build ML dataset for a single year or range (e.g. 2023 or '2020-2023')."""
    t0 = time.perf_counter()

    # --- Load ---
    parquet_path = os.path.join(
        _PROJECT_ROOT, "data", "historical_data", "BTCUSDT_1m_full.parquet"
    )
    df = pd.read_parquet(parquet_path)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Support year ranges: "2020-2023" or single year "2024"
    year_str = str(year)
    if "-" in year_str:
        y_start, y_end = year_str.split("-")
        start = f"{y_start}-01-01"
        end = f"{int(y_end)+1}-01-01"
    else:
        start = f"{year}-01-01"
        end = f"{int(year)+1}-01-01"

    mask = (df["datetime"] >= start) & (df["datetime"] < end)
    df = df.loc[mask].copy()
    n_bars = len(df)
    label = f"{year}"
    print(f"  [{label}] {n_bars:,} bars ({df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]})")

    # --- Indicators (20-day Turtle params + daily ATR for normalization) ---
    df = add_indicators(df, entry_period=28800, exit_period=14400, atr_period=28800, vol_period=20)

    # --- Trigger (20-day breakout, filter weak breakouts) ---
    trigger = make_turtle_breakout_trigger(
        entry_period=28800, atr_period=28800, atr_mult=4.0,
        intensity_threshold=0.5, signed=True,
    )
    pipeline = default_feature_pipeline()  # 14 features (includes trend_filter)

    # --- Fixed-horizon labeling (10-day forward return / daily ATR) ---
    labeling_config = {
        "method": "fixed_horizon",
        "horizon": 14400,  # 10 days forward
    }

    X, meta = build_ml_dataset(df, trigger, pipeline, labeling_config, verbose=False)

    elapsed = time.perf_counter() - t0
    print(f"  [{year}] {len(X):,} events, {len(meta['feature_names'])} features, "
          f"{elapsed:.1f}s")

    # Print label distribution
    rates = meta["label_rates"]
    print(f"  [{year}] Labels: TP={rates.get(1,0)*100:.1f}%  SL={rates.get(-1,0)*100:.1f}%  "
          f"Timeout={rates.get(0,0)*100:.1f}%")

    return X, meta


def main():
    t_start = time.perf_counter()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("Building 2020-2023 (train) + 2024 (test) ML datasets")
    print("  Method: fixed-horizon, 10-day forward return / daily ATR")
    print("  Entry: 20-day Donchian breakout, intensity >= 0.5")
    print("  Features: 14 (incl. trend_filter 200MA)")
    print("=" * 60)

    for year, role in [("2020-2023", "train"), (2024, "test")]:
        print(f"\n--- {year} ({role}) ---")
        X, meta = build_dataset_for_year(year)

        # Save (use role for filename: train/test)
        fname = f"X_{role}.parquet"
        x_path = os.path.join(out_dir, fname)
        X.to_parquet(x_path, index=True)
        print(f"  [{year}] -> {os.path.basename(x_path)} "
              f"({os.path.getsize(x_path) / 1e6:.1f} MB)")

        # Save metadata
        meta_path = os.path.join(out_dir, f"meta_{year}.json")
        meta_clean = {}
        for k, v in meta.items():
            if isinstance(v, dict):
                meta_clean[k] = {
                    str(k2): float(v2) if hasattr(v2, "item") else v2
                    for k2, v2 in v.items()
                }
            elif hasattr(v, "item"):
                meta_clean[k] = float(v)
            elif isinstance(v, list):
                meta_clean[k] = [str(x) for x in v]
            else:
                meta_clean[k] = v
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_clean, f, indent=2, default=str)

    total_time = time.perf_counter() - t_start
    print(f"\n{'=' * 60}")
    print(f"Done in {total_time:.1f}s")
    print(f"Train: research/outputs/X_train.parquet (2020-2023)")
    print(f"Test:  research/outputs/X_test.parquet (2024)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
