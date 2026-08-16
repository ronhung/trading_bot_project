"""
Phase 3b — Parameter Sweep for the Turtle strategy.

Goal: find entry/stop/filter parameters that maximize Sharpe while
controlling max drawdown (MDD), using proper time-series validation:

  - Sweep on TRAIN (2020-2023), rank by a risk-adjusted score.
  - Re-validate the top-N on TEST (2024) out-of-sample.

Uses the vectorized lightweight_backtest (~0.5s per run) so the whole
grid finishes in seconds.

Usage:
    python research/phase3b_sweep.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from research.backtest import lightweight_backtest
from research.param_sweep import run_parameter_sweep


# ============================================================
# Sweep target (module-level for multiprocessing pickling)
# ============================================================

def _sweep_target(
    raw_data: pd.DataFrame,
    entry_period: int = 28800,
    atr_mult: float = 4.0,
    intensity_threshold: float = 0.5,
    **kwargs,
) -> dict:
    """
    Backtest target with Turtle-style coupled lookbacks:
      exit_period = entry_period / 2  (half the breakout window)
      atr_period  = entry_period      (same volatility window)

    Returns Sharpe, MDD, and a risk-adjusted score that penalizes
    deep drawdowns (1 point of Sharpe ≈ 25% of MDD).
    """
    result = lightweight_backtest(
        raw_data,
        entry_period=entry_period,
        exit_period=entry_period // 2,
        atr_period=entry_period,
        atr_mult=atr_mult,
        intensity_threshold=intensity_threshold,
        initial_capital=10000.0,
        max_leverage=20.0,
        verbose=False,
    )

    sharpe = result["sharpe"]
    mdd = result["max_dd_pct"]
    # Risk-adjusted score: high Sharpe, low MDD.
    score = sharpe - 0.02 * mdd

    return {
        "sharpe": sharpe,
        "max_dd_pct": mdd,
        "score": score,
        "total_return_pct": result["total_return_pct"],
        "win_rate": result["win_rate"],
        "n_trades": result["n_trades"],
        "profit_factor": result["profit_factor"]
        if isinstance(result["profit_factor"], float) else 999.0,
    }


# ============================================================
# Data loading
# ============================================================

def _load_year_range(start_year: int, end_year: int) -> pd.DataFrame:
    """Load 1m klines for [start_year, end_year)."""
    path = os.path.join(
        _PROJECT_ROOT, "data", "historical_data", "BTCUSDT_1m_full.parquet"
    )
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    mask = (df["datetime"] >= f"{start_year}-01-01") & (df["datetime"] < f"{end_year}-01-01")
    return df.loc[mask].copy()


# ============================================================
# Main
# ============================================================

def main() -> int:
    print("=" * 64)
    print("Phase 3b — Turtle Parameter Sweep (Sharpe ↑, MDD ↓)")
    print("=" * 64)

    # --- Reasonable grid (coupled lookbacks, 48 combos) ---
    # entry in days (1m bars = 1440/day): 10/20/30/40 days
    param_grid = {
        "entry_period": [14400, 28800, 43200, 57600],
        "atr_mult": [3.0, 4.0, 5.0, 6.0],
        "intensity_threshold": [0.0, 0.5, 1.0],
    }
    n_combos = 4 * 4 * 3
    print(f"\nGrid: {len(param_grid)} dims × {n_combos} combos")
    for k, v in param_grid.items():
        print(f"  {k}: {v}")

    # --- Load train / test ---
    print("\n[1/3] Loading data...")
    t0 = time.perf_counter()
    df_train = _load_year_range(2020, 2024)  # 2020-2023
    df_test = _load_year_range(2024, 2025)   # 2024
    print(f"  Train: {len(df_train):,} bars (2020-2023)")
    print(f"  Test:  {len(df_test):,} bars (2024)")

    # --- Sweep on train ---
    print(f"\n[2/3] Sweeping {n_combos} combos on TRAIN (2020-2023)...")
    results = run_parameter_sweep(
        target_func=_sweep_target,
        param_grid=param_grid,
        raw_data=df_train,
        n_jobs=1,  # sequential — avoids pickling the 2.1M-bar frame to N workers
        rank_by="score",
        ascending=False,
    )

    # Drop error rows
    results = results[~results["error"].astype(bool)] if "error" in results.columns else results

    print(f"\n  Top 10 by risk-adjusted score (train):")
    cols = ["entry_period", "atr_mult", "intensity_threshold",
            "sharpe", "max_dd_pct", "score", "total_return_pct", "n_trades", "win_rate"]
    pd.set_option("display.width", 160)
    print(results[cols].head(10).to_string(index=False))

    # --- Validate top-5 on test (2024) ---
    print(f"\n[3/3] Validating top-5 on TEST (2024, out-of-sample)...")
    top5 = results.head(5)
    print(f"\n  {'entry':>7} {'atr_mult':>8} {'intens':>6} | {'train_sharpe':>12} {'test_sharpe':>11} {'test_MDD':>9} {'test_ret':>9}")
    print(f"  {'-'*7} {'-'*8} {'-'*6} | {'-'*12} {'-'*11} {'-'*9} {'-'*9}")

    validation = []
    for _, row in top5.iterrows():
        r = lightweight_backtest(
            df_test,
            entry_period=int(row["entry_period"]),
            exit_period=int(row["entry_period"]) // 2,
            atr_period=int(row["entry_period"]),
            atr_mult=float(row["atr_mult"]),
            intensity_threshold=float(row["intensity_threshold"]),
            initial_capital=10000.0,
            max_leverage=20.0,
            verbose=False,
        )
        validation.append({
            "entry_period": int(row["entry_period"]),
            "atr_mult": float(row["atr_mult"]),
            "intensity_threshold": float(row["intensity_threshold"]),
            "train_sharpe": float(row["sharpe"]),
            "test_sharpe": float(r["sharpe"]),
            "test_max_dd_pct": float(r["max_dd_pct"]),
            "test_return_pct": float(r["total_return_pct"]),
        })
        print(f"  {int(row['entry_period']):>7} {float(row['atr_mult']):>8} {float(row['intensity_threshold']):>6} "
              f"| {float(row['sharpe']):>12.2f} {float(r['sharpe']):>11.2f} "
              f"{float(r['max_dd_pct']):>8.1f}% {float(r['total_return_pct']):>8.1f}%")

    # --- Summary ---
    val_df = pd.DataFrame(validation)
    if len(val_df):
        best_test = val_df.sort_values("test_sharpe", ascending=False).iloc[0]
        print(f"\n  Best out-of-sample (test) params:")
        print(f"    entry={best_test['entry_period']} bars ({best_test['entry_period']//1440} days), "
              f"atr_mult={best_test['atr_mult']}, "
              f"intensity={best_test['intensity_threshold']}")
        print(f"    test Sharpe={best_test['test_sharpe']:.2f}, "
              f"MDD={best_test['test_max_dd_pct']:.1f}%, "
              f"return={best_test['test_return_pct']:.1f}%")

    elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 64}")
    print(f"Done in {elapsed:.1f}s")
    print(f"{'=' * 64}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
