"""
Phase 3b — Parameter Sweep for the Adam breakout strategy.

Mode A (no ML): sweep the trigger lookback (period) + triple-barrier exit params
    (upper/lower/horizon), rank by Sharpe.
Mode B (with ML): sweep the entry threshold `ml_threshold` with a FIXED trigger
    (the period the classifier was trained on, 43200) + the trained classifier.

Usage:
    python research/adam_sweep.py
"""

import os
import sys
import json
import time

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from research.param_sweep import run_parameter_sweep


# ============================================================
# Sweep target (module-level for pickling)
# ============================================================

def _adam_backtest_target(
    raw_data: pd.DataFrame,
    period: int = 43200,
    upper_barrier: float = 0.05,
    lower_barrier: float = -0.02,
    horizon: int = 1440,
    ml_threshold: float = 0.0,
    ml_model=None,
    ml_feature_cols=None,
    indicator_params=None,
    **kwargs,
) -> dict:
    from research.triggers.adam_breakout import AdamBreakoutTrigger
    from research.labeling import TripleBarrierLabeler
    from research.backtest import lightweight_backtest

    trigger = AdamBreakoutTrigger(period=int(period), signed=True)
    exit_labeler = TripleBarrierLabeler(
        upper_barrier=float(upper_barrier), lower_barrier=float(lower_barrier),
        horizon=int(horizon), barrier_mode="pct",
    )
    result = lightweight_backtest(
        raw_data,
        trigger=trigger,
        exit_labeler=exit_labeler,
        indicator_params=indicator_params,
        ml_model=ml_model,
        ml_threshold=float(ml_threshold),
        ml_feature_cols=ml_feature_cols,
        stop_pct=abs(float(lower_barrier)),
        verbose=False,
    )
    pf = result["profit_factor"]
    return {
        "sharpe": float(result["sharpe"]),
        "win_rate": float(result["win_rate"]),
        "total_return_pct": float(result["total_return_pct"]),
        "max_dd_pct": float(result["max_dd_pct"]),
        "n_trades": int(result["n_trades"]),
        "profit_factor": float(pf) if isinstance(pf, float) else 999.0,
    }


# ============================================================
# Data
# ============================================================

def _load_segments():
    path = os.path.join(_PROJECT_ROOT, "data", "historical_data", "BTCUSDT_1m_full.parquet")
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    train = df[df["datetime"] < "2024-01-01"].reset_index(drop=True)
    test = df[df["datetime"] >= "2024-01-01"].reset_index(drop=True)
    return train, test


# ============================================================
# Main
# ============================================================

def main() -> int:
    print("=" * 72)
    print("Phase 3b — Adam Parameter Sweep")
    print("=" * 72)

    train, test = _load_segments()
    print(f"\nTrain (2020-2023): {len(train):,} bars   Test (2024-now): {len(test):,} bars")

    # ---------------------------------------------------------
    # Mode A: sweep trigger + barrier (no ML)
    # ---------------------------------------------------------
    grid_A = {
        "period": [10080, 14400, 20160, 28800, 43200, 64800, 86400],   # 7/10/14/20/30/45/60 days
        "upper_barrier": [0.03, 0.05, 0.08],
        "lower_barrier": [-0.01, -0.02, -0.03],
        "horizon": [720, 1440, 2880, 4320],                            # 0.5/1/2/3 days
    }
    n_A = 7 * 3 * 3 * 4
    print(f"\n[Mode A] Sweeping {n_A} combos (period x upper x lower x horizon) on TRAIN...")
    t0 = time.perf_counter()
    res_A = run_parameter_sweep(
        target_func=_adam_backtest_target,
        param_grid=grid_A,
        raw_data=train,
        n_jobs=1,
        rank_by="sharpe",
    )
    res_A = res_A[~res_A.get("error", pd.Series(False, index=res_A.index)).astype(bool)] \
        if "error" in res_A.columns else res_A
    print(f"  sweep done in {time.perf_counter()-t0:.0f}s")

    cols = ["period", "upper_barrier", "lower_barrier", "horizon",
            "sharpe", "max_dd_pct", "total_return_pct", "n_trades", "win_rate"]
    print("\n  Top 10 by Sharpe (train):")
    print(res_A[cols].head(10).to_string(index=False))

    # validate top-5 on test
    print("\n  Validating top-5 on TEST (out-of-sample)...")
    print(f"  {'period':>7} {'upper':>6} {'lower':>7} {'horizon':>7} | "
          f"{'train_sharpe':>12} {'test_sharpe':>11} {'test_ret':>9} {'test_MDD':>9}")
    for _, row in res_A.head(5).iterrows():
        r = _adam_backtest_target(
            test,
            period=int(row["period"]), upper_barrier=float(row["upper_barrier"]),
            lower_barrier=float(row["lower_barrier"]), horizon=int(row["horizon"]),
        )
        print(f"  {int(row['period']):>7} {float(row['upper_barrier']):>6} "
              f"{float(row['lower_barrier']):>7} {int(row['horizon']):>7} | "
              f"{float(row['sharpe']):>12.2f} {r['sharpe']:>11.2f} "
              f"{r['total_return_pct']:>8.1f}% {r['max_dd_pct']:>8.1f}%")

    # ---------------------------------------------------------
    # Mode B: sweep ml_threshold with fixed trigger + trained model
    # ---------------------------------------------------------
    print("\n" + "=" * 72)
    print("[Mode B] Sweep ml_threshold (fixed period=43200 + trained classifier)")
    print("=" * 72)

    out_dir = os.path.join(_PROJECT_ROOT, "research", "outputs")
    model_path = os.path.join(out_dir, "adam_breakout_model.json")
    feat_path = os.path.join(out_dir, "adam_breakout_features.json")
    if not os.path.exists(model_path) or not os.path.exists(feat_path):
        print(f"  [SKIP] classifier not found ({model_path}). Run pipeline_runner first.")
    else:
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        with open(feat_path, "r") as f:
            feature_cols = json.load(f)
        print(f"  loaded classifier + {len(feature_cols)} features")

        fixed_B = {
            "period": 43200,
            "upper_barrier": 0.05,
            "lower_barrier": -0.02,
            "horizon": 1440,
            "ml_model": model,
            "ml_feature_cols": feature_cols,
            "indicator_params": {"entry_period": 43200, "exit_period": 21600,
                                 "atr_period": 43200, "vol_period": 1440, "ma_period": 288000},
        }
        grid_B = {"ml_threshold": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}
        t0 = time.perf_counter()
        res_B = run_parameter_sweep(
            target_func=_adam_backtest_target,
            param_grid=grid_B,
            raw_data=train,
            fixed_kwargs=fixed_B,
            n_jobs=1,
            rank_by="sharpe",
        )
        print(f"  sweep done in {time.perf_counter()-t0:.0f}s")
        print("\n  ml_threshold sweep (train), sorted by Sharpe:")
        print(res_B[["ml_threshold", "sharpe", "n_trades", "win_rate", "total_return_pct", "max_dd_pct"]]
              .to_string(index=False))

        best_thr = float(res_B.iloc[0]["ml_threshold"])
        r_test_ml = _adam_backtest_target(
            test, period=43200, upper_barrier=0.05, lower_barrier=-0.02, horizon=1440,
            ml_model=model, ml_feature_cols=feature_cols, ml_threshold=best_thr,
            indicator_params=fixed_B["indicator_params"],
        )
        r_test_no_ml = _adam_backtest_target(
            test, period=43200, upper_barrier=0.05, lower_barrier=-0.02, horizon=1440,
        )
        print(f"\n  Best threshold={best_thr} validated on TEST (out-of-sample):")
        print(f"    with ML filter : sharpe={r_test_ml['sharpe']:.2f}, win_rate={r_test_ml['win_rate']*100:.1f}%, "
              f"trades={r_test_ml['n_trades']}, ret={r_test_ml['total_return_pct']:.1f}%, "
              f"mdd={r_test_ml['max_dd_pct']:.1f}%")
        print(f"    without ML     : sharpe={r_test_no_ml['sharpe']:.2f}, win_rate={r_test_no_ml['win_rate']*100:.1f}%, "
              f"trades={r_test_no_ml['n_trades']}, ret={r_test_no_ml['total_return_pct']:.1f}%, "
              f"mdd={r_test_no_ml['max_dd_pct']:.1f}%")

    print("\n" + "=" * 72)
    print("Done")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
