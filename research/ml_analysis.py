"""
Phase 3: Fixed-Horizon Regression — XGBoost + Spearman Rank IC + Decile.

Flow:
  1. Load X_2023.parquet (train) + X_2024.parquet (test)
  2. Target: y_norm (continuous, forward return / daily ATR)
  3. Train XGBoost regressor on 2023
  4. Predict y_norm on 2024 (out-of-sample)
  5. Spearman Rank IC: correlation(predicted, actual y_norm) > 0.03
  6. Decile Mean y_norm: 5 quintiles, Q5 mean > Q1 mean, monotonic staircase

Usage:
    python research/ml_analysis.py
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ============================================================
# Config
# ============================================================

XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    verbosity=0,
)

IC_THRESHOLD = 0.03
N_QUINTILES = 5


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 64)
    print("Phase 3: Fixed-Horizon Regression — Spearman IC + Decile")
    print("=" * 64)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

    # ---- 1. Load datasets ----
    print("\n[1/5] Loading datasets...")
    X_train = pd.read_parquet(os.path.join(out_dir, "X_train.parquet"))
    X_test = pd.read_parquet(os.path.join(out_dir, "X_test.parquet"))

    train = X_train.copy()
    test = X_test.copy()

    print(f"  Train (2020-2023): {len(train):,} events")
    print(f"  Test  (2024):      {len(test):,} events")
    print(f"  Target y_norm: mean={train['y_norm'].mean():.3f}, std={train['y_norm'].std():.3f}")
    if len(train) < 500:
        print(f"  [WARN] Small sample: 20-day breakouts on 1 year = ~{len(train)} events. "
              f"Decile staircase may be noisy with small N.")

    # ---- 2. Prepare features ----
    print("\n[2/5] Preparing features...")
    exclude_cols = {
        "label", "barrier_hit", "exit_idx", "n_bars_held",
        "entry_price", "exit_price", "actual_return", "return_atr",
        "raw_return", "y_norm", "truncated",
        "event_time", "side",
    }
    feature_cols = [c for c in train.columns if c not in exclude_cols]
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    X_tr = np.nan_to_num(train[feature_cols].values.astype(np.float32), nan=0.0)
    y_tr = train["y_norm"].values.astype(np.float32)

    X_te = np.nan_to_num(test[feature_cols].values.astype(np.float32), nan=0.0)
    y_te = test["y_norm"].values.astype(np.float32)

    # ---- 3. Train XGBoost regressor ----
    print("\n[3/5] Training XGBoost regressor...")
    import xgboost as xgb
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_tr, y_tr, verbose=False)
    print(f"  Trained on {len(X_tr):,} samples")

    # Save model + feature list for backtest filter
    model_path = os.path.join(out_dir, "xgb_model.json")
    model.save_model(model_path)
    import json
    with open(os.path.join(out_dir, "xgb_features.json"), "w") as f:
        json.dump(feature_cols, f)
    print(f"  Model + feature list saved to {out_dir}")

    # ---- 4. Predict + Spearman Rank IC ----
    print("\n[4/5] Predicting + Spearman Rank IC...")
    test["pred_y"] = model.predict(X_te)

    valid = ~np.isnan(y_te)
    ic, ic_pval = spearmanr(test.loc[valid, "pred_y"], y_te[valid])
    ic_pass = abs(ic) > IC_THRESHOLD

    print(f"  Spearman Rank IC: {ic:.4f}  (p={ic_pval:.4f})")
    print(f"  {'PASS' if ic_pass else 'FAIL'}  (|rho| > {IC_THRESHOLD})")

    # ---- 5. Decile Mean y_norm ----
    print(f"\n[5/5] Decile Mean y_norm ({N_QUINTILES} quintiles)...")
    test["quintile"] = pd.qcut(
        test["pred_y"], q=N_QUINTILES, labels=False, duplicates="drop"
    )

    print(f"\n  {'Quintile':<12} {'N_Events':>8} {'Mean y_norm':>12} {'Med y_norm':>11} {'Prob_Range':>22}")
    print(f"  {'-'*12} {'-'*8} {'-'*12} {'-'*11} {'-'*22}")

    quintile_means = []
    for q in range(N_QUINTILES):
        mask = test["quintile"] == q
        if mask.sum() == 0:
            continue
        subset = test[mask]
        mean_y = subset["y_norm"].mean()
        med_y = subset["y_norm"].median()
        p_min = subset["pred_y"].min()
        p_max = subset["pred_y"].max()
        n = len(subset)
        quintile_means.append(mean_y)
        label = ""
        if q == 0: label = "(low) "
        elif q == N_QUINTILES - 1: label = "(high)"
        print(f"  Q{q+1} {label:<6} {n:>8} {mean_y:>11.4f} {med_y:>10.4f} "
              f"[{p_min:.3f}, {p_max:.3f}]")

    q1_mean = quintile_means[0]
    q5_mean = quintile_means[-1]
    spread = q5_mean - q1_mean
    monotonic = all(
        quintile_means[i] <= quintile_means[i+1]
        for i in range(len(quintile_means) - 1)
    )
    decile_pass = spread > 0 and monotonic
    print(f"\n  Q5 - Q1 spread: {spread:+.4f}  monotonic={monotonic}  {'PASS' if decile_pass else 'FAIL'}")

    # ---- Feature Importance ----
    print(f"\n{'=' * 64}")
    print("Feature Importance (XGBoost gain)")
    print(f"{'=' * 64}")
    importance = model.get_booster().get_score(importance_type="gain")
    imp_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for rank, (fname, gain) in enumerate(imp_sorted, 1):
        idx = int(fname.replace("f", ""))
        feat_name = feature_cols[idx] if idx < len(feature_cols) else fname
        print(f"  {rank:>2}. {feat_name:<25s} {gain:>10.1f}")

    # ---- Verdict ----
    print(f"\n{'=' * 64}")
    print("PHASE 3 VERDICT")
    print(f"{'=' * 64}")
    checks = [
        ("Spearman Rank IC > 0.03", ic_pass),
        ("Decile monotonic staircase", decile_pass),
    ]
    for desc, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}")

    all_pass = all(ok for _, ok in checks)
    if all_pass:
        print(f"\n  >> PHASE 3 PASSED — features predict forward return in ATR units.")
    else:
        print(f"\n  >> PHASE 3 NOT YET PASSED.")
    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    main()
