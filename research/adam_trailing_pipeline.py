"""
Adam trailing-stop pipeline: Phase 1 -> 3b with the trailing-stop + fixed-loss logic.

  Phase 1 : AdamBreakoutTrigger (breakout entry, long + short)
  Phase 2 : 8 features + TrailingExitLabeler -> ML dataset (label = profitable)
  Phase 3 : XGBoost classifier -> predict "profitable trailing trade", AUC
  Phase 3b: Mode A sweep (period x trail_period, no ML), Mode B sweep
            (ml_threshold with fixed config + classifier), walk-forward

Backtest uses TrailingExitLabeler + risk_pct=0.10 (fixed 10% risk per the
actual initial-stop distance).
"""

import os
import sys
import time

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from research.param_sweep import run_parameter_sweep

IND_FULL = {"entry_period": 43200, "exit_period": 14400, "atr_period": 43200,
            "vol_period": 1440, "ma_period": 288000}


def _trailing_backtest_target(raw_data, period=43200, trail_period=14400, risk_pct=0.10,
                              ml_threshold=0.0, ml_model=None, ml_feature_cols=None,
                              indicator_params=None, max_dd_pct=None, risk_window=129600, **kwargs):
    from research.triggers.adam_breakout import AdamBreakoutTrigger
    from research.labeling import TrailingExitLabeler
    from research.backtest import lightweight_backtest
    from execution.sizers import FixedRiskSizer

    trigger = AdamBreakoutTrigger(period=int(period), signed=True)
    exit_labeler = TrailingExitLabeler(trail_period=int(trail_period))
    if indicator_params is None:
        indicator_params = {"exit_period": int(trail_period)}
    sizer = FixedRiskSizer(risk_pct=float(risk_pct), max_leverage=100.0)
    risk_manager = None
    if max_dd_pct is not None:
        from execution.risk_managers import MaxDrawdownRiskManager
        risk_manager = MaxDrawdownRiskManager(max_dd_pct=float(max_dd_pct))
    result = lightweight_backtest(
        raw_data, trigger=trigger, exit_labeler=exit_labeler,
        position_sizer=sizer, risk_manager=risk_manager, risk_window=int(risk_window),
        indicator_params=indicator_params,
        ml_model=ml_model, ml_threshold=float(ml_threshold), ml_feature_cols=ml_feature_cols,
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


def _build_Xy(df, trail_period=14400):
    from research.features import add_indicators, default_feature_set
    from research.triggers.adam_breakout import AdamBreakoutTrigger
    from research.labeling import TrailingExitLabeler
    from research.evaluator import ModelEvaluator

    ind = add_indicators(df, entry_period=43200, exit_period=trail_period, atr_period=43200,
                         vol_period=1440, ma_period=288000)
    trigger = AdamBreakoutTrigger(period=43200, signed=True)
    events = trigger.generate_signals(ind)
    feature_df = default_feature_set().compute(ind, events)
    labels_df = TrailingExitLabeler(trail_period=trail_period).compute_labels(ind, events)
    X = feature_df.join(labels_df, how="left")

    evaluator = ModelEvaluator(target="label")
    X_np, _, feature_names = evaluator.prepare_features(X)
    r_mult = X["r_multiple"].values
    y_np = (r_mult > 2.0).astype(np.float32)   # 1 = big winner (> 2R)
    return X_np, y_np, feature_names, len(X)


def main():
    from sklearn.metrics import roc_auc_score

    print("=" * 78)
    print("Adam Trailing-Stop Pipeline — Phase 1 → 3b")
    print("=" * 78)

    path = os.path.join(_PROJECT_ROOT, "data", "historical_data", "BTCUSDT_1m_full.parquet")
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    train = df[df["datetime"] < "2024-01-01"].reset_index(drop=True)
    test = df[df["datetime"] >= "2024-01-01"].reset_index(drop=True)
    print(f"\n資料：train {len(train):,} bars (2020-2023) / test {len(test):,} bars (2024-now)")

    # ---- Phase 1 ----
    from research.triggers.adam_breakout import AdamBreakoutTrigger
    from research.features import add_indicators
    trigger = AdamBreakoutTrigger(period=43200, signed=True)
    ev_tr = trigger.generate_signals(add_indicators(train, **IND_FULL))
    ev_te = trigger.generate_signals(add_indicators(test, **IND_FULL))
    n_tr = int((np.asarray(ev_tr.values) != 0).sum())
    n_te = int((np.asarray(ev_te.values) != 0).sum())
    print(f"\n[Phase 1] AdamBreakoutTrigger(period=43200): train {n_tr} 事件 / test {n_te} 事件")

    # ---- Phase 2 ----
    X_tr, y_tr, feature_names, n_rows_tr = _build_Xy(train, trail_period=14400)
    X_te, y_te, _, n_rows_te = _build_Xy(test, trail_period=14400)
    print(f"\n[Phase 2] 資料集：{len(feature_names)} 個特徵，train {n_rows_tr} 列 / test {n_rows_te} 列")
    print(f"  label 分佈（train）：profitable={int((y_tr == 1).sum())} ({100*(y_tr == 1).mean():.1f}%)  "
          f"loss={int((y_tr == 0).sum())}")
    print(f"  特徵：{feature_names}")

    # ---- Phase 3 ----
    from research.evaluator import ModelEvaluator
    evaluator = ModelEvaluator(target="label")
    model = evaluator.train_classifier(X_tr, y_tr)
    y_prob = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, y_prob)
    print(f"\n[Phase 3] XGBoost classifier → 預測「這筆 trailing 交易會不會賺」")
    print(f"  AUC-ROC (test 2024-now): {auc:.4f}")

    # ---- Phase 3b: Mode A (sweep period x trail_period, no ML) ----
    print("\n" + "=" * 78)
    print("[Phase 3b Mode A] 掃 period × trail_period（無 ML），risk_pct=10%")
    print("=" * 78)
    grid_A = {
        "period": [10080, 20160, 43200, 64800, 86400],        # 7/14/30/45/60 天
        "trail_period": [4320, 7200, 14400, 21600, 28800],    # 3/5/10/15/20 天
    }
    t0 = time.perf_counter()
    res_A = run_parameter_sweep(target_func=_trailing_backtest_target, param_grid=grid_A,
                                raw_data=train, n_jobs=1, rank_by="sharpe")
    res_A = res_A[~res_A.get("error", pd.Series(False, index=res_A.index)).astype(bool)] \
        if "error" in res_A.columns else res_A
    print(f"  ({len(res_A)} combos, {time.perf_counter()-t0:.0f}s)")
    cols = ["period", "trail_period", "sharpe", "max_dd_pct", "total_return_pct", "n_trades", "win_rate"]
    print(res_A[cols].head(10).to_string(index=False))

    print("\n  Top-5 在 TEST 驗證：")
    for _, row in res_A.head(5).iterrows():
        r = _trailing_backtest_target(test, period=int(row["period"]), trail_period=int(row["trail_period"]))
        print(f"    period={int(row['period']):>6} trail={int(row['trail_period']):>6} | "
              f"train_sharpe={float(row['sharpe']):>6.2f} → test_sharpe={r['sharpe']:>6.2f} "
              f"ret={r['total_return_pct']:>6.1f}% mdd={r['max_dd_pct']:>5.1f}%")

    # ---- Phase 3b: Mode B (sweep ml_threshold, fixed config + classifier) ----
    print("\n" + "=" * 78)
    print("[Phase 3b Mode B] 掃 ml_threshold（固定 period=43200/trail=14400 + classifier）")
    print("=" * 78)
    grid_B = {"ml_threshold": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}
    res_B = run_parameter_sweep(
        target_func=_trailing_backtest_target, param_grid=grid_B, raw_data=train,
        fixed_kwargs={"period": 43200, "trail_period": 14400, "risk_pct": 0.10,
                      "ml_model": model, "ml_feature_cols": feature_names, "indicator_params": IND_FULL},
        n_jobs=1, rank_by="sharpe",
    )
    print(res_B[["ml_threshold", "sharpe", "n_trades", "win_rate", "total_return_pct", "max_dd_pct"]].to_string(index=False))

    best_thr = float(res_B.iloc[0]["ml_threshold"])
    r_ml = _trailing_backtest_target(test, period=43200, trail_period=14400, risk_pct=0.10,
                                     ml_model=model, ml_feature_cols=feature_names,
                                     ml_threshold=best_thr, indicator_params=IND_FULL)
    r_no = _trailing_backtest_target(test, period=43200, trail_period=14400, risk_pct=0.10)
    print(f"\n  最佳 threshold={best_thr} 在 TEST（out-of-sample）：")
    print(f"    帶 ML  : sharpe={r_ml['sharpe']:.2f} win={r_ml['win_rate']*100:.1f}% trades={r_ml['n_trades']} "
          f"ret={r_ml['total_return_pct']:.1f}% mdd={r_ml['max_dd_pct']:.1f}%")
    print(f"    不帶 ML: sharpe={r_no['sharpe']:.2f} win={r_no['win_rate']*100:.1f}% trades={r_no['n_trades']} "
          f"ret={r_no['total_return_pct']:.1f}% mdd={r_no['max_dd_pct']:.1f}%")

    # ---- Walk-forward ----
    print("\n" + "=" * 78)
    print("[Walk-Forward] 每年重訓 → 測下一年（帶 ML vs 不帶 ML）")
    print("=" * 78)
    windows = [
        ("2020-01-01", "2023-01-01", "2023-01-01", "2024-01-01"),
        ("2020-01-01", "2024-01-01", "2024-01-01", "2025-01-01"),
        ("2020-01-01", "2025-01-01", "2025-01-01", "2026-01-01"),
        ("2020-01-01", "2026-01-01", "2026-01-01", "2026-10-03"),
    ]
    for tr_s, tr_e, te_s, te_e in windows:
        tr_df = df[(df["datetime"] >= tr_s) & (df["datetime"] < tr_e)].reset_index(drop=True)
        te_df = df[(df["datetime"] >= te_s) & (df["datetime"] < te_e)].reset_index(drop=True)
        Xa, ya, fn, _ = _build_Xy(tr_df, trail_period=14400)
        m = ModelEvaluator(target="label").train_classifier(Xa, ya)
        Xb, yb, _, _ = _build_Xy(te_df, trail_period=14400)
        # evaluate AUC on this test year
        prob = m.predict_proba(Xb)[:, 1]
        a = roc_auc_score(yb, prob)
        r_ml_w = _trailing_backtest_target(te_df, period=43200, trail_period=14400, risk_pct=0.10,
                                           ml_model=m, ml_feature_cols=fn, ml_threshold=0.4,
                                           indicator_params=IND_FULL)
        r_no_w = _trailing_backtest_target(te_df, period=43200, trail_period=14400, risk_pct=0.10)
        print(f"  test {te_s[:4]}: AUC={a:.3f} | 不帶ML sharpe={r_no_w['sharpe']:>6.2f} ret={r_no_w['total_return_pct']:>6.1f}% "
              f"({r_no_w['n_trades']:>3}t) | 帶ML sharpe={r_ml_w['sharpe']:>6.2f} ret={r_ml_w['total_return_pct']:>6.1f}% "
              f"({r_ml_w['n_trades']:>3}t)")

    # ---- Phase 4: 風控門檻掃描 ----
    print("\n" + "=" * 78)
    print("[Phase 4] 風控門檻掃描（MaxDrawdownRiskManager, rolling peak 90 天）on TEST")
    print("=" * 78)
    for max_dd in [None, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
        label = "無風控" if max_dd is None else f"{max_dd*100:.0f}%"
        r = _trailing_backtest_target(test, period=43200, trail_period=14400, risk_pct=0.10,
                                      max_dd_pct=max_dd, risk_window=129600)
        print(f"  {label:>6}: sharpe={r['sharpe']:>6.2f}  ret={r['total_return_pct']:>7.1f}%  "
              f"mdd={r['max_dd_pct']:>5.1f}%  trades={r['n_trades']:>3}  win={r['win_rate']*100:.0f}%")

    print("\n" + "=" * 78)
    print("Done")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
