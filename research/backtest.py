"""
Vectorized Lightweight Backtester — pure pandas/numpy, ~50-200x faster than Backtrader.

Design:
  1. Pluggable: a BaseEventTrigger defines entries, a BaseLabeler defines exits.
  2. A single event-driven engine simulates position tracking (non-overlapping,
     one position at a time) and builds a mark-to-market equity curve.
  3. Metrics (Sharpe, drawdown, win rate) computed vectorized from the equity curve.

Shares add_indicators() from research.features — the indicator formulas
exactly mirror shared/core_logic/turtle_math.py.

Zero Backtrader dependency.  Suitable for large parameter sweeps.
"""

import os
import sys
import math
import numpy as np
import pandas as pd
from typing import Dict, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from research.features import add_indicators, default_feature_pipeline
from core.position_sizer import BasePositionSizer
from core.risk_manager import BaseRiskManager


# ============================================================
# Helper: extract features at a bar for ML filtering
# ============================================================

def _get_features_at_bar(df, bar_idx, feature_cols, pipeline):
    """Compute feature vector at a specific bar for ML prediction."""
    feat_dict = pipeline(df, bar_idx)
    if feat_dict is None:
        return None
    # Build array in the correct feature order
    vals = [feat_dict.get(c, 0.0) for c in feature_cols]
    return np.array([vals], dtype=np.float32)


# ============================================================
# 1. Main backtest function (pluggable)
# ============================================================

def lightweight_backtest(
    df: pd.DataFrame,
    trigger,                          # BaseEventTrigger — entry signals (-1/0/1)
    exit_labeler,                     # BaseLabeler — per-event exit (exit_idx/exit_price/barrier_hit)
    indicator_params: Optional[Dict] = None,
    ml_model=None,                    # optional entry filter (classifier or regressor)
    ml_threshold: float = 0.0,
    ml_feature_cols: Optional[list] = None,  # explicit feature order for the model
    position_sizer: Optional[BasePositionSizer] = None,
    risk_manager: Optional[BaseRiskManager] = None,
    initial_capital: float = 10000.0,
    commission: float = 0.0005,
    max_leverage: float = 20.0,
    risk_pct: float = 0.02,
    stop_pct: float = 0.02,
    verbose: bool = False,
) -> dict:
    """
    Pluggable event-driven backtest.

    Parameters
    ----------
    df : OHLCV DataFrame (open/high/low/close/volume), chronological (oldest first).
    trigger : BaseEventTrigger — generates entry events (-1=short, 1=long, 0=flat).
    exit_labeler : BaseLabeler — computes per-event exit (columns: exit_idx,
        exit_price, entry_price, barrier_hit, n_bars_held).
    indicator_params : dict passed to add_indicators() so the exit labeler (and
        the optional ML filter) see the right columns. Must match what the ML
        model was trained on when ml_model is used.
    ml_model : optional model to filter entries. Classifiers are scored via
        predict_proba()[:, 1]; regressors via predict().
    ml_threshold : minimum score to enter; entries below are skipped.
    position_sizer : BasePositionSizer, or None for inline risk-based sizing.
    risk_manager : BaseRiskManager, or None.
    initial_capital, commission, max_leverage, risk_pct, stop_pct : sizing/fee knobs.

    Returns
    -------
    dict with sharpe/win_rate/total_return_pct/max_dd_pct/n_trades/... plus
    equity_curve, trades_df, final_capital.
    """
    # --- 1. Indicators (for the exit labeler + optional ML filter) ---
    ind = add_indicators(df, **(indicator_params or {}))

    # --- 2. Entry events ---
    events = trigger.generate_signals(ind)
    sig_vals = np.asarray(events.values)
    event_idx = np.flatnonzero(sig_vals != 0)
    sides = sig_vals[event_idx].astype(int)
    n = len(ind)
    close = ind["close"].values

    # --- 3. Exits ---
    labels = exit_labeler.compute_labels(ind, events)
    if len(labels) == 0:
        empty = _compute_metrics(np.array([initial_capital]), [], initial_capital, 0)
        empty["equity_curve"] = np.array([initial_capital])
        empty["trades_df"] = pd.DataFrame()
        empty["final_capital"] = initial_capital
        return empty

    exit_idx_arr = labels["exit_idx"].values.astype(int)
    exit_price_arr = labels["exit_price"].values.astype(float)
    entry_price_arr = labels["entry_price"].values.astype(float)
    barrier_hit_arr = labels["barrier_hit"].values.astype(str)

    # --- 4. ML filter setup ---
    _ml_feature_cols = ml_feature_cols  # explicit order, else auto-detect (sorted)
    ml_feature_pipeline = None
    is_classifier = False
    if ml_model is not None:
        ml_feature_pipeline = default_feature_pipeline()
        is_classifier = hasattr(ml_model, "predict_proba")
        if _ml_feature_cols is None:
            sample = ml_feature_pipeline(ind, min(200, n - 1))
            _ml_feature_cols = sorted(sample.keys())

    # --- 5. Process events sequentially (non-overlapping) ---
    capital = initial_capital
    trades = []
    last_exit_idx = -1

    for k in range(len(event_idx)):
        ei = int(event_idx[k])
        side = int(sides[k])
        if ei <= last_exit_idx:
            continue  # overlapping entry — skip (one position at a time)

        xidx = int(exit_idx_arr[k])
        eprice = entry_price_arr[k]
        xprice = exit_price_arr[k]
        reason = barrier_hit_arr[k]

        # ML filter
        if ml_model is not None:
            feats = _get_features_at_bar(ind, ei, _ml_feature_cols, ml_feature_pipeline)
            if feats is None:
                continue
            if is_classifier:
                score = float(ml_model.predict_proba(feats)[0][1])
            else:
                score = float(ml_model.predict(feats)[0])
            if score <= ml_threshold:
                if verbose:
                    print(f"[{ind.index[ei]}] ML FILTER skip {side}, "
                          f"score={score:.3f} <= {ml_threshold}")
                continue

        # risk manager
        if risk_manager is not None:
            drawdown = (capital - initial_capital) / initial_capital
            if not risk_manager.check_risk_limits(
                {"current_drawdown": min(drawdown, 0.0), "current_position": 0.0}
            ):
                continue

        # sizing
        if position_sizer is not None:
            size = position_sizer.calculate_size(
                signal_strength=1.0,
                current_atr=1.0,
                account_equity=capital,
                entry_price=eprice,
            )
        else:
            risk = eprice * stop_pct
            if risk > 0 and capital > 0:
                size_risk = (capital * risk_pct) / risk
                size_cap = (capital * max_leverage) / eprice
                size = math.floor(min(size_risk, size_cap) * 1000) / 1000.0
            else:
                size = 0.0
        if size <= 0.0:
            continue

        # PnL
        raw_pnl = side * (xprice - eprice) * size
        fees = commission * (eprice + xprice) * size
        pnl = raw_pnl - fees

        trades.append({
            "entry_idx": ei,
            "exit_idx": xidx,
            "side": side,
            "entry_price": eprice,
            "exit_price": xprice,
            "size": size,
            "pnl": pnl,
            "entry_time": ind.index[ei],
            "exit_time": ind.index[xidx],
            "bars_held": xidx - ei,
            "exit_reason": reason,
            "pnl_pct": pnl / initial_capital * 100.0,
        })
        last_exit_idx = xidx

    # --- 6. Mark-to-market equity curve ---
    capital = initial_capital
    equity = []
    open_trade = None
    tptr = 0
    for i in range(n):
        if open_trade is not None and i >= open_trade["exit_idx"]:
            capital += open_trade["pnl"]
            open_trade = None
        if open_trade is None and tptr < len(trades) and trades[tptr]["entry_idx"] == i:
            open_trade = trades[tptr]
            tptr += 1
        unrealized = 0.0
        if open_trade is not None:
            unrealized = open_trade["side"] * (close[i] - open_trade["entry_price"]) * open_trade["size"]
        equity.append(capital + unrealized)

    equity_arr = np.array(equity)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["entry_idx", "exit_idx", "side", "entry_price", "exit_price", "size",
                 "pnl", "entry_time", "exit_time", "bars_held", "exit_reason", "pnl_pct"])

    # --- 7. Metrics ---
    trades_metrics = [
        {"pnl": t["pnl"], "bars_held": t["bars_held"],
         "entry_price": t["entry_price"], "exit_price": t["exit_price"], "size": t["size"]}
        for t in trades
    ]
    metrics = _compute_metrics(equity_arr, trades_metrics, initial_capital, warmup=20)

    metrics["equity_curve"] = equity_arr
    metrics["trades_df"] = trades_df
    metrics["final_capital"] = capital

    return metrics


# ============================================================
# 2. Metrics computation (pure vectorized)
# ============================================================

def _compute_metrics(
    equity: np.ndarray,
    trades: list,
    initial_capital: float,
    warmup: int,
) -> dict:
    """Compute performance metrics from equity curve and trade log."""
    total_return_pct = (equity[-1] / initial_capital - 1.0) * 100

    # Per-bar returns (skip warmup for Sharpe to avoid flat-start bias)
    returns = np.diff(equity[warmup:]) / equity[warmup:-1]

    # Annualized Sharpe (365 * 24 * 60 = 525600 minutes/year)
    periods_per_year = 365 * 24 * 60
    if len(returns) > 1:
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        sharpe = (mean_ret / std_ret) * np.sqrt(periods_per_year) if std_ret > 0 else 0.0
    else:
        sharpe = 0.0
        mean_ret = 0.0
        std_ret = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    drawdowns = (equity - peak) / peak * 100
    max_dd_pct = abs(np.min(drawdowns))  # stored as positive number

    # Trade statistics
    if trades:
        pnls = np.array([t["pnl"] for t in trades])
        n_trades = len(pnls)
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        n_wins = len(wins)
        n_losses = len(losses)
        win_rate = n_wins / n_trades if n_trades > 0 else 0.0

        profit_factor = (wins.sum() / abs(losses.sum())) if len(losses) > 0 and abs(losses.sum()) > 0 else float("inf")
        avg_trade_pnl = float(np.mean(pnls))
        avg_win = float(np.mean(wins)) if n_wins > 0 else 0.0
        avg_loss = float(np.mean(losses)) if n_losses > 0 else 0.0
        avg_bars_held = float(np.mean([t["bars_held"] for t in trades]))
        total_fees = sum(
            (t["entry_price"] + t["exit_price"]) * t["size"] * 0.0005
            for t in trades
        )
    else:
        n_trades = 0
        n_wins = 0
        n_losses = 0
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade_pnl = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        avg_bars_held = 0.0
        total_fees = 0.0

    return {
        "sharpe": round(sharpe, 4),
        "win_rate": round(win_rate, 4),
        "total_return_pct": round(total_return_pct, 2),
        "max_dd_pct": round(max_dd_pct, 2),
        "n_trades": n_trades,
        "n_wins": n_wins,
        "n_losses": n_losses,
        "profit_factor": round(profit_factor, 2) if np.isfinite(profit_factor) else "inf",
        "avg_trade_pnl": round(avg_trade_pnl, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "avg_bars_held": round(avg_bars_held, 1),
        "total_fees": round(total_fees, 2),
    }


# ============================================================
# __main__ demo
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vectorized Lightweight Backtester")
    parser.add_argument("--real", action="store_true",
                        help="Use real BTCUSDT data instead of synthetic")
    parser.add_argument("--year", type=int, default=2024,
                        help="Year to backtest (default: 2024)")
    parser.add_argument("--entry", type=int, default=20)
    parser.add_argument("--exit", type=int, default=10)
    parser.add_argument("--atr-period", type=int, default=20)
    parser.add_argument("--atr-mult", type=float, default=2.0)
    parser.add_argument("--capital", type=float, default=10000.0)
    parser.add_argument("--risk-pct", type=float, default=0.02)
    parser.add_argument("--ml-filter", action="store_true",
                        help="Use trained XGBoost model to filter entries")
    parser.add_argument("--ml-threshold", type=float, default=0.0,
                        help="Min predicted score to enter (default: 0)")
    args = parser.parse_args()

    from research.triggers.turtle_breakout import TurtleBreakoutTrigger
    from research.labeling import TurtleExitLabeler

    print("=" * 60)
    print("Vectorized Lightweight Backtester")
    print("=" * 60)

    if args.real:
        parquet_path = os.path.join(
            _PROJECT_ROOT, "data", "historical_data", "BTCUSDT_1m_full.parquet"
        )
        print(f"\n[1] Loading real BTCUSDT data ({args.year})...")
        df = pd.read_parquet(parquet_path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        start = f"{args.year}-01-01"
        end = f"{args.year + 1}-01-01"
        mask = (df["datetime"] >= start) & (df["datetime"] < end)
        df = df.loc[mask].copy()
        print(f"    {len(df):,} bars ({df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]})")
    else:
        from research.dataset_builder import make_synthetic_ohlcv
        print("\n[1] Generating synthetic OHLCV data (20,000 bars)...")
        df = make_synthetic_ohlcv(n_bars=20000, seed=42)

    # Load ML model if requested
    ml_model = None
    ml_threshold = 0.0
    if args.ml_filter:
        import xgboost as xgb
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
        model_path = os.path.join(out_dir, "xgb_model.json")
        if not os.path.exists(model_path):
            print("[ERROR] xgb_model.json not found. Run ml_analysis.py first.")
            sys.exit(1)
        ml_model = xgb.XGBRegressor()
        ml_model.load_model(model_path)
        ml_threshold = args.ml_threshold
        print(f"\n[ML] Loaded XGBoost model, threshold={ml_threshold}")

    trigger = TurtleBreakoutTrigger(
        entry_period=args.entry, atr_period=args.atr_period,
        atr_mult=args.atr_mult, intensity_threshold=0.0, signed=True,
    )
    exit_labeler = TurtleExitLabeler(
        exit_period=args.exit, atr_period=args.atr_period, atr_mult=args.atr_mult,
    )

    import time
    print(f"\n[2] Running backtest: entry={args.entry}, exit={args.exit}, "
          f"atr_period={args.atr_period}, atr_mult={args.atr_mult}, "
          f"capital={args.capital}, risk_pct={args.risk_pct}")
    t0 = time.perf_counter()
    result = lightweight_backtest(
        df,
        trigger=trigger,
        exit_labeler=exit_labeler,
        indicator_params={"entry_period": args.entry, "exit_period": args.exit,
                          "atr_period": args.atr_period},
        ml_model=ml_model,
        ml_threshold=ml_threshold,
        initial_capital=args.capital,
        risk_pct=args.risk_pct,
    )
    elapsed = time.perf_counter() - t0

    print(f"\n[3] Results ({elapsed:.3f}s):")
    print(f"    Sharpe:          {result['sharpe']}")
    print(f"    Win Rate:        {result['win_rate']*100:.1f}%")
    print(f"    Total Return:    {result['total_return_pct']:.2f}%")
    print(f"    Max Drawdown:    {result['max_dd_pct']:.2f}%")
    print(f"    Trades:          {result['n_trades']} (W:{result['n_wins']} L:{result['n_losses']})")
    print(f"    Profit Factor:   {result['profit_factor']}")
    print(f"    Avg Trade PnL:   ${result['avg_trade_pnl']}")
    print(f"    Avg Bars Held:   {result['avg_bars_held']}")
    print(f"    Final Capital:   ${result['final_capital']:.2f}")

    if len(result["trades_df"]) > 0:
        print(f"\n[4] Trade log (first 10 of {len(result['trades_df'])}):")
        pd.set_option("display.max_columns", 10)
        pd.set_option("display.width", 140)
        print(result["trades_df"].head(10).to_string())

    print("\n" + "=" * 60)
    print("Done")
    print("=" * 60)
