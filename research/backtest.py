"""
Vectorized Lightweight Backtester — pure pandas/numpy, ~50-200x faster than Backtrader.

Design:
  1. Signals computed once for ALL bars via vectorized pandas operations.
  2. A tight state-machine loop simulates position tracking.
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
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from research.features import add_indicators, default_feature_pipeline


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
# 1. Main backtest function
# ============================================================

def lightweight_backtest(
    df: pd.DataFrame,
    entry_period: int = 20,
    exit_period: int = 10,
    atr_period: int = 20,
    atr_mult: float = 2.0,
    intensity_threshold: float = 0.0,
    initial_capital: float = 10000.0,
    risk_pct: float = 0.02,
    commission: float = 0.0005,
    max_leverage: float = 20.0,
    verbose: bool = False,
    ml_model=None,
    ml_threshold: float = 0.0,
) -> dict:
    """
    Pure vectorized turtle-trend-following backtest.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with columns: open, high, low, close.
        Sorted chronologically (oldest first).
    entry_period, exit_period, atr_period : int
        Strategy lookback parameters.
    atr_mult : float
        ATR multiplier for stop-loss distance.
    intensity_threshold : float
        Minimum breakout intensity to enter (0.0 = any breakout).
    initial_capital : float
        Starting capital in quote currency (USDT).
    risk_pct : float
        Fraction of capital risked per trade (0.02 = 2%).
    commission : float
        Fee per trade as fraction of notional (0.0005 = 5 bps).
    verbose : bool
        If True, prints trade log.

    Returns
    -------
    dict with keys:
      sharpe, win_rate, total_return_pct, max_dd_pct,
      n_trades, profit_factor, avg_trade_pnl, avg_bars_held,
      final_capital, equity_curve, trades_df
    """
    # --- Ensure indicators are computed ---
    required_cols = {"entry_high", "entry_low", "exit_high", "exit_low", "atr"}
    if not required_cols.issubset(df.columns):
        df = add_indicators(df, entry_period=entry_period,
                            exit_period=exit_period,
                            atr_period=atr_period)

    # --- ML filter setup ---
    ml_feature_cols = []
    ml_feature_pipeline = None
    if ml_model is not None:
        ml_feature_pipeline = default_feature_pipeline()
        # Detect feature columns from the pipeline
        sample = ml_feature_pipeline(df, max(entry_period, atr_period) + 2)
        ml_feature_cols = sorted(sample.keys())

    n = len(df)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    entry_high = df["entry_high"].values
    entry_low = df["entry_low"].values
    exit_high = df["exit_high"].values
    exit_low = df["exit_low"].values
    atr = df["atr"].values

    # --- Step 1: Vectorized signal computation ---
    # Mirrors turtle_math.py exactly
    valid_atr = np.where((~np.isnan(atr)) & (atr > 0), atr, 1.0)

    long_intensity = (close - entry_high) / valid_atr
    short_intensity = (entry_low - close) / valid_atr

    long_entry_mask = (close > entry_high) & (long_intensity > intensity_threshold)
    short_entry_mask = (close < entry_low) & (short_intensity > intensity_threshold)
    # Exit only where there's no entry (matches turtle_math.py elif priority)
    exit_long_mask = (close < exit_low) & ~long_entry_mask & ~short_entry_mask
    exit_short_mask = (close > exit_high) & ~long_entry_mask & ~short_entry_mask

    # Combine into single signal array
    # 0=hold, 1=long_entry, -1=short_entry, 2=exit_long, -2=exit_short
    signals = np.zeros(n, dtype=int)
    signals[long_entry_mask] = 1
    signals[short_entry_mask] = -1
    signals[exit_long_mask] = 2
    signals[exit_short_mask] = -2

    # Entry priority over exit when both trigger on same bar (only relevant when flat)
    # Exit priority when in position already handled by state machine

    # Pre-computed stop levels (vectorized)
    long_stop = close - atr * atr_mult
    short_stop = close + atr * atr_mult

    # --- Step 2: State machine ---
    warmup = max(entry_period, atr_period) + 1

    position = 0       # 0=flat, 1=long, -1=short
    entry_price = 0.0
    stop_price = 0.0
    size = 0.0
    entry_bar = 0
    capital = initial_capital
    equity = [capital]

    trades: List[dict] = []

    for i in range(warmup, n):
        sig = signals[i]
        c = close[i]
        h = high[i]
        l = low[i]

        if position == 0:
            if sig == 1:  # long entry
                # ML filter: skip if model predicts below threshold
                if ml_model is not None:
                    feats = _get_features_at_bar(df, i, ml_feature_cols, ml_feature_pipeline)
                    if feats is not None:
                        pred = float(ml_model.predict(feats)[0])
                        if pred <= ml_threshold:
                            if verbose:
                                print(f"[{df.index[i]}] ML FILTER: skip LONG, pred_y={pred:.2f} <= {ml_threshold}")
                            continue

                entry_price = c
                stop_price = long_stop[i]
                risk = abs(entry_price - stop_price)
                if risk > 0 and capital > 0:
                    size_risk = (capital * risk_pct) / risk
                    # Cap at max leverage (Binance USD-M: 20x)
                    size_leverage_cap = (capital * max_leverage) / entry_price
                    size_new = math.floor(min(size_risk, size_leverage_cap) * 1000) / 1000.0
                else:
                    size_new = 0.0
                # Skip entry if size rounds to zero (account too small for this risk)
                if size_new <= 0.0:
                    continue
                position = 1
                size = size_new
                entry_bar = i

            elif sig == -1:  # short entry
                # ML filter: skip if model predicts below threshold
                if ml_model is not None:
                    feats = _get_features_at_bar(df, i, ml_feature_cols, ml_feature_pipeline)
                    if feats is not None:
                        pred = float(ml_model.predict(feats)[0])
                        if pred <= ml_threshold:
                            if verbose:
                                print(f"[{df.index[i]}] ML FILTER: skip SHORT, pred_y={pred:.2f} <= {ml_threshold}")
                            continue

                entry_price = c
                stop_price = short_stop[i]
                risk = abs(entry_price - stop_price)
                if risk > 0 and capital > 0:
                    size_risk = (capital * risk_pct) / risk
                    size_leverage_cap = (capital * max_leverage) / entry_price
                    size_new = math.floor(min(size_risk, size_leverage_cap) * 1000) / 1000.0
                else:
                    size_new = 0.0
                if size_new <= 0.0:
                    continue
                position = -1
                size = size_new
                entry_bar = i

        elif position == 1:  # holding long
            exit_flag = False
            exit_price = c
            reason = "signal"

            # Check stop-loss first (intra-bar low touches stop)
            if l <= stop_price:
                exit_flag = True
                exit_price = min(c, stop_price)  # conservative: take worse price
                reason = "stop"
            elif sig == 2:  # exit signal
                exit_flag = True
                exit_price = c
                reason = "signal"

            if exit_flag:
                # PnL with commission on both sides
                raw_pnl = (exit_price - entry_price) * size
                fees = commission * (entry_price + exit_price) * size
                pnl = raw_pnl - fees
                capital += pnl

                trades.append({
                    "entry_time": df.index[entry_bar],
                    "exit_time": df.index[i],
                    "side": "long",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "stop_price": stop_price,
                    "size": size,
                    "pnl": pnl,
                    "pnl_pct": pnl / initial_capital * 100,
                    "bars_held": i - entry_bar,
                    "exit_reason": reason,
                })

                if verbose:
                    print(f"[{df.index[i]}] CLOSE LONG: entry={entry_price:.1f} "
                          f"exit={exit_price:.1f} pnl={pnl:.2f} reason={reason}")

                position = 0

        elif position == -1:  # holding short (symmetric)
            exit_flag = False
            exit_price = c
            reason = "signal"

            if h >= stop_price:
                exit_flag = True
                exit_price = max(c, stop_price)  # conservative: take worse price
                reason = "stop"
            elif sig == -2:  # exit short signal
                exit_flag = True
                exit_price = c
                reason = "signal"

            if exit_flag:
                raw_pnl = (entry_price - exit_price) * size
                fees = commission * (entry_price + exit_price) * size
                pnl = raw_pnl - fees
                capital += pnl

                trades.append({
                    "entry_time": df.index[entry_bar],
                    "exit_time": df.index[i],
                    "side": "short",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "stop_price": stop_price,
                    "size": size,
                    "pnl": pnl,
                    "pnl_pct": pnl / initial_capital * 100,
                    "bars_held": i - entry_bar,
                    "exit_reason": reason,
                })

                if verbose:
                    print(f"[{df.index[i]}] CLOSE SHORT: entry={entry_price:.1f} "
                          f"exit={exit_price:.1f} pnl={pnl:.2f} reason={reason}")

                position = 0

        # Update equity curve even when holding (mark-to-market)
        if position == 1:
            unrealized = (c - entry_price) * size
        elif position == -1:
            unrealized = (entry_price - c) * size
        else:
            unrealized = 0.0
        equity.append(capital + unrealized)

    # --- Step 3: Compute metrics ---
    equity_arr = np.array(equity)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["entry_time", "exit_time", "side", "entry_price", "exit_price",
                 "stop_price", "size", "pnl", "pnl_pct", "bars_held", "exit_reason"])

    metrics = _compute_metrics(equity_arr, trades, initial_capital, warmup)

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
                        help="Min predicted y_norm to enter (default: 0)")
    args = parser.parse_args()

    print("=" * 60)
    print("Vectorized Lightweight Backtester")
    print("=" * 60)

    if args.real:
        # Load real data
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

    # Run backtest
    import time
    print(f"\n[2] Running backtest: entry={args.entry}, exit={args.exit}, "
          f"atr_period={args.atr_period}, atr_mult={args.atr_mult}, "
          f"capital={args.capital}, risk_pct={args.risk_pct}")
    t0 = time.perf_counter()
    result = lightweight_backtest(
        df,
        entry_period=args.entry,
        exit_period=args.exit,
        atr_period=args.atr_period,
        atr_mult=args.atr_mult,
        initial_capital=args.capital,
        risk_pct=args.risk_pct,
        ml_model=ml_model,
        ml_threshold=ml_threshold,
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

    # Trade log
    if len(result["trades_df"]) > 0:
        print(f"\n[4] Trade log (first 10 of {len(result['trades_df'])}):")
        pd.set_option("display.max_columns", 10)
        pd.set_option("display.width", 140)
        print(result["trades_df"].head(10).to_string())

    print("\n" + "=" * 60)
    print("Done")
    print("=" * 60)
