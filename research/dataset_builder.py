"""
ML Dataset Builder — transforms raw K-line data into (X, y) supervised learning datasets.

Core API:
    build_ml_dataset(raw_data, event_trigger, feature_pipeline, labeling_config)
    -> (dataset_X_y, metadata)

Includes:
    - Synthetic OHLCV generator for demos (make_synthetic_ohlcv)
    - Turtle breakout event trigger (turtle_breakout_trigger)

All operations are purely pandas/numpy — zero Backtrader dependency.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Tuple, Optional, Union

# Ensure project root is importable (mirrors existing project convention)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from research.labeling import apply_triple_barrier, fixed_horizon_label
from research.features import add_indicators, make_feature_pipeline, FeatureFunc


# ============================================================
# 1. Synthetic data generator (for demos)
# ============================================================

def make_synthetic_ohlcv(
    n_bars: int = 20000,
    start_price: float = 20000.0,
    sigma: float = 0.0008,
    seed: int = 42,
    freq: str = "1min",
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data with trend regimes so Donchian breakouts occur.

    Uses seeded geometric Brownian motion with alternating drift regimes
    to create realistic trending and ranging periods.

    Parameters
    ----------
    n_bars : int
        Number of 1-minute bars to generate.
    start_price : float
        Initial close price.
    sigma : float
        Per-bar volatility (log-return std).
    seed : int
        Random seed for reproducibility.
    freq : str
        Pandas frequency string for the DatetimeIndex.

    Returns
    -------
    pd.DataFrame with columns matching the real Binance parquet schema.
    """
    rng = np.random.default_rng(seed)

    # Generate returns with regime-switching drift
    returns = np.zeros(n_bars)

    # Create alternating trend/flat regimes
    regime_len_mean = 2000  # ~33 hours per regime
    i = 0
    while i < n_bars:
        # Random regime: trending (drift != 0) or ranging (drift ~ 0)
        is_trending = rng.random() < 0.55  # 55% chance trending
        length = max(200, int(rng.exponential(regime_len_mean)))

        if is_trending:
            drift = rng.uniform(-0.00015, 0.00015)
        else:
            drift = 0.0

        end = min(i + length, n_bars)
        if end > i:
            returns[i:end] = rng.normal(drift, sigma, end - i)
        i = end

    # Cumulative to prices
    close = start_price * np.exp(np.cumsum(returns))
    close = np.maximum(close, 1.0)

    # Generate OHLC from close path
    open_arr = np.roll(close, 1)
    open_arr[0] = start_price

    bar_range = close * rng.uniform(0.0005, 0.003, n_bars)
    high = np.maximum(open_arr, close) + bar_range * rng.uniform(0.2, 1.0, n_bars)
    low = np.minimum(open_arr, close) - bar_range * rng.uniform(0.2, 1.0, n_bars)
    low = np.maximum(low, 0.01)

    volume = rng.lognormal(8.0, 0.6, n_bars)
    taker_buy_base = volume * rng.uniform(0.35, 0.65, n_bars)
    quote_volume = close * volume

    # DatetimeIndex + explicit datetime column (matches real parquet schema)
    dates = pd.date_range("2023-01-01", periods=n_bars, freq=freq)
    df = pd.DataFrame({
        "datetime": dates,
        "open": open_arr,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "quote_volume": quote_volume,
        "trades_count": rng.integers(50, 600, n_bars),
        "taker_buy_base": taker_buy_base,
        "taker_buy_quote": taker_buy_base * close,
    }, index=dates)
    df.index.name = "timestamp"

    return df


# ============================================================
# 2. Event triggers
# ============================================================

def make_turtle_breakout_trigger(
    entry_period: int = 20,
    atr_period: int = 20,
    atr_mult: float = 2.0,
    intensity_threshold: float = 0.0,
    signed: bool = True,
) -> Callable[[pd.DataFrame], pd.Series]:
    """
    Factory: returns a callable event trigger for Turtle breakout detection.

    The returned function has signature fn(df) -> pd.Series and is suitable
    for passing as the `event_trigger` parameter to build_ml_dataset().

    Parameters
    ----------
    entry_period : int
        Donchian channel lookback for entry breakouts.
    atr_period : int
        ATR smoothing period.
    atr_mult : float
        ATR multiplier (used for stop_price computation, not event detection).
    intensity_threshold : float
        Minimum breakout intensity in ATR units. 0.0 = any breakout.
    signed : bool
        If True, returns Series with values in {-1, 0, 1}.
        If False, returns boolean Series.

    Returns
    -------
    callable: fn(df) -> pd.Series
    """
    def trigger(df: pd.DataFrame) -> pd.Series:
        # Ensure indicators are computed
        ind = add_indicators(df, entry_period=entry_period, atr_period=atr_period)

        close = ind["close"].values
        entry_high = ind["entry_high"].values
        entry_low = ind["entry_low"].values
        atr = ind["atr"].values

        # turtle_math fallback: zero/NaN ATR -> use 1.0
        valid_atr = np.where((~np.isnan(atr)) & (atr > 0), atr, 1.0)

        long_breakout = (close > entry_high) & (~np.isnan(entry_high))
        short_breakout = (close < entry_low) & (~np.isnan(entry_low))

        long_intensity = (close - entry_high) / valid_atr
        short_intensity = (entry_low - close) / valid_atr

        long_event = long_breakout & (long_intensity > intensity_threshold)
        short_event = short_breakout & (short_intensity > intensity_threshold)

        if signed:
            result = pd.Series(0, index=ind.index, dtype=int)
            result[long_event] = 1
            result[short_event] = -1
        else:
            result = pd.Series(False, index=ind.index)
            result[long_event | short_event] = True

        return result

    return trigger


# Backward-compatible alias
def turtle_breakout_trigger(
    df: pd.DataFrame,
    entry_period: int = 20,
    atr_period: int = 20,
    atr_mult: float = 2.0,
    intensity_threshold: float = 0.0,
    signed: bool = True,
) -> pd.Series:
    """
    One-shot turtle breakout event detection.
    Convenience wrapper: calls make_turtle_breakout_trigger(...)(df) directly.
    """
    return make_turtle_breakout_trigger(
        entry_period=entry_period,
        atr_period=atr_period,
        atr_mult=atr_mult,
        intensity_threshold=intensity_threshold,
        signed=signed,
    )(df)


# ============================================================
# 3. Main dataset builder
# ============================================================

def build_ml_dataset(
    raw_data: pd.DataFrame,
    event_trigger: Callable[[pd.DataFrame], pd.Series],
    feature_pipeline: FeatureFunc,
    labeling_config: dict,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Transform raw K-line data into an ML-ready (X, y) dataset.

    Pipeline:
      1. Run event_trigger to find entry events
      2. For each event, compute features via feature_pipeline
      3. Apply triple-barrier labeling to produce labels and returns
      4. Assemble (X, y) DataFrame + metadata statistics

    Parameters
    ----------
    raw_data : pd.DataFrame
        OHLCV data. Required columns: open, high, low, close, volume.
        Must be sorted chronologically (oldest first).
    event_trigger : callable
        fn(df) -> pd.Series.
        Returns boolean (True at events) or signed {-1, 0, 1} Series.
        If boolean, all events are treated as long (side=1).
    feature_pipeline : callable
        fn(df, event_idx) -> {feature_name: value}.
        Computes all features for one event at the given integer position.
    labeling_config : dict
        Triple-barrier parameters. Keys:
          - upper_barrier: float (default 0.02, take-profit return as decimal)
          - lower_barrier: float (default -0.01, stop-loss return)
          - horizon: int (default 288, max holding bars)
          - exit_price_mode: str (default "close")
    verbose : bool
        If True, print progress and summary statistics.

    Returns
    -------
    dataset : pd.DataFrame
        One row per event. Columns include all feature columns plus:
          event_time, side, entry_price, label, barrier_hit,
          exit_idx, n_bars_held, exit_price, actual_return
    metadata : dict
        Summary statistics: n_events, label_counts, label_rates,
        barrier_hit_counts, mean_actual_return, etc.
    """
    # --- Validate input ---
    required = {"open", "high", "low", "close"}
    missing = required - set(raw_data.columns)
    if missing:
        raise ValueError(f"raw_data missing required columns: {missing}")

    df = raw_data.copy()

    # --- Extract config with defaults ---
    method = labeling_config.get("method", "triple_barrier")
    ub = labeling_config.get("upper_barrier", 0.02)
    lb = labeling_config.get("lower_barrier", -0.01)
    horizon = labeling_config.get("horizon", 288)
    barrier_mode = labeling_config.get("barrier_mode", "pct")

    # --- Step 1: Run event trigger ---
    event_series = event_trigger(df)
    event_series = event_series.reindex(df.index).fillna(0)

    # Detect if signed or boolean
    unique_vals = set(event_series.unique()) - {0, False}
    is_signed = unique_vals.issubset({-1, 1})

    if is_signed:
        event_mask = event_series != 0
        sides = event_series[event_mask].values.astype(int)
    else:
        event_mask = event_series.astype(bool)
        sides = np.ones(event_mask.sum(), dtype=int)

    event_indices = np.flatnonzero(event_mask.values)
    n_events = len(event_indices)

    if verbose:
        print(f"[build_ml_dataset] Event trigger found {n_events} events "
              f"({n_events / len(df) * 100:.2f}% of {len(df)} bars)")

    if n_events == 0:
        empty_df = pd.DataFrame(columns=["event_time", "side", "entry_price",
                                          "label", "barrier_hit", "exit_idx",
                                          "n_bars_held", "exit_price", "actual_return"])
        empty_meta = {"n_events": 0, "error": "No events detected"}
        return empty_df, empty_meta

    # --- Step 2: Compute features per event ---
    feature_rows = []
    entry_prices = []
    event_times = []
    for idx in event_indices:
        features = feature_pipeline(df, idx)
        feature_rows.append(features)
        event_times.append(df.index[idx])
        entry_prices.append(df["close"].iloc[idx])

    features_df = pd.DataFrame(feature_rows, index=event_indices)
    features_df.index.name = "event_idx"

    if verbose:
        print(f"[build_ml_dataset] Computed {len(features_df.columns)} features "
              f"for {n_events} events")

    # --- Step 3: Triple-barrier labeling ---
    close_arr = df["close"].values
    high_arr = df["high"].values
    low_arr = df["low"].values

    # --- Step 3: Labeling ---
    if method == "fixed_horizon":
        if "atr_daily" not in df.columns:
            raise ValueError("method='fixed_horizon' requires 'atr_daily' column. "
                             "Run add_indicators() first.")
        labels_df = fixed_horizon_label(
            close=close_arr,
            events=event_indices,
            sides=sides,
            daily_atr=df["atr_daily"].values,
            horizon=horizon,
        )
        # Rename columns to match triple-barrier convention
        labels_df["label"] = np.where(labels_df["y_norm"] > 0, 1,
                               np.where(labels_df["y_norm"] < 0, -1, 0))
        labels_df["barrier_hit"] = "fixed_horizon"
        labels_df["n_bars_held"] = np.minimum(horizon, len(close_arr) - 1 - event_indices)
        labels_df["actual_return"] = labels_df["raw_return"]
        labels_df["return_atr"] = labels_df["y_norm"]
        labels_df["truncated"] = (event_indices + horizon) >= len(close_arr)
    else:
        # ATR values for barrier_mode="atr"
        atr_vals = None
        if barrier_mode == "atr":
            if "atr" not in df.columns:
                raise ValueError("barrier_mode='atr' requires 'atr' column in raw_data. "
                                 "Run add_indicators() first.")
            atr_vals = df["atr"].values

        labels_df = apply_triple_barrier(
            close=close_arr,
            high=high_arr,
            low=low_arr,
            events=event_indices,
            upper_barrier=ub,
            lower_barrier=lb,
            horizon=horizon,
            side=sides,
            barrier_mode=barrier_mode,
            atr_values=atr_vals,
        )

    if verbose:
        vc = labels_df["label"].value_counts().to_dict()
        print(f"[build_ml_dataset] Labels: TP={vc.get(1,0)}, SL={vc.get(-1,0)}, "
              f"Timeout={vc.get(0,0)}")

    # --- Step 4: Assemble final dataset ---
    dataset = features_df.join(labels_df, how="left")
    dataset["side"] = sides
    dataset["entry_price"] = entry_prices
    dataset["event_time"] = event_times

    # Reorder columns: metadata first, then features, then label
    meta_cols = ["event_time", "side", "entry_price",
                 "label", "barrier_hit", "exit_idx",
                 "n_bars_held", "exit_price",
                 "actual_return", "return_atr", "truncated"]
    feature_cols = [c for c in dataset.columns if c not in meta_cols]
    dataset = dataset[meta_cols + feature_cols]

    # Remove internal index column if it leaked in (event_idx should already be the index)
    if "event_idx" in dataset.columns:
        dataset = dataset.drop(columns=["event_idx"])

    # --- Compute metadata ---
    vc = labels_df["label"].value_counts()
    barrier_vc = labels_df["barrier_hit"].value_counts()
    total = len(labels_df)
    metadata = {
        "n_events": n_events,
        "n_events_labeled": int(labels_df["label"].notna().sum()),
        "n_truncated": int(labels_df["truncated"].sum()),
        "coverage": n_events / len(df),
        "label_counts": {int(k): int(v) for k, v in vc.items()},
        "label_rates": {int(k): v / total for k, v in vc.items()},
        "barrier_hit_counts": {str(k): int(v) for k, v in barrier_vc.items()},
        "mean_actual_return": float(labels_df["actual_return"].mean()),
        "median_actual_return": float(labels_df["actual_return"].median()),
        "mean_n_bars_held": float(labels_df["n_bars_held"].mean()),
        "feature_names": feature_cols,
        "config": labeling_config,
    }

    return dataset, metadata


# ============================================================
# __main__ demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing research.dataset_builder -- build_ml_dataset()")
    print("=" * 60)

    # Generate synthetic data
    print("\n[1] Generating synthetic OHLCV data (10,000 bars)...")
    df = make_synthetic_ohlcv(n_bars=10000, seed=7)

    # Precompute indicators (fast path for features)
    print("[2] Precomputing indicators...")
    df = add_indicators(df, entry_period=20, exit_period=10,
                         atr_period=20, vol_period=20)

    # Build event trigger (factory returns a callable)
    print("[3] Building event trigger...")
    trigger = make_turtle_breakout_trigger(entry_period=20, atr_period=20,
                                            intensity_threshold=0.0, signed=True)

    # Build feature pipeline
    from research.features import default_feature_pipeline
    pipeline = default_feature_pipeline()

    # Run dataset builder
    print("[4] Running build_ml_dataset()...")
    labeling_config = {
        "upper_barrier": 0.02,   # +2% take-profit
        "lower_barrier": -0.01,  # -1% stop-loss
        "horizon": 288,          # max 288 bars (~4.8 hours at 1m)
    }
    X, meta = build_ml_dataset(df, trigger, pipeline, labeling_config, verbose=True)

    # Print results
    print(f"\n[5] Results:")
    print(f"    Dataset shape: {X.shape[0]} rows x {X.shape[1]} cols")
    print(f"    Feature columns: {meta['feature_names']}")
    print(f"    Label distribution:")
    for label, rate in meta["label_rates"].items():
        label_name = {1: "Take-Profit", -1: "Stop-Loss", 0: "Timeout"}[label]
        print(f"      {label_name}: {rate*100:.1f}% ({meta['label_counts'][label]} events)")
    print(f"    Barrier hits: {meta['barrier_hit_counts']}")
    print(f"    Mean actual return: {meta['mean_actual_return']:.4f}")
    print(f"    Mean bars held: {meta['mean_n_bars_held']:.1f}")
    print(f"    Coverage: {meta['coverage']*100:.2f}% of bars are events")

    # Show first few rows
    print(f"\n[6] First 5 rows of dataset:")
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 160)
    print(X.head(5).to_string())

    print("\n" + "=" * 60)
    print("Demo completed successfully")
    print("(Run 'python -m research.features' for rigorous zero-lookahead verification)")
    print("=" * 60)
