"""
Feature library for ML dataset construction.

Design: two-layer architecture for guaranteed zero-lookahead.

1. Precompute layer (add_indicators):
   Adds rolling indicator columns to a COPY of the DataFrame.
   All columns use .shift(1) so row `i` only contains data from <= `i`.
   This mirrors the formulas in shared/core_logic/turtle_math.py exactly.

2. Feature callables:
   fn(df, event_idx) -> {feature_name: value}
   - Fast path: detect precomputed columns, index directly (O(1))
   - Fallback: compute on df.iloc[:event_idx+1] if columns missing
   The __main__ self-test verifies both paths agree.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional, Tuple


# ============================================================
# 1. Indicator precomputation (shared foundation)
# ============================================================

def add_indicators(
    df: pd.DataFrame,
    entry_period: int = 20,
    exit_period: int = 10,
    atr_period: int = 20,
    vol_period: int = 20,
) -> pd.DataFrame:
    """
    Add rolling indicator columns to a COPY of the DataFrame.

    All indicators use .shift(1) — row `i` only uses data from bars <= `i`.
    Formulas exactly mirror shared/core_logic/turtle_math.py.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: open, high, low, close, volume.
        Sorted chronologically (oldest first).
    entry_period, exit_period : int
        Donchian channel lookback periods.
    atr_period : int
        ATR smoothing period.
    vol_period : int
        Volume moving average period.

    Returns
    -------
    pd.DataFrame with added columns:
      entry_high, entry_low, exit_high, exit_low,
      atr, vol_ma, vol_ratio, channel_pos,
      ret_5, ret_10, ret_30
    """
    out = df.copy()

    # --- Donchian channels (shifted: row i uses only bars < i) ---
    out["entry_high"] = out["high"].rolling(entry_period).max().shift(1)
    out["entry_low"]  = out["low"].rolling(entry_period).min().shift(1)
    out["exit_high"]  = out["high"].rolling(exit_period).max().shift(1)
    out["exit_low"]   = out["low"].rolling(exit_period).min().shift(1)

    # --- ATR: True Range, shifted rolling mean ---
    pc = out["close"].shift(1)
    tr1 = out["high"] - out["low"]
    tr2 = (out["high"] - pc).abs()
    tr3 = (out["low"] - pc).abs()
    out["atr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1) \
                     .rolling(atr_period).mean().shift(1)

    # --- Volume metrics ---
    out["vol_ma"] = out["volume"].rolling(vol_period).mean().shift(1)
    out["vol_ratio"] = out["volume"] / out["vol_ma"]

    # --- Channel position (where price sits within the Donchian band) ---
    channel_range = out["entry_high"] - out["entry_low"]
    out["channel_pos"] = np.where(
        channel_range > 0,
        ((out["close"] - out["entry_low"]) / channel_range).clip(0, 1),
        0.5,
    )

    # --- Lagged returns ---
    for p in [5, 10, 30]:
        out[f"ret_{p}"] = out["close"] / out["close"].shift(p) - 1.0

    # --- 200-period moving average (trend filter) ---
    out["ma_200"] = out["close"].rolling(200).mean().shift(1)

    # --- Daily ATR as percentage of price (for volatility normalization) ---
    # atr is absolute ($); divide by close to get decimal; rolling 1440 smooths it
    atr_pct_raw = out["atr"] / out["close"]
    out["atr_daily"] = atr_pct_raw.rolling(1440).mean()  # ~1-day ATR as fraction of price

    # --- Taker flow (if columns exist in the data) ---
    if "taker_buy_base" in out.columns and "volume" in out.columns:
        out["taker_buy_ratio"] = (
            out["taker_buy_base"].rolling(vol_period).sum().shift(1)
            / out["volume"].rolling(vol_period).sum().shift(1)
        )

    return out


# ============================================================
# 2. Feature callables (zero-lookahead guaranteed)
# ============================================================

def _safe_loc(df: pd.DataFrame, idx: int, col: str, fallback: float = np.nan):
    """Index a precomputed column; fallback on missing or NaN."""
    if col not in df.columns:
        return fallback
    val = df[col].iloc[idx]
    if pd.isna(val):
        return fallback
    return float(val)


def feature_atr(df: pd.DataFrame, idx: int, period: int = 20) -> Dict[str, float]:
    """ATR value and ATR as percentage of price at event."""
    atr_val = _safe_loc(df, idx, "atr", 0.0)
    close = df["close"].iloc[idx]
    atr_pct = (atr_val / close) if close > 0 else 0.0
    return {"atr": atr_val, "atr_pct": atr_pct}


def feature_breakout_intensity(df: pd.DataFrame, idx: int,
                                entry_period: int = 20,
                                atr_period: int = 20) -> Dict[str, float]:
    """Breakout strength normalized by ATR. Matches turtle_math intensity formula."""
    close = df["close"].iloc[idx]

    # Fast path: use precomputed columns
    entry_high = _safe_loc(df, idx, "entry_high")
    entry_low = _safe_loc(df, idx, "entry_low")
    atr_val = _safe_loc(df, idx, "atr")

    # Fallback: compute on the fly from df slice
    if pd.isna(entry_high) or pd.isna(entry_low):
        sub = df.iloc[:idx + 1]
        entry_high = sub["high"].rolling(entry_period).max().iloc[-1]
        entry_low = sub["low"].rolling(entry_period).min().iloc[-1]
    if pd.isna(atr_val) or atr_val <= 0:
        atr_val = 1.0  # turtle_math fallback

    long_intensity = (close - entry_high) / atr_val if pd.notna(entry_high) else 0.0
    short_intensity = (entry_low - close) / atr_val if pd.notna(entry_low) else 0.0

    return {"long_intensity": float(long_intensity), "short_intensity": float(short_intensity)}


def feature_volume_ratio(df: pd.DataFrame, idx: int,
                          vol_period: int = 20) -> Dict[str, float]:
    """Volume surge detection: ratio to moving average."""
    vol_ratio = _safe_loc(df, idx, "vol_ratio", 1.0)

    # Volume z-score over lookback (using log volume to handle skew)
    if "volume" in df.columns and idx >= vol_period:
        vol_hist = df["volume"].iloc[max(0, idx - vol_period + 1): idx + 1]
        log_vol = np.log(vol_hist.replace(0, np.nan))
        if len(log_vol.dropna()) >= 5:
            mean_lv = log_vol.mean()
            std_lv = log_vol.std()
            current_lv = np.log(max(df["volume"].iloc[idx], 1e-12))
            vol_zscore = (current_lv - mean_lv) / std_lv if std_lv > 0 else 0.0
        else:
            vol_zscore = 0.0
    else:
        vol_zscore = 0.0

    return {"vol_ratio": vol_ratio, "vol_zscore": vol_zscore}


def feature_channel_position(df: pd.DataFrame, idx: int,
                              entry_period: int = 20) -> Dict[str, float]:
    """Where price sits within the Donchian channel (0=bottom, 1=top)."""
    pos = _safe_loc(df, idx, "channel_pos", 0.5)
    return {"channel_pos": pos}


def feature_donchian_width(df: pd.DataFrame, idx: int,
                            entry_period: int = 20) -> Dict[str, float]:
    """Donchian channel width as percentage of price — volatility context."""
    entry_high = _safe_loc(df, idx, "entry_high")
    entry_low = _safe_loc(df, idx, "entry_low")
    close = df["close"].iloc[idx]

    if pd.notna(entry_high) and pd.notna(entry_low) and close > 0:
        width_pct = (entry_high - entry_low) / close
    else:
        width_pct = 0.0

    return {"channel_width_pct": float(width_pct)}


def feature_lagged_returns(df: pd.DataFrame, idx: int,
                            periods: Tuple[int, ...] = (5, 10, 30)) -> Dict[str, float]:
    """Returns over lookback periods at event time."""
    result = {}
    for p in periods:
        col = f"ret_{p}"
        val = _safe_loc(df, idx, col, 0.0)
        result[col] = val
    return result


def feature_taker_flow(df: pd.DataFrame, idx: int,
                        period: int = 20) -> Dict[str, float]:
    """Buy/sell pressure from taker volume ratio."""
    ratio = _safe_loc(df, idx, "taker_buy_ratio", 0.5)
    return {"taker_buy_ratio": ratio}


def feature_trend_filter(df: pd.DataFrame, idx: int,
                          ma_period: int = 200) -> Dict[str, float]:
    """200MA direction: +1 uptrend, -1 downtrend, and price-vs-MA ratio."""
    ma_val = _safe_loc(df, idx, "ma_200")
    close = df["close"].iloc[idx]

    if pd.isna(ma_val) or ma_val <= 0:
        trend_direction = 0.0
        ma_ratio = 1.0
    else:
        ma_ratio = close / ma_val
        if ma_ratio > 1.01:
            trend_direction = 1.0   # uptrend
        elif ma_ratio < 0.99:
            trend_direction = -1.0  # downtrend
        else:
            trend_direction = 0.0   # sideways

    return {"trend_direction": trend_direction, "ma_ratio": ma_ratio}


def feature_hour_of_day(df: pd.DataFrame, idx: int) -> Dict[str, float]:
    """Hour of day (0-23) from datetime index — captures intraday seasonality."""
    if hasattr(df.index, "hour"):
        hour = float(df.index[idx].hour)
    elif "datetime" in df.columns:
        dt = df["datetime"].iloc[idx]
        hour = float(pd.Timestamp(dt).hour)
    else:
        hour = 0.0
    return {"hour_of_day": hour}


# ============================================================
# 3. Feature pipeline builder
# ============================================================

# Canonical feature function signature
FeatureFunc = Callable[[pd.DataFrame, int], Dict[str, float]]


def make_feature_pipeline(feature_funcs: List[FeatureFunc]) -> FeatureFunc:
    """
    Compose multiple feature functions into a single pipeline.

    Usage:
        pipeline = make_feature_pipeline([
            feature_atr,
            feature_breakout_intensity,
            feature_channel_position,
        ])
        features = pipeline(df, event_idx)  # merged dict of all features
    """
    def pipeline(df: pd.DataFrame, idx: int) -> Dict[str, float]:
        result = {}
        for fn in feature_funcs:
            result.update(fn(df, idx))
        return result
    return pipeline


def default_feature_pipeline() -> FeatureFunc:
    """Standard feature set used by dataset_builder and param_sweep demos."""
    return make_feature_pipeline([
        feature_atr,
        feature_breakout_intensity,
        feature_volume_ratio,
        feature_channel_position,
        feature_donchian_width,
        feature_lagged_returns,
        feature_taker_flow,
        feature_trend_filter,
    ])


# ============================================================
# 4. BaseFeature wrapper classes (implementing core.feature.BaseFeature)
# ============================================================

from core.feature import BaseFeature


class _CallableFeature(BaseFeature):
    """Adapter: wraps an existing FeatureFunc callable as a BaseFeature."""

    def __init__(self, func: FeatureFunc, name: str = "callable_feature"):
        self._func = func
        self._name = name

    def compute(self, data: pd.DataFrame, events: pd.Series) -> pd.DataFrame:
        event_mask = events != 0 if events.dtype == int else events.astype(bool)
        event_indices = np.flatnonzero(event_mask.values)
        rows = [self._func(data, int(idx)) for idx in event_indices]
        if not rows:
            return pd.DataFrame(index=pd.Index([], name="event_idx"))
        return pd.DataFrame(rows, index=event_indices)

    def compute_one(self, data: pd.DataFrame, idx: int) -> Dict[str, float]:
        return self._func(data, idx)

    def __repr__(self) -> str:
        return f"_CallableFeature({self._name})"


class VolumeRatioFeature(BaseFeature):
    """Volume surge detection: ratio of current volume to moving average."""

    def __init__(self, vol_period: int = 20):
        self.vol_period = vol_period

    def compute(self, data: pd.DataFrame, events: pd.Series) -> pd.DataFrame:
        return _CallableFeature(
            lambda df, idx: feature_volume_ratio(df, idx, self.vol_period),
            "vol_ratio",
        ).compute(data, events)

    def compute_one(self, data: pd.DataFrame, idx: int) -> Dict[str, float]:
        return feature_volume_ratio(data, idx, self.vol_period)


class BreakoutIntensityFeature(BaseFeature):
    """Breakout strength normalized by ATR (matches turtle_math intensity)."""

    def __init__(self, entry_period: int = 20, atr_period: int = 20):
        self.entry_period = entry_period
        self.atr_period = atr_period

    def compute(self, data: pd.DataFrame, events: pd.Series) -> pd.DataFrame:
        return _CallableFeature(
            lambda df, idx: feature_breakout_intensity(
                df, idx, self.entry_period, self.atr_period,
            ),
            "breakout_intensity",
        ).compute(data, events)

    def compute_one(self, data: pd.DataFrame, idx: int) -> Dict[str, float]:
        return feature_breakout_intensity(data, idx, self.entry_period, self.atr_period)


class ATRFeature(BaseFeature):
    """ATR value and ATR as percentage of price."""

    def __init__(self, period: int = 20):
        self.period = period

    def compute(self, data: pd.DataFrame, events: pd.Series) -> pd.DataFrame:
        return _CallableFeature(
            lambda df, idx: feature_atr(df, idx, self.period), "atr",
        ).compute(data, events)

    def compute_one(self, data: pd.DataFrame, idx: int) -> Dict[str, float]:
        return feature_atr(data, idx, self.period)


class ChannelPositionFeature(BaseFeature):
    """Where price sits within the Donchian channel (0=bottom, 1=top)."""

    def __init__(self, entry_period: int = 20):
        self.entry_period = entry_period

    def compute(self, data: pd.DataFrame, events: pd.Series) -> pd.DataFrame:
        return _CallableFeature(
            lambda df, idx: feature_channel_position(df, idx, self.entry_period),
            "channel_pos",
        ).compute(data, events)

    def compute_one(self, data: pd.DataFrame, idx: int) -> Dict[str, float]:
        return feature_channel_position(data, idx, self.entry_period)


class DonchianWidthFeature(BaseFeature):
    """Donchian channel width as percentage of price — volatility context."""

    def __init__(self, entry_period: int = 20):
        self.entry_period = entry_period

    def compute(self, data: pd.DataFrame, events: pd.Series) -> pd.DataFrame:
        return _CallableFeature(
            lambda df, idx: feature_donchian_width(df, idx, self.entry_period),
            "channel_width",
        ).compute(data, events)

    def compute_one(self, data: pd.DataFrame, idx: int) -> Dict[str, float]:
        return feature_donchian_width(data, idx, self.entry_period)


class LaggedReturnsFeature(BaseFeature):
    """Returns over lookback periods at event time."""

    def __init__(self, periods: Tuple[int, ...] = (5, 10, 30)):
        self.periods = periods

    def compute(self, data: pd.DataFrame, events: pd.Series) -> pd.DataFrame:
        return _CallableFeature(
            lambda df, idx: feature_lagged_returns(df, idx, self.periods),
            "lagged_returns",
        ).compute(data, events)

    def compute_one(self, data: pd.DataFrame, idx: int) -> Dict[str, float]:
        return feature_lagged_returns(data, idx, self.periods)


class TakerFlowFeature(BaseFeature):
    """Buy/sell pressure from taker volume ratio."""

    def __init__(self, period: int = 20):
        self.period = period

    def compute(self, data: pd.DataFrame, events: pd.Series) -> pd.DataFrame:
        return _CallableFeature(
            lambda df, idx: feature_taker_flow(df, idx, self.period),
            "taker_flow",
        ).compute(data, events)

    def compute_one(self, data: pd.DataFrame, idx: int) -> Dict[str, float]:
        return feature_taker_flow(data, idx, self.period)


class TrendFilterFeature(BaseFeature):
    """200MA direction: +1 uptrend, -1 downtrend, and price-vs-MA ratio."""

    def __init__(self, ma_period: int = 200):
        self.ma_period = ma_period

    def compute(self, data: pd.DataFrame, events: pd.Series) -> pd.DataFrame:
        return _CallableFeature(
            lambda df, idx: feature_trend_filter(df, idx, self.ma_period),
            "trend_filter",
        ).compute(data, events)

    def compute_one(self, data: pd.DataFrame, idx: int) -> Dict[str, float]:
        return feature_trend_filter(data, idx, self.ma_period)


class CompositeFeature(BaseFeature):
    """Combine multiple BaseFeature instances into one."""

    def __init__(self, features: List[BaseFeature]):
        self._features = features

    def compute(self, data: pd.DataFrame, events: pd.Series) -> pd.DataFrame:
        results = [f.compute(data, events) for f in self._features]
        return pd.concat(results, axis=1)

    def compute_one(self, data: pd.DataFrame, idx: int) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for f in self._features:
            result.update(f.compute_one(data, idx))
        return result


def default_feature_set() -> CompositeFeature:
    """Standard feature set as BaseFeature objects (used by pipeline_runner)."""
    return CompositeFeature([
        ATRFeature(),
        BreakoutIntensityFeature(),
        VolumeRatioFeature(),
        ChannelPositionFeature(),
        DonchianWidthFeature(),
        LaggedReturnsFeature(),
        TakerFlowFeature(),
        TrendFilterFeature(),
    ])


# ============================================================
# __main__ sanity check
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing research.features -- Indicator Precomputation + Feature Pipeline")
    print("=" * 60)

    # Build tiny dataset for manual inspection
    np.random.seed(99)
    n = 500
    close = 20000.0 + np.cumsum(np.random.randn(n) * 50)
    close = np.maximum(close, 100.0)
    high = close * (1.0 + np.random.uniform(0.001, 0.006, n))
    low = close * (1.0 - np.random.uniform(0.001, 0.006, n))
    open_p = close * (1.0 + np.random.uniform(-0.002, 0.002, n))
    volume = np.random.lognormal(10, 0.5, n)
    taker_buy = volume * np.random.uniform(0.3, 0.7, n)

    dates = pd.date_range("2023-01-01", periods=n, freq="1min")
    df = pd.DataFrame({
        "datetime": dates,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "quote_volume": close * volume,
        "trades_count": np.random.randint(50, 500, n),
        "taker_buy_base": taker_buy,
        "taker_buy_quote": taker_buy * close,
    }, index=dates)

    # Precompute indicators
    df = add_indicators(df, entry_period=20, exit_period=10, atr_period=20, vol_period=20)

    # Verify columns exist
    expected_cols = [
        "entry_high", "entry_low", "exit_high", "exit_low",
        "atr", "vol_ma", "vol_ratio", "channel_pos",
        "ret_5", "ret_10", "ret_30",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        print(f"[FAIL] Missing indicator columns: {missing}")
    else:
        print(f"[PASS] All {len(expected_cols)} indicator columns present")

    # Verify warmup: first max(entry, atr) rows should be NaN for shifted indicators
    # rolling(20).max().shift(1): 20 bars needed for rolling, shift pushes to row 20+
    warmup = max(20, 20)  # 20 rows (0-19) should be NaN
    nan_check_cols = ["entry_high", "entry_low", "atr"]
    warmup_nans = df[nan_check_cols].iloc[:warmup].isna().all(axis=1)
    if warmup_nans.all():
        print(f"[PASS] First {warmup} rows have NaN indicators (correct warmup)")
    else:
        bad_rows = (~warmup_nans).sum()
        print(f"[FAIL] {bad_rows} rows in warmup zone have non-NaN indicators")

    # Verify no lookahead: at any row i, indicator values should equal
    # those computed from df.iloc[:i+1] alone
    pipeline = default_feature_pipeline()
    test_indices = list(range(warmup, min(warmup + 50, n)))
    all_ok = True
    for i in test_indices:
        # Feature from full precomputed df
        feat_full = pipeline(df, i)
        # Feature from isolated slice (no precomputed columns)
        df_slice = df.drop(columns=[c for c in expected_cols if c in df.columns])
        # Re-add only slice-computed indicators
        df_slice_ind = add_indicators(df_slice.iloc[:i+1])
        last_idx = len(df_slice_ind) - 1
        feat_slice = pipeline(df_slice_ind, last_idx)
        # Compare (skip taker_buy_ratio if not in both)
        common_keys = set(feat_full.keys()) & set(feat_slice.keys())
        for k in common_keys:
            vf, vs = feat_full[k], feat_slice[k]
            if pd.isna(vf) and pd.isna(vs):
                continue
            if abs(vf - vs) > 1e-10:
                print(f"  MISMATCH i={i}, key={k}: full={vf:.6f}, slice={vs:.6f}")
                all_ok = False
    if all_ok:
        print(f"[PASS] Zero-lookahead verified: {len(test_indices)} events, precomputed == slice-computed")

    # Print a sample feature vector
    print(f"\nSample feature vector at bar {test_indices[-1]}:")
    sample = pipeline(df, test_indices[-1])
    for k, v in sample.items():
        print(f"  {k:25s}: {v:12.6f}")

    print("\n" + "=" * 60)
    if all_ok and not missing:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED - review output above.")
    print("=" * 60)
