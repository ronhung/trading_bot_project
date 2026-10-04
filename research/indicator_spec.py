"""
Declarative indicator spec — single source of truth for rolling indicators.

Both backends are thin interpreters of ROLLING_SPEC:
  * research.features.add_indicators()      — vectorized (pandas, whole frame)
  * research.features_incremental.IncrementalIndicators — streaming (deque, per bar)

Adding a new rolling indicator = add one entry to ROLLING_SPEC; both backends
pick it up automatically. The input transforms live here too (input_series for
the vectorized backend, input_value for the streaming backend).

Non-rolling "derived" columns (vol_ratio, vol_zscore, channel_pos, ret_*, …)
are element-wise formulas computed from the rolling outputs + the current bar.
They are NOT duplicated rolling logic, so they live next to where they're used.
"""

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# Rolling indicator definitions.
#   op     : max | min | mean | std | sum
#   input  : a bar field, or a transform ("log_volume", "true_range")
#   period : name of the period parameter (resolved from the caller's params)
#
# Every entry uses .shift(1) semantics: row i uses only bars < i.
ROLLING_SPEC: List[Dict[str, str]] = [
    {"name": "entry_high", "op": "max",  "input": "high",           "period": "entry_period"},
    {"name": "entry_low",  "op": "min",  "input": "low",            "period": "entry_period"},
    {"name": "exit_high",  "op": "max",  "input": "high",           "period": "exit_period"},
    {"name": "exit_low",   "op": "min",  "input": "low",            "period": "exit_period"},
    {"name": "atr",        "op": "mean", "input": "true_range",     "period": "atr_period"},
    {"name": "vol_ma",     "op": "mean", "input": "volume",         "period": "vol_period"},
    {"name": "vol_mean",   "op": "mean", "input": "log_volume",     "period": "vol_period"},
    {"name": "vol_std",    "op": "std",  "input": "log_volume",     "period": "vol_period"},
    {"name": "ma",         "op": "mean", "input": "close",          "period": "ma_period"},
    {"name": "taker_sum",  "op": "sum",  "input": "taker_buy_base", "period": "vol_period"},
    {"name": "vol_sum",    "op": "sum",  "input": "volume",         "period": "vol_period"},
]


def input_series(df: pd.DataFrame, input_name: str) -> pd.Series:
    """Vectorized input series for a spec `input` name."""
    if input_name == "log_volume":
        return np.log(df["volume"].replace(0, np.nan))
    if input_name == "true_range":
        prev_close = df["close"].shift(1)
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - prev_close).abs()
        tr3 = (df["low"] - prev_close).abs()
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    if input_name not in df.columns:
        # Optional field (e.g. taker_buy_base) absent — all-NaN keeps the
        # rolling window NaN, and downstream guards skip the derived column.
        return pd.Series(np.nan, index=df.index)
    return df[input_name]


def input_value(bar: Dict[str, Any], prev_close: Optional[float], input_name: str) -> float:
    """Streaming input value for a spec `input` name (current bar)."""
    if input_name == "log_volume":
        v = float(bar.get("volume", 0.0))
        return math.log(v) if v > 0 else float("nan")
    if input_name == "true_range":
        high = float(bar["high"])
        low = float(bar["low"])
        if prev_close is None:
            return high - low
        return max(high - low, abs(high - prev_close), abs(low - prev_close))
    val = bar.get(input_name)
    if val is None:
        return float("nan")
    return float(val)
