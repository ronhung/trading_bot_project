"""
Incremental streaming indicators — O(1) per bar.

Mirrors research.features.add_indicators() column-for-column, but maintains
rolling state so each bar costs O(1) amortized instead of rescanning the whole
lookback window. This is what makes a full-history streaming backtest (the
Phase 5 Python brain + C++ engine) feasible with the real 30-day lookbacks.

Only produces the columns the streaming StrategyWrapper actually reads:
  close, entry_high, entry_low, exit_high, exit_low, atr, vol_ratio,
  vol_zscore, channel_pos, ret_5/10/30, taker_buy_ratio, ma.

The remaining feature values (atr_pct, long/short_intensity, channel_width_pct,
trend_direction, ma_ratio) are derived by the feature functions from these
columns, so they need no separate state.

Semantics match add_indicators exactly:
  * rolling().shift(1) — the current bar's value uses the previous N bars.
  * population std (ddof=0) for vol_zscore.
"""

import math
from collections import deque
from typing import Dict, Optional

from research.indicator_spec import ROLLING_SPEC, input_value


def _make_tracker(op: str, period: int):
    """Instantiate the rolling tracker for a ROLLING_SPEC `op`."""
    if op == "max":
        return _RollingMax(period)
    if op == "min":
        return _RollingMin(period)
    if op == "mean":
        return _RollingMean(period)
    if op == "sum":
        return _RollingSum(period)
    if op == "std":
        return _RollingStd(period)
    raise ValueError(f"unknown rolling op: {op}")


class _RollingMax:
    """O(1) amortized rolling max via a monotonic-decreasing deque."""
    __slots__ = ("period", "_dq", "_seq")

    def __init__(self, period: int):
        self.period = period
        self._dq = deque()   # (value, seq) — decreasing by value
        self._seq = 0

    def push(self, value: float) -> None:
        while self._dq and self._dq[-1][0] <= value:
            self._dq.pop()
        self._dq.append((value, self._seq))
        self._seq += 1
        cutoff = self._seq - self.period
        while self._dq and self._dq[0][1] < cutoff:
            self._dq.popleft()

    def ready(self) -> bool:
        return self._seq >= self.period

    def value(self) -> float:
        if not self.ready():
            return float("nan")
        return self._dq[0][0] if self._dq else float("nan")


class _RollingMin:
    __slots__ = ("period", "_dq", "_seq")

    def __init__(self, period: int):
        self.period = period
        self._dq = deque()   # (value, seq) — increasing by value
        self._seq = 0

    def push(self, value: float) -> None:
        while self._dq and self._dq[-1][0] >= value:
            self._dq.pop()
        self._dq.append((value, self._seq))
        self._seq += 1
        cutoff = self._seq - self.period
        while self._dq and self._dq[0][1] < cutoff:
            self._dq.popleft()

    def ready(self) -> bool:
        return self._seq >= self.period

    def value(self) -> float:
        if not self.ready():
            return float("nan")
        return self._dq[0][0] if self._dq else float("nan")


class _RollingMean:
    __slots__ = ("period", "_vals", "_sum", "_seq")

    def __init__(self, period: int):
        self.period = period
        self._vals = deque()
        self._sum = 0.0
        self._seq = 0

    def push(self, value: float) -> None:
        self._vals.append(value)
        self._sum += value
        self._seq += 1
        if len(self._vals) > self.period:
            self._sum -= self._vals.popleft()

    def ready(self) -> bool:
        return self._seq >= self.period

    def value(self) -> float:
        return self._sum / self.period if self.ready() else float("nan")


class _RollingSum:
    __slots__ = ("period", "_vals", "_sum", "_seq")

    def __init__(self, period: int):
        self.period = period
        self._vals = deque()
        self._sum = 0.0
        self._seq = 0

    def push(self, value: float) -> None:
        self._vals.append(value)
        self._sum += value
        self._seq += 1
        if len(self._vals) > self.period:
            self._sum -= self._vals.popleft()

    def ready(self) -> bool:
        return self._seq >= self.period

    def value(self) -> float:
        return self._sum if self.ready() else float("nan")


class _RollingStd:
    """Population std (ddof=0) via running sum + sum of squares."""
    __slots__ = ("period", "_vals", "_sum", "_sumsq", "_seq")

    def __init__(self, period: int):
        self.period = period
        self._vals = deque()
        self._sum = 0.0
        self._sumsq = 0.0
        self._seq = 0

    def push(self, value: float) -> None:
        self._vals.append(value)
        self._sum += value
        self._sumsq += value * value
        self._seq += 1
        if len(self._vals) > self.period:
            old = self._vals.popleft()
            self._sum -= old
            self._sumsq -= old * old

    def ready(self) -> bool:
        return self._seq >= self.period

    def mean(self) -> float:
        return self._sum / self.period if self.ready() else float("nan")

    def std(self) -> float:
        if not self.ready():
            return float("nan")
        mean = self._sum / self.period
        var = self._sumsq / self.period - mean * mean
        return math.sqrt(var) if var > 0 else 0.0

    def value(self) -> float:
        return self.std()


class IncrementalIndicators:
    """
    Streaming counterpart to add_indicators().

    Usage per bar:
        ind = IncrementalIndicators(entry_period=..., exit_period=..., ...)
        for bar in bars:
            row = ind.update(bar)   # dict of the current bar's indicator values
    """

    def __init__(
        self,
        entry_period: int = 20,
        exit_period: int = 10,
        atr_period: int = 20,
        vol_period: int = 20,
        ma_period: int = 200,
    ):
        params = {
            "entry_period": entry_period, "exit_period": exit_period,
            "atr_period": atr_period, "vol_period": vol_period, "ma_period": ma_period,
        }
        self._trackers: Dict[str, object] = {}
        for spec in ROLLING_SPEC:
            period = params[spec["period"]]
            self._trackers[spec["name"]] = _make_tracker(spec["op"], period)

        self._ret_periods = (5, 10, 30)
        self._max_ret = max(self._ret_periods)
        self._close_hist: deque = deque()
        self._prev_close: Optional[float] = None

    def update(self, bar: Dict) -> Dict[str, float]:
        close = float(bar["close"])
        volume = float(bar.get("volume", 0.0))
        has_taker = "taker_buy_base" in bar

        # ---- shifted rolling values (previous N bars, excluding current) ----
        vals = {name: trk.value() for name, trk in self._trackers.items()}
        entry_high = vals["entry_high"]
        entry_low = vals["entry_low"]
        exit_high = vals["exit_high"]
        exit_low = vals["exit_low"]
        atr = vals["atr"]
        vol_ma = vals["vol_ma"]
        vol_mean = vals["vol_mean"]
        vol_std = vals["vol_std"]
        ma = vals["ma"]

        # ---- derived (element-wise) columns ----
        log_vol = math.log(volume) if volume > 0 else float("nan")
        vol_ratio = (volume / vol_ma
                     if (not math.isnan(vol_ma) and vol_ma != 0) else float("nan"))
        if (not math.isnan(log_vol) and not math.isnan(vol_mean)
                and not math.isnan(vol_std) and vol_std != 0):
            vol_zscore = (log_vol - vol_mean) / vol_std
        else:
            vol_zscore = float("nan")

        channel_range = entry_high - entry_low
        if not math.isnan(channel_range) and channel_range > 0:
            channel_pos = min(max((close - entry_low) / channel_range, 0.0), 1.0)
        else:
            channel_pos = 0.5

        result: Dict[str, float] = {
            "close": close,
            "entry_high": entry_high,
            "entry_low": entry_low,
            "exit_high": exit_high,
            "exit_low": exit_low,
            "atr": atr,
            "vol_ratio": vol_ratio,
            "vol_zscore": vol_zscore,
            "channel_pos": channel_pos,
            "ma": ma,
        }

        for p in self._ret_periods:
            if len(self._close_hist) >= p:
                prev = self._close_hist[-p]
                result[f"ret_{p}"] = (close / prev - 1.0) if prev != 0 else float("nan")
            else:
                result[f"ret_{p}"] = float("nan")

        if has_taker:
            ts = vals["taker_sum"]
            vs = vals["vol_sum"]
            result["taker_buy_ratio"] = (ts / vs
                                         if (not math.isnan(ts) and not math.isnan(vs) and vs != 0)
                                         else float("nan"))

        # ---- push current bar into rolling state ----
        for spec in ROLLING_SPEC:
            value = input_value(bar, self._prev_close, spec["input"])
            if not math.isnan(value):
                self._trackers[spec["name"]].push(value)

        self._close_hist.append(close)
        while len(self._close_hist) > self._max_ret:
            self._close_hist.popleft()

        self._prev_close = close
        return result
