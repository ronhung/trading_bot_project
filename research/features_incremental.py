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
        self._entry_high = _RollingMax(entry_period)
        self._entry_low = _RollingMin(entry_period)
        self._exit_high = _RollingMax(exit_period)
        self._exit_low = _RollingMin(exit_period)
        self._atr = _RollingMean(atr_period)
        self._vol_ma = _RollingMean(vol_period)
        self._log_vol_mean = _RollingMean(vol_period)
        self._log_vol_std = _RollingStd(vol_period)
        self._ma = _RollingMean(ma_period)
        self._taker_sum = _RollingSum(vol_period)
        self._vol_sum = _RollingSum(vol_period)

        self._ret_periods = (5, 10, 30)
        self._max_ret = max(self._ret_periods)
        self._close_hist: deque = deque()
        self._prev_close: Optional[float] = None

    def update(self, bar: Dict) -> Dict[str, float]:
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])
        volume = float(bar.get("volume", 0.0))
        has_taker = "taker_buy_base" in bar
        taker = float(bar.get("taker_buy_base", 0.0))

        # ---- shifted indicators (previous N bars, excluding current) ----
        entry_high = self._entry_high.value()
        entry_low = self._entry_low.value()
        exit_high = self._exit_high.value()
        exit_low = self._exit_low.value()
        atr = self._atr.value()
        vol_ma = self._vol_ma.value()
        log_vol_mean = self._log_vol_mean.value()
        log_vol_std = self._log_vol_std.std()
        ma = self._ma.value()

        # ---- non-shifted indicators (mix current + shifted state) ----
        log_vol = math.log(volume) if volume > 0 else float("nan")
        vol_ratio = (volume / vol_ma
                     if (not math.isnan(vol_ma) and vol_ma != 0) else float("nan"))
        if (not math.isnan(log_vol) and not math.isnan(log_vol_mean)
                and not math.isnan(log_vol_std) and log_vol_std != 0):
            vol_zscore = (log_vol - log_vol_mean) / log_vol_std
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
            ts = self._taker_sum.value()
            vs = self._vol_sum.value()
            result["taker_buy_ratio"] = (ts / vs
                                         if (not math.isnan(ts) and not math.isnan(vs) and vs != 0)
                                         else float("nan"))

        # ---- push current bar into rolling state ----
        self._entry_high.push(high)
        self._entry_low.push(low)
        self._exit_high.push(high)
        self._exit_low.push(low)

        if self._prev_close is None:
            tr = high - low
        else:
            tr = max(high - low, abs(high - self._prev_close), abs(low - self._prev_close))
        self._atr.push(tr)

        self._vol_ma.push(volume)
        if not math.isnan(log_vol):
            self._log_vol_mean.push(log_vol)
            self._log_vol_std.push(log_vol)

        self._ma.push(close)

        if has_taker:
            self._taker_sum.push(taker)
            self._vol_sum.push(volume)

        self._close_hist.append(close)
        while len(self._close_hist) > self._max_ret:
            self._close_hist.popleft()

        self._prev_close = close
        return result
