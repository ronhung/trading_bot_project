"""
Dynamic VWAP Band Reversion Trigger.

Microstructure Alpha utilizing institutional VWAP bands & OFI confirmation:
1. Calculates rolling Volume-Weighted Average Price (VWAP) and volume-weighted standard deviation (sigma_vwap).
2. Generates dynamic volatility bands:
   - UpperBand = VWAP + k * sigma_vwap
   - LowerBand = VWAP - k * sigma_vwap
3. Microstructure confirmation:
   - Long: Price <= LowerBand and OFI > +ofi_threshold (buying pressure emerges at lower boundary).
   - Short: Price >= UpperBand and OFI < -ofi_threshold (selling pressure emerges at upper boundary).
"""

import numpy as np
import pandas as pd

from core.trigger import BaseEventTrigger
from core.strategy_factory import register_trigger


@register_trigger("dynamic_vwap_band")
class DynamicVWAPBandTrigger(BaseEventTrigger):
    """
    Dynamic VWAP Band Reversion Trigger.

    Parameters:
        vwap_window: Lookback window for rolling VWAP & sigma (default 240 = 4 hours).
        band_mult: Standard deviation multiplier k (default 2.0).
        ofi_window: Lookback window for smoothing OFI (default 15).
        ofi_threshold: Minimum OFI threshold for boundary rejection confirmation (default 0.02).
        cooldown: Minimum bars between consecutive signals (default 60).
        allow_long: Allow long entries at LowerBand (default True).
        allow_short: Allow short entries at UpperBand (default True).
    """

    def __init__(
        self,
        vwap_window: int = 240,
        band_mult: float = 2.0,
        ofi_window: int = 15,
        ofi_threshold: float = 0.02,
        cooldown: int = 60,
        allow_long: bool = True,
        allow_short: bool = True,
        **kwargs,
    ):
        self.vwap_window = int(vwap_window)
        self.band_mult = float(band_mult)
        self.ofi_window = int(ofi_window)
        self.ofi_threshold = float(ofi_threshold)
        self.cooldown = int(cooldown)
        self.allow_long = bool(allow_long)
        self.allow_short = bool(allow_short)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        vol = data["volume"]

        # Vectorized Rolling Volume-Weighted Statistics
        pv = close * vol
        pv2 = (close ** 2) * vol

        roll_pv = pv.rolling(self.vwap_window).sum()
        roll_pv2 = pv2.rolling(self.vwap_window).sum()
        roll_vol = vol.rolling(self.vwap_window).sum().replace(0, np.nan)

        vwap = (roll_pv / roll_vol).bfill()
        vwap_var = np.maximum(0.0, (roll_pv2 / roll_vol) - (vwap ** 2)).fillna(0.0)
        vwap_std = np.sqrt(vwap_var)

        upper_band = vwap + self.band_mult * vwap_std
        lower_band = vwap - self.band_mult * vwap_std

        # Microstructure OFI
        if "ofi" in data.columns:
            ofi = data["ofi"]
        elif "taker_ratio" in data.columns:
            ofi = (data["taker_ratio"] - 0.5) * 2.0
        else:
            ofi = pd.Series(0.0, index=data.index)

        rolling_ofi = ofi.rolling(self.ofi_window).mean()

        # Signals
        long_cond = (
            self.allow_long
            & (close <= lower_band)
            & (rolling_ofi > self.ofi_threshold)
        )

        short_cond = (
            self.allow_short
            & (close >= upper_band)
            & (rolling_ofi < -self.ofi_threshold)
        )

        signals = pd.Series(0, index=data.index, dtype=int)
        signals[long_cond] = 1
        signals[short_cond] = -1

        # Apply cooldown
        if self.cooldown > 0:
            sig_arr = signals.values.copy()
            last_sig_idx = -self.cooldown - 1
            for i in range(len(sig_arr)):
                if sig_arr[i] != 0:
                    if i - last_sig_idx < self.cooldown:
                        sig_arr[i] = 0
                    else:
                        last_sig_idx = i
            signals = pd.Series(sig_arr, index=data.index)

        return signals
