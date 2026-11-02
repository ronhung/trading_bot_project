"""
Liquidation Climax Fading Trigger (Mean Reversion after Forced Liquidations).

Microstructure Alpha targeting retail liquidation cascades:
1. Detects Trades Count bursts: TradesCount_t > mean(TradesCount) + trades_std_mult * std(TradesCount).
2. Measures extreme price dislocation from rolling VWAP normalized by ATR:
   - Long (panic selling climax): (Close - VWAP) / ATR < -deviation_mult
   - Short (short squeeze climax): (Close - VWAP) / ATR > +deviation_mult
3. Reversal confirmation: OFI or taker flow stops cascading and rebounds.
4. Generates Long (+1) on panic selling bottoms and Short (-1) on short squeezes.
"""

import numpy as np
import pandas as pd

from core.trigger import BaseEventTrigger
from core.strategy_factory import register_trigger


@register_trigger("liquidation_climax")
class LiquidationClimaxTrigger(BaseEventTrigger):
    """
    Liquidation Climax Fading Trigger.

    Parameters:
        trades_window: Lookback window for trades count statistics (default 120 = 2 hours).
        trades_std_mult: Multiplier for trades count burst (default 2.5).
        deviation_mult: Minimum z-score deviation of (Close - VWAP) / ATR (default 2.5).
        atr_period: ATR lookback period (default 60).
        ofi_window: Lookback window for smoothing OFI (default 5).
        cooldown: Minimum bars between consecutive signals (default 60).
        allow_long: Allow fading selling climaxes (default True).
        allow_short: Allow fading buying climaxes (default True).
    """

    def __init__(
        self,
        trades_window: int = 120,
        trades_std_mult: float = 2.5,
        deviation_mult: float = 2.5,
        atr_period: int = 60,
        ofi_window: int = 5,
        cooldown: int = 60,
        allow_long: bool = True,
        allow_short: bool = True,
        **kwargs,
    ):
        self.trades_window = int(trades_window)
        self.trades_std_mult = float(trades_std_mult)
        self.deviation_mult = float(deviation_mult)
        self.atr_period = int(atr_period)
        self.ofi_window = int(ofi_window)
        self.cooldown = int(cooldown)
        self.allow_long = bool(allow_long)
        self.allow_short = bool(allow_short)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        # Calculate True Range & ATR
        tr = np.maximum(high - low, np.maximum((high - close.shift(1)).abs(), (low - close.shift(1)).abs()))
        atr = tr.rolling(self.atr_period).mean().bfill()
        safe_atr = atr.replace(0, np.nan)

        # VWAP
        if "vwap" in data.columns:
            vwap = data["vwap"]
        else:
            vol = data["volume"]
            vwap = (data["turnover"] / vol).fillna(close) if "turnover" in data.columns else close

        # Price deviation from VWAP in units of ATR
        vwap_dev = (close - vwap) / safe_atr

        # Trades Count Burst
        if "trades" in data.columns:
            trades = data["trades"]
        else:
            trades = data["volume"]

        trades_mean = trades.rolling(self.trades_window).mean()
        trades_std = trades.rolling(self.trades_window).std().replace(0, 1.0)
        trades_burst = trades > (trades_mean + self.trades_std_mult * trades_std)

        # Microstructure OFI
        if "ofi" in data.columns:
            ofi = data["ofi"]
        elif "taker_ratio" in data.columns:
            ofi = (data["taker_ratio"] - 0.5) * 2.0
        else:
            ofi = pd.Series(0.0, index=data.index)

        rolling_ofi = ofi.rolling(self.ofi_window).mean()

        # Long Signal: Extreme downward dislocation + trades count spike + flow bottoming
        long_cond = (
            self.allow_long
            & trades_burst
            & (vwap_dev < -self.deviation_mult)
            & (rolling_ofi > -0.10)
        )

        # Short Signal: Extreme upward dislocation + trades count spike + flow topping
        short_cond = (
            self.allow_short
            & trades_burst
            & (vwap_dev > self.deviation_mult)
            & (rolling_ofi < 0.10)
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
