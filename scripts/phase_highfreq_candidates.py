"""Test high-frequency strategy candidates for profitability (Phase 3b)."""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd

from research.backtest import lightweight_backtest
from research.labeling import TrailingExitLabeler, FixedHorizonLabeler
from research.triggers.trend_breakout import TrendFilteredBreakoutTrigger
from research.triggers.bollinger_mean_reversion import BollingerMeanReversionTrigger
from execution.sizers import FixedRiskSizer
from execution.risk_managers import MaxDrawdownRiskManager


def load():
    df = pd.read_parquet(os.path.join(_PROJECT_ROOT, "data", "historical_data", "BTCUSDT_1m_full.parquet"))
    df["datetime"] = pd.to_datetime(df["datetime"])
    return (df[df["datetime"] < "2024-01-01"].reset_index(drop=True),
            df[df["datetime"] >= "2024-01-01"].reset_index(drop=True))


def bt(df, name, trig, lab, ind, rp=0.01):
    r = lightweight_backtest(df, trigger=trig, exit_labeler=lab,
        position_sizer=FixedRiskSizer(risk_pct=rp, max_leverage=20.0),
        risk_manager=MaxDrawdownRiskManager(max_dd_pct=0.30),
        indicator_params=ind, initial_capital=100000.0)
    print(f"{name:30s} sharpe={r['sharpe']:+6.2f} ret={r['total_return_pct']:+7.1f}% "
          f"dd={r['max_dd_pct']:5.1f}% trades={r['n_trades']:5d} win={r['win_rate']*100:3.0f}%")
    return r


def main():
    train, test = load()

    print("=== short-horizon trend breakout (momentum) ===")
    for ep in (720, 1440):
        for tag, data in (("TRAIN", train), ("TEST", test)):
            bt(data, f"Breakout ep={ep} {tag}",
               TrendFilteredBreakoutTrigger(entry_period=ep, trend_period=ep * 4),
               TrailingExitLabeler(trail_period=ep // 2),
               {"entry_period": ep, "exit_period": ep // 2, "atr_period": ep, "ma_period": ep * 4})

    print("\n=== short-horizon momentum (breakout, no trend filter) ===")
    for ep in (120, 240, 480):
        for tag, data in (("TRAIN", train), ("TEST", test)):
            bt(data, f"Momentum ep={ep} {tag}",
               TrendFilteredBreakoutTrigger(entry_period=ep, trend_period=ep * 4),
               TrailingExitLabeler(trail_period=ep // 2),
               {"entry_period": ep, "exit_period": ep // 2, "atr_period": ep, "ma_period": ep * 4})


if __name__ == "__main__":
    main()
