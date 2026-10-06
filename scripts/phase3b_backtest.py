"""Phase 3b: vectorized backtest (profitability check) for both strategies.

Usage:
  python scripts/phase3b_backtest.py
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd

from research.backtest import lightweight_backtest
from research.labeling import MeanReversionLabeler, TrailingExitLabeler
from research.triggers.bollinger_mean_reversion import BollingerMeanReversionTrigger
from research.triggers.trend_breakout import TrendFilteredBreakoutTrigger
from execution.sizers import FixedRiskSizer
from execution.risk_managers import MaxDrawdownRiskManager


def load():
    path = os.path.join(_PROJECT_ROOT, "data", "historical_data", "BTCUSDT_1m_full.parquet")
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    train = df[df["datetime"] < "2024-01-01"].reset_index(drop=True)
    test = df[df["datetime"] >= "2024-01-01"].reset_index(drop=True)
    return train, test


def bt(df, name, trigger, labeler, indicator_params, risk_pct=0.01):
    result = lightweight_backtest(
        df,
        trigger=trigger,
        exit_labeler=labeler,
        position_sizer=FixedRiskSizer(risk_pct=risk_pct, max_leverage=20.0),
        risk_manager=MaxDrawdownRiskManager(max_dd_pct=0.30),
        indicator_params=indicator_params,
        initial_capital=100000.0,
    )
    print(f"[{name}] sharpe={result['sharpe']:+.2f} ret={result['total_return_pct']:+.1f}% "
          f"dd={result['max_dd_pct']:.1f}% trades={result['n_trades']} win={result['win_rate']*100:.0f}%")
    return result


def main():
    train, test = load()
    print(f"train {len(train):,} bars | test {len(test):,} bars\n")

    # Strategy A — high-freq Bollinger mean reversion
    a_trigger = BollingerMeanReversionTrigger(period=120, num_std=2.0)
    a_labeler = MeanReversionLabeler(period=120, num_std=2.0, max_hold=240)
    a_ind = {"entry_period": 120, "exit_period": 60, "atr_period": 120, "ma_period": 120}

    # Strategy B — low-freq trend breakout
    b_trigger = TrendFilteredBreakoutTrigger(entry_period=7200, trend_period=28800)
    b_labeler = TrailingExitLabeler(trail_period=2880)
    b_ind = {"entry_period": 7200, "exit_period": 2880, "atr_period": 7200, "ma_period": 28800}

    print("=== TRAIN (2020-2023) ===")
    bt(train, "A: Bollinger MR", a_trigger, a_labeler, a_ind)
    bt(train, "B: Trend breakout", b_trigger, b_labeler, b_ind)

    print("\n=== TEST (2024+) ===")
    bt(test, "A: Bollinger MR", a_trigger, a_labeler, a_ind)
    bt(test, "B: Trend breakout", b_trigger, b_labeler, b_ind)


if __name__ == "__main__":
    main()
