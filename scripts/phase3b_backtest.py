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


import argparse
import yaml
import importlib
from core.strategy_factory import build_strategy_from_config
from core.labeler import BaseLabeler


def _resolve_labeler(labeler_cfg: dict) -> BaseLabeler:
    l_type = labeler_cfg.get("type", "research.labeling.TrailingExitLabeler")
    l_params = labeler_cfg.get("params", {})
    if "." in l_type:
        mod_name, cls_name = l_type.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        cls = getattr(mod, cls_name)
    elif "trail" in l_type.lower():
        cls = TrailingExitLabeler
    else:
        cls = MeanReversionLabeler
    return cls(**l_params)


def run_from_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_source = cfg.get("data", {}).get("source", "data/historical_data/BTCUSDT_1m_full.parquet")
    if not os.path.isabs(data_source):
        data_source = os.path.join(_PROJECT_ROOT, data_source)

    df = pd.read_parquet(data_source)
    df["datetime"] = pd.to_datetime(df.get("datetime", pd.to_datetime(df.get("open_time", 0), unit="ms")))

    wrapper, meta = build_strategy_from_config(cfg)
    trigger = wrapper._trigger
    sizer = wrapper._sizer
    risk_mgr = wrapper._risk_manager
    ind_params = meta["indicator_params"]

    labeler_cfg = cfg.get("labeler", {"type": "research.labeling.TrailingExitLabeler", "params": {"trail_period": ind_params["exit_period"]}})
    labeler = _resolve_labeler(labeler_cfg)

    train_range = cfg.get("data", {}).get("train_range", None)
    test_range = cfg.get("data", {}).get("test_range", None)

    exec_cfg = cfg.get("execution", {})
    entry_cfg = exec_cfg.get("entry", {})
    entry_order_type = entry_cfg.get("order_type", "market")
    maker_comm = float(exec_cfg.get("maker_commission", 0.0002))
    taker_comm = float(exec_cfg.get("taker_commission", 0.0005))
    limit_timeout = int(entry_cfg.get("timeout_bars", 2))

    name = cfg.get("strategy", {}).get("name", "Strategy")
    print(f"\n=======================================================")
    print(f"Phase 3b Backtest: {name} (Config-Driven Vectorized)")
    print(f"=======================================================")

    exec_kwargs = {
        "entry_order_type": entry_order_type,
        "maker_commission": maker_comm,
        "taker_commission": taker_comm,
        "limit_timeout_bars": limit_timeout,
    }

    if train_range and len(train_range) == 2:
        train_df = df[(df["datetime"] >= train_range[0]) & (df["datetime"] < train_range[1])].reset_index(drop=True)
        print(f"\n--- TRAIN SET ({train_range[0]} to {train_range[1]}) : {len(train_df):,} bars ---")
        bt(train_df, f"{name} [TRAIN]", trigger, labeler, ind_params, sizer=sizer, risk_mgr=risk_mgr, **exec_kwargs)

    if test_range and len(test_range) == 2:
        test_df = df[(df["datetime"] >= test_range[0]) & (df["datetime"] <= test_range[1])].reset_index(drop=True)
        print(f"\n--- TEST SET ({test_range[0]} to {test_range[1]}) : {len(test_df):,} bars ---")
        bt(test_df, f"{name} [TEST]", trigger, labeler, ind_params, sizer=sizer, risk_mgr=risk_mgr, **exec_kwargs)

    print(f"\n--- FULL 3-YEAR DATASET ({df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}) : {len(df):,} bars ---")
    bt(df, f"{name} [FULL 3-YEAR]", trigger, labeler, ind_params, sizer=sizer, risk_mgr=risk_mgr, **exec_kwargs)


def bt(df, name, trigger, labeler, indicator_params, sizer=None, risk_mgr=None, risk_pct=0.01, **kwargs):
    result = lightweight_backtest(
        df,
        trigger=trigger,
        exit_labeler=labeler,
        position_sizer=sizer or FixedRiskSizer(risk_pct=risk_pct, max_leverage=20.0),
        risk_manager=risk_mgr or MaxDrawdownRiskManager(max_dd_pct=0.30),
        indicator_params=indicator_params,
        initial_capital=100000.0,
        **kwargs,
    )
    print(f"[{name}] sharpe={result['sharpe']:+.2f} ret={result['total_return_pct']:+.1f}% "
          f"dd={result['max_dd_pct']:.1f}% trades={result['n_trades']} win={result['win_rate']*100:.0f}%")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="Path to strategy YAML config")
    args = ap.parse_args()

    if args.config:
        run_from_config(args.config)
        return

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
