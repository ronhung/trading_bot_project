"""
Microstructure 1-minute Bar Aggregator.

Aggregates raw tick aggTrades (250M+ rows) into compact, highly informative
1-minute microstructure bars containing:
  - OHLCV (open, high, low, close, volume)
  - Taker Flow (taker_buy_base, taker_sell_base)
  - Order Flow Imbalance (ofi = (buy - sell) / total)
  - Volume Weighted Average Price (vwap)
  - Trade counts & turnover

Converts 3 years of raw ticks (~12 GB compressed Parquet) into a blazing-fast
~45 MB unified Parquet dataset ready for Phase 1 to Phase 5 simulation.
"""

import os
import glob
import sys
import argparse
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(_PROJECT_ROOT, "data", "historical_data", "aggtrades")
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "data", "historical_data", "microstructure_1m")
FINAL_FILE = os.path.join(_PROJECT_ROOT, "data", "historical_data", "BTCUSDT_1m_microstructure_3yr.parquet")


def aggregate_monthly_file(parquet_path: str) -> pd.DataFrame:
    """Read a raw monthly aggTrades parquet file and aggregate to 1m bars."""
    print(f"[Aggregating] {os.path.basename(parquet_path)} ...", flush=True)
    df = pd.read_parquet(parquet_path, columns=["price", "quantity", "transact_time", "is_buyer_maker"])
    
    # 1-minute bucket (in milliseconds)
    df["bar_time"] = (df["transact_time"] // 60000) * 60000
    df["quote_vol"] = df["price"] * df["quantity"]
    df["taker_buy_qty"] = np.where(~df["is_buyer_maker"], df["quantity"], 0.0).astype(np.float32)
    df["taker_buy_quote"] = np.where(~df["is_buyer_maker"], df["quote_vol"], 0.0).astype(np.float32)

    # Fast groupby aggregation
    grouped = df.groupby("bar_time")
    
    agg_df = pd.DataFrame({
        "open_time": grouped["bar_time"].first(),
        "open": grouped["price"].first(),
        "high": grouped["price"].max(),
        "low": grouped["price"].min(),
        "close": grouped["price"].last(),
        "volume": grouped["quantity"].sum(),
        "turnover": grouped["quote_vol"].sum(),
        "trades": grouped["price"].count().astype(np.int32),
        "taker_buy_base": grouped["taker_buy_qty"].sum(),
        "taker_buy_quote": grouped["taker_buy_quote"].sum(),
    }).reset_index(drop=True)

    agg_df["close_time"] = agg_df["open_time"] + 59999
    agg_df["datetime"] = pd.to_datetime(agg_df["open_time"], unit="ms")
    
    # Microstructural signals
    vol = agg_df["volume"].replace(0, np.nan)
    agg_df["vwap"] = (agg_df["turnover"] / vol).fillna(agg_df["close"]).astype(np.float32)
    taker_sell = agg_df["volume"] - agg_df["taker_buy_base"]
    agg_df["ofi"] = ((agg_df["taker_buy_base"] - taker_sell) / (agg_df["volume"] + 1e-9)).astype(np.float32)
    agg_df["taker_ratio"] = (agg_df["taker_buy_base"] / (agg_df["volume"] + 1e-9)).astype(np.float32)

    return agg_df


def process_all_files():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "BTCUSDT-aggTrades-*.parquet")))
    if not files:
        print("[Error] No aggTrades parquet files found in", INPUT_DIR)
        return

    monthly_dfs = []
    for fp in files:
        month_str = os.path.basename(fp).replace("BTCUSDT-aggTrades-", "").replace(".parquet", "")
        month_out = os.path.join(OUTPUT_DIR, f"BTCUSDT-1m-{month_str}.parquet")
        if os.path.exists(month_out):
            print(f"[Skip] {os.path.basename(month_out)} exists", flush=True)
            m_df = pd.read_parquet(month_out)
        else:
            m_df = aggregate_monthly_file(fp)
            m_df.to_parquet(month_out, compression="zstd")
            print(f"   Saved {os.path.basename(month_out)}: {len(m_df):,} bars", flush=True)
        monthly_dfs.append(m_df)

    if monthly_dfs:
        print("\n[Merging] Combining all monthly 1m bars into 3-year unified dataset...", flush=True)
        full_df = pd.concat(monthly_dfs, ignore_index=True)
        full_df.sort_values("open_time", inplace=True)
        full_df.drop_duplicates(subset=["open_time"], inplace=True)
        full_df.reset_index(drop=True, inplace=True)

        full_df.to_parquet(FINAL_FILE, compression="zstd")
        mb = os.path.getsize(FINAL_FILE) / 1024 / 1024
        print(f"[Done] Created {FINAL_FILE} with {len(full_df):,} bars ({full_df['datetime'].iloc[0]} to {full_df['datetime'].iloc[-1]}) - {mb:.1f} MB", flush=True)


if __name__ == "__main__":
    process_all_files()
