"""
Export microstructure Parquet bars to C++ DataReplayer CSV format.
"""

import os
import sys
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_PATH = os.path.join(_PROJECT_ROOT, "data", "historical_data", "BTCUSDT_1m_microstructure_3yr.parquet")
CSV_PATH = os.path.join(_PROJECT_ROOT, "data", "historical_data", "BTCUSDT_1m_microstructure_sample.csv")


def export_sample(n_bars: int = 10000, start_date: str = "2023-01-01"):
    df = pd.read_parquet(PARQUET_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    if start_date:
        df = df[df["datetime"] >= start_date].reset_index(drop=True)
    
    if n_bars > 0:
        df = df.iloc[:n_bars].copy()

    # Required 11 columns for C++ DataReplayer:
    # open_time,open,high,low,close,volume,close_time,quote_volume,trades_count,taker_buy_base,taker_buy_quote
    out_df = pd.DataFrame({
        "open_time": df["open_time"].astype(int),
        "open": df["open"],
        "high": df["high"],
        "low": df["low"],
        "close": df["close"],
        "volume": df["volume"],
        "close_time": df["close_time"].astype(int),
        "quote_volume": df["turnover"],
        "trades_count": df["trades"].astype(int),
        "taker_buy_base": df["taker_buy_base"],
        "taker_buy_quote": df["taker_buy_quote"],
    })

    out_df.to_csv(CSV_PATH, index=False)
    print(f"[Exported] {len(out_df):,} bars to {CSV_PATH}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    export_sample(n_bars=n)
