"""Data feed conversion for Backtrader."""

import backtrader as bt
import pandas as pd
import os

# ==========================================
# 1. custom Backtrader DataFeed for Binance Kline Data
# ==========================================
class BinanceKlineData(bt.feeds.PandasData):

    # Additional lines for Binance Kline Data
    lines = ('quote_volume', 'trades_count', 'taker_buy_base', 'taker_buy_quote',)

    # Mapping of DataFrame columns to Backtrader lines
    params = (
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', None), 
        
        ('quote_volume', 'quote_volume'),
        ('trades_count', 'trades_count'),
        ('taker_buy_base', 'taker_buy_base'),
        ('taker_buy_quote', 'taker_buy_quote'),
    )

# ==========================================
# 2. custom Backtrader DataFeed for Binance Funding Rate Data
# ==========================================
class BinanceFundingData(bt.feeds.PandasData):

    # Additional line for funding rate
    lines = ('funding_rate',)

    params = (
        ('datetime', None),
        ('open', None),
        ('high', None),
        ('low', None),
        ('close', None),
        ('volume', None),
        ('openinterest', None),
        
        ('funding_rate', 'funding_rate'),
    )

# ==========================================
# 3. Utility function to load Parquet files into Backtrader DataFeed
# ==========================================
def load_parquet_feed(filepath, feed_class):
    """
    Load a Parquet file and convert it into a Backtrader DataFeed.
    :param filepath: Path to the Parquet file.
    :param feed_class: The Backtrader DataFeed class to use (BinanceKlineData or BinanceFundingData).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_parquet(filepath)

    if 'datetime' in df.columns:
        # convert 'datetime' column to pandas datetime if it's not already
        df['datetime'] = pd.to_datetime(df['datetime'])
        # set 'datetime' as the index
        df.set_index('datetime', inplace=True)
    
    # sort the DataFrame by index to ensure chronological order
    df.sort_index(inplace=True)


    data_feed = feed_class(dataname=df)
    return data_feed


