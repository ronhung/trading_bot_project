import pandas as pd
import numpy as np

def calculate_turtle_signals(df: pd.DataFrame, entry_period: int, exit_period: int, atr_period: int, atr_mult: float):
    """
    Pure math function: calculate Turtle Trading entry/exit signals and stop price.
    Returns: (signal, stop_price)
             signal: 1 (long), -1 (short), 2 (close long), -2 (close short), 0 (hold)
    """
    if len(df) < max(entry_period, atr_period) + 1:
        return 0, None

    # 1. Calculate channel breakouts (Shift(1) avoids lookahead bias)
    df['entry_high'] = df['high'].rolling(entry_period).max().shift(1)
    df['entry_low'] = df['low'].rolling(entry_period).min().shift(1)
    df['exit_high'] = df['high'].rolling(exit_period).max().shift(1)
    df['exit_low'] = df['low'].rolling(exit_period).min().shift(1)

    # 2. Calculate ATR (simple True Range average)
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = (df['high'] - df['prev_close']).abs()
    df['tr3'] = (df['low'] - df['prev_close']).abs()
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].rolling(atr_period).mean().shift(1) # use the previous bar's ATR

    current = df.iloc[-1]
    
    # 3. Determine the signal
    signal = 0
    stop_price = None

    if current['close'] > current['entry_high']:
        signal = 1
        stop_price = current['close'] - (current['atr'] * atr_mult)
    elif current['close'] < current['entry_low']:
        signal = -1
        stop_price = current['close'] + (current['atr'] * atr_mult)
    elif current['close'] < current['exit_low']:
        signal = 2 # close long signal
    elif current['close'] > current['exit_high']:
        signal = -2 # close short signal

    return signal, stop_price