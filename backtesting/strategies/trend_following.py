import backtrader as bt
import pandas as pd
from .base_strategy import BaseStrategy
from shared.core_logic.turtle_math import calculate_turtle_signals

class TurtleDonchianStrategy(BaseStrategy):
    params = (
        ('entry_period', 20),
        ('exit_period', 10),
        ('atr_period', 20),
        ('atr_multiplier', 2.0),
    )

    def __init__(self):
        super().__init__()
        # Removed all bt.indicators declarations
        self.stop_price = None
        self.max_lookback = max(self.p.entry_period, self.p.atr_period) + 2

    def on_bar(self):
        # Ensure there is enough K-line history available
        if len(self.kline_1d) < self.max_lookback:
            return
        if self.order:
            return

        # 1. Extract Backtrader's internal stream data into arrays
        highs = self.kline_1d.high.get(size=self.max_lookback)
        lows = self.kline_1d.low.get(size=self.max_lookback)
        closes = self.kline_1d.close.get(size=self.max_lookback)

        # 2. Pack into a DataFrame
        df = pd.DataFrame({
            'high': highs,
            'low': lows,
            'close': closes
        })

        # 3. Call the shared brain (the live engine calls the same function)
        signal, new_stop_price = calculate_turtle_signals(
            df, 
            self.p.entry_period, 
            self.p.exit_period, 
            self.p.atr_period, 
            self.p.atr_multiplier
        )

        current_close = self.kline_1m.close[0]

        # 4. Execute actions based on the brain's signal
        if not self.position:
            if signal == 1:
                self.stop_price = new_stop_price
                target_size = self.calculate_size_by_risk(current_close, self.stop_price)
                if target_size > 0: 
                    self.order = self.buy(size=target_size)
            elif signal == -1:
                self.stop_price = new_stop_price
                target_size = self.calculate_size_by_risk(current_close, self.stop_price)
                if target_size > 0: 
                    self.order = self.sell(size=target_size)
        else:
            # When holding a position, check the shared brain's close signal or absolute stop loss
            if self.position.size > 0 and (signal == 2 or current_close <= self.stop_price):
                self.order = self.close()
            elif self.position.size < 0 and (signal == -2 or current_close >= self.stop_price):
                self.order = self.close()