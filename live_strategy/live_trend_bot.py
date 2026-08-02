import os
import sys
import pandas as pd
from collections import deque
from datetime import datetime

# ==========================================
# 1. Path navigation magic (same as the backtesting side)
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from zmq_client import BinanceZmqClient
# ✅ Import the shared math brain across folders
from shared.core_logic.turtle_math import calculate_turtle_signals 

class LiveTurtleBot:
    def __init__(self, symbol="BTCUSDT"):
        self.symbol = symbol
        
        # live trading parameters (recommended: align these values with your best Backtrader results)
        self.entry_period = 2
        self.exit_period = 1
        self.atr_period = 2
        self.atr_mult = 2.0
        
        # buffer size setting (keep enough history for Donchian channels and ATR)
        self.max_len = max(self.entry_period, self.atr_period) + 1
        self.kline_buffer = deque(maxlen=self.max_len)

        # Real state synced from C++ RiskManager (updated on every kline)
        self.current_position = 0.0
        self.available_balance = 0.0
        self.current_stop = 0.0
        
        self.client = BinanceZmqClient(sub_port=5555, push_port=5556)
        self.client.set_kline_callback(self.on_handle_market_data)
        self.client.set_order_update_callback(self.on_order_update)
        
        print(f"🐢 [Live brain] {self.symbol} Turtle trading bot is online!")
        print("-" * 50)

    def on_handle_market_data(self, data):
        # Sync real position/balance state from C++ RiskManager on every message
        self.current_position = data.get("current_position", 0.0)
        self.available_balance = data.get("available_balance", 0.0)
        self.current_stop = data.get("stop_price", 0.0)

        if data.get("is_closed") == True:
            # ensure the high, low, close fields required by the brain are extracted
            cleaned_kline = {
                "close_time": data["close_time"],
                "high": data["high"],
                "low": data["low"],
                "close": data["close"]
            }
            self.kline_buffer.append(cleaned_kline)
            self.execute_strategy_logic()

    def on_order_update(self, data):
        """Handle order status updates pushed from C++ (FILLED, CANCELED, etc.)."""
        status = data.get("status", "?")
        cid = data.get("client_order_id", "?")
        side = data.get("side", "?")
        qty = data.get("quantity", 0)
        px = data.get("price", 0)
        reason = data.get("reason", "")
        reduce_only = data.get("reduce_only", False)
        extra = f" reason={reason}" if reason else ""
        ro = " [reduceOnly]" if reduce_only else ""
        print(f"📋 [Order] {cid}: {status} {side} qty={qty} @ {px}{ro}{extra}")

    def execute_strategy_logic(self):
        # if there's not enough history to calculate the channels, wait
        if len(self.kline_buffer) < max(self.entry_period, self.atr_period) + 1:
            print(f"⏳ Collecting/warming up historical data... ({len(self.kline_buffer)}/{self.max_len})")
            return

        # convert to DataFrame
        df = pd.DataFrame(self.kline_buffer)

        # ==========================================
        # 🧠 Call the shared brain! (completely free of live execution logic)
        # ==========================================
        signal, stop_price = calculate_turtle_signals(
            df,
            self.entry_period,
            self.exit_period,
            self.atr_period,
            self.atr_mult
        )

        current = df.iloc[-1]
        dt_str = datetime.fromtimestamp(current['close_time'] / 1000.0).strftime('%H:%M:%S')
        price = current['close']

        # ==========================================
        # Execute the command issued by the brain
        # (gated on real position state from C++ RiskManager)
        # ==========================================
        flat = abs(self.current_position) < 1e-8

        # 🔍 DEBUG: always print state so we can verify C++ state sync
        print(f"[{dt_str}] 📡 C++ state: pos={self.current_position:.4f} | bal={self.available_balance:.2f} | stop={self.current_stop:.4f} | flat={flat} | signal={signal}")

        if signal == 1:
            if not flat:
                print(f"[{dt_str}] 🔒 Long breakout but pos={self.current_position:.4f} (not flat); skip BUY")
                return
            print(f"[{dt_str}] 🚀 [Long signal] broke above upper band! price: {price:.2f} | stop set at: {stop_price:.2f}")
            self.client.send_order_signal("BUY", self.symbol, price, stop_price)

        elif signal == -1:
            if not flat:
                print(f"[{dt_str}] 🔒 Short breakout but pos={self.current_position:.4f} (not flat); skip SELL")
                return
            print(f"[{dt_str}] 💥 [Short signal] broke below lower band! price: {price:.2f} | stop set at: {stop_price:.2f}")
            self.client.send_order_signal("SELL", self.symbol, price, stop_price)

        elif signal == 2:
            if self.current_position <= 0:
                print(f"[{dt_str}] ⚪ Close-Long signal but pos={self.current_position:.4f}; skip")
                return
            print(f"[{dt_str}] 🛑 [Close Long] fell below short-term low! price: {price:.2f}")
            self.client.send_order_signal("CLOSE_LONG", self.symbol, price, 0)

        elif signal == -2:
            if self.current_position >= 0:
                print(f"[{dt_str}] ⚪ Close-Short signal but pos={self.current_position:.4f}; skip")
                return
            print(f"[{dt_str}] 🛑 [Close Short] broke above short-term high! price: {price:.2f}")
            self.client.send_order_signal("CLOSE_SHORT", self.symbol, price, 0)
        else:
            print(f"[{dt_str}] ⚪ Standing by... (close: {price:.2f} | pos={self.current_position:.4f} | bal={self.available_balance:.2f})")

    def start(self):
        self.client.connect()
        self.client.start_listening()

if __name__ == "__main__":
    bot = LiveTurtleBot()
    bot.start()