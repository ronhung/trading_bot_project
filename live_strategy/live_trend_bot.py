import os
import sys
import pandas as pd
import requests
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
    # Field names matching C++ KLineData struct in live_engine/src/core/kline_data.h.
    # Every bar stored in kline_buffer uses exactly this set of keys.
    KLINE_FIELDS = [
        "symbol", "open_time", "close_time",
        "open", "high", "low", "close",
        "volume", "quote_volume",
        "taker_buy_base", "taker_buy_quote",
        "trades_count", "is_closed",
    ]

    @staticmethod
    def _kline_from_zmq(data: dict) -> dict:
        """Extract KLineData fields from a C++ ZMQ kline message."""
        return {k: data[k] for k in LiveTurtleBot.KLINE_FIELDS if k in data}

    @staticmethod
    def _kline_from_rest(k: list, symbol: str) -> dict:
        """Build a KLineData dict from a Binance REST /fapi/v1/klines row.
        Array indices: [0:open_time, 1:open, 2:high, 3:low, 4:close,
        5:volume, 6:close_time, 7:quote_volume, 8:trades_count,
        9:taker_buy_base, 10:taker_buy_quote, 11:ignore]
        REST only returns closed bars, so is_closed is always True.
        """
        return {
            "symbol": symbol,
            "open_time": k[0],
            "close_time": k[6],
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "quote_volume": float(k[7]),
            "taker_buy_base": float(k[9]),
            "taker_buy_quote": float(k[10]),
            "trades_count": k[8],
            "is_closed": True,
        }

    def __init__(self, symbol="BTCUSDT", warmup=True):
        self.symbol = symbol
        self._do_warmup = warmup
        
        # live trading parameters (recommended: align these values with your best Backtrader results)
        self.entry_period = 2
        self.exit_period = 1
        self.atr_period = 2
        self.atr_mult = 2.0
        self.intensity_threshold = 0.0
        
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
            # Store the full KLineData struct (all 13 fields) so any future
            # strategy can access every field C++ sends.
            kline = self._kline_from_zmq(data)
            self.kline_buffer.append(kline)
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

    def warmup_buffer(self):
        """
        Fetch historical 1m klines from Binance REST API to pre-fill the
        Donchian channel buffer.  Eliminates the cold-start waiting period.

        Strategy window = max(entry_period, atr_period) + 1 bars.
        We fetch a few extra bars to absorb any timing gap between the
        REST snapshot and the first ZMQ bar from C++.
        """
        window_size = max(self.entry_period, self.atr_period) + 1
        fetch_limit = window_size + 5

        url = "https://fapi.binance.com/fapi/v1/klines"
        params = {
            "symbol": self.symbol,
            "interval": "1m",
            "limit": fetch_limit
        }

        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            raw_klines = resp.json()
        except Exception as e:
            print(f"⚠️  [Warmup] REST fetch failed: {e}")
            print("    Starting cold — waiting for real-time bars to fill buffer...")
            return

        if not raw_klines:
            print("⚠️  [Warmup] REST returned empty klines array")
            return

        # Binance includes the still-forming (in-progress) candle as the last
        # element. Drop it so we only buffer fully-closed bars — using the
        # partial bar would inject a duplicate/revised row into the strategy
        # buffer and skew the Donchian/ATR channels.
        closed_klines = raw_klines[:-1] if len(raw_klines) > 1 else raw_klines
        for k in closed_klines:
            self.kline_buffer.append(self._kline_from_rest(k, self.symbol))

        last_ct = self.kline_buffer[-1]["close_time"]
        last_dt = datetime.fromtimestamp(last_ct / 1000.0).strftime("%Y-%m-%d %H:%M:%S")
        print(f"🔥 [Warmup] Fetched {len(closed_klines)} closed bars "
              f"({len(raw_klines) - len(closed_klines)} forming bar skipped), "
              f"{len(self.kline_buffer)} kept in buffer (maxlen={self.kline_buffer.maxlen}), "
              f"last closed @ {last_dt}")

        # If the buffer is already full after warmup, the next real-time bar
        # will immediately trigger execute_strategy_logic().

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
            self.atr_mult,
            self.intensity_threshold
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
        if self._do_warmup:
            # Pre-fill the Donchian buffer so we can trade immediately.
            # Skip in backtest mode (--no-warmup) to avoid mixing live
            # REST bars with historical CSV replay data.
            self.warmup_buffer()
        self.client.connect()
        self.client.start_listening()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Live Turtle Trading Bot")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip REST warmup (use in C++ backtest mode)")
    args = parser.parse_args()
    bot = LiveTurtleBot(warmup=not args.no_warmup)
    bot.start()