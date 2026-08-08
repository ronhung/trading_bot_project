"""
BinanceZmqDataFeeder — ZMQ-based market data feed implementing LiveDataFeeder ABC.

Extracts buffer management, warmup, and C++ state sync from the original
LiveTurtleBot. Wraps the unchanged BinanceZmqClient for ZMQ transport.
"""

from collections import deque
from datetime import datetime
from typing import Any, Callable, Dict, Optional

import pandas as pd
import requests

from core.data_feeder import LiveDataFeeder
from live_strategy.zmq_client import BinanceZmqClient


class BinanceZmqDataFeeder(LiveDataFeeder):
    """
    Live data feeder over ZMQ from the C++ engine.

    Buffers kline data for strategy computation, syncs C++ RiskManager
    state (position, balance, stop_price) on every bar, and supports
    REST warmup to pre-fill the Donchian buffer.
    """

    KLINE_FIELDS = [
        "symbol", "open_time", "close_time",
        "open", "high", "low", "close",
        "volume", "quote_volume",
        "taker_buy_base", "taker_buy_quote",
        "trades_count", "is_closed",
    ]

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        entry_period: int = 20,
        atr_period: int = 20,
        warmup: bool = True,
        host: str = "localhost",
        sub_port: int = 5555,
        push_port: int = 5556,
    ):
        """
        Args:
            symbol: Trading pair.
            entry_period: Donchian lookback (bars) — determines buffer size.
            atr_period: ATR smoothing period.
            warmup: If True, REST fetch historical bars to pre-fill buffer.
            host: ZMQ host.
            sub_port: ZMQ SUB port (market data from C++).
            push_port: ZMQ PUSH port (signals to C++).
        """
        self.symbol = symbol
        self._entry_period = entry_period
        self._atr_period = atr_period
        self._do_warmup = warmup

        # Buffer for strategy computation
        self._max_len = max(entry_period, atr_period) + 1
        self.kline_buffer: deque = deque(maxlen=self._max_len)

        # C++ RiskManager state (synced on every kline)
        self.current_position: float = 0.0
        self.available_balance: float = 0.0
        self.current_stop: float = 0.0

        # ZMQ client
        self._client = BinanceZmqClient(
            host=host, sub_port=sub_port, push_port=push_port,
        )
        self._on_bar_callback: Optional[Callable] = None

    # -- LiveDataFeeder interface -------------------------------------------

    def connect(self) -> None:
        self._client.set_kline_callback(self._on_kline)
        self._client.set_order_update_callback(self._on_order_update)
        self._client.connect()

    def start(self, on_bar_callback: Callable) -> None:
        """
        Begin listening for market data.

        Args:
            on_bar_callback: Called as on_bar_callback(bar_data, portfolio_state)
                for each closed bar.
        """
        self._on_bar_callback = on_bar_callback
        if self._do_warmup:
            self._warmup_buffer()
        self._client.start_listening()

    # -- Warmup -------------------------------------------------------------

    def warmup(self) -> None:
        """Public warmup (can be called before start())."""
        self._warmup_buffer()

    def _warmup_buffer(self) -> None:
        """
        Fetch historical 1m klines from Binance REST to pre-fill the
        Donchian channel buffer. Eliminates cold-start waiting period.
        """
        window_size = max(self._entry_period, self._atr_period) + 1
        fetch_limit = window_size + 5

        url = "https://fapi.binance.com/fapi/v1/klines"
        params = {
            "symbol": self.symbol,
            "interval": "1m",
            "limit": fetch_limit,
        }

        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            raw_klines = resp.json()
        except Exception as e:
            print(f"  [Warmup] REST fetch failed: {e}")
            print("    Starting cold — waiting for real-time bars...")
            return

        if not raw_klines:
            print("  [Warmup] REST returned empty klines array")
            return

        # Drop the still-forming (in-progress) candle
        closed = raw_klines[:-1] if len(raw_klines) > 1 else raw_klines
        for k in closed:
            self.kline_buffer.append(self._kline_from_rest(k, self.symbol))

        last_ct = self.kline_buffer[-1]["close_time"]
        last_dt = datetime.fromtimestamp(last_ct / 1000.0).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        print(
            f"  [Warmup] {len(closed)} bars buffered, "
            f"last closed @ {last_dt}"
        )

    # -- ZMQ callbacks ------------------------------------------------------

    def _on_kline(self, data: Dict[str, Any]) -> None:
        """Called by BinanceZmqClient on each kline message."""
        # Sync C++ state
        self.current_position = data.get("current_position", 0.0)
        self.available_balance = data.get("available_balance", 0.0)
        self.current_stop = data.get("stop_price", 0.0)

        if data.get("is_closed") is True:
            kline = self._kline_from_zmq(data)
            self.kline_buffer.append(kline)

            if self._on_bar_callback is not None:
                portfolio_state = {
                    "current_position": self.current_position,
                    "available_balance": self.available_balance,
                    "stop_price": self.current_stop,
                    "equity": self.available_balance,  # simplified
                    "current_drawdown": 0.0,            # C++ provides this
                }
                self._on_bar_callback(kline, portfolio_state)

    def _on_order_update(self, data: Dict[str, Any]) -> None:
        """Called by BinanceZmqClient on order status updates."""
        status = data.get("status", "?")
        cid = data.get("client_order_id", "?")
        side = data.get("side", "?")
        qty = data.get("quantity", 0)
        px = data.get("price", 0)
        reason = data.get("reason", "")
        extra = f" reason={reason}" if reason else ""
        print(f"  [Order] {cid}: {status} {side} qty={qty} @ {px}{extra}")

    # -- Kline parsing (moved verbatim from LiveTurtleBot) ------------------

    @staticmethod
    def _kline_from_zmq(data: dict) -> dict:
        """Extract KLineData fields from a C++ ZMQ kline message."""
        return {
            k: data[k]
            for k in BinanceZmqDataFeeder.KLINE_FIELDS if k in data
        }

    @staticmethod
    def _kline_from_rest(k: list, symbol: str) -> dict:
        """Build a KLineData dict from Binance REST /fapi/v1/klines row."""
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

    @property
    def client(self) -> BinanceZmqClient:
        """Access the underlying ZMQ client (for gateway sharing)."""
        return self._client
