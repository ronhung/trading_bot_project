"""
Live Turtle Trading Bot — composition shell using the unified framework.

Wires together:
  - BinanceZmqDataFeeder   (implements LiveDataFeeder)
  - BinanceZmqExecutionGateway (implements LiveExecutionGateway)
  - StrategyWrapper         (implements entry logic + bracket order protocol)

The bot itself is thin — all strategy logic lives in the injected components.
Switching from backtest to live is a config change, not a code change.
"""

import argparse
import json
import os
import sys

# Project root for cross-package imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.order_payload import TrailingExitIndicator
from core.strategy_wrapper import StrategyWrapper
from execution.sizers import VolatilityTargetingSizer
from execution.risk_managers import MaxDrawdownRiskManager
from research.features import default_feature_set
from research.triggers.turtle_breakout import TurtleBreakoutTrigger
from live_strategy.zmq_feeder import BinanceZmqDataFeeder
from live_strategy.zmq_gateway import BinanceZmqExecutionGateway


class LiveTurtleBot:
    """
    Composition root for the live trading bot.

    Owns the feeder, gateway, and StrategyWrapper. The feeder drives
    the event loop; on each bar, StrategyWrapper.on_bar() decides whether
    to send an OrderPayload. The gateway transmits it to C++.
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        entry_period: int = 20,
        exit_period: int = 10,
        atr_period: int = 20,
        atr_mult: float = 2.0,
        intensity_threshold: float = 0.0,
        risk_pct: float = 0.02,
        max_leverage: float = 20.0,
        max_dd_pct: float = 0.05,
        warmup: bool = True,
        model_path: str | None = None,
        feature_list_path: str | None = None,
        signal_threshold: float = 0.0,
    ):
        self.symbol = symbol

        # -- Data feeder --
        self.feeder = BinanceZmqDataFeeder(
            symbol=symbol,
            entry_period=entry_period,
            atr_period=atr_period,
            warmup=warmup,
        )

        # -- Execution gateway (shares ZMQ client with feeder) --
        self.gateway = BinanceZmqExecutionGateway(self.feeder.client)

        # -- Strategy components (DI) --
        trigger = TurtleBreakoutTrigger(
            entry_period=entry_period,
            atr_period=atr_period,
            atr_mult=atr_mult,
            intensity_threshold=intensity_threshold,
        )
        features = default_feature_set()
        sizer = VolatilityTargetingSizer(
            risk_pct=risk_pct, max_leverage=max_leverage,
        )
        risk_manager = MaxDrawdownRiskManager(max_dd_pct=max_dd_pct)

        # -- ML model (optional) --
        model = None
        feature_names: list = []
        if model_path is not None and feature_list_path is not None:
            import xgboost as xgb
            model = xgb.XGBRegressor()
            model.load_model(model_path)
            with open(feature_list_path, "r") as f:
                feature_names = json.load(f)
            print(f"  [ML] Loaded model: {model_path}")
            print(f"  [ML] Features ({len(feature_names)}): {feature_names}")

        # -- StrategyWrapper --
        self.strategy = StrategyWrapper(
            trigger=trigger,
            feature=features,
            model=model,
            feature_names=feature_names,
            sizer=sizer,
            risk_manager=risk_manager,
            signal_threshold=signal_threshold,
            symbol=symbol,
            trailing_exit_indicator=TrailingExitIndicator.DONCHIAN_LOW,
            trailing_exit_period=exit_period,
        )

        # Wire position_closed callback
        self.gateway.set_position_closed_callback(
            self.strategy.on_position_closed
        )

    def start(self) -> None:
        """Connect and begin the event loop."""
        self.feeder.connect()
        self.feeder.start(self._on_bar)

    def _on_bar(self, bar_data: dict, portfolio_state: dict) -> None:
        """
        Called by feeder on each closed bar.

        Delegates to StrategyWrapper for entry decision. If an OrderPayload
        is returned, sends it via the gateway.
        """
        order = self.strategy.on_bar(bar_data, portfolio_state)
        if order is not None:
            self.gateway.send_order(order)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Turtle Trading Bot")
    parser.add_argument(
        "--no-warmup", action="store_true",
        help="Skip REST warmup (use in C++ backtest mode)",
    )
    parser.add_argument(
        "--entry", type=int, default=20,
        help="Entry Donchian period (bars)",
    )
    parser.add_argument(
        "--exit", type=int, default=10,
        help="Exit Donchian period (bars)",
    )
    parser.add_argument(
        "--atr-period", type=int, default=20,
        help="ATR smoothing period",
    )
    parser.add_argument(
        "--atr-mult", type=float, default=2.0,
        help="ATR multiplier for stop distance",
    )
    parser.add_argument(
        "--risk-pct", type=float, default=0.02,
        help="Risk per trade as decimal",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to XGBoost model JSON",
    )
    parser.add_argument(
        "--features", type=str, default=None,
        help="Path to feature list JSON",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.0,
        help="ML signal threshold",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("Live Turtle Trading Bot")
    print(f"  Symbol: BTCUSDT")
    print(f"  Params: entry={args.entry}, exit={args.exit}, "
          f"atr_period={args.atr_period}, atr_mult={args.atr_mult}")
    print("=" * 50)

    bot = LiveTurtleBot(
        entry_period=args.entry,
        exit_period=args.exit,
        atr_period=args.atr_period,
        atr_mult=args.atr_mult,
        risk_pct=args.risk_pct,
        warmup=not args.no_warmup,
        model_path=args.model,
        feature_list_path=args.features,
        signal_threshold=args.threshold,
    )
    bot.start()
