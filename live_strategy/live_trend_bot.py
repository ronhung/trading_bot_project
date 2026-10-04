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

from core.order_payload import TrailingExitIndicator, BracketExit
from core.strategy_wrapper import StrategyWrapper
from execution.sizers import FixedRiskSizer
from execution.risk_managers import MaxDrawdownRiskManager
from research.features import default_feature_set
from research.triggers.adam_breakout import AdamBreakoutTrigger
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
        period: int = 43200,
        trail_period: int = 14400,
        hard_stop_mode: str = "donchian",
        atr_mult: float = 2.0,
        risk_pct: float = 0.10,
        max_leverage: float = 100.0,
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
            entry_period=period,
            atr_period=period,
            warmup=warmup,
        )

        # -- Execution gateway (shares ZMQ client with feeder) --
        self.gateway = BinanceZmqExecutionGateway(self.feeder.client)

        # -- Strategy components (DI — swap objects to change strategy) --
        trigger = AdamBreakoutTrigger(period=period)
        features = default_feature_set()
        sizer = FixedRiskSizer(risk_pct=risk_pct, max_leverage=max_leverage)
        risk_manager = MaxDrawdownRiskManager(max_dd_pct=max_dd_pct)
        bracket_exit = BracketExit(
            trailing_exit_indicator=TrailingExitIndicator.DONCHIAN_LOW,
            trailing_exit_period=trail_period,
            hard_stop_mode=hard_stop_mode,
            atr_mult=atr_mult,
        )
        indicator_params = {
            "entry_period": period,
            "exit_period": trail_period,
            "atr_period": period,
        }

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
            bracket_exit=bracket_exit,
            indicator_params=indicator_params,
            signal_threshold=signal_threshold,
            symbol=symbol,
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
    parser = argparse.ArgumentParser(description="Adam Trailing Trading Bot")
    parser.add_argument(
        "--no-warmup", action="store_true",
        help="Skip REST warmup (use in C++ backtest mode)",
    )
    parser.add_argument(
        "--period", type=int, default=43200,
        help="Breakout lookback (bars; 43200 = 30 days at 1m)",
    )
    parser.add_argument(
        "--trail-period", type=int, default=14400,
        help="Trailing exit lookback (bars; 14400 = 10 days at 1m)",
    )
    parser.add_argument(
        "--hard-stop-mode", type=str, default="donchian",
        help="Hard stop mode: donchian | atr | none",
    )
    parser.add_argument(
        "--atr-mult", type=float, default=2.0,
        help="ATR multiplier for hard stop (only used in atr mode)",
    )
    parser.add_argument(
        "--risk-pct", type=float, default=0.10,
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
    print("Adam Trailing Trading Bot")
    print(f"  Symbol: BTCUSDT")
    print(f"  Params: period={args.period}, trail={args.trail_period}, "
          f"hard_stop={args.hard_stop_mode}, risk_pct={args.risk_pct}")
    print("=" * 50)

    bot = LiveTurtleBot(
        period=args.period,
        trail_period=args.trail_period,
        hard_stop_mode=args.hard_stop_mode,
        atr_mult=args.atr_mult,
        risk_pct=args.risk_pct,
        warmup=not args.no_warmup,
        model_path=args.model,
        feature_list_path=args.features,
        signal_threshold=args.threshold,
    )
    bot.start()
