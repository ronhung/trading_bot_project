"""
Phase 5-6: StrategyWrapper — bridges research (Phase 3) to execution (Phase 5/6).

Loads a trained ML model and assembles trigger + features + sizer + risk
from dependency-injected components. Called per-bar by the execution engine.

RESPONSIBILITY BOUNDARY:
  Python handles ENTRY only. Once an OrderPayload is sent, C++ manages the
  full bracket-order lifecycle. Python waits for POSITION_CLOSED callback
  before re-enabling entry detection.
"""

from enum import Enum, auto
from typing import Optional, Dict, Any, List
import logging

import numpy as np
import pandas as pd

from core.order_payload import OrderPayload, Action, TrailingExitIndicator
from core.trigger import BaseEventTrigger
from core.feature import BaseFeature
from core.position_sizer import BasePositionSizer
from core.risk_manager import BaseRiskManager
from shared.core_logic.turtle_math import calculate_turtle_signals

logger = logging.getLogger(__name__)


class StrategyState(Enum):
    """Python-side position tracking state."""
    IDLE = auto()            # detecting entries
    WAITING_CLOSE = auto()   # position open; C++ manages exit


class StrategyWrapper:
    """
    Bridge: Phase 3 trained model → Phase 5/6 execution engine.

    On each bar:
      1. If WAITING_CLOSE → return None (C++ is managing the exit).
      2. Check risk limits → block if exceeded.
      3. Compute signal via trigger.
      4. If entry signal → compute feature vector → ML model predict.
      5. If prediction > threshold → calculate size via sizer.
      6. Assemble OrderPayload with bracket exit parameters.
      7. Set state = WAITING_CLOSE, return OrderPayload.

    C++ sends POSITION_CLOSED → on_position_closed() → state = IDLE.
    """

    def __init__(
        self,
        trigger: BaseEventTrigger,
        feature: BaseFeature,
        model: Optional[Any],
        feature_names: List[str],
        sizer: BasePositionSizer,
        risk_manager: BaseRiskManager,
        signal_threshold: float = 0.0,
        symbol: str = "BTCUSDT",
        trailing_exit_indicator: TrailingExitIndicator = TrailingExitIndicator.DONCHIAN_LOW,
        trailing_exit_period: int = 14400,
        entry_period: int = 20,
        exit_period: int = 10,
        atr_period: int = 20,
        atr_mult: float = 2.0,
        intensity_threshold: float = 0.0,
    ):
        """
        Args:
            trigger: Event trigger for entry detection (research path).
            feature: Feature computer for ML input vector.
            model: Trained ML model with .predict(X) method.
            feature_names: Ordered list of feature names matching model input.
            sizer: Position size calculator.
            risk_manager: Risk gate (False = block all entries).
            signal_threshold: Minimum model prediction to fire.
            symbol: Trading pair.
            trailing_exit_indicator: Exit rule for C++ bracket order.
            trailing_exit_period: Lookback bars for trailing exit.
            entry_period, exit_period, atr_period, atr_mult,
            intensity_threshold: Passed to calculate_turtle_signals()
                (shared/core_logic/turtle_math.py) — the single source
                of truth for execution-path signal generation.
        """
        self._trigger = trigger
        self._feature = feature
        self._model = model
        self._feature_names = feature_names
        self._sizer = sizer
        self._risk_manager = risk_manager
        self._signal_threshold = signal_threshold
        self._symbol = symbol
        self._trailing_exit_indicator = trailing_exit_indicator
        self._trailing_exit_period = trailing_exit_period

        # Turtle params for shared brain (execution path)
        self._entry_period = entry_period
        self._exit_period = exit_period
        self._atr_period = atr_period
        self._atr_mult = atr_mult
        self._intensity_threshold = intensity_threshold

        self._state: StrategyState = StrategyState.IDLE
        self._kline_buffer: List[Dict[str, Any]] = []

    # -- public API -------------------------------------------------------

    @property
    def state(self) -> StrategyState:
        return self._state

    def on_bar(
        self, bar_data: Dict[str, Any], portfolio_state: Dict[str, Any]
    ) -> Optional[OrderPayload]:
        """
        Called each time step by the execution engine.

        Args:
            bar_data: Dict with OHLCV fields (open, high, low, close, volume,
                      open_time, close_time, ...). Matches C++ KLineData struct.
            portfolio_state: Dict with C++ RiskManager state:
                current_position, available_balance, stop_price.

        Returns:
            OrderPayload if an entry should be placed, None otherwise.
        """
        # --- Gate: waiting for C++ to close position ---
        if self._state == StrategyState.WAITING_CLOSE:
            return None

        # --- Gate: risk limits ---
        if not self._risk_manager.check_risk_limits(portfolio_state):
            logger.debug("Risk limits blocked entry")
            return None

        # --- Build DataFrame from buffer + current bar ---
        self._kline_buffer.append(bar_data)
        df = pd.DataFrame(self._kline_buffer)

        # --- Compute entry signal via shared brain (single source of truth) ---
        signal, stop_price = calculate_turtle_signals(
            df,
            self._entry_period,
            self._exit_period,
            self._atr_period,
            self._atr_mult,
            self._intensity_threshold,
        )

        if signal == 0 or stop_price is None:
            return None  # no event at this bar

        # Map turtle_math signal codes to Action
        # 1=long entry, -1=short entry, 2=close long, -2=close short
        if signal == 1:
            action = Action.BUY
        elif signal == -1:
            action = Action.SELL
        else:
            return None  # exit signals (2, -2) not handled here — C++ manages exits

        # --- Compute features at current bar ---
        feat_dict = self._feature.compute_one(df, len(df) - 1)

        # --- ML filter ---
        ml_score: float = 0.0
        if self._model is not None:
            feature_vec = np.array(
                [[feat_dict.get(name, 0.0) for name in self._feature_names]],
                dtype=np.float32,
            )
            ml_score = float(self._model.predict(feature_vec)[0])
            if ml_score <= self._signal_threshold:
                logger.debug(
                    "ML filter suppressed %s: pred=%.4f <= threshold=%.2f",
                    action.value, ml_score, self._signal_threshold,
                )
                return None

        # --- Compute position size ---
        close = float(bar_data["close"])
        atr_val = feat_dict.get("atr", 0.0)
        if atr_val <= 0:
            atr_val = 1.0  # defensive fallback

        equity = float(portfolio_state.get("available_balance", 0.0))
        size = self._sizer.calculate_size(
            signal_strength=2.0,   # Turtle: ATR multiplier for stop
            current_atr=atr_val,
            account_equity=equity,
            entry_price=close,
        )
        if size <= 0.0:
            logger.debug("Sizer returned zero size; skipping entry")
            return None

        # --- Compute bracket exit parameters ---
        # Use stop_price from calculate_turtle_signals() — the shared brain
        hard_stop = stop_price

        # --- Assemble OrderPayload ---
        order = OrderPayload(
            action=action,
            symbol=self._symbol,
            quantity=size,
            entry_price=close,
            hard_stop_loss=hard_stop,
            trailing_exit_indicator=self._trailing_exit_indicator,
            trailing_exit_period=self._trailing_exit_period,
        )

        # --- Transition to WAITING_CLOSE ---
        self._state = StrategyState.WAITING_CLOSE
        logger.info(
            "ENTRY %s: price=%.2f size=%.4f stop=%.2f ml_score=%.4f",
            action.value, close, size, hard_stop, ml_score,
        )

        return order

    def on_position_closed(self, close_info: Dict[str, Any]) -> None:
        """
        Callback from execution engine: position has been fully closed.

        Args:
            close_info: Dict with reason, entry_price, exit_price, pnl.
        """
        reason = close_info.get("reason", "unknown")
        pnl = close_info.get("pnl", 0.0)
        logger.info("POSITION_CLOSED: reason=%s pnl=%.2f", reason, pnl)
        self._state = StrategyState.IDLE

    def reset(self) -> None:
        """Reset state and clear buffer (e.g., for backtest restart)."""
        self._state = StrategyState.IDLE
        self._kline_buffer.clear()

    @classmethod
    def from_yaml(cls, config_path: str) -> "StrategyWrapper":
        """
        Factory: instantiate StrategyWrapper from a YAML config file.

        The config must contain: trigger, features, position_sizer,
        risk_manager, model (optional), bracket, execution sections.

        Uses the same component resolution as pipeline_runner.py
        (importlib-based dynamic instantiation from type paths).
        """
        import importlib
        import json

        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        def _instantiate(spec: dict):
            type_path = spec["type"]
            params = spec.get("params", {})
            module_path, class_name = type_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            cls_ = getattr(module, class_name)
            return cls_(**params)

        # Trigger (research path; execution uses turtle_math directly)
        trigger = _instantiate(cfg["trigger"])

        # Features
        feature_specs = cfg["features"]
        if isinstance(feature_specs, list):
            from research.features import CompositeFeature
            feature = CompositeFeature([_instantiate(s) for s in feature_specs])
        else:
            feature = _instantiate(feature_specs)

        # Sizer + risk
        sizer = _instantiate(cfg["position_sizer"])
        risk_manager = _instantiate(cfg["risk_manager"])

        # Model (optional)
        model = None
        feature_names: list = []
        model_cfg = cfg.get("model", {})
        if model_cfg.get("path"):
            import xgboost as xgb
            model = xgb.XGBRegressor()
            model.load_model(model_cfg["path"])
            with open(model_cfg["feature_list"], "r") as f:
                feature_names = json.load(f)

        # Execution
        exec_cfg = cfg.get("execution", {})
        bracket_cfg = cfg.get("bracket", {})

        trailing_map = {
            "donchian_low": TrailingExitIndicator.DONCHIAN_LOW,
            "donchian_high": TrailingExitIndicator.DONCHIAN_HIGH,
            "moving_average": TrailingExitIndicator.MOVING_AVERAGE,
        }

        return cls(
            trigger=trigger,
            feature=feature,
            model=model,
            feature_names=feature_names,
            sizer=sizer,
            risk_manager=risk_manager,
            signal_threshold=model_cfg.get("threshold", 0.0),
            symbol=exec_cfg.get("symbol", "BTCUSDT"),
            trailing_exit_indicator=trailing_map.get(
                bracket_cfg.get("trailing_exit_indicator", "donchian_low"),
                TrailingExitIndicator.DONCHIAN_LOW,
            ),
            trailing_exit_period=bracket_cfg.get("trailing_exit_period", 14400),
            entry_period=cfg["trigger"]["params"].get("entry_period", 20),
            exit_period=bracket_cfg.get("trailing_exit_period", 10),
            atr_period=cfg["trigger"]["params"].get("atr_period", 20),
            atr_mult=cfg["trigger"]["params"].get("atr_mult", 2.0),
            intensity_threshold=cfg["trigger"]["params"].get("intensity_threshold", 0.0),
        )
