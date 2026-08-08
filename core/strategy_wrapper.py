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
    ):
        """
        Args:
            trigger: Event trigger for entry detection.
            feature: Feature computer for ML input vector.
            model: Trained ML model with .predict(X) method (or None for no ML filter).
            feature_names: Ordered list of feature names matching model input.
            sizer: Position size calculator.
            risk_manager: Risk gate (False = block all entries).
            signal_threshold: Minimum model prediction to fire.
            symbol: Trading pair.
            trailing_exit_indicator: Exit rule for C++ bracket order.
            trailing_exit_period: Lookback bars for trailing exit.
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

        # --- Compute entry signal ---
        signal_series = self._trigger.generate_signals(df)
        signal = int(signal_series.iloc[-1])

        if signal == 0:
            return None  # no event at this bar

        # Determine action from signal
        if signal == 1:
            action = Action.BUY
        elif signal == -1:
            action = Action.SELL
        else:
            return None

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
        stop_distance = 2.0 * atr_val
        if action == Action.BUY:
            hard_stop = close - stop_distance
        else:
            hard_stop = close + stop_distance

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
