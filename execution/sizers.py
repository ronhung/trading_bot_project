"""Concrete position sizer implementations."""

import math

from core.position_sizer import BasePositionSizer


class VolatilityTargetingSizer(BasePositionSizer):
    """
    Turtle-style volatility-targeted position sizing.

    Risk per trade = risk_pct * account_equity.
    Stop distance = atr_mult * ATR.
    Size = risk_amount / stop_distance, capped by max_leverage.

    This reproduces the sizing logic in:
      - C++ RiskManager::calculate_target_size()
      - backtesting/strategies/base_strategy.py::calculate_size_by_risk()
      - research/backtest.py::lightweight_backtest() inline sizing
    """

    def __init__(
        self,
        risk_pct: float = 0.02,
        max_leverage: float = 20.0,
        min_size: float = 0.001,
    ):
        """
        Args:
            risk_pct: Fraction of equity risked per trade (0.02 = 2%).
            max_leverage: Maximum notional leverage cap.
            min_size: Minimum trade size (exchange minimum).
        """
        self.risk_pct = risk_pct
        self.max_leverage = max_leverage
        self.min_size = min_size

    def calculate_size(
        self,
        signal_strength: float,
        current_atr: float,
        account_equity: float,
        entry_price: float | None = None,
        stop_distance: float | None = None,
    ) -> float:
        """
        Compute position size using Turtle N-value logic.

        Args:
            signal_strength: ATR multiplier for stop distance (e.g., 2.0).
            current_atr: Current ATR value in price units.
            account_equity: Available balance in quote currency.
            entry_price: Entry price for leverage cap.

        Returns:
            Position size in base units (e.g., BTC). 0.0 if invalid inputs.
        """
        if account_equity <= 0.0:
            return 0.0
        if current_atr <= 0.0 or signal_strength <= 0.0:
            return 0.0

        stop_distance = signal_strength * current_atr
        if stop_distance <= 0.0:
            return 0.0

        # Risk-based size
        risk_amount = account_equity * self.risk_pct
        size = risk_amount / stop_distance

        # Leverage cap
        if entry_price is not None and entry_price > 0.0:
            max_notional = account_equity * self.max_leverage
            size = min(size, max_notional / entry_price)

        # Floor to min_size precision
        size = math.floor(size / self.min_size) * self.min_size

        return size if size >= self.min_size else 0.0


class FixedRiskSizer(BasePositionSizer):
    """
    Fixed-risk position sizing: risk a fixed fraction of equity per the actual
    stop distance supplied by the exit labeler.

        size = (account_equity * risk_pct) / stop_distance, capped by max_leverage.

    Used for trailing-stop / barrier exits where the stop distance is a known
    per-event value (entry → initial stop), not an ATR-derived quantity.
    """

    def __init__(self, risk_pct: float = 0.10, max_leverage: float = 100.0, min_size: float = 0.001):
        self.risk_pct = risk_pct
        self.max_leverage = max_leverage
        self.min_size = min_size

    def calculate_size(
        self,
        signal_strength: float,
        current_atr: float,
        account_equity: float,
        entry_price: float | None = None,
        stop_distance: float | None = None,
    ) -> float:
        if account_equity <= 0.0:
            return 0.0
        if stop_distance is None or stop_distance <= 0.0:
            return 0.0

        size = (account_equity * self.risk_pct) / stop_distance

        if entry_price is not None and entry_price > 0.0:
            max_notional = account_equity * self.max_leverage
            size = min(size, max_notional / entry_price)

        size = math.floor(size / self.min_size) * self.min_size
        return size if size >= self.min_size else 0.0



class HalfKellySizer(BasePositionSizer):
    """
    Half-Kelly Position Sizer with Hard Leverage/Risk Caps.

    Calculates trade risk percentage based on the Half-Kelly criterion:
        f* = (p * b - q) / b
        f_half = 0.5 * f*
        risk_pct = min(max(0.0, f_half), max_risk_cap)
        size = (account_equity * risk_pct) / stop_distance

    Where:
        p = win rate (or dynamic ML predicted probability)
        b = payout ratio (average win / average loss)
        q = 1 - p
    """

    def __init__(
        self,
        win_rate: float = 0.45,
        profit_factor: float = 2.0,
        max_risk_cap: float = 0.05,
        max_leverage: float = 10.0,
        min_size: float = 0.001,
        **kwargs,
    ):
        self.win_rate = float(win_rate)
        self.profit_factor = float(profit_factor)
        self.max_risk_cap = float(max_risk_cap)
        self.max_leverage = float(max_leverage)
        self.min_size = float(min_size)

    def calculate_size(
        self,
        signal_strength: float,
        current_atr: float,
        account_equity: float,
        entry_price: float | None = None,
        stop_distance: float | None = None,
    ) -> float:
        if account_equity <= 0.0 or stop_distance is None or stop_distance <= 0.0:
            return 0.0

        p = signal_strength if 0.0 < signal_strength < 1.0 else self.win_rate
        b = self.profit_factor
        q = 1.0 - p

        full_kelly = (p * b - q) / b if b > 0.0 else 0.0
        if full_kelly <= 0.0:
            return 0.0

        half_kelly = 0.5 * full_kelly
        risk_pct = min(half_kelly, self.max_risk_cap)

        size = (account_equity * risk_pct) / stop_distance

        if entry_price is not None and entry_price > 0.0:
            max_notional = account_equity * self.max_leverage
            size = min(size, max_notional / entry_price)

        size = math.floor(size / self.min_size) * self.min_size
        return size if size >= self.min_size else 0.0
