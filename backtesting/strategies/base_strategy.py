"""Base strategy with common risk and position management."""

import backtrader as bt
import math

class BaseStrategy(bt.Strategy):
    """
    Base class for quantitative strategies.
    Responsible for handling non-alpha housekeeping tasks:
    1. Funding rate settlement
    2. Order status tracking and logging
    3. Dynamic position sizing
    """
    # Default parameters for risk management and logging
    params = (
        ('risk_pct', 0.02),       # max risk per trade (2% of total capital)
        ('print_log', True),      # whether to print trading logs
    )

    def __init__(self):
        # 1. bind data feeds
        self.kline_1m = self.datas[0]
        self.kline_1d = self.datas[1]

        self.funding = self.datas[2] if len(self.datas) > 2 else None
        
        # 2. state tracking variables
        self.order = None              # current active order
        self.last_funding_time = None  # timestamp of last funding payment
        self.total_funding_paid = 0.0  # total funding paid (for analytics)

    def next(self):
        # 1. check funding rate and apply fees
        self._check_funding_rate()

        # 2. hand control to subclass-implemented strategy logic
        self.on_bar()

    def on_bar(self):
        """
        Subclasses (e.g., TrendFollowingStrategy) must override this method
        to implement indicator checks and buy/sell conditions.
        """
        raise NotImplementedError("Subclasses must implement on_bar()!")

    def _check_funding_rate(self):
        """Core logic: simulate funding rate settlement for Binance USD-M futures."""
        # If there is no position or no funding data, skip
        if not self.position or not self.funding:
            return

        current_time = self.funding.datetime.datetime(0)
        
        # Funding is settled every 8 hours on Binance USD-M futures. We check if the current bar's timestamp is different from the last funding settlement time.
        if self.last_funding_time != current_time:
            self.last_funding_time = current_time

            # funding_fee = position value * current funding rate
            # For Binance USD-M: position value = size * current contract price
            position_value = self.position.size * self.kline_1m.close[0]
            current_rate = self.funding.funding_rate[0]
            
            # Calculate funding fee. Positive means longs pay shorts; negative means shorts pay longs.
            funding_fee = position_value * current_rate
            
            # Directly adjust the broker's cash balance.
            # This is the standard way to simulate funding in Backtrader.
            self.broker.add_cash(-funding_fee)
            self.total_funding_paid += funding_fee

            # Optional: log large funding deductions
            # self.log(f"⏰ Funding settlement: rate {current_rate*100:.4f}%, amount {-funding_fee:.2f} USDT")

    def calculate_size_by_risk(self, entry_price, stop_loss_price):
        """
        Risk management: calculate safe order size based on stop loss distance
        and total account value.
        Formula: size = (account_value * risk_pct) / loss_per_unit
        """
        if entry_price == stop_loss_price:
            return 0.0
            
        account_value = self.broker.getvalue()
        risk_amount = account_value * self.p.risk_pct
        
        # Calculate the distance between entry and stop loss
        stop_distance = abs(entry_price - stop_loss_price)
        
        # Calculate the target size based on risk amount and stop distance
        target_size = risk_amount / stop_distance
        
        # Round down to 3 decimal places to avoid fractional contracts
        target_size = math.floor(target_size * 1000) / 1000.0
        return target_size

    def notify_order(self, order):
        """Triggered when an order status changes (pushed by Broker)."""
        if order.status in [order.Submitted, order.Accepted]:
            # Order is still pending, do nothing
            return

        if order.status in [order.Completed]:
            direction = 'Buy (Long)' if order.isbuy() else 'Sell (Short)'
            self.log(f"✅ {direction} executed: price {order.executed.price:.2f}, size {order.executed.size}, cost {order.executed.comm:.4f}")
            self.order = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"❌ Order failed/canceled: status {order.getstatusname()}")
            self.order = None

    def notify_trade(self, trade):
        """Triggered when a trade is fully closed (entry then exit)."""
        if not trade.isclosed:
            return
        self.log(f"💰 Trade closed - Net PnL (after fees): {trade.pnlcomm:.2f} USDT")

    def log(self, txt, dt=None):
        """Unified log output format."""
        if self.p.print_log:
            dt = dt or self.kline_1m.datetime.datetime(0)
            print(f"[{dt}] {txt}")