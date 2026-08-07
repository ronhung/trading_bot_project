"""Main backtesting program to run Cerebro and generate performance reports."""

import backtrader as bt
import pandas as pd
import datetime
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import our custom modules
from data_feeds import load_parquet_feed, BinanceKlineData, BinanceFundingData
from strategies.trend_following import TurtleDonchianStrategy

def main():
    # 1. Initialize Cerebro engine
    cerebro = bt.Cerebro()

    print("⏳ Loading Parquet data, please wait...")
    
    # Define the test time range (recommended: run 3 months first, then full 6 years)
    # When loading DataFeed, you can pass fromdate and todate directly
    start_date = datetime.datetime(2023, 1, 1)
    end_date = datetime.datetime(2024, 1, 1)

    # Resolve current script directory and build data paths dynamically
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kline_path = os.path.join(base_dir, "data", "historical_data", "BTCUSDT_1m_full.parquet")
    funding_path = os.path.join(base_dir, "data", "historical_data", "BTCUSDT_funding_rate_full.parquet")

    # 2. Load and add K-line data
    kline_feed_1m = load_parquet_feed(kline_path, BinanceKlineData)
    # Backtrader's PandasData supports slicing directly
    kline_feed_1m.p.fromdate = start_date
    kline_feed_1m.p.todate = end_date
    cerebro.adddata(kline_feed_1m, name="BTCUSDT_1m")

    kline_feed_1d = load_parquet_feed(kline_path, BinanceKlineData)
    kline_feed_1d.p.fromdate = start_date
    kline_feed_1d.p.todate = end_date
    cerebro.resampledata(
        kline_feed_1d, 
        timeframe=bt.TimeFrame.Minutes, 
        compression=1440, 
        name="BTCUSDT_1d"
    )


    # 3. Load and add funding rate data
    funding_feed = load_parquet_feed(funding_path, BinanceFundingData)
    funding_feed.p.fromdate = start_date
    funding_feed.p.todate = end_date
    cerebro.adddata(funding_feed, name="FundingRate")

    # 4. Add strategy (override parameters here for testing)
    cerebro.addstrategy(
        TurtleDonchianStrategy,
        entry_period=20,     # entry: breakout of 20-bar high/low
        exit_period=10,      # exit: breakdown of 10-bar high/low
        atr_period=20,
        atr_multiplier=2.0
    )

    # 5. Configure broker starting cash and commission
    # Assume starting cash is 10,000 USDT, Binance taker fees are about 0.04% - 0.05%
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.0005) 

    # 6. Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, annualize=True, riskfreerate=0.0, factor=365.0)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='strategy_returns')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, data=kline_feed_1d, _name='benchmark_bnh_returns')
    print(f"💰 Starting portfolio value: {cerebro.broker.getvalue():.2f} USDT")
    print("🚀 Starting backtest...")

    # 7. Run the backtest engine
    results = cerebro.run()
    strat = results[0]

    # 8. Print performance report
    print(f"💰 Final portfolio value: {cerebro.broker.getvalue():.2f} USDT")
    
    # Extract analyzer results
    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    trades = strat.analyzers.trades.get_analysis()
    strategy_returns = strat.analyzers.strategy_returns.get_analysis()
    benchmark_bnh_returns = strat.analyzers.benchmark_bnh_returns.get_analysis()

    print("\n========== 📊 Performance Report ==========")
    print(f"Annualized Sharpe Ratio: {sharpe.get('sharperatio', 'N/A')}")
    print(f"Max Drawdown: {drawdown.get('max', {}).get('drawdown', 0):.2f}%")
    print(f"Total Return (Strategy): {((1 + pd.Series(strategy_returns)).prod() - 1) * 100:.2f}%")
    print(f"Total Return (Buy & Hold): {((1 + pd.Series(benchmark_bnh_returns)).prod() - 1) * 100:.2f}%")
    total_trades = trades.get('total', {}).get('closed', 0)
    won_trades = trades.get('won', {}).get('total', 0)
    win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
    
    print(f"Total trades: {total_trades}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Total funding paid: {strat.total_funding_paid:.2f} USDT")
    print("===================================")

    # 9. Plot charts (warning: 1m data is huge; comment this out for full dataset to avoid crashing)
    # cerebro.plot(style='candlestick', volume=False, barup='green', bardown='red')

if __name__ == '__main__':
    main()