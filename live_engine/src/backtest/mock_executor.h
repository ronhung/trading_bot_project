#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <deque>
#include "../core/i_order_executor.h"
#include "../core/risk_manager.h"
#include "../core/kline_data.h"
#include "../core/trailing_stop.h"

struct TradeRecord {
    uint64_t timestamp_ms = 0;
    std::string symbol;
    std::string side;
    double quantity = 0.0;
    double price = 0.0;
    double fee = 0.0;
    double pnl = 0.0;
    double balance_after = 0.0;
    double position_after = 0.0;
    std::string reason;
};

struct PendingLimitOrder {
    std::string symbol;
    std::string side;
    double price = 0.0;
    double quantity = 0.0;
    double queue_ahead = 0.0;
    bool reduce_only = false;
    ExecutionSpec execution;
    uint64_t created_time_ms = 0;
};

class MockExecutor : public IOrderExecutor {
public:
    explicit MockExecutor(RiskManager* risk_manager,
                          double fee_rate = 0.0005,
                          double slippage_bps = 1.0);

    bool send_order(const OrderRequest& req) override;

    void set_market(const KLineData& bar);
    bool get_current_position(const std::string& symbol, double& out_position) override;
    bool check_and_execute_stop(const std::string& symbol);
    void arm_trailing_exit(const std::string& indicator, int period) override;
    void set_exit_execution(const ExecutionSpec& exit_exec) override;

    void export_trades_csv(const std::string& path) const;
    const std::vector<TradeRecord>& trades() const { return trades_; }

private:
    double apply_slippage(const std::string& side, double price, double notional = 0.0) const;
    void record_trade(const std::string& symbol, const std::string& side,
                      double qty, double fill_price, double fee, double pnl,
                      const std::string& reason);
    bool execute_fill(const std::string& symbol, const std::string& side,
                      double quantity, double fill_price, bool reduce_only,
                      const std::string& reason);
    void process_pending_orders(const KLineData& bar);

    RiskManager* risk_;
    double fee_rate_;
    double slippage_bps_; // 1 bps = 0.01%
    KLineData current_bar_;
    TrailingStop trailing_stop_;
    ExecutionSpec exit_execution_;
    std::vector<TradeRecord> trades_;
    std::deque<PendingLimitOrder> pending_orders_;
};
