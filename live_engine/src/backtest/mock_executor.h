#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include "../core/i_order_executor.h"
#include "../core/risk_manager.h"
#include "../core/kline_data.h"

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

class MockExecutor : public IOrderExecutor {
public:
    explicit MockExecutor(RiskManager* risk_manager,
                          double fee_rate = 0.0005,
                          double slippage_bps = 1.0);

    void send_order(const std::string& symbol,
                    const std::string& side,
                    double quantity,
                    double price,
                    bool reduce_only = false) override;

    void set_market(const KLineData& bar);
    // Returns true if a stop was triggered and a close was filled.
    bool check_and_execute_stop(const std::string& symbol);

    void export_trades_csv(const std::string& path) const;
    const std::vector<TradeRecord>& trades() const { return trades_; }

private:
    double apply_slippage(const std::string& side, double price) const;
    void record_trade(const std::string& symbol, const std::string& side,
                      double qty, double fill_price, double fee, double pnl,
                      const std::string& reason);

    RiskManager* risk_;
    double fee_rate_;
    double slippage_bps_; // 1 bps = 0.01%
    KLineData current_bar_;
    std::vector<TradeRecord> trades_;
};
