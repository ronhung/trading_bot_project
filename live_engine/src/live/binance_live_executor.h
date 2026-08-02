#pragma once
#include <string>
#include "../core/i_order_executor.h"

class BinanceLiveExecutor : public IOrderExecutor {
public:
    BinanceLiveExecutor(const std::string& api_key, const std::string& secret_key);

    void send_order(const std::string& symbol,
                    const std::string& side,
                    double quantity,
                    double price,
                    bool reduce_only = false) override;

    bool get_initial_state(double& out_usdt_balance, double& out_btcusdt_position);
    std::string get_listen_key();

private:
    std::string api_key_;
    std::string secret_key_;
    std::string generate_signature(const std::string& query_string);
};
