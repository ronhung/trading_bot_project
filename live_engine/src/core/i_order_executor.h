#pragma once
#include <string>

// Abstract execution interface — live and backtest share the same call site.
class IOrderExecutor {
public:
    virtual ~IOrderExecutor() = default;

    // side: "BUY" or "SELL"
    virtual void send_order(const std::string& symbol,
                            const std::string& side,
                            double quantity,
                            double price,
                            bool reduce_only = false) = 0;
};
