#pragma once
#include <string>
#include <cstdint>
#include <functional>

// Shared order status update struct — used by both live and backtest paths.
// Lives in core/ so IpcServer can reference it without depending on live/.
struct OrderStatusUpdate {
    std::string symbol;
    std::string client_order_id;   // our client-generated ID (primary correlation key)
    int64_t  order_id = 0;         // server-assigned, parsed from REST/WS for audit only
    std::string side;              // BUY / SELL
    std::string order_type;        // LIMIT / MARKET
    double   quantity = 0.0;
    double   price = 0.0;
    std::string status;            // FILLED / CANCELED / EXPIRED / REJECTED (terminal states)
    bool     reduce_only = false;
    std::string reason;            // "", "timeout", "timeout_exhausted", ...
};

// Abstract execution interface — live and backtest share the same call site.
class IOrderExecutor {
public:
    virtual ~IOrderExecutor() = default;

    // side: "BUY" or "SELL"
    // Returns true if the order was accepted and is now tracked; false if rejected.
    virtual bool send_order(const std::string& symbol,
                            const std::string& side,
                            double quantity,
                            double price,
                            bool reduce_only = false) = 0;

    // Set a callback for order status updates (FILLED, CANCELED, etc.).
    // Default no-op — MockExecutor doesn't need it.
    virtual void set_order_status_callback(std::function<void(const OrderStatusUpdate&)>) {}

    // Returns true if there is any open (NEW / PARTIALLY_FILLED) order.
    // Used by IpcServer to reject duplicate open signals.
    virtual bool has_open_order() const { return false; }
};
