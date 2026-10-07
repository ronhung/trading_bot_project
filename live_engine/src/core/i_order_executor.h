#pragma once
#include <string>
#include <cstdint>
#include <functional>

// Execution parameters carried on an order: how to place it + how to handle an
// unfilled (resting) order. Serialized from Python's ExecutionSpec and consumed
// by both the live executor and the backtest mock so they stay in parity.
struct ExecutionSpec {
    std::string order_type = "MARKET";       // MARKET / LIMIT / LIMIT_MAKER
    std::string time_in_force = "GTC";       // GTC / IOC / FOK / GTX (LIMIT only)
    int timeout_ms = 0;                      // 0 = never chase (MARKET fills immediately)
    std::string unfilled_policy = "CANCEL";  // CANCEL / REPRICE / MARKET
    int max_reprice_attempts = 2;            // REPRICE only
};

// A single order to place, plus the execution spec governing it.
struct OrderRequest {
    std::string symbol;
    std::string side;               // BUY / SELL
    double quantity = 0.0;
    double price = 0.0;
    bool reduce_only = false;
    ExecutionSpec execution;
};

// Shared order status update struct — used by both live and backtest paths.
// Lives in core/ so IpcServer can reference it without depending on live/.
struct OrderStatusUpdate {
    std::string symbol;
    std::string client_order_id;   // our client-generated ID (primary correlation key)
    int64_t  order_id = 0;         // server-assigned, parsed from REST/WS for audit only
    std::string side;              // BUY / SELL
    std::string order_type;        // LIMIT / MARKET
    double   quantity = 0.0;
    double   filled_quantity = 0.0;// cumulative filled qty (0 for a never-filled cancel)
    double   price = 0.0;
    std::string status;            // FILLED / CANCELED / EXPIRED / REJECTED (terminal states)
    bool     reduce_only = false;
    std::string reason;            // "", "timeout_reprice", "timeout_market", "timeout_abandoned", ...
};

// Abstract execution interface — live and backtest share the same call site.
class IOrderExecutor {
public:
    virtual ~IOrderExecutor() = default;

    // Returns true if the order was accepted and is now tracked; false if rejected.
    virtual bool send_order(const OrderRequest& req) = 0;

    // Set a callback for order status updates (FILLED, CANCELED, etc.).
    // Default no-op — MockExecutor doesn't need it.
    virtual void set_order_status_callback(std::function<void(const OrderStatusUpdate&)>) {}

    // Returns true if there is any open (NEW / PARTIALLY_FILLED) order.
    // Used by IpcServer to reject duplicate open signals.
    virtual bool has_open_order() const { return false; }

    // Arm a trailing exit for the just-opened position. The executor reads
    // the current side + stop from its RiskManager. Default no-op.
    virtual void arm_trailing_exit(const std::string& indicator, int period) {
        (void)indicator; (void)period;
    }

    // Store the EXIT execution spec (travels with the entry order). Used later
    // when the executor places the trailing-stop / hard-stop close on its own.
    virtual void set_exit_execution(const ExecutionSpec& exit_exec) {
        (void)exit_exec;
    }

    // Query the current position on the exchange for the given symbol.
    // Returns false if the executor cannot verify the position.
    virtual bool get_current_position(const std::string& symbol, double& out_position) {
        (void)symbol;
        out_position = 0.0;
        return false;
    }
};
