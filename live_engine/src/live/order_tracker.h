#pragma once
#include <string>
#include <unordered_map>
#include <mutex>
#include <chrono>
#include <functional>
#include <vector>
#include "../core/i_order_executor.h"

enum class TrackedOrderStatus { NEW, PARTIALLY_FILLED, FILLED, CANCELED, EXPIRED, REJECTED };

struct TrackedOrder {
    std::string client_order_id;
    int64_t order_id = 0;          // server-assigned id from REST response (audit)
    std::string symbol;
    std::string side;              // "BUY" or "SELL"
    double quantity = 0.0;
    double filled_quantity = 0.0;  // cumulative filled qty from WS (o.z)
    double price = 0.0;
    bool reduce_only = false;
    TrackedOrderStatus status = TrackedOrderStatus::NEW;
    int reprice_attempts = 0;
    std::chrono::steady_clock::time_point created_at;
    bool reported_terminal = false;  // dedupe publishes across threads
};

class OrderTracker {
public:
    using OrderUpdateCallback = std::function<void(const OrderStatusUpdate&)>;

    void set_update_callback(OrderUpdateCallback cb);

    // Called right after a successful send_order POST — registers the order.
    void register_order(const TrackedOrder& order);

    // Called from ORDER_TRADE_UPDATE WebSocket handler.
    // o: the "o" sub-object from Binance's ORDER_TRADE_UPDATE payload.
    void on_order_update(const std::string& client_order_id,
                         const std::string& status,        // NEW, PARTIALLY_FILLED, FILLED, CANCELED...
                         double filled_qty);

    // Returns orders that have been in NEW or PARTIALLY_FILLED state for > timeout_ms.
    std::vector<TrackedOrder> get_timed_out_orders(int timeout_ms);

    // Mark an order as cancelled after the watchdog's cancel succeeded.
    // Marks terminal only (does NOT publish) — the caller publishes the single
    // CANCELED update; this flag de-dupes the subsequent WS CANCELED event.
    void mark_cancel_requested(const std::string& client_order_id);

    // Check if any order is currently open (NEW or PARTIALLY_FILLED).
    bool has_open_order() const;

    // Prune orders that are terminal and older than age_ms.
    void prune(int age_ms = 600000);  // default 10 minutes

private:
    void fire_update(const TrackedOrder& order);

    mutable std::mutex mtx_;
    std::unordered_map<std::string, TrackedOrder> orders_;
    OrderUpdateCallback on_update_;
};
