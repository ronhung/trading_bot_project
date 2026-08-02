#pragma once
#include <string>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <functional>
#include "../core/i_order_executor.h"
#include "order_tracker.h"

class BinanceLiveExecutor : public IOrderExecutor {
public:
    BinanceLiveExecutor(const std::string& api_key, const std::string& secret_key);
    ~BinanceLiveExecutor() override;

    // IOrderExecutor overrides
    bool send_order(const std::string& symbol,
                    const std::string& side,
                    double quantity,
                    double price,
                    bool reduce_only = false) override;

    void set_order_status_callback(std::function<void(const OrderStatusUpdate&)>) override;
    bool has_open_order() const override;

    // Initial state query (called once at startup)
    bool get_initial_state(double& out_usdt_balance, double& out_btcusdt_position);

    // User data stream listen key
    std::string get_listen_key();

    // Cancel an order by clientOrderId
    bool cancel_order(const std::string& symbol, const std::string& client_order_id);

    // Order monitor lifecycle (called from main)
    void start_order_monitor();
    void stop_order_monitor();

    // Called from private WS ORDER_TRADE_UPDATE handler
    void on_order_update(const std::string& client_order_id,
                         const std::string& status,
                         double filled_qty);

private:
    // Signature generation for Binance REST API
    std::string generate_signature(const std::string& query_string);

    // Generate a unique clientOrderId
    std::string next_client_order_id();

    // Get current market price via REST ticker
    double get_market_price(const std::string& symbol);

    // Internal: place a LIMIT order (used by send_order and reprice)
    bool place_order_internal(const std::string& symbol,
                              const std::string& side,
                              double quantity,
                              double price,
                              bool reduce_only,
                              int reprice_attempts);

    // Reprice a timed-out order at current market
    void reprice_order(const TrackedOrder& ord);

    // Monitor thread: periodic timeout check + cancel + reprice
    void monitor_loop();

    // Fire status update through the callback
    void publish_status(const OrderStatusUpdate& u);

    std::string api_key_;
    std::string secret_key_;

    // Order tracking
    OrderTracker order_tracker_;
    std::function<void(const OrderStatusUpdate&)> on_order_status_;

    // Monitor thread
    std::thread monitor_thread_;
    std::mutex monitor_mtx_;
    std::condition_variable monitor_cv_;
    std::atomic<bool> monitor_running_{false};

    // Client order ID sequence
    std::atomic<uint64_t> cid_counter_{0};

    static constexpr std::chrono::minutes kOrderTimeout{3};
    static constexpr int kMaxRepriceAttempts = 2;
};
