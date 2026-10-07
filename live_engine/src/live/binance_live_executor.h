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
#include "../core/trailing_stop.h"
#include "order_tracker.h"

class RiskManager;

class BinanceLiveExecutor : public IOrderExecutor {
public:
    BinanceLiveExecutor(const std::string& api_key, const std::string& secret_key);
    ~BinanceLiveExecutor() override;

    // IOrderExecutor overrides
    bool send_order(const OrderRequest& req) override;

    void set_order_status_callback(std::function<void(const OrderStatusUpdate&)>) override;
    bool has_open_order() const override;

    // Wire the shared RiskManager (position + stop price source for trailing exit).
    void set_risk_manager(RiskManager* risk);

    // Arm the client-side trailing exit for a just-opened position.
    // Same TrailingStop class as the backtest MockExecutor.
    void arm_trailing_exit(const std::string& indicator, int period) override;

    // Store the EXIT execution spec (used when check_trailing_exit places the close).
    void set_exit_execution(const ExecutionSpec& exit_exec) override;

    // Per-bar trailing-exit check. Called from the main loop on each closed
    // kline; fires a reduce-only close when the trailing stop is hit.
    void check_trailing_exit(const KLineData& bar);

    // Initial state query (called once at startup)
    bool get_initial_state(double& out_usdt_balance, double& out_btcusdt_position);

    // User data stream listen key
    std::string get_listen_key();       // POST — create new listenKey
    bool keep_alive_listen_key(const std::string& listen_key);  // PUT — extend TTL

    // Query the current exchange position for a symbol.
    bool get_current_position(const std::string& symbol, double& out_position) override;

    // Cancel an order by clientOrderId.
    // Returns true only if the cancel was accepted AND the order transitioned
    // to CANCELED.  Returns false if the order was already terminal (filled /
    // expired / not found) or a network error occurred.
    bool cancel_order(const std::string& symbol, const std::string& client_order_id);

    // Query the current exchange status of an order by clientOrderId.
    // out_status is set to the Binance status string (NEW, FILLED, CANCELED, …).
    // Returns false on network/parse failure.
    bool query_order_status(const std::string& symbol,
                            const std::string& client_order_id,
                            std::string& out_status);

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

    // Internal: place an order (used by send_order and reprice) according to
    // req.execution (order_type / time_in_force).
    bool place_order_internal(const OrderRequest& req, int reprice_attempts);

    // Reprice a timed-out order at current market
    void reprice_order(const TrackedOrder& ord);

    // Monitor thread: periodic timeout check + cancel + reprice
    void monitor_loop();

    // Fire status update through the callback
    void publish_status(const OrderStatusUpdate& u);

    // Arm the trailing stop now if a position is filled (assumes trailing_mtx_ held).
    void arm_now_locked();

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

    // Trailing exit (client-side, shared TrailingStop class with backtest).
    RiskManager* risk_ = nullptr;
    TrailingStop trailing_stop_;
    bool trailing_fired_ = false;   // close already submitted; wait for position to clear
    std::mutex trailing_mtx_;       // guards trailing_stop_ across rx-thread vs main-loop
    ExecutionSpec exit_execution_;  // how the trailing-stop / hard-stop close is placed
};
