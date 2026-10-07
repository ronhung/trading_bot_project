#include "binance_live_executor.h"
#include "../core/risk_manager.h"
#include <httplib.h>
#include <openssl/hmac.h>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

BinanceLiveExecutor::BinanceLiveExecutor(const std::string& api_key, const std::string& secret_key)
    : api_key_(api_key), secret_key_(secret_key) {
    std::cout << "🛡️ [Execution] BinanceLiveExecutor initialized (bound to Binance Testnet)." << std::endl;
}

BinanceLiveExecutor::~BinanceLiveExecutor() {
    stop_order_monitor();
}

std::string BinanceLiveExecutor::generate_signature(const std::string& query_string) {
    unsigned char hash[32];
    unsigned int length = 32;

    HMAC_CTX* hmac = HMAC_CTX_new();
    HMAC_Init_ex(hmac, secret_key_.c_str(), static_cast<int>(secret_key_.length()), EVP_sha256(), NULL);
    HMAC_Update(hmac, reinterpret_cast<const unsigned char*>(query_string.c_str()), query_string.length());
    HMAC_Final(hmac, hash, &length);
    HMAC_CTX_free(hmac);

    std::stringstream ss;
    for (unsigned int i = 0; i < length; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(hash[i]);
    }
    return ss.str();
}

std::string BinanceLiveExecutor::next_client_order_id() {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    uint64_t seq = ++cid_counter_;
    return "BOT" + std::to_string(ms) + "_" + std::to_string(seq);
}

void BinanceLiveExecutor::set_order_status_callback(std::function<void(const OrderStatusUpdate&)> cb) {
    on_order_status_ = std::move(cb);
    order_tracker_.set_update_callback([this](const OrderStatusUpdate& u) {
        publish_status(u);
    });
}

bool BinanceLiveExecutor::has_open_order() const {
    return order_tracker_.has_open_order();
}

void BinanceLiveExecutor::on_order_update(const std::string& client_order_id,
                                           const std::string& status,
                                           double filled_qty) {
    order_tracker_.on_order_update(client_order_id, status, filled_qty);
}

void BinanceLiveExecutor::publish_status(const OrderStatusUpdate& u) {
    if (on_order_status_) {
        on_order_status_(u);
    }
}

// ---------------------------------------------------------------------------
// Core order placement (used by send_order and reprice)
// ---------------------------------------------------------------------------
bool BinanceLiveExecutor::place_order_internal(const OrderRequest& req, int reprice_attempts) {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    std::string cid = next_client_order_id();

    // Round to Binance tick/step sizes (BTCUSDT: price=0.01, qty=0.001)
    double rounded_price = std::round(req.price * 100.0) / 100.0;
    double rounded_qty   = std::floor(req.quantity * 1000.0) / 1000.0;
    if (rounded_qty <= 0.0) {
        std::cerr << "⚠️ [BinanceLiveExecutor] Rounded quantity is zero; rejecting order." << std::endl;
        return false;
    }

    // Map the execution spec to Binance type + timeInForce.
    std::string type;
    std::string tif;
    const std::string& ot = req.execution.order_type;
    if (ot == "MARKET") {
        type = "MARKET";
    } else if (ot == "LIMIT_MAKER") {
        type = "LIMIT";
        tif = "GTX";                       // post-only
    } else {
        type = "LIMIT";
        tif = req.execution.time_in_force;
    }

    std::stringstream query_ss;
    query_ss << "symbol=" << req.symbol
             << "&side=" << req.side
             << "&type=" << type;
    if (!tif.empty()) {
        query_ss << "&timeInForce=" << tif;
    }
    query_ss << "&quantity=" << rounded_qty;
    if (type != "MARKET") {
        query_ss << "&price=" << rounded_price;  // MARKET ignores price
    }
    query_ss << "&newClientOrderId=" << cid
             << "&timestamp=" << ms;

    if (req.reduce_only) {
        query_ss << "&reduceOnly=true";
    }
    std::string query_string = query_ss.str();
    std::string signature = generate_signature(query_string);
    std::string payload = query_string + "&signature=" + signature;

    httplib::Headers headers = {
        {"X-MBX-APIKEY", api_key_}
    };

    httplib::Client cli("https://testnet.binancefuture.com");
    cli.set_connection_timeout(5);

    std::cout << "\n🔫 [BinanceLiveExecutor] Sending " << req.side << " " << req.symbol
              << " @ " << rounded_price << " (qty: " << rounded_qty
              << ", type: " << type << (tif.empty() ? "" : "/" + tif)
              << ", cid: " << cid
              << (req.reduce_only ? ", reduceOnly" : "")
              << (reprice_attempts > 0 ? ", reprice#" + std::to_string(reprice_attempts) : "")
              << ")" << std::endl;

    auto res = cli.Post("/fapi/v1/order", headers, payload, "application/x-www-form-urlencoded");

    if (res && res->status == 200) {
        try {
            json j = json::parse(res->body);
            int64_t order_id = j.value("orderId", 0LL);
            std::string status = j.value("status", "NEW");
            double executed_qty = std::stod(j.value("executedQty", "0"));

            TrackedOrder tracked;
            tracked.order_id = order_id;
            tracked.client_order_id = cid;
            tracked.symbol = req.symbol;
            tracked.side = req.side;
            tracked.quantity = rounded_qty;
            tracked.filled_quantity = executed_qty;
            tracked.price = rounded_price;
            tracked.reduce_only = req.reduce_only;
            tracked.reprice_attempts = reprice_attempts;
            tracked.created_at = std::chrono::steady_clock::now();
            tracked.execution = req.execution;
            if (status == "NEW") tracked.status = TrackedOrderStatus::NEW;
            else if (status == "PARTIALLY_FILLED") tracked.status = TrackedOrderStatus::PARTIALLY_FILLED;
            else if (status == "FILLED") tracked.status = TrackedOrderStatus::FILLED;
            else if (status == "CANCELED") tracked.status = TrackedOrderStatus::CANCELED;
            else if (status == "EXPIRED") tracked.status = TrackedOrderStatus::EXPIRED;
            else if (status == "REJECTED") tracked.status = TrackedOrderStatus::REJECTED;

            order_tracker_.register_order(tracked);

            // If the exchange already reached a terminal state in the POST
            // response (e.g. an instant fill), publish it immediately so Python
            // learns even if the private-WS event is delayed or the stream drops.
            if (status == "FILLED" || status == "CANCELED" ||
                status == "EXPIRED" || status == "REJECTED") {
                order_tracker_.on_order_update(cid, status, executed_qty);
            }

            std::cout << "✅ [BinanceLiveExecutor] Order accepted! orderId=" << order_id
                      << " cid=" << cid << " status=" << status << std::endl;
            return true;

        } catch (const std::exception& e) {
            std::cerr << "🔥 [BinanceLiveExecutor] Failed to parse order response: " << e.what() << std::endl;
            return false;
        }
    }

    if (res) {
        std::cout << "❌ [BinanceLiveExecutor] Order rejected! Status: " << res->status
                  << " | Response: " << res->body << std::endl;
    } else {
        auto err = res.error();
        std::cout << "🔥 [BinanceLiveExecutor] Network error! Error: "
                  << httplib::to_string(err) << std::endl;
    }
    return false;
}

bool BinanceLiveExecutor::send_order(const OrderRequest& req) {
    return place_order_internal(req, 0);
}

// ---------------------------------------------------------------------------
// Trailing exit (client-side, same TrailingStop class as the backtest engine)
// ---------------------------------------------------------------------------
void BinanceLiveExecutor::set_risk_manager(RiskManager* risk) {
    risk_ = risk;
}

void BinanceLiveExecutor::set_exit_execution(const ExecutionSpec& exit_exec) {
    std::lock_guard<std::mutex> lk(trailing_mtx_);
    exit_execution_ = exit_exec;
}

void BinanceLiveExecutor::arm_now_locked() {
    // Assumes trailing_mtx_ is held.
    if (!risk_ || trailing_stop_.active()) return;
    double pos = risk_->get_current_position();
    if (pos == 0.0) return;
    int side = (pos > 0.0) ? 1 : -1;
    double hard_stop = risk_->get_stop_price();
    trailing_stop_.on_entry(side, hard_stop);
}

void BinanceLiveExecutor::arm_trailing_exit(const std::string& indicator, int period) {
    // Called from the IpcServer receive thread right after the entry order is
    // accepted. The exchange fill is async, so we may not have a position yet;
    // configure now and let check_trailing_exit() arm lazily once filled.
    std::lock_guard<std::mutex> lk(trailing_mtx_);
    trailing_fired_ = false;
    trailing_stop_.configure(indicator, period);
    arm_now_locked();
}

void BinanceLiveExecutor::check_trailing_exit(const KLineData& bar) {
    if (!risk_) return;

    double pos = risk_->get_current_position();
    bool fire = false;
    double exit_price = 0.0;
    std::string close_side;
    ExecutionSpec exit_spec;

    {
        std::lock_guard<std::mutex> lk(trailing_mtx_);
        exit_spec = exit_execution_;

        if (pos == 0.0) {
            // Position cleared (exchange ACCOUNT_UPDATE) — disarm.
            trailing_stop_.reset();
            trailing_fired_ = false;
        } else if (!trailing_fired_) {
            // A close order is already in flight; do not re-arm on a stale position.
            // Lazy-arm: position just filled after arm_trailing_exit().
            arm_now_locked();

            std::string reason;
            if (trailing_stop_.on_bar(bar, exit_price, reason)) {
                trailing_fired_ = true;
                fire = true;
                close_side = (pos > 0.0) ? "SELL" : "BUY";
            }
        }

        // Always feed the rolling window so the Donchian lookback stays
        // full-history (matches the Python TrailingExitLabeler .shift(1)).
        trailing_stop_.observe(bar);
    }

    if (fire) {
        std::cout << "🛑 [BinanceLiveExecutor] " << "trailing_stop"
                  << " hit @ " << exit_price << std::endl;
        OrderRequest close_req;
        close_req.symbol = bar.symbol;
        close_req.side = close_side;
        close_req.quantity = std::abs(pos);
        close_req.price = exit_price;
        close_req.reduce_only = true;
        close_req.execution = exit_spec;
        send_order(close_req);
    }
}

// ---------------------------------------------------------------------------
// Query order status by clientOrderId (REST fallback when WS is delayed/lost)
// ---------------------------------------------------------------------------
bool BinanceLiveExecutor::query_order_status(const std::string& symbol,
                                              const std::string& client_order_id,
                                              std::string& out_status) {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    std::string query_string = "symbol=" + symbol
                             + "&origClientOrderId=" + client_order_id
                             + "&timestamp=" + std::to_string(ms);
    std::string signature = generate_signature(query_string);
    std::string path = "/fapi/v1/order?" + query_string + "&signature=" + signature;

    httplib::Headers headers = {
        {"X-MBX-APIKEY", api_key_}
    };

    httplib::Client cli("https://testnet.binancefuture.com");
    cli.set_connection_timeout(5);

    auto res = cli.Get(path.c_str(), headers);
    if (res && res->status == 200) {
        try {
            json j = json::parse(res->body);
            out_status = j.value("status", "");
            return true;
        } catch (const std::exception& e) {
            std::cerr << "🔥 Failed to parse order query: " << e.what() << std::endl;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Cancel an order by clientOrderId
// ---------------------------------------------------------------------------
bool BinanceLiveExecutor::cancel_order(const std::string& symbol, const std::string& client_order_id) {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    std::string query_string = "symbol=" + symbol
                             + "&origClientOrderId=" + client_order_id
                             + "&timestamp=" + std::to_string(ms);
    std::string signature = generate_signature(query_string);
    std::string payload = query_string + "&signature=" + signature;

    httplib::Headers headers = {
        {"X-MBX-APIKEY", api_key_}
    };

    httplib::Client cli("https://testnet.binancefuture.com");
    cli.set_connection_timeout(5);

    std::cout << "❌ [BinanceLiveExecutor] Canceling order " << client_order_id << " ..." << std::endl;

    auto res = cli.Delete("/fapi/v1/order", headers, payload, "application/x-www-form-urlencoded");

    if (res && res->status == 200) {
        try {
            json j = json::parse(res->body);
            std::string status = j.value("status", "");
            std::cout << "✅ [BinanceLiveExecutor] Cancel response: " << client_order_id
                      << " -> " << status << std::endl;
            // Only return true if the order was genuinely canceled.
            // If the order already filled/expired, return false so the
            // monitor loop does NOT reprice (would create a duplicate).
            return (status == "CANCELED");
        } catch (const std::exception& e) {
            std::cerr << "🔥 Failed to parse cancel response: " << e.what() << std::endl;
            return false;
        }
    }

    if (res) {
        std::cout << "⚠️ [BinanceLiveExecutor] Cancel failed. Status: " << res->status
                  << " | " << res->body << std::endl;
    } else {
        std::cout << "🔥 [BinanceLiveExecutor] Cancel network error: "
                  << httplib::to_string(res.error()) << std::endl;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Market price query (for reprice)
// ---------------------------------------------------------------------------
double BinanceLiveExecutor::get_market_price(const std::string& symbol) {
    httplib::Client cli("https://testnet.binancefuture.com");
    cli.set_connection_timeout(3);

    std::string path = "/fapi/v1/ticker/price?symbol=" + symbol;
    auto res = cli.Get(path.c_str());

    if (res && res->status == 200) {
        try {
            json j = json::parse(res->body);
            double px = std::stod(j["price"].get<std::string>());
            return px;
        } catch (const std::exception& e) {
            std::cerr << "🔥 Failed to parse ticker price: " << e.what() << std::endl;
        }
    }
    return 0.0;
}

// ---------------------------------------------------------------------------
// Reprice a timed-out order
// ---------------------------------------------------------------------------
void BinanceLiveExecutor::reprice_order(const TrackedOrder& ord) {
    double remaining = ord.quantity - ord.filled_quantity;
    if (remaining <= 0.0) {
        std::cout << "⚪ [Reprice] Order " << ord.client_order_id
                  << " fully filled, nothing to reprice." << std::endl;
        return;
    }

    double mkt = get_market_price(ord.symbol);
    if (mkt <= 0.0) {
        std::cerr << "⚠️ [Reprice] Cannot get market price for " << ord.symbol
                  << "; skipping reprice." << std::endl;
        return;
    }

    // Round to 0.01 (BTCUSDT tick size)
    double new_price = std::round(mkt * 100.0) / 100.0;

    std::cout << "🔄 [Reprice] " << ord.client_order_id
              << " | old px=" << ord.price << " -> new px=" << new_price
              << " | remaining=" << remaining << " | attempt=" << (ord.reprice_attempts + 1)
              << std::endl;

    OrderRequest req;
    req.symbol = ord.symbol;
    req.side = ord.side;
    req.quantity = remaining;
    req.price = new_price;
    req.reduce_only = ord.reduce_only;
    req.execution = ord.execution;  // same order_type / timeout / policy; chase again
    place_order_internal(req, ord.reprice_attempts + 1);
}

// ---------------------------------------------------------------------------
// Order monitor thread
// ---------------------------------------------------------------------------
void BinanceLiveExecutor::start_order_monitor() {
    if (monitor_running_.load()) return;
    monitor_running_ = true;
    monitor_thread_ = std::thread(&BinanceLiveExecutor::monitor_loop, this);
    std::cout << "⏱️ [OrderMonitor] Started (per-order timeout + unfilled policy)" << std::endl;
}

void BinanceLiveExecutor::stop_order_monitor() {
    if (!monitor_running_.load()) return;
    monitor_running_ = false;
    monitor_cv_.notify_all();
    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }
    std::cout << "⏱️ [OrderMonitor] Stopped." << std::endl;
}

void BinanceLiveExecutor::monitor_loop() {
    while (monitor_running_) {
        // Wait 10 seconds or until stopped
        {
            std::unique_lock<std::mutex> lk(monitor_mtx_);
            monitor_cv_.wait_for(lk, std::chrono::seconds(10),
                                 [this] { return !monitor_running_.load(); });
        }
        if (!monitor_running_) break;

        // Prune old terminal orders
        order_tracker_.prune(600000);  // 10 min

        // Check for timed-out orders (per-order timeout from the execution spec)
        auto timed_out = order_tracker_.get_timed_out_orders();

        for (auto& ord : timed_out) {
            // ── Guard #1: verify the order still exists on the exchange ──
            // Without this check, the monitor fires a cancel at an order that
            // already filled/expired on the exchange but whose ORDER_TRADE_UPDATE
            // hasn't arrived yet (or never will, e.g. listenKey expired).
            // Binance responds with -2011 "Unknown order sent" and we never
            // update the tracker → infinite retry loop every 10 s.
            std::string exchange_status;
            bool queried = query_order_status(ord.symbol, ord.client_order_id,
                                              exchange_status);

            if (queried && exchange_status != "NEW" &&
                exchange_status != "PARTIALLY_FILLED") {
                // Already terminal on the exchange — update tracker in-place
                // and publish the real status so Python knows.  Do NOT cancel
                // or reprice.
                std::cout << "⚡ [OrderMonitor] " << ord.client_order_id
                          << " already " << exchange_status
                          << " on exchange; syncing tracker (no cancel)."
                          << std::endl;

                order_tracker_.on_order_update(ord.client_order_id,
                                               exchange_status, ord.filled_quantity);
                continue;  // ← skip cancel entirely
            }

            // ── Guard #2: decide the timeout action from this order's policy ──
            const std::string& policy = ord.execution.unfilled_policy;
            bool exhausted = false;
            std::string action;  // "cancel" | "reprice" | "market"
            if (policy == "MARKET") {
                action = "market";
            } else if (policy == "REPRICE") {
                exhausted = (ord.reprice_attempts >= ord.execution.max_reprice_attempts);
                action = exhausted ? "cancel" : "reprice";
            } else {  // CANCEL
                action = "cancel";
            }

            std::cout << "⏰ [OrderMonitor] " << ord.client_order_id
                      << " timed out (policy=" << policy
                      << ", reprice#" << ord.reprice_attempts << ") -> " << action
                      << std::endl;

            bool was_canceled = cancel_order(ord.symbol, ord.client_order_id);

            if (!was_canceled) {
                // Cancel was rejected — even though we just queried and it was
                // open, a fill may have raced in. Sync from the exchange so the
                // tracker won't retry, and do NOT follow up (would double-fill).
                std::cout << "⚠️ [OrderMonitor] Cancel rejected for "
                          << ord.client_order_id
                          << " — syncing from exchange; will not follow up."
                          << std::endl;

                std::string post_status;
                if (query_order_status(ord.symbol, ord.client_order_id,
                                       post_status)) {
                    order_tracker_.on_order_update(ord.client_order_id,
                                                   post_status, 0.0);
                }
                continue;
            }

            order_tracker_.mark_cancel_requested(ord.client_order_id);

            // Publish the CANCELED update. The reason tells Python whether the
            // entry is still pending (another order follows) or abandoned.
            OrderStatusUpdate cu;
            cu.client_order_id = ord.client_order_id;
            cu.symbol = ord.symbol;
            cu.side = ord.side;
            cu.order_type = ord.execution.order_type;
            cu.quantity = ord.quantity;
            cu.filled_quantity = ord.filled_quantity;
            cu.price = ord.price;
            cu.status = "CANCELED";
            cu.reduce_only = ord.reduce_only;
            if (action == "reprice") {
                cu.reason = "timeout_reprice";
            } else if (action == "market") {
                cu.reason = "timeout_market";
            } else {
                cu.reason = "timeout_abandoned";
            }
            publish_status(cu);

            // Follow-up: chase (reprice) or hard-eat at market.
            if (action == "reprice") {
                reprice_order(ord);
            } else if (action == "market") {
                double remaining = ord.quantity - ord.filled_quantity;
                if (remaining > 0.0) {
                    std::cout << "🍽️ [OrderMonitor] Hard-eating remaining "
                              << remaining << " at market." << std::endl;
                    OrderRequest mkt;
                    mkt.symbol = ord.symbol;
                    mkt.side = ord.side;
                    mkt.quantity = remaining;
                    mkt.price = ord.price;
                    mkt.reduce_only = ord.reduce_only;
                    mkt.execution = ord.execution;
                    mkt.execution.order_type = "MARKET";
                    mkt.execution.timeout_ms = 0;
                    mkt.execution.unfilled_policy = "CANCEL";
                    send_order(mkt);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Utility: listen key + initial state (unchanged logic)
// ---------------------------------------------------------------------------
bool BinanceLiveExecutor::get_current_position(const std::string& symbol, double& out_position) {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    std::string query_string = "timestamp=" + std::to_string(ms);
    std::string signature = generate_signature(query_string);
    std::string path = "/fapi/v2/account?" + query_string + "&signature=" + signature;

    httplib::Headers headers = {
        {"X-MBX-APIKEY", api_key_}
    };

    httplib::Client cli("https://testnet.binancefuture.com");
    cli.set_connection_timeout(5);

    auto res = cli.Get(path.c_str(), headers);
    if (res && res->status == 200) {
        try {
            json j = json::parse(res->body);
            out_position = 0.0;
            for (auto& pos : j["positions"]) {
                if (pos["symbol"].get<std::string>() == symbol) {
                    out_position = std::stod(pos["positionAmt"].get<std::string>());
                    return true;
                }
            }
            return true;
        } catch (const std::exception& e) {
            std::cerr << "🔥 Failed to parse exchange position: " << e.what() << std::endl;
            return false;
        }
    }

    std::cerr << "❌ Failed to query exchange position! Status: "
              << (res ? std::to_string(res->status) : "connection error") << std::endl;
    return false;
}

// ---------------------------------------------------------------------------
// Keepalive: PUT /fapi/v1/listenKey  (Binance requires every 30 min)
// ---------------------------------------------------------------------------
bool BinanceLiveExecutor::keep_alive_listen_key(const std::string& listen_key) {
    httplib::Client cli("https://testnet.binancefuture.com");
    cli.set_connection_timeout(5);

    httplib::Headers headers = {
        {"X-MBX-APIKEY", api_key_}
    };

    auto res = cli.Put("/fapi/v1/listenKey", headers, "", "application/x-www-form-urlencoded");

    if (res && res->status == 200) {
        return true;
    }

    std::cerr << "⚠️  [Keepalive] PUT listenKey failed: "
              << (res ? std::to_string(res->status) + " " + res->body : "connection error")
              << std::endl;
    return false;
}

std::string BinanceLiveExecutor::get_listen_key() {
    httplib::Client cli("https://testnet.binancefuture.com");
    cli.set_connection_timeout(5);

    httplib::Headers headers = {
        {"X-MBX-APIKEY", api_key_}
    };

    std::cout << "🔑 [BinanceLiveExecutor] Requesting private listenKey from Binance..." << std::endl;

    auto res = cli.Post("/fapi/v1/listenKey", headers, "", "application/x-www-form-urlencoded");

    if (res && res->status == 200) {
        try {
            json j = json::parse(res->body);
            std::string listen_key = j["listenKey"];
            std::cout << "✅ [BinanceLiveExecutor] Successfully obtained listenKey!" << std::endl;
            return listen_key;
        } catch (const std::exception& e) {
            std::cerr << "🔥 Failed to parse listenKey: " << e.what() << std::endl;
        }
    } else {
        std::cerr << "❌ Failed to obtain listenKey! Status: "
                  << (res ? std::to_string(res->status) : "connection error") << std::endl;
    }
    return "";
}

bool BinanceLiveExecutor::get_initial_state(double& out_usdt_balance, double& out_btcusdt_position) {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    std::string query_string = "timestamp=" + std::to_string(ms);
    std::string signature = generate_signature(query_string);
    std::string path = "/fapi/v2/account?" + query_string + "&signature=" + signature;

    httplib::Headers headers = {
        {"X-MBX-APIKEY", api_key_}
    };

    httplib::Client cli("https://testnet.binancefuture.com");
    cli.set_connection_timeout(5);

    std::cout << "🔍 [BinanceLiveExecutor] Querying initial account state..." << std::endl;

    auto res = cli.Get(path.c_str(), headers);

    if (res && res->status == 200) {
        try {
            json j = json::parse(res->body);
            out_usdt_balance = 0.0;
            out_btcusdt_position = 0.0;

            for (auto& asset : j["assets"]) {
                if (asset["asset"] == "USDT") {
                    out_usdt_balance = std::stod(asset["walletBalance"].get<std::string>());
                }
            }

            for (auto& pos : j["positions"]) {
                if (pos["symbol"] == "BTCUSDT") {
                    out_btcusdt_position = std::stod(pos["positionAmt"].get<std::string>());
                }
            }
            return true;
        } catch (const std::exception& e) {
            std::cerr << "🔥 Failed to parse initial account state: " << e.what() << std::endl;
            return false;
        }
    }

    std::cerr << "❌ Initial state query failed! Status: "
              << (res ? std::to_string(res->status) : "connection error") << std::endl;
    if (res) std::cerr << "Response: " << res->body << std::endl;
    return false;
}
