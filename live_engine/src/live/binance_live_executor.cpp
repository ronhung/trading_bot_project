#include "binance_live_executor.h"
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
bool BinanceLiveExecutor::place_order_internal(const std::string& symbol,
                                                const std::string& side,
                                                double quantity,
                                                double price,
                                                bool reduce_only,
                                                int reprice_attempts) {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    std::string cid = next_client_order_id();

    std::stringstream query_ss;
    query_ss << "symbol=" << symbol
             << "&side=" << side
             << "&type=LIMIT"
             << "&timeInForce=GTC"
             << "&quantity=" << quantity
             << "&price=" << price
             << "&newClientOrderId=" << cid
             << "&timestamp=" << ms;

    if (reduce_only) {
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

    std::cout << "\n🔫 [BinanceLiveExecutor] Sending " << side << " " << symbol
              << " @ " << price << " (qty: " << quantity
              << ", cid: " << cid
              << (reduce_only ? ", reduceOnly" : "")
              << (reprice_attempts > 0 ? ", reprice#" + std::to_string(reprice_attempts) : "")
              << ")" << std::endl;

    auto res = cli.Post("/fapi/v1/order", headers, payload, "application/x-www-form-urlencoded");

    if (res && res->status == 200) {
        try {
            json j = json::parse(res->body);
            int64_t order_id = j.value("orderId", 0LL);
            std::string status = j.value("status", "NEW");

            TrackedOrder tracked;
            tracked.client_order_id = cid;
            tracked.symbol = symbol;
            tracked.side = side;
            tracked.quantity = quantity;
            tracked.price = price;
            tracked.reduce_only = reduce_only;
            tracked.reprice_attempts = reprice_attempts;
            tracked.created_at = std::chrono::steady_clock::now();
            if (status == "NEW") tracked.status = TrackedOrderStatus::NEW;
            else if (status == "FILLED") tracked.status = TrackedOrderStatus::FILLED;

            order_tracker_.register_order(tracked);

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

bool BinanceLiveExecutor::send_order(const std::string& symbol,
                                     const std::string& side,
                                     double quantity,
                                     double price,
                                     bool reduce_only) {
    return place_order_internal(symbol, side, quantity, price, reduce_only, 0);
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
            std::string status = j.value("status", "CANCELED");
            std::cout << "✅ [BinanceLiveExecutor] Cancel accepted: " << client_order_id
                      << " -> " << status << std::endl;
            return true;
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

    place_order_internal(ord.symbol, ord.side, remaining, new_price,
                         ord.reduce_only, ord.reprice_attempts + 1);
}

// ---------------------------------------------------------------------------
// Order monitor thread
// ---------------------------------------------------------------------------
void BinanceLiveExecutor::start_order_monitor() {
    if (monitor_running_.load()) return;
    monitor_running_ = true;
    monitor_thread_ = std::thread(&BinanceLiveExecutor::monitor_loop, this);
    std::cout << "⏱️ [OrderMonitor] Started (timeout=" << kOrderTimeout.count() << "min, max reprice="
              << kMaxRepriceAttempts << ")" << std::endl;
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

        // Check for timed-out orders
        int timeout_ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
            kOrderTimeout).count());
        auto timed_out = order_tracker_.get_timed_out_orders(timeout_ms);

        for (auto& ord : timed_out) {
            if (ord.reprice_attempts >= kMaxRepriceAttempts) {
                // Exhausted: cancel and report
                std::cout << "⏰ [OrderMonitor] " << ord.client_order_id
                          << " timed out with " << ord.reprice_attempts
                          << " reprice attempts — canceling (exhausted)." << std::endl;

                cancel_order(ord.symbol, ord.client_order_id);
                order_tracker_.mark_cancel_requested(ord.client_order_id);

                // Publish exhausted status
                OrderStatusUpdate u;
                u.client_order_id = ord.client_order_id;
                u.symbol = ord.symbol;
                u.side = ord.side;
                u.order_type = "LIMIT";
                u.quantity = ord.quantity;
                u.price = ord.price;
                u.status = "CANCELED";
                u.reduce_only = ord.reduce_only;
                u.reason = "timeout_exhausted";
                publish_status(u);

            } else {
                // Cancel and reprice
                std::cout << "⏰ [OrderMonitor] " << ord.client_order_id
                          << " timed out (" << ord.reprice_attempts
                          << " prev reprice(s)) — cancel + reprice." << std::endl;

                bool canceled = cancel_order(ord.symbol, ord.client_order_id);

                // Mark the old order as cancelled in tracker
                order_tracker_.mark_cancel_requested(ord.client_order_id);

                // Publish cancel status
                OrderStatusUpdate cu;
                cu.client_order_id = ord.client_order_id;
                cu.symbol = ord.symbol;
                cu.side = ord.side;
                cu.order_type = "LIMIT";
                cu.quantity = ord.quantity;
                cu.price = ord.price;
                cu.status = "CANCELED";
                cu.reduce_only = ord.reduce_only;
                cu.reason = "timeout";
                publish_status(cu);

                if (canceled) {
                    // Reprice at current market
                    reprice_order(ord);
                } else {
                    std::cout << "⚠️ [OrderMonitor] Cancel failed for " << ord.client_order_id
                              << " — awaiting WS event; will not reprice." << std::endl;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Utility: listen key + initial state (unchanged logic)
// ---------------------------------------------------------------------------
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
