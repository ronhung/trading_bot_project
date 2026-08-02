#include "ipc_server.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

IpcServer::IpcServer(int pub_port, int pull_port,
                     IOrderExecutor* executor, RiskManager* risk_manager,
                     bool sync_mode)
    : context(1),
      publisher(context, zmq::socket_type::pub),
      receiver(context, zmq::socket_type::pull),
      running(true),
      sync_mode_(sync_mode),
      executor_(executor),
      risk_manager_(risk_manager)
{
    std::string pub_addr = "tcp://*:" + std::to_string(pub_port);
    publisher.bind(pub_addr);
    std::cout << "📡 [IPC] Market publisher (PUB) bound to: " << pub_addr << std::endl;

    std::string pull_addr = "tcp://*:" + std::to_string(pull_port);
    receiver.bind(pull_addr);
    receiver.set(zmq::sockopt::rcvtimeo, sync_mode_ ? 50 : 1000);
    std::cout << "🎯 [IPC] Command receiver (PULL) bound to: " << pull_addr
              << (sync_mode_ ? " [SYNC/backtest]" : " [ASYNC/live]") << std::endl;

    if (!sync_mode_) {
        rx_thread = std::thread(&IpcServer::receive_loop, this);
    }

    // Wire the executor's order status callback to our thread-safe queue.
    // (MockExecutor's default is a no-op, so backtest is unaffected.)
    if (executor_) {
        executor_->set_order_status_callback([this](const OrderStatusUpdate& u) {
            queue_order_update(u);
        });
    }
}

IpcServer::~IpcServer() {
    running = false;
    if (rx_thread.joinable()) {
        rx_thread.join();
    }
}

void IpcServer::publish_kline(const KLineData& kline) {
    json j;
    j["type"] = "kline";
    j["symbol"] = kline.symbol;
    j["open_time"] = kline.open_time;
    j["close_time"] = kline.close_time;
    j["open"] = kline.open;
    j["high"] = kline.high;
    j["low"] = kline.low;
    j["close"] = kline.close;
    j["volume"] = kline.volume;
    j["quote_volume"] = kline.quote_volume;
    j["taker_buy_base"] = kline.taker_buy_base;
    j["taker_buy_quote"] = kline.taker_buy_quote;
    j["trades_count"] = kline.trades_count;
    j["is_closed"] = kline.is_closed;

    // Inject real RiskManager state so Python brain knows actual position
    j["current_position"]  = risk_manager_ ? risk_manager_->get_current_position() : 0.0;
    j["available_balance"] = risk_manager_ ? risk_manager_->get_current_balance()   : 0.0;
    j["stop_price"]        = risk_manager_ ? risk_manager_->get_stop_price()        : 0.0;

    std::string message = j.dump();
    zmq::message_t zmq_msg(message.begin(), message.end());
    publisher.send(zmq_msg, zmq::send_flags::none);
}

// ---------------------------------------------------------------------------
// Thread-safe order update queue → PUB socket
// Called from any thread; actual PUB send only from main-loop via pump.
// ---------------------------------------------------------------------------
void IpcServer::queue_order_update(const OrderStatusUpdate& u) {
    json j;
    j["type"] = "order_update";
    j["symbol"] = u.symbol;
    j["client_order_id"] = u.client_order_id;
    j["order_id"] = u.order_id;
    j["side"] = u.side;
    j["order_type"] = u.order_type;
    j["quantity"] = u.quantity;
    j["price"] = u.price;
    j["status"] = u.status;
    j["reduce_only"] = u.reduce_only;
    j["reason"] = u.reason;
    order_update_queue_.push(j.dump());
}

void IpcServer::pump_order_updates() {
    std::string msg;
    while (order_update_queue_.try_pop(msg)) {
        zmq::message_t zmq_msg(msg.begin(), msg.end());
        publisher.send(zmq_msg, zmq::send_flags::none);
    }
}

bool IpcServer::recv_until_ack(int timeout_ms) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    receiver.set(zmq::sockopt::rcvtimeo, 50);

    while (std::chrono::steady_clock::now() < deadline) {
        zmq::message_t request;
        auto res = receiver.recv(request, zmq::recv_flags::none);
        if (!res) {
            continue;
        }
        std::string msg_str(static_cast<char*>(request.data()), request.size());
        try {
            json j = json::parse(msg_str);
            if (j.value("type", "") == "ack") {
                return true;
            }
            handle_message(msg_str);
        } catch (const std::exception& e) {
            std::cerr << "Failed to parse Python message: " << e.what() << std::endl;
        }
    }
    return false;
}

bool IpcServer::publish_kline_and_wait(const KLineData& kline, int timeout_ms) {
    publish_kline(kline);
    return recv_until_ack(timeout_ms);
}

bool IpcServer::wait_for_python(int timeout_ms) {
    std::cout << "⏳ [IPC] Waiting for Python brain to connect (start live_trend_bot.py)..." << std::endl;

    KLineData ping;
    ping.symbol = "BTCUSDT";
    ping.open_time = 0;
    ping.close_time = 0;
    ping.open = ping.high = ping.low = ping.close = 1.0;
    ping.volume = 0.0;
    ping.quote_volume = 0.0;
    ping.taker_buy_base = 0.0;
    ping.taker_buy_quote = 0.0;
    ping.trades_count = 0;
    ping.is_closed = false; // warmup only — strategy ignores non-closed bars

    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        publish_kline(ping);
        if (recv_until_ack(500)) {
            std::cout << "✅ [IPC] Python brain connected!" << std::endl;
            return true;
        }
    }
    std::cerr << "❌ [IPC] Timed out waiting for Python brain." << std::endl;
    return false;
}

void IpcServer::handle_message(const std::string& msg_str) {
    json j = json::parse(msg_str);

    if (j.value("type", "") == "ack") {
        return;
    }

    std::string action = j.at("action").get<std::string>();
    std::string symbol = j.value("symbol", "BTCUSDT");
    double price = j.value("price", 0.0);
    double stop_price = j.value("stop_price", 0.0);

    std::cout << "\n⚡ [C++ Received command] action: " << action
              << " | symbol: " << symbol
              << " | trigger price: " << price
              << " | stop price: " << stop_price << std::endl;

    if (!executor_ || !risk_manager_) {
        return;
    }

    if (action == "BUY" || action == "SELL") {
        // One-position rule: refuse to pyramid / flip without an explicit CLOSE_*.
        double cur_pos = risk_manager_->get_current_position();
        if (std::abs(cur_pos) > 1e-12) {
            std::cout << "🚫 [IPC] Already in position (" << cur_pos
                      << "); ignore open until CLOSE_*." << std::endl;
            return;
        }

        // Also refuse if there's already a pending open order
        if (executor_->has_open_order()) {
            std::cout << "🚫 [IPC] Open order already pending; ignore duplicate open." << std::endl;
            return;
        }

        double safe_quantity = risk_manager_->calculate_target_size(action, price, stop_price);
        if (safe_quantity <= 0.0) {
            std::cout << "🚫 [IPC] Insufficient balance or invalid stop, cancel open position." << std::endl;
            return;
        }
        bool ok = executor_->send_order(symbol, action, safe_quantity, price, false);
        if (ok) {
            risk_manager_->set_stop_price(stop_price);
        } else {
            std::cout << "❌ [IPC] Order rejected by executor (not tracked)." << std::endl;
        }
    } else if (action == "CLOSE_LONG" || action == "CLOSE_SHORT") {
        double current_pos = risk_manager_->get_current_position();
        double close_quantity = std::abs(current_pos);
        if (close_quantity <= 0.0) {
            std::cout << "⚪ [IPC] No current position; ignoring close command." << std::endl;
            return;
        }
        if (action == "CLOSE_LONG") {
            std::cout << "🛡️ [Close Long] Current position: " << current_pos
                      << ", sending SELL + reduceOnly" << std::endl;
            executor_->send_order(symbol, "SELL", close_quantity, price, true);
        } else {
            std::cout << "🛡️ [Close Short] Current position: " << current_pos
                      << ", sending BUY + reduceOnly" << std::endl;
            executor_->send_order(symbol, "BUY", close_quantity, price, true);
        }
        risk_manager_->clear_stop();
    }
}

void IpcServer::receive_loop() {
    while (running) {
        zmq::message_t request;
        auto res = receiver.recv(request, zmq::recv_flags::none);
        if (!res) {
            continue;
        }
        std::string msg_str(static_cast<char*>(request.data()), request.size());
        try {
            handle_message(msg_str);
        } catch (const std::exception& e) {
            std::cerr << "Failed to parse Python command: " << e.what() << std::endl;
        }
    }
}
