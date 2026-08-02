#include <iostream>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocket.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>

#include "binance_ws.h"
#include "binance_live_executor.h"
#include "../core/thread_safe_queue.h"
#include "../core/ipc_server.h"
#include "../core/risk_manager.h"

using json = nlohmann::json;
namespace fs = std::filesystem;

static fs::path find_project_root() {
    wchar_t buf[MAX_PATH];
    GetModuleFileNameW(nullptr, buf, MAX_PATH);
    fs::path p = fs::path(buf).parent_path();
    for (int i = 0; i < 6; ++i) {
        if (fs::exists(p / "shared" / "config.json")) return p;
        if (!p.has_parent_path() || p == p.parent_path()) break;
        p = p.parent_path();
    }
    p = fs::current_path();
    for (int i = 0; i < 4; ++i) {
        if (fs::exists(p / "shared" / "config.json")) return p;
        if (!p.has_parent_path() || p == p.parent_path()) break;
        p = p.parent_path();
    }
    return fs::current_path();
}

int main() {
    try {
        SetConsoleOutputCP(CP_UTF8);
        ix::initNetSystem();

        fs::path root = find_project_root();
        std::string config_path = (root / "shared" / "config.json").string();
        std::ifstream config_file(config_path);
        if (!config_file.is_open()) {
            std::cerr << "❌ Cannot open config: " << config_path << std::endl;
            return 1;
        }
        json config;
        config_file >> config;
        std::string API_KEY = config["api_key"];
        std::string SECRET_KEY = config["secret_key"];

        int pub_port = config.value("/zmq/market_feed_port"_json_pointer, 5555);
        int pull_port = config.value("/zmq/signal_port"_json_pointer, 5556);

        BinanceLiveExecutor executor(API_KEY, SECRET_KEY);
        RiskManager risk_manager(0.02);
        double init_balance = 0.0;
        double init_position = 0.0;

        if (executor.get_initial_state(init_balance, init_position)) {
            risk_manager.update_balance(init_balance);
            risk_manager.update_position(init_position);
            std::cout << "✅ [System warmup] Initial USDT balance: " << init_balance
                      << " | Initial BTCUSDT position: " << init_position << "\n" << std::endl;
        } else {
            std::cout << "⚠️ [System warmup] State query failed, risk manager will start from 0!\n" << std::endl;
        }

        std::cout << "👑 Quantitative live engine starting..." << std::endl;
        IpcServer ipc(pub_port, pull_port, &executor, &risk_manager, false);

        // Start the order monitor thread (timeout → cancel → reprice)
        executor.start_order_monitor();

        std::string listen_key = executor.get_listen_key();
        ix::WebSocket user_ws;

        if (!listen_key.empty()) {
            std::string user_ws_url = "wss://stream.binancefuture.com/ws/" + listen_key;
            user_ws.setUrl(user_ws_url);

            user_ws.setOnMessageCallback([&risk_manager, &executor](const ix::WebSocketMessagePtr& msg) {
                if (msg->type == ix::WebSocketMessageType::Message) {
                    try {
                        json j = json::parse(msg->str);
                        if (j.contains("e") && j["e"] == "ORDER_TRADE_UPDATE") {
                            // Handle order status updates (fills, cancels, etc.)
                            if (j.contains("o")) {
                                auto& o = j["o"];
                                std::string client_id = o.value("c", "");
                                std::string status = o.value("X", "");  // NEW, PARTIALLY_FILLED, FILLED, CANCELED...
                                double filled_qty = std::stod(o.value("z", "0"));
                                if (!client_id.empty()) {
                                    executor.on_order_update(client_id, status, filled_qty);
                                }
                            }
                        } else if (j.contains("e") && j["e"] == "ACCOUNT_UPDATE") {
                            auto& account = j["a"];
                            if (account.contains("B")) {
                                for (auto& balance : account["B"]) {
                                    if (balance["a"] == "USDT") {
                                        double current_balance = std::stod(balance["wb"].get<std::string>());
                                        risk_manager.update_balance(current_balance);
                                        std::cout << "💰 [Radar] USDT balance updated: " << current_balance << std::endl;
                                        break;
                                    }
                                }
                            }
                            if (account.contains("P")) {
                                for (auto& position : account["P"]) {
                                    if (position["s"] == "BTCUSDT") {
                                        double current_position = std::stod(position["pa"].get<std::string>());
                                        risk_manager.update_position(current_position);
                                        std::cout << "📦 [Radar] BTCUSDT position updated: " << current_position << std::endl;
                                        break;
                                    }
                                }
                            }
                        }
                    } catch (const std::exception& e) {
                        std::cerr << "🔥 Failed to parse private radar JSON: " << e.what() << std::endl;
                    }
                } else if (msg->type == ix::WebSocketMessageType::Open) {
                    std::cout << "🟢 [Private radar] Connected to Binance server!" << std::endl;
                }
            });
            user_ws.start();
        }

        ThreadSafeQueue<KLineData> market_queue;
        BinanceWebSocket ws;

        ws.set_kline_callback([&market_queue](const KLineData& data) {
            market_queue.push(data);
        });

        ws.connect_and_stream("BTCUSDT");

        KLineData current_kline;
        std::cout << "⚙️ Engine main loop is ready, waiting for market data..." << std::endl;

        while (true) {
            // Flush any queued order status updates to Python (PUB socket, main thread only)
            ipc.pump_order_updates();

            // Wait for the next closed kline with a 250ms timeout so we can pump updates
            if (market_queue.wait_and_pop(current_kline, 250)) {
                if (!current_kline.is_closed) {
                    continue;
                }
                std::cout << "[Main] Closed kline close=" << current_kline.close
                          << " -> Python brain" << std::endl;
                ipc.publish_kline(current_kline);
            }
        }

        ix::uninitNetSystem();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception occurred: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Unknown error occurred!" << std::endl;
        return 1;
    }
}
