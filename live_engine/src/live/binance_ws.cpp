#include "binance_ws.h"
#include <iostream>
#include <cctype>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

BinanceWebSocket::BinanceWebSocket() {}

BinanceWebSocket::~BinanceWebSocket() {
    webSocket.stop();
}

void BinanceWebSocket::connect_and_stream(const std::string& symbol) {
    std::string symbol_lower = symbol;
    for (char& c : symbol_lower) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    std::string url = "wss://fstream.binance.com/market/ws/" + symbol_lower + "@kline_1m";
    webSocket.setUrl(url);

    std::cout << "🚀 Preparing to connect to Binance WebSocket (" << symbol << ")..." << std::endl;

    webSocket.setOnMessageCallback([this](const ix::WebSocketMessagePtr& msg) {
        if (msg->type == ix::WebSocketMessageType::Open) {
            std::cout << "✅ Connection opened! Starting to receive real-time klines..." << std::endl;
        } else if (msg->type == ix::WebSocketMessageType::Error) {
            std::cerr << "❌ Connection error: " << msg->errorInfo.reason << std::endl;
        } else if (msg->type == ix::WebSocketMessageType::Message) {
            try {
                json j = json::parse(msg->str);
                if (j.contains("k")) {
                    auto kline = j["k"];
                    KLineData data;
                    data.symbol = j["s"].get<std::string>();
                    data.open_time = kline["t"].get<uint64_t>();
                    data.close_time = kline["T"].get<uint64_t>();
                    data.trades_count = kline["n"].get<uint64_t>();
                    data.open = std::stod(kline["o"].get<std::string>());
                    data.high = std::stod(kline["h"].get<std::string>());
                    data.low = std::stod(kline["l"].get<std::string>());
                    data.close = std::stod(kline["c"].get<std::string>());
                    data.volume = std::stod(kline["v"].get<std::string>());
                    data.quote_volume = std::stod(kline["q"].get<std::string>());
                    data.taker_buy_base = std::stod(kline["V"].get<std::string>());
                    data.taker_buy_quote = std::stod(kline["Q"].get<std::string>());
                    data.is_closed = kline["x"].get<bool>();

                    if (this->on_kline_received) {
                        this->on_kline_received(data);
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "⚠️ Exception occurred: " << e.what() << std::endl;
            }
        } else if (msg->type == ix::WebSocketMessageType::Close) {
            std::cout << "⚠️ Connection closed by server. Reason: " << msg->errorInfo.reason << std::endl;
        }
    });

    webSocket.start();
}
