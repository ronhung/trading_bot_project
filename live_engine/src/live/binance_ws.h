#pragma once
#include <string>
#include <cstdint>
#include <ixwebsocket/IXWebSocket.h>
#include <functional>
#include "../core/kline_data.h"

class BinanceWebSocket {
public:
    BinanceWebSocket();
    ~BinanceWebSocket();

    void connect_and_stream(const std::string& symbol);
    void set_kline_callback(std::function<void(const KLineData&)> callback) {
        on_kline_received = callback;
    }

private:
    ix::WebSocket webSocket;
    std::function<void(const KLineData&)> on_kline_received;
};
