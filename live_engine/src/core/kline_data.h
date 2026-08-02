#pragma once
#include <string>
#include <cstdint>

struct KLineData {
    std::string symbol;

    uint64_t open_time = 0;
    uint64_t close_time = 0;

    double open = 0.0;
    double high = 0.0;
    double low = 0.0;
    double close = 0.0;

    double volume = 0.0;
    double quote_volume = 0.0;

    double taker_buy_base = 0.0;
    double taker_buy_quote = 0.0;

    uint64_t trades_count = 0;
    bool is_closed = false;
};
