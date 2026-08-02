#include "data_replayer.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

DataReplayer::DataReplayer(const std::string& csv_path,
                           IpcServer* ipc,
                           MockExecutor* executor,
                           RiskManager* risk_manager)
    : csv_path_(csv_path), ipc_(ipc), executor_(executor), risk_(risk_manager) {}

bool DataReplayer::parse_line(const std::string& line, KLineData& out) const {
    // Expected header (no datetime):
    // open_time,open,high,low,close,volume,close_time,quote_volume,trades_count,taker_buy_base,taker_buy_quote
    std::stringstream ss(line);
    std::string token;
    std::string fields[11];
    int idx = 0;
    while (std::getline(ss, token, ',') && idx < 11) {
        fields[idx++] = token;
    }
    if (idx < 11) {
        return false;
    }

    try {
        out.symbol = "BTCUSDT";
        out.open_time = std::stoull(fields[0]);
        out.open = std::stod(fields[1]);
        out.high = std::stod(fields[2]);
        out.low = std::stod(fields[3]);
        out.close = std::stod(fields[4]);
        out.volume = std::stod(fields[5]);
        out.close_time = std::stoull(fields[6]);
        out.quote_volume = std::stod(fields[7]);
        out.trades_count = std::stoull(fields[8]);
        out.taker_buy_base = std::stod(fields[9]);
        out.taker_buy_quote = std::stod(fields[10]);
        out.is_closed = true;
        return true;
    } catch (...) {
        return false;
    }
}

std::size_t DataReplayer::run() {
    std::ifstream in(csv_path_);
    if (!in.is_open()) {
        std::cerr << "❌ [DataReplayer] Cannot open CSV: " << csv_path_ << std::endl;
        return 0;
    }

    std::string header;
    std::getline(in, header);
    std::cout << "📂 [DataReplayer] Loaded " << csv_path_ << std::endl;
    std::cout << "   Header: " << header << std::endl;

    if (!ipc_->wait_for_python()) {
        return 0;
    }

    std::cout << "🚀 [DataReplayer] Floodgates open — replaying bars..." << std::endl;
    auto t0 = std::chrono::steady_clock::now();

    std::size_t count = 0;
    std::string line;
    KLineData bar;

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (!parse_line(line, bar)) {
            std::cerr << "⚠️ [DataReplayer] Skip bad line at #" << count << std::endl;
            continue;
        }

        executor_->set_market(bar);
        executor_->check_and_execute_stop(bar.symbol);

        if (!ipc_->publish_kline_and_wait(bar, 60000)) {
            std::cerr << "❌ [DataReplayer] ACK timeout at bar #" << count
                      << " open_time=" << bar.open_time << std::endl;
            break;
        }

        ++count;
        if (count % 100000 == 0) {
            auto now = std::chrono::steady_clock::now();
            double sec = std::chrono::duration<double>(now - t0).count();
            std::cout << "… replayed " << count << " bars ("
                      << (count / std::max(sec, 1e-6)) << " bars/s)"
                      << " | bal=" << risk_->get_current_balance()
                      << " | pos=" << risk_->get_current_position() << std::endl;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "🏁 [DataReplayer] Done. Bars=" << count
              << " | elapsed=" << sec << "s"
              << " | final balance=" << risk_->get_current_balance()
              << " | final position=" << risk_->get_current_position()
              << std::endl;
    return count;
}
