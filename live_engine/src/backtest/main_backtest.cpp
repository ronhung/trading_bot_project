#include <iostream>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>

#include "../core/ipc_server.h"
#include "../core/risk_manager.h"
#include "mock_executor.h"
#include "data_replayer.h"

using json = nlohmann::json;
namespace fs = std::filesystem;

static fs::path find_project_root() {
    wchar_t buf[MAX_PATH];
    GetModuleFileNameW(nullptr, buf, MAX_PATH);
    fs::path p = fs::path(buf).parent_path();
    for (int i = 0; i < 6; ++i) {
        if (fs::exists(p / "shared" / "config.json") ||
            fs::exists(p / "data" / "historical_data")) {
            return p;
        }
        if (!p.has_parent_path() || p == p.parent_path()) break;
        p = p.parent_path();
    }
    p = fs::current_path();
    for (int i = 0; i < 4; ++i) {
        if (fs::exists(p / "shared" / "config.json") ||
            fs::exists(p / "data" / "historical_data")) {
            return p;
        }
        if (!p.has_parent_path() || p == p.parent_path()) break;
        p = p.parent_path();
    }
    return fs::current_path();
}

int main(int argc, char** argv) {
    try {
        SetConsoleOutputCP(CP_UTF8);

        fs::path root = find_project_root();
        std::string config_path = (root / "shared" / "config.json").string();
        std::string csv_path = (root / "data" / "historical_data" / "BTCUSDT_1m_full.csv").string();
        std::string trades_out = (root / "data" / "historical_data" / "backtest_trades.csv").string();
        double initial_balance = 100000.0;

        if (argc >= 2) csv_path = argv[1];
        if (argc >= 3) trades_out = argv[2];
        if (argc >= 4) initial_balance = std::stod(argv[3]);

        std::ifstream config_file(config_path);
        int pub_port = 5555;
        int pull_port = 5556;
        if (config_file.is_open()) {
            json config;
            config_file >> config;
            pub_port = config.value("/zmq/market_feed_port"_json_pointer, 5555);
            pull_port = config.value("/zmq/signal_port"_json_pointer, 5556);
            if (config.contains("backtest") && config["backtest"].contains("initial_balance")) {
                initial_balance = config["backtest"]["initial_balance"].get<double>();
            }
        } else {
            std::cout << "⚠️ Config not found at " << config_path << " (using defaults)" << std::endl;
        }

        RiskManager risk_manager(0.02, 20.0);
        risk_manager.update_balance(initial_balance);
        risk_manager.update_position(0.0);

        MockExecutor mock(&risk_manager, 0.0005 /* 0.05% */, 1.0 /* 1 bps */);
        IpcServer ipc(pub_port, pull_port, &mock, &risk_manager, true /* sync */);

        std::cout << "👑 Backtest engine starting..." << std::endl;
        std::cout << "   Project root: " << root.string() << std::endl;
        std::cout << "   CSV: " << csv_path << std::endl;
        std::cout << "   Initial balance: " << initial_balance << " USDT" << std::endl;

        DataReplayer replayer(csv_path, &ipc, &mock, &risk_manager);
        std::size_t bars = replayer.run();

        if (bars == 0) {
            std::cerr << "❌ Backtest produced no bars." << std::endl;
            return 1;
        }

        std::cout << "\n======= BACKTEST REPORT =======" << std::endl;
        std::cout << "Final balance : " << risk_manager.get_current_balance() << " USDT" << std::endl;
        std::cout << "Final position: " << risk_manager.get_current_position() << " BTC" << std::endl;
        std::cout << "Trades        : " << mock.trades().size() << std::endl;
        std::cout << "===============================\n" << std::endl;

        mock.export_trades_csv(trades_out);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception: " << e.what() << std::endl;
        return 1;
    }
}
