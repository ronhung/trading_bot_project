#pragma once
#include <string>
#include "../core/ipc_server.h"
#include "mock_executor.h"

class DataReplayer {
public:
    DataReplayer(const std::string& csv_path,
                 IpcServer* ipc,
                 MockExecutor* executor,
                 RiskManager* risk_manager);

    // Returns number of bars replayed. Blocks until Python connects, then floods.
    std::size_t run();

private:
    bool parse_line(const std::string& line, KLineData& out) const;

    std::string csv_path_;
    IpcServer* ipc_;
    MockExecutor* executor_;
    RiskManager* risk_;
};
