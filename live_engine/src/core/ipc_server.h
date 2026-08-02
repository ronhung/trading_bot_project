#pragma once
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <zmq.hpp>
#include "kline_data.h"
#include "i_order_executor.h"
#include "risk_manager.h"

class IpcServer {
public:
    // sync_mode=true: backtest — publish then block until Python ACK
    // sync_mode=false: live — background PULL thread
    IpcServer(int pub_port, int pull_port,
              IOrderExecutor* executor, RiskManager* risk_manager,
              bool sync_mode = false);
    ~IpcServer();

    void publish_kline(const KLineData& kline);

    // Backtest: publish kline and wait until Python ACKs (orders applied first).
    bool publish_kline_and_wait(const KLineData& kline, int timeout_ms = 30000);

    // Backtest handshake: ping until Python brain connects and ACKs.
    bool wait_for_python(int timeout_ms = 300000);

private:
    void receive_loop();
    void handle_message(const std::string& msg_str);
    bool recv_until_ack(int timeout_ms);

    zmq::context_t context;
    zmq::socket_t publisher;
    zmq::socket_t receiver;

    std::thread rx_thread;
    std::atomic<bool> running;
    bool sync_mode_;

    IOrderExecutor* executor_;
    RiskManager* risk_manager_;
};
