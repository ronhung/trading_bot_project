#pragma once
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <zmq.hpp>
#include "kline_data.h"
#include "i_order_executor.h"
#include "risk_manager.h"
#include "thread_safe_queue.h"

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

    // Live: drain queued order-update messages onto the PUB socket.
    // MUST be called only from the main-loop thread (ZMQ socket not thread-safe).
    void pump_order_updates();

private:
    void receive_loop();
    void handle_message(const std::string& msg_str);
    bool recv_until_ack(int timeout_ms);

    // Thread-safe enqueue of an order status update JSON string.
    void queue_order_update(const OrderStatusUpdate& u);

    zmq::context_t context;
    zmq::socket_t publisher;
    zmq::socket_t receiver;

    std::thread rx_thread;
    std::atomic<bool> running;
    bool sync_mode_;

    IOrderExecutor* executor_;
    RiskManager* risk_manager_;

    // Queue for order status updates to be published from the main-loop thread.
    ThreadSafeQueue<std::string> order_update_queue_;
};
