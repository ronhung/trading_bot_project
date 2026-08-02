#include "order_tracker.h"
#include <iostream>
#include <algorithm>

void OrderTracker::set_update_callback(OrderUpdateCallback cb) {
    std::lock_guard<std::mutex> lock(mtx_);
    on_update_ = std::move(cb);
}

void OrderTracker::register_order(const TrackedOrder& order) {
    std::lock_guard<std::mutex> lock(mtx_);
    orders_[order.client_order_id] = order;
    std::cout << "📝 [OrderTracker] Registered " << order.client_order_id
              << " | " << order.side << " qty=" << order.quantity
              << " @ " << order.price << std::endl;
}

void OrderTracker::on_order_update(const std::string& client_order_id,
                                    const std::string& status,
                                    double filled_qty) {
    TrackedOrder* ord = nullptr;
    bool is_terminal = false;
    bool should_fire = false;

    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = orders_.find(client_order_id);
        if (it != orders_.end()) {
            ord = &it->second;

            // Map Binance status string to our enum
            if (status == "NEW") ord->status = TrackedOrderStatus::NEW;
            else if (status == "PARTIALLY_FILLED") ord->status = TrackedOrderStatus::PARTIALLY_FILLED;
            else if (status == "FILLED") ord->status = TrackedOrderStatus::FILLED;
            else if (status == "CANCELED") ord->status = TrackedOrderStatus::CANCELED;
            else if (status == "EXPIRED") ord->status = TrackedOrderStatus::EXPIRED;
            else if (status == "REJECTED") ord->status = TrackedOrderStatus::REJECTED;

            ord->filled_quantity = filled_qty;

            is_terminal = (status == "FILLED" || status == "CANCELED" ||
                          status == "EXPIRED" || status == "REJECTED");

            if (is_terminal && !ord->reported_terminal) {
                ord->reported_terminal = true;
                should_fire = true;
            }
        }
    }

    if (should_fire && ord) {
        fire_update(*ord);
    }

    if (ord) {
        std::cout << "📋 [OrderTracker] " << client_order_id
                  << " -> " << status
                  << (is_terminal ? " (terminal)" : "")
                  << " | filled=" << filled_qty << std::endl;
    }
}

std::vector<TrackedOrder> OrderTracker::get_timed_out_orders(int timeout_ms) {
    std::vector<TrackedOrder> result;
    auto now = std::chrono::steady_clock::now();

    std::lock_guard<std::mutex> lock(mtx_);
    for (auto& [cid, ord] : orders_) {
        if ((ord.status == TrackedOrderStatus::NEW ||
             ord.status == TrackedOrderStatus::PARTIALLY_FILLED) &&
            !ord.reported_terminal) {
            auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - ord.created_at).count();
            if (age >= timeout_ms) {
                result.push_back(ord);
            }
        }
    }
    return result;
}

void OrderTracker::mark_cancel_requested(const std::string& client_order_id) {
    TrackedOrder ord_copy;
    bool should_fire = false;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        auto it = orders_.find(client_order_id);
        if (it != orders_.end()) {
            it->second.status = TrackedOrderStatus::CANCELED;
            if (!it->second.reported_terminal) {
                it->second.reported_terminal = true;
                ord_copy = it->second;
                should_fire = true;
            }
        }
    }
    if (should_fire) {
        fire_update(ord_copy);
    }
}

bool OrderTracker::has_open_order() const {
    std::lock_guard<std::mutex> lock(mtx_);
    for (const auto& [cid, ord] : orders_) {
        if ((ord.status == TrackedOrderStatus::NEW ||
             ord.status == TrackedOrderStatus::PARTIALLY_FILLED) &&
            !ord.reported_terminal) {
            return true;
        }
    }
    return false;
}

const TrackedOrder* OrderTracker::find(const std::string& client_order_id) const {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = orders_.find(client_order_id);
    if (it != orders_.end()) return &it->second;
    return nullptr;
}

void OrderTracker::prune(int age_ms) {
    auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto it = orders_.begin(); it != orders_.end(); ) {
        if (it->second.reported_terminal) {
            auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - it->second.created_at).count();
            if (age >= age_ms) {
                it = orders_.erase(it);
                continue;
            }
        }
        ++it;
    }
}

void OrderTracker::fire_update(const TrackedOrder& order) {
    OrderStatusUpdate u;
    u.client_order_id = order.client_order_id;
    u.symbol = order.symbol;
    u.side = order.side;
    u.order_type = "LIMIT";
    u.quantity = order.quantity;
    u.price = order.price;
    u.reduce_only = order.reduce_only;

    switch (order.status) {
        case TrackedOrderStatus::FILLED:    u.status = "FILLED";    break;
        case TrackedOrderStatus::CANCELED:  u.status = "CANCELED";  break;
        case TrackedOrderStatus::EXPIRED:   u.status = "EXPIRED";   break;
        case TrackedOrderStatus::REJECTED:  u.status = "REJECTED";  break;
        default: u.status = "NEW"; break;
    }

    // Fire callback outside lock (caller holds no lock)
    OrderUpdateCallback cb;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        cb = on_update_;
    }
    if (cb) {
        cb(u);
    }
}
