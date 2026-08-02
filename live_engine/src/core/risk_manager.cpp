#include "risk_manager.h"

RiskManager::RiskManager(double risk_pct, double max_leverage)
    : risk_pct_(risk_pct),
      max_leverage_(max_leverage),
      current_balance_(0.0),
      current_position_(0.0),
      stop_price_(0.0),
      entry_price_(0.0) {}

void RiskManager::update_balance(double new_balance) {
    std::lock_guard<std::mutex> lock(mtx_);
    current_balance_ = new_balance;
}

void RiskManager::update_position(double new_position_size) {
    std::lock_guard<std::mutex> lock(mtx_);
    current_position_ = new_position_size;
    if (std::abs(current_position_) < 1e-12) {
        stop_price_ = 0.0;
        entry_price_ = 0.0;
    }
}

void RiskManager::set_stop_price(double stop_price) {
    std::lock_guard<std::mutex> lock(mtx_);
    stop_price_ = stop_price;
}

void RiskManager::clear_stop() {
    std::lock_guard<std::mutex> lock(mtx_);
    stop_price_ = 0.0;
}

double RiskManager::get_stop_price() {
    std::lock_guard<std::mutex> lock(mtx_);
    return stop_price_;
}

double RiskManager::get_current_position() {
    std::lock_guard<std::mutex> lock(mtx_);
    return current_position_;
}

double RiskManager::get_current_balance() {
    std::lock_guard<std::mutex> lock(mtx_);
    return current_balance_;
}

double RiskManager::get_entry_price() {
    std::lock_guard<std::mutex> lock(mtx_);
    return entry_price_;
}

void RiskManager::set_entry_price(double entry_price) {
    std::lock_guard<std::mutex> lock(mtx_);
    entry_price_ = entry_price;
}

double RiskManager::calculate_target_size(const std::string& action, double current_price, double stop_price) {
    std::lock_guard<std::mutex> lock(mtx_);

    if (current_price <= 0.0) {
        std::cerr << "⚠️ [RiskCenter] Order rejected: invalid market price!" << std::endl;
        return 0.0;
    }

    if (stop_price <= 0.0 || current_price == stop_price) {
        std::cerr << "⚠️ [RiskCenter] Order rejected: invalid stop price!" << std::endl;
        return 0.0;
    }

    // Direction sanity: long stop must be below market; short stop must be above.
    if (action == "BUY" && stop_price >= current_price) {
        std::cerr << "⚠️ [RiskCenter] Order rejected: BUY stop loss must be BELOW current price!" << std::endl;
        return 0.0;
    }
    if (action == "SELL" && stop_price <= current_price) {
        std::cerr << "⚠️ [RiskCenter] Order rejected: SELL stop loss must be ABOVE current price!" << std::endl;
        return 0.0;
    }

    // Size = (Balance * Risk%) / |Entry - Stop|
    double risk_amount = current_balance_ * risk_pct_;
    double stop_distance = std::abs(current_price - stop_price);
    double target_size = risk_amount / stop_distance;

    // Cap notional by max leverage so tiny stops cannot explode size.
    double max_notional_value = current_balance_ * max_leverage_;
    double max_size = max_notional_value / current_price;
    if (target_size > max_size) {
        std::cout << "⚠️ [RiskCenter] Target size exceeds max leverage; capping to "
                  << max_size << " BTC" << std::endl;
        target_size = max_size;
    }

    // Binance BTCUSDT min qty step 0.001 — floor so we never overshoot risk.
    target_size = std::floor(target_size * 1000.0) / 1000.0;

    std::cout << "🛡️ [RiskCenter] Evaluation -> Balance: " << current_balance_
              << " | Stop distance: " << stop_distance
              << " | Approved order size: " << target_size << " BTC" << std::endl;

    return target_size;
}
