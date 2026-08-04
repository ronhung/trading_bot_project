#pragma once
#include <string>
#include <mutex>
#include <cmath>
#include <iostream>

class RiskManager {
public:
    explicit RiskManager(double risk_pct = 0.02, double max_leverage = 20.0);

    void update_balance(double new_balance);
    void update_position(double new_position_size);

    // Turtle / ATR stop for the active position (0 = none).
    void set_stop_price(double stop_price);
    void clear_stop();
    double get_stop_price();

    double calculate_target_size(const std::string& action, double current_price, double stop_price);

    double get_current_position();
    double get_current_balance();
    double get_entry_price();
    void set_entry_price(double entry_price);

    static constexpr double kMinTradeQty = 0.001;
    static bool is_effective_position(double position) {
        return std::abs(position) >= kMinTradeQty;
    }

private:
    double risk_pct_;
    double max_leverage_;
    double current_balance_;
    double current_position_;
    double stop_price_;
    double entry_price_;
    std::mutex mtx_;
};
