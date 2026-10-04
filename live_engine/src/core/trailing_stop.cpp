#include "trailing_stop.h"
#include <algorithm>

void TrailingStop::configure(const std::string& indicator, int period) {
    indicator_ = indicator;
    period_ = period;
}

void TrailingStop::on_entry(int side, double hard_stop) {
    side_ = side;
    hard_stop_ = hard_stop;
    lows_.clear();
    highs_.clear();
    closes_.clear();
}

void TrailingStop::reset() {
    side_ = 0;
    hard_stop_ = 0.0;
    lows_.clear();
    highs_.clear();
    closes_.clear();
}

bool TrailingStop::on_bar(const KLineData& bar, double& exit_price, std::string& reason) {
    if (side_ == 0) return false;

    // Keep a rolling window of the last `period_` bars.
    lows_.push_back(bar.low);
    highs_.push_back(bar.high);
    closes_.push_back(bar.close);
    if (period_ > 0) {
        while (static_cast<int>(lows_.size()) > period_) {
            lows_.pop_front();
            highs_.pop_front();
            closes_.pop_front();
        }
    }

    // Compute the trailing level (if the rule is set).
    double trailing = 0.0;
    bool has_trailing = false;
    if (indicator_ == "donchian_low" && !lows_.empty()) {
        trailing = *std::min_element(lows_.begin(), lows_.end());
        has_trailing = true;
    } else if (indicator_ == "donchian_high" && !highs_.empty()) {
        trailing = *std::max_element(highs_.begin(), highs_.end());
        has_trailing = true;
    } else if (indicator_ == "moving_average" && !closes_.empty()) {
        double sum = 0.0;
        for (double c : closes_) sum += c;
        trailing = sum / static_cast<double>(closes_.size());
        has_trailing = true;
    }

    if (side_ > 0) {
        double stop = hard_stop_;
        if (has_trailing) stop = std::max(stop, trailing);  // trails up, never down
        if (bar.low <= stop) {
            exit_price = std::min(bar.close, stop);
            reason = "trailing_stop";
            reset();
            return true;
        }
    } else {
        double stop = hard_stop_;
        if (has_trailing) stop = std::min(stop, trailing);  // trails down, never up
        if (bar.high >= stop) {
            exit_price = std::max(bar.close, stop);
            reason = "trailing_stop";
            reset();
            return true;
        }
    }
    return false;
}
