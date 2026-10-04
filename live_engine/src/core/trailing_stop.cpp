#include "trailing_stop.h"
#include <algorithm>

void TrailingStop::configure(const std::string& indicator, int period) {
    indicator_ = indicator;
    period_ = period;
}

void TrailingStop::observe(const KLineData& bar) {
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
}

double TrailingStop::trailing_level() const {
    if (indicator_ == "donchian_low" && !lows_.empty()) {
        return *std::min_element(lows_.begin(), lows_.end());
    }
    if (indicator_ == "donchian_high" && !highs_.empty()) {
        return *std::max_element(highs_.begin(), highs_.end());
    }
    if (indicator_ == "moving_average" && !closes_.empty()) {
        double sum = 0.0;
        for (double c : closes_) sum += c;
        return sum / static_cast<double>(closes_.size());
    }
    return 0.0;
}

void TrailingStop::on_entry(int side, double hard_stop) {
    side_ = side;
    hard_stop_ = hard_stop;
    extreme_ = hard_stop;  // running extreme starts at the initial stop level
}

void TrailingStop::reset() {
    side_ = 0;
    hard_stop_ = 0.0;
    extreme_ = 0.0;
    // NOTE: the rolling window is intentionally NOT cleared — it must persist
    // across positions so the N-bar lookback stays full-history.
}

bool TrailingStop::on_bar(const KLineData& bar, double& exit_price, std::string& reason) {
    if (side_ == 0) return false;

    // Trailing level from the window of the previous N bars (observe() is
    // called *after* this check, so the current bar is excluded).
    double trailing = trailing_level();

    if (side_ > 0) {
        double stop = std::max(hard_stop_, extreme_);  // trails up, never down
        if (bar.low <= stop) {
            exit_price = std::min(bar.close, stop);
            reason = "trailing_stop";
            reset();
            return true;
        }
        extreme_ = std::max(extreme_, trailing);
    } else {
        double stop = std::min(hard_stop_, extreme_);  // trails down, never up
        if (bar.high >= stop) {
            exit_price = std::max(bar.close, stop);
            reason = "trailing_stop";
            reset();
            return true;
        }
        extreme_ = std::min(extreme_, trailing);
    }
    return false;
}
