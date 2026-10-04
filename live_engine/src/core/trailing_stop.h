#pragma once
#include <string>
#include <deque>
#include "kline_data.h"

// One-directional trailing stop for a single position.
//
// The exit is "price hits the stop": the stop trails in the favorable
// direction and never reverses. Supports the bracket-order rules the
// OrderPayload can request:
//   "donchian_low"   -> long stops trail the N-bar low  (running max)
//   "donchian_high"  -> short stops trail the N-bar high (running min)
//   "moving_average" -> stop trails the N-bar close average
//   "none"           -> fixed hard stop only
class TrailingStop {
public:
    void configure(const std::string& indicator, int period);
    void on_entry(int side, double hard_stop);   // +1 long / -1 short
    // Returns true when the position should exit; fills exit_price + reason.
    bool on_bar(const KLineData& bar, double& exit_price, std::string& reason);
    void reset();
    bool active() const { return side_ != 0; }

private:
    int side_ = 0;
    double hard_stop_ = 0.0;
    std::string indicator_ = "none";
    int period_ = 0;
    std::deque<double> lows_;
    std::deque<double> highs_;
    std::deque<double> closes_;
};
