#pragma once
#include <string>
#include <deque>
#include "kline_data.h"

// One-directional trailing stop for a single position, shared by the backtest
// MockExecutor and the live BinanceLiveExecutor.
//
// Semantics match the Python TrailingExitLabeler exactly:
//   * The trailing level is the Donchian channel of the *previous* N bars
//     (add_indicators computes exit_low/exit_high with .shift(1)).
//   * The stop trails in the favorable direction and NEVER reverses:
//       long:  stop = running MAX of the N-bar low
//       short: stop = running MIN of the N-bar high
//
// Ordering contract: per bar, call on_bar() (check exit using the window of
// the previous N bars) and then observe() (append the current bar to the
// window). The window persists across positions so the lookback is always the
// full N bars of history, not "bars since entry".
class TrailingStop {
public:
    void configure(const std::string& indicator, int period);
    // Append a bar to the rolling lookback window (always, even when flat).
    void observe(const KLineData& bar);
    // Arm for a position: +1 long / -1 short. Seeds the extreme from the hard
    // stop (== the entry Donchian level for donchian mode; == ATR stop floor
    // for atr mode). The max(hard_stop, extreme) flooring makes the seed exact.
    void on_entry(int side, double hard_stop);
    // Returns true when the position should exit; fills exit_price + reason.
    bool on_bar(const KLineData& bar, double& exit_price, std::string& reason);
    void reset();
    bool active() const { return side_ != 0; }

private:
    double trailing_level() const;

    int side_ = 0;
    double hard_stop_ = 0.0;
    double extreme_ = 0.0;   // running max (long) / min (short) of trailing level
    std::string indicator_ = "none";
    int period_ = 0;
    std::deque<double> lows_;
    std::deque<double> highs_;
    std::deque<double> closes_;
};
