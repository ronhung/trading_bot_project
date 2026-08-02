#include "mock_executor.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>

MockExecutor::MockExecutor(RiskManager* risk_manager, double fee_rate, double slippage_bps)
    : risk_(risk_manager), fee_rate_(fee_rate), slippage_bps_(slippage_bps) {
    std::cout << "🧪 [MockExecutor] Initialized | fee=" << (fee_rate_ * 100.0)
              << "% | slippage=" << slippage_bps_ << " bps" << std::endl;
}

void MockExecutor::set_market(const KLineData& bar) {
    current_bar_ = bar;
}

double MockExecutor::apply_slippage(const std::string& side, double price) const {
    const double slip = slippage_bps_ / 10000.0;
    if (side == "BUY") {
        return price * (1.0 + slip);
    }
    return price * (1.0 - slip);
}

void MockExecutor::record_trade(const std::string& symbol, const std::string& side,
                                double qty, double fill_price, double fee, double pnl,
                                const std::string& reason) {
    TradeRecord rec;
    rec.timestamp_ms = current_bar_.close_time;
    rec.symbol = symbol;
    rec.side = side;
    rec.quantity = qty;
    rec.price = fill_price;
    rec.fee = fee;
    rec.pnl = pnl;
    rec.balance_after = risk_->get_current_balance();
    rec.position_after = risk_->get_current_position();
    rec.reason = reason;
    trades_.push_back(rec);
}

bool MockExecutor::send_order(const std::string& symbol,
                              const std::string& side,
                              double quantity,
                              double price,
                              bool reduce_only) {
    if (!risk_ || quantity <= 0.0 || price <= 0.0) {
        return false;
    }

    double fill_price = apply_slippage(side, price);
    double notional = fill_price * quantity;
    double fee = notional * fee_rate_;

    double pos = risk_->get_current_position();
    double bal = risk_->get_current_balance();
    double entry = risk_->get_entry_price();
    double pnl = 0.0;

    if (reduce_only || (side == "SELL" && pos > 0.0) || (side == "BUY" && pos < 0.0)) {
        // Closing / reducing
        double close_qty = std::min(quantity, std::abs(pos));
        if (close_qty <= 0.0) {
            std::cout << "⚪ [MockExecutor] Nothing to close." << std::endl;
            return false;
        }
        notional = fill_price * close_qty;
        fee = notional * fee_rate_;

        if (pos > 0.0) {
            // close long
            pnl = (fill_price - entry) * close_qty;
            pos -= close_qty;
        } else {
            // close short
            pnl = (entry - fill_price) * close_qty;
            pos += close_qty;
        }

        bal += pnl - fee;
        if (std::abs(pos) < 1e-12) {
            pos = 0.0;
            risk_->clear_stop();
            risk_->set_entry_price(0.0);
        }

        risk_->update_balance(bal);
        risk_->update_position(pos);
        record_trade(symbol, side, close_qty, fill_price, fee, pnl, "close");
        std::cout << "✅ [MockExecutor] CLOSE " << side << " qty=" << close_qty
                  << " @ " << fill_price << " | pnl=" << pnl << " fee=" << fee
                  << " | bal=" << bal << std::endl;
        return true;
    }

    // Opening / adding in direction of side
    if (side == "BUY") {
        // If currently short, this would flip — for simplicity require flat or same dir
        if (pos < 0.0) {
            std::cout << "⚠️ [MockExecutor] Already short; ignore BUY open (use CLOSE_SHORT)." << std::endl;
            return false;
        }
        double new_pos = pos + quantity;
        double new_entry = (pos <= 0.0)
            ? fill_price
            : ((entry * pos) + (fill_price * quantity)) / new_pos;
        bal -= fee;
        risk_->update_balance(bal);
        risk_->update_position(new_pos);
        risk_->set_entry_price(new_entry);
        record_trade(symbol, side, quantity, fill_price, fee, 0.0, "open");
        std::cout << "✅ [MockExecutor] OPEN LONG qty=" << quantity << " @ " << fill_price
                  << " fee=" << fee << " | bal=" << bal << std::endl;
        return true;
    } else if (side == "SELL") {
        if (pos > 0.0) {
            std::cout << "⚠️ [MockExecutor] Already long; ignore SELL open (use CLOSE_LONG)." << std::endl;
            return false;
        }
        double abs_pos = std::abs(pos);
        double new_abs = abs_pos + quantity;
        double new_entry = (abs_pos <= 0.0)
            ? fill_price
            : ((entry * abs_pos) + (fill_price * quantity)) / new_abs;
        bal -= fee;
        risk_->update_balance(bal);
        risk_->update_position(-(new_abs));
        risk_->set_entry_price(new_entry);
        record_trade(symbol, side, quantity, fill_price, fee, 0.0, "open");
        std::cout << "✅ [MockExecutor] OPEN SHORT qty=" << quantity << " @ " << fill_price
                  << " fee=" << fee << " | bal=" << bal << std::endl;
        return true;
    }
    return false; // unknown side
}

bool MockExecutor::check_and_execute_stop(const std::string& symbol) {
    double pos = risk_->get_current_position();
    double stop = risk_->get_stop_price();
    if (pos == 0.0 || stop <= 0.0) {
        return false;
    }

    if (pos > 0.0 && current_bar_.low <= stop) {
        std::cout << "🛑 [MockExecutor] LONG stop hit @ " << stop << std::endl;
        send_order(symbol, "SELL", std::abs(pos), stop, true);
        return true;
    }
    if (pos < 0.0 && current_bar_.high >= stop) {
        std::cout << "🛑 [MockExecutor] SHORT stop hit @ " << stop << std::endl;
        send_order(symbol, "BUY", std::abs(pos), stop, true);
        return true;
    }
    return false;
}

void MockExecutor::export_trades_csv(const std::string& path) const {
    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "❌ Cannot write trades CSV: " << path << std::endl;
        return;
    }
    out << "timestamp_ms,symbol,side,quantity,price,fee,pnl,balance_after,position_after,reason\n";
    out << std::setprecision(10);
    for (const auto& t : trades_) {
        out << t.timestamp_ms << ','
            << t.symbol << ','
            << t.side << ','
            << t.quantity << ','
            << t.price << ','
            << t.fee << ','
            << t.pnl << ','
            << t.balance_after << ','
            << t.position_after << ','
            << t.reason << '\n';
    }
    std::cout << "📝 [MockExecutor] Wrote " << trades_.size() << " trades -> " << path << std::endl;
}
