#include "mock_executor.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>

MockExecutor::MockExecutor(RiskManager* risk_manager, double fee_rate, double slippage_bps)
    : risk_(risk_manager), fee_rate_(fee_rate), slippage_bps_(slippage_bps) {
    std::cout << "🧪 [MockExecutor] Initialized | fee=" << (fee_rate_ * 100.0)
              << "% | slippage=" << slippage_bps_ << " bps (Queue Simulation Active)" << std::endl;
}

void MockExecutor::set_market(const KLineData& bar) {
    current_bar_ = bar;
    trailing_stop_.observe(bar);
    process_pending_orders(bar);
}

void MockExecutor::set_exit_execution(const ExecutionSpec& exit_exec) {
    exit_execution_ = exit_exec;
}

bool MockExecutor::get_current_position(const std::string& symbol, double& out_position) {
    (void)symbol;
    if (!risk_) {
        out_position = 0.0;
        return false;
    }
    out_position = risk_->get_current_position();
    return true;
}

double MockExecutor::apply_slippage(const std::string& side, double price, double notional) const {
    const double base_slip = slippage_bps_ / 10000.0;
    // Market impact: orders larger than 500k USDT encounter non-linear slippage penalty
    const double impact_mult = 1.0 + std::min(5.0, notional / 500000.0);
    const double effective_slip = base_slip * impact_mult;
    if (side == "BUY") {
        return price * (1.0 + effective_slip);
    }
    return price * (1.0 - effective_slip);
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

bool MockExecutor::execute_fill(const std::string& symbol, const std::string& side,
                                double quantity, double fill_price, bool reduce_only,
                                const std::string& reason) {
    if (!risk_ || quantity <= 0.0 || fill_price <= 0.0) {
        return false;
    }

    double notional = fill_price * quantity;
    double fee = notional * fee_rate_;
    double pos = risk_->get_current_position();
    double bal = risk_->get_current_balance();
    double entry = risk_->get_entry_price();
    double pnl = 0.0;

    if (reduce_only || (side == "SELL" && pos > 0.0) || (side == "BUY" && pos < 0.0)) {
        double close_qty = std::min(quantity, std::abs(pos));
        if (close_qty <= 0.0) {
            return false;
        }
        notional = fill_price * close_qty;
        fee = notional * fee_rate_;

        if (pos > 0.0) {
            pnl = (fill_price - entry) * close_qty;
            pos -= close_qty;
        } else {
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
        record_trade(symbol, side, close_qty, fill_price, fee, pnl, reason);
        std::cout << "✅ [MockExecutor] FILL CLOSE " << side << " qty=" << close_qty
                  << " @ " << fill_price << " | pnl=" << pnl << " fee=" << fee
                  << " | bal=" << bal << " (" << reason << ")" << std::endl;
        return true;
    }

    if (side == "BUY") {
        if (pos < 0.0) {
            std::cout << "⚠️ [MockExecutor] Already short; cannot BUY open." << std::endl;
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
        record_trade(symbol, side, quantity, fill_price, fee, 0.0, reason);
        std::cout << "✅ [MockExecutor] FILL OPEN LONG qty=" << quantity << " @ " << fill_price
                  << " fee=" << fee << " | bal=" << bal << " (" << reason << ")" << std::endl;
        return true;
    }

    if (side == "SELL") {
        if (pos > 0.0) {
            std::cout << "⚠️ [MockExecutor] Already long; cannot SELL open." << std::endl;
            return false;
        }
        double new_pos = pos - quantity;
        double new_entry = (pos >= 0.0)
            ? fill_price
            : ((entry * std::abs(pos)) + (fill_price * quantity)) / std::abs(new_pos);
        bal -= fee;
        risk_->update_balance(bal);
        risk_->update_position(new_pos);
        risk_->set_entry_price(new_entry);
        record_trade(symbol, side, quantity, fill_price, fee, 0.0, reason);
        std::cout << "✅ [MockExecutor] FILL OPEN SHORT qty=" << quantity << " @ " << fill_price
                  << " fee=" << fee << " | bal=" << bal << " (" << reason << ")" << std::endl;
        return true;
    }

    return false;
}

void MockExecutor::process_pending_orders(const KLineData& bar) {
    if (pending_orders_.empty()) return;

    std::deque<PendingLimitOrder> still_pending;
    while (!pending_orders_.empty()) {
        auto ord = pending_orders_.front();
        pending_orders_.pop_front();

        bool filled = false;
        if (ord.side == "BUY") {
            // Price penetrated strictly below limit price -> guaranteed full fill
            if (bar.low < ord.price) {
                execute_fill(ord.symbol, ord.side, ord.quantity, ord.price, ord.reduce_only, "limit_penetrate");
                filled = true;
            } else if (bar.low == ord.price) {
                // Touched exact limit price: consume queue by available bar volume
                ord.queue_ahead -= bar.volume;
                if (ord.queue_ahead <= 0.0) {
                    execute_fill(ord.symbol, ord.side, ord.quantity, ord.price, ord.reduce_only, "limit_queue_cleared");
                    filled = true;
                }
            }
        } else if (ord.side == "SELL") {
            // Price penetrated strictly above limit price -> guaranteed full fill
            if (bar.high > ord.price) {
                execute_fill(ord.symbol, ord.side, ord.quantity, ord.price, ord.reduce_only, "limit_penetrate");
                filled = true;
            } else if (bar.high == ord.price) {
                ord.queue_ahead -= bar.volume;
                if (ord.queue_ahead <= 0.0) {
                    execute_fill(ord.symbol, ord.side, ord.quantity, ord.price, ord.reduce_only, "limit_queue_cleared");
                    filled = true;
                }
            }
        }

        if (!filled) {
            still_pending.push_back(ord);
        }
    }
    pending_orders_ = std::move(still_pending);
}

bool MockExecutor::send_order(const OrderRequest& req) {
    const std::string& symbol = req.symbol;
    const std::string& side = req.side;
    const double quantity = req.quantity;
    const double price = req.price;
    const bool reduce_only = req.reduce_only;
    const std::string& ot = req.execution.order_type;

    if (!risk_ || quantity <= 0.0 || price <= 0.0) {
        return false;
    }

    // 1. MARKET or IOC orders execute immediately with impact slippage
    if (ot == "MARKET" || ot == "IOC") {
        double fill_price = apply_slippage(side, price, price * quantity);
        return execute_fill(symbol, side, quantity, fill_price, reduce_only, "market_order");
    }

    // 2. LIMIT / LIMIT_MAKER: Queue position simulation
    // If limit order aggressively crosses current market, execute immediately
    if (side == "BUY" && price >= current_bar_.close) {
        double fill_price = apply_slippage(side, current_bar_.close, current_bar_.close * quantity);
        return execute_fill(symbol, side, quantity, fill_price, reduce_only, "limit_cross_market");
    }
    if (side == "SELL" && price <= current_bar_.close) {
        double fill_price = apply_slippage(side, current_bar_.close, current_bar_.close * quantity);
        return execute_fill(symbol, side, quantity, fill_price, reduce_only, "limit_cross_market");
    }

    // Resting passive limit order: place in queue
    PendingLimitOrder pending;
    pending.symbol = symbol;
    pending.side = side;
    pending.price = price;
    pending.quantity = quantity;
    // Conservative queue position: estimated 10% of typical 1m volume waiting ahead
    pending.queue_ahead = std::max(10.0, current_bar_.volume * 0.10);
    pending.reduce_only = reduce_only;
    pending.execution = req.execution;
    pending.created_time_ms = current_bar_.close_time;

    pending_orders_.push_back(pending);
    std::cout << "⏳ [MockExecutor] Resting " << ot << " " << side << " placed at " << price
              << " | Queue ahead=" << pending.queue_ahead << std::endl;
    return true;
}

void MockExecutor::arm_trailing_exit(const std::string& indicator, int period) {
    if (indicator == "none" || period <= 0) return;
    trailing_stop_.configure(indicator, period);
    double pos = risk_ ? risk_->get_current_position() : 0.0;
    int side = (pos > 0) ? 1 : ((pos < 0) ? -1 : 0);
    double initial_stop = risk_ ? risk_->get_stop_price() : 0.0;
    trailing_stop_.on_entry(side, initial_stop);
}

bool MockExecutor::check_and_execute_stop(const std::string& symbol) {
    if (!risk_) return false;
    double pos = risk_->get_current_position();
    if (std::abs(pos) < 1e-12) return false;

    double exit_price = 0.0;
    std::string reason;
    if (trailing_stop_.on_bar(current_bar_, exit_price, reason)) {
        std::string close_side = (pos > 0) ? "SELL" : "BUY";
        double fill_price = apply_slippage(close_side, exit_price, exit_price * std::abs(pos));
        return execute_fill(symbol, close_side, std::abs(pos), fill_price, true, reason);
    }
    return false;
}

void MockExecutor::export_trades_csv(const std::string& path) const {
    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "❌ Failed to open trades export file: " << path << std::endl;
        return;
    }

    out << "timestamp_ms,symbol,side,quantity,price,fee,pnl,balance_after,position_after,reason\n";
    for (const auto& t : trades_) {
        out << t.timestamp_ms << ","
            << t.symbol << ","
            << t.side << ","
            << std::fixed << std::setprecision(6) << t.quantity << ","
            << std::setprecision(2) << t.price << ","
            << std::setprecision(4) << t.fee << ","
            << std::setprecision(2) << t.pnl << ","
            << std::setprecision(2) << t.balance_after << ","
            << std::setprecision(6) << t.position_after << ","
            << t.reason << "\n";
    }
    std::cout << "📝 [MockExecutor] Exported " << trades_.size() << " trades to " << path << std::endl;
}
