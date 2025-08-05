#include "order_book.h"
#include <algorithm>
#include <stdexcept>

namespace hf_arbitrage {

OrderBook::OrderBook(SymbolId symbol_id) 
    : symbol_id_(symbol_id), stats_() {
}

void OrderBook::addOrder(Side side, Price price, Quantity quantity, OrderId order_id) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    withLock([&]() {
        auto& levels = (side == Side::BUY) ? bids_ : asks_;
        levels[price] += quantity;
        order_map_[order_id] = std::make_pair(side, price);
    });
    
    auto end_time = std::chrono::high_resolution_clock::now();
    updateStats(end_time - start_time);
}

void OrderBook::modifyOrder(Side side, Price price, Quantity quantity, OrderId order_id) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    withLock([&]() {
        auto& levels = (side == Side::BUY) ? bids_ : asks_;
        
        // Find existing order
        auto order_it = order_map_.find(order_id);
        if (order_it != order_map_.end()) {
            Price old_price = order_it->second.second;
            
            // Remove from old price level
            auto level_it = levels.find(old_price);
            if (level_it != levels.end()) {
                level_it->second = std::max(Quantity(0), level_it->second - quantity);
                if (level_it->second == 0) {
                    levels.erase(level_it);
                }
            }
        }
        
        // Add to new price level
        levels[price] += quantity;
        order_map_[order_id] = std::make_pair(side, price);
    });
    
    auto end_time = std::chrono::high_resolution_clock::now();
    updateStats(end_time - start_time);
}

void OrderBook::cancelOrder(Side side, Price price, OrderId order_id) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    withLock([&]() {
        auto& levels = (side == Side::BUY) ? bids_ : asks_;
        
        auto order_it = order_map_.find(order_id);
        if (order_it != order_map_.end()) {
            Price order_price = order_it->second.second;
            auto level_it = levels.find(order_price);
            
            if (level_it != levels.end()) {
                level_it->second = std::max(Quantity(0), level_it->second - 1);
                if (level_it->second == 0) {
                    levels.erase(level_it);
                }
            }
            
            order_map_.erase(order_it);
        }
    });
    
    auto end_time = std::chrono::high_resolution_clock::now();
    updateStats(end_time - start_time);
}

void OrderBook::processTrade(Price price, Quantity quantity, Side side) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    withLock([&]() {
        auto& levels = (side == Side::SELL) ? bids_ : asks_;  // Opposite side gets hit
        
        auto level_it = levels.begin();
        if (level_it != levels.end()) {
            Quantity remaining_quantity = quantity;
            
            while (remaining_quantity > 0 && level_it != levels.end()) {
                if (level_it->second <= remaining_quantity) {
                    remaining_quantity -= level_it->second;
                    level_it = levels.erase(level_it);
                } else {
                    level_it->second -= remaining_quantity;
                    remaining_quantity = 0;
                }
            }
        }
    });
    
    auto end_time = std::chrono::high_resolution_clock::now();
    updateStats(end_time - start_time);
}

OrderBookSnapshot OrderBook::getSnapshot() const {
    return withLock([this]() {
        OrderBookSnapshot snapshot;
        snapshot.symbol_id = symbol_id_;
        snapshot.timestamp = std::chrono::high_resolution_clock::now();
        
        // Fill bid levels
        uint32_t bid_count = 0;
        for (const auto& [price, quantity] : bids_) {
            if (bid_count >= MAX_ORDER_BOOK_LEVELS) break;
            snapshot.bids[bid_count] = createLevel(price, quantity);
            bid_count++;
        }
        snapshot.bid_levels = bid_count;
        
        // Fill ask levels
        uint32_t ask_count = 0;
        for (const auto& [price, quantity] : asks_) {
            if (ask_count >= MAX_ORDER_BOOK_LEVELS) break;
            snapshot.asks[ask_count] = createLevel(price, quantity);
            ask_count++;
        }
        snapshot.ask_levels = ask_count;
        
        return snapshot;
    });
}

void OrderBook::clear() {
    withLock([this]() {
        bids_.clear();
        asks_.clear();
        order_map_.clear();
    });
}

Price OrderBook::getBestBid() const {
    return withLock([this]() {
        return bids_.empty() ? 0.0 : bids_.begin()->first;
    });
}

Price OrderBook::getBestAsk() const {
    return withLock([this]() {
        return asks_.empty() ? 0.0 : asks_.begin()->first;
    });
}

Quantity OrderBook::getBestBidQuantity() const {
    return withLock([this]() {
        return bids_.empty() ? 0 : bids_.begin()->second;
    });
}

Quantity OrderBook::getBestAskQuantity() const {
    return withLock([this]() {
        return asks_.empty() ? 0 : asks_.begin()->second;
    });
}

double OrderBook::getSpread() const {
    return withLock([this]() {
        if (bids_.empty() || asks_.empty()) return 0.0;
        return asks_.begin()->first - bids_.begin()->first;
    });
}

double OrderBook::getMidPrice() const {
    return withLock([this]() {
        if (bids_.empty() || asks_.empty()) return 0.0;
        return (bids_.begin()->first + asks_.begin()->first) / 2.0;
    });
}

uint32_t OrderBook::getBidLevels() const {
    return withLock([this]() {
        return static_cast<uint32_t>(bids_.size());
    });
}

uint32_t OrderBook::getAskLevels() const {
    return withLock([this]() {
        return static_cast<uint32_t>(asks_.size());
    });
}

uint64_t OrderBook::getTotalBidQuantity() const {
    return withLock([this]() {
        uint64_t total = 0;
        for (const auto& [price, quantity] : bids_) {
            total += quantity;
        }
        return total;
    });
}

uint64_t OrderBook::getTotalAskQuantity() const {
    return withLock([this]() {
        uint64_t total = 0;
        for (const auto& [price, quantity] : asks_) {
            total += quantity;
        }
        return total;
    });
}

void OrderBook::updateStats(const Nanoseconds& processing_time) {
    MarketDataStats current_stats = stats_.load();
    current_stats.messages_processed++;
    current_stats.total_processing_time += processing_time;
    current_stats.min_processing_time = std::min(current_stats.min_processing_time, processing_time);
    current_stats.max_processing_time = std::max(current_stats.max_processing_time, processing_time);
    current_stats.avg_processing_time = current_stats.total_processing_time / current_stats.messages_processed;
    stats_.store(current_stats);
}

MarketDataStats OrderBook::getStats() const {
    return stats_.load();
}

OrderBookLevel OrderBook::createLevel(Price price, Quantity quantity) const {
    return OrderBookLevel(price, quantity, 1);  // Simplified order count
}

// OrderBookManager implementation
OrderBookManager::OrderBookManager(uint32_t max_symbols) 
    : max_symbols_(max_symbols) {
    order_books_.reserve(max_symbols);
    for (uint32_t i = 0; i < max_symbols; ++i) {
        order_books_.push_back(std::make_unique<OrderBook>(i));
    }
}

OrderBook* OrderBookManager::getOrderBook(SymbolId symbol_id) {
    if (symbol_id >= max_symbols_) return nullptr;
    return order_books_[symbol_id].get();
}

const OrderBook* OrderBookManager::getOrderBook(SymbolId symbol_id) const {
    if (symbol_id >= max_symbols_) return nullptr;
    return order_books_[symbol_id].get();
}

void OrderBookManager::processMessage(const TradeMessage& trade) {
    if (OrderBook* ob = getOrderBook(trade.symbol_id)) {
        ob->processTrade(trade.price, trade.quantity, trade.side);
    }
}

void OrderBookManager::processMessage(const QuoteMessage& quote) {
    // Quote messages don't directly modify order book
    // They're used for reference pricing
}

MarketDataStats OrderBookManager::getTotalStats() const {
    MarketDataStats total_stats;
    
    for (const auto& order_book : order_books_) {
        MarketDataStats book_stats = order_book->getStats();
        total_stats.messages_processed += book_stats.messages_processed;
        total_stats.trades_processed += book_stats.trades_processed;
        total_stats.quotes_processed += book_stats.quotes_processed;
        total_stats.total_processing_time += book_stats.total_processing_time;
        
        if (book_stats.min_processing_time < total_stats.min_processing_time) {
            total_stats.min_processing_time = book_stats.min_processing_time;
        }
        
        if (book_stats.max_processing_time > total_stats.max_processing_time) {
            total_stats.max_processing_time = book_stats.max_processing_time;
        }
    }
    
    if (total_stats.messages_processed > 0) {
        total_stats.avg_processing_time = total_stats.total_processing_time / total_stats.messages_processed;
    }
    
    return total_stats;
}

std::vector<OrderBookSnapshot> OrderBookManager::getAllSnapshots() const {
    std::vector<OrderBookSnapshot> snapshots;
    snapshots.reserve(order_books_.size());
    
    for (const auto& order_book : order_books_) {
        snapshots.push_back(order_book->getSnapshot());
    }
    
    return snapshots;
}

} // namespace hf_arbitrage 