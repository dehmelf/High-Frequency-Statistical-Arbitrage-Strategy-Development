#pragma once

#include "types.h"
#include <map>
#include <mutex>
#include <atomic>
#include <vector>

namespace hf_arbitrage {

class OrderBook {
public:
    OrderBook(SymbolId symbol_id);
    ~OrderBook() = default;

    // Disable copy constructor and assignment
    OrderBook(const OrderBook&) = delete;
    OrderBook& operator=(const OrderBook&) = delete;

    // Order book operations
    void addOrder(Side side, Price price, Quantity quantity, OrderId order_id);
    void modifyOrder(Side side, Price price, Quantity quantity, OrderId order_id);
    void cancelOrder(Side side, Price price, OrderId order_id);
    void processTrade(Price price, Quantity quantity, Side side);

    // Snapshot operations
    OrderBookSnapshot getSnapshot() const;
    void clear();

    // Market data access
    Price getBestBid() const;
    Price getBestAsk() const;
    Quantity getBestBidQuantity() const;
    Quantity getBestAskQuantity() const;
    double getSpread() const;
    double getMidPrice() const;

    // Statistics
    uint32_t getBidLevels() const;
    uint32_t getAskLevels() const;
    uint64_t getTotalBidQuantity() const;
    uint64_t getTotalAskQuantity() const;

    // Performance tracking
    void updateStats(const Nanoseconds& processing_time);
    MarketDataStats getStats() const;

private:
    SymbolId symbol_id_;
    mutable std::mutex mutex_;
    
    // Order book levels using price-time priority
    std::map<Price, Quantity, std::greater<Price>> bids_;  // Descending for bids
    std::map<Price, Quantity, std::less<Price>> asks_;     // Ascending for asks
    
    // Order tracking
    std::map<OrderId, std::pair<Side, Price>> order_map_;
    
    // Statistics
    mutable std::atomic<MarketDataStats> stats_;
    
    // Helper methods
    void updateLevel(std::map<Price, Quantity>& levels, Price price, Quantity quantity);
    void removeLevel(std::map<Price, Quantity>& levels, Price price);
    OrderBookLevel createLevel(Price price, Quantity quantity) const;
    
    // Thread-safe operations
    template<typename Func>
    auto withLock(Func&& func) const -> decltype(func()) {
        std::lock_guard<std::mutex> lock(mutex_);
        return func();
    }
};

// Order book manager for multiple symbols
class OrderBookManager {
public:
    OrderBookManager(uint32_t max_symbols = MAX_SYMBOLS);
    ~OrderBookManager() = default;

    // Order book access
    OrderBook* getOrderBook(SymbolId symbol_id);
    const OrderBook* getOrderBook(SymbolId symbol_id) const;
    
    // Batch operations
    void processMessage(const TradeMessage& trade);
    void processMessage(const QuoteMessage& quote);
    
    // Statistics
    MarketDataStats getTotalStats() const;
    std::vector<OrderBookSnapshot> getAllSnapshots() const;

private:
    std::vector<std::unique_ptr<OrderBook>> order_books_;
    mutable std::mutex manager_mutex_;
    uint32_t max_symbols_;
};

} // namespace hf_arbitrage 