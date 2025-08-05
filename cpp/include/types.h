#pragma once

#include <cstdint>
#include <array>
#include <string>
#include <chrono>

namespace hf_arbitrage {

// High-resolution timestamp type
using Timestamp = std::chrono::time_point<std::chrono::high_resolution_clock>;
using Nanoseconds = std::chrono::nanoseconds;

// Market data types
using Price = double;
using Quantity = uint64_t;
using OrderId = uint64_t;
using SymbolId = uint32_t;
using SequenceNumber = uint64_t;

// Order book depth
constexpr size_t MAX_ORDER_BOOK_LEVELS = 10;
constexpr size_t MAX_SYMBOLS = 1000;
constexpr size_t RING_BUFFER_SIZE = 1000000;

// Market data message types
enum class MessageType : uint8_t {
    TRADE = 1,
    QUOTE = 2,
    ORDER_UPDATE = 3,
    HEARTBEAT = 4
};

// Order side
enum class Side : uint8_t {
    BUY = 0,
    SELL = 1
};

// Order book level structure
struct OrderBookLevel {
    Price price;
    Quantity quantity;
    uint32_t order_count;
    
    OrderBookLevel() : price(0.0), quantity(0), order_count(0) {}
    OrderBookLevel(Price p, Quantity q, uint32_t count) 
        : price(p), quantity(q), order_count(count) {}
};

// Order book snapshot
struct OrderBookSnapshot {
    SymbolId symbol_id;
    Timestamp timestamp;
    std::array<OrderBookLevel, MAX_ORDER_BOOK_LEVELS> bids;
    std::array<OrderBookLevel, MAX_ORDER_BOOK_LEVELS> asks;
    uint32_t bid_levels;
    uint32_t ask_levels;
    
    OrderBookSnapshot() : symbol_id(0), bid_levels(0), ask_levels(0) {}
};

// Trade message
struct TradeMessage {
    SymbolId symbol_id;
    Timestamp timestamp;
    Price price;
    Quantity quantity;
    Side side;
    OrderId trade_id;
    
    TradeMessage() : symbol_id(0), price(0.0), quantity(0), side(Side::BUY), trade_id(0) {}
};

// Quote message
struct QuoteMessage {
    SymbolId symbol_id;
    Timestamp timestamp;
    Price bid_price;
    Quantity bid_quantity;
    Price ask_price;
    Quantity ask_quantity;
    
    QuoteMessage() : symbol_id(0), bid_price(0.0), bid_quantity(0), 
                    ask_price(0.0), ask_quantity(0) {}
};

// Market data statistics
struct MarketDataStats {
    uint64_t messages_processed;
    uint64_t trades_processed;
    uint64_t quotes_processed;
    Nanoseconds total_processing_time;
    Nanoseconds min_processing_time;
    Nanoseconds max_processing_time;
    Nanoseconds avg_processing_time;
    
    MarketDataStats() : messages_processed(0), trades_processed(0), quotes_processed(0),
                       total_processing_time(0), min_processing_time(Nanoseconds::max()),
                       max_processing_time(0), avg_processing_time(0) {}
};

// Performance metrics
struct PerformanceMetrics {
    double sharpe_ratio;
    double max_drawdown;
    double total_return;
    double volatility;
    uint64_t total_trades;
    double win_rate;
    double avg_trade_duration;
    
    PerformanceMetrics() : sharpe_ratio(0.0), max_drawdown(0.0), total_return(0.0),
                          volatility(0.0), total_trades(0), win_rate(0.0), avg_trade_duration(0.0) {}
};

// Configuration structure
struct Config {
    std::string data_source_ip;
    uint16_t data_source_port;
    std::string output_directory;
    uint32_t max_symbols;
    uint32_t ring_buffer_size;
    bool enable_logging;
    std::string log_level;
    
    Config() : data_source_port(0), max_symbols(MAX_SYMBOLS), 
               ring_buffer_size(RING_BUFFER_SIZE), enable_logging(true), log_level("INFO") {}
};

} // namespace hf_arbitrage 