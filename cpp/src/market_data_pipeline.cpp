#include "market_data_pipeline.h"
#include "order_book.h"
#include "tick_processor.h"
#include "udp_receiver.h"
#include "market_data_writer.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <signal.h>

namespace hf_arbitrage {

MarketDataPipeline::MarketDataPipeline(const Config& config)
    : config_(config),
      order_book_manager_(config.max_symbols),
      tick_processor_(order_book_manager_),
      udp_receiver_(config.data_source_ip, config.data_source_port),
      market_data_writer_(config.output_directory),
      running_(false),
      stats_thread_(),
      processing_thread_() {
}

MarketDataPipeline::~MarketDataPipeline() {
    stop();
}

void MarketDataPipeline::start() {
    if (running_) {
        std::cerr << "Pipeline already running" << std::endl;
        return;
    }
    
    running_ = true;
    
    // Start UDP receiver
    if (!udp_receiver_.start()) {
        std::cerr << "Failed to start UDP receiver" << std::endl;
        return;
    }
    
    // Start processing thread
    processing_thread_ = std::thread(&MarketDataPipeline::processingLoop, this);
    
    // Start statistics thread
    stats_thread_ = std::thread(&MarketDataPipeline::statsLoop, this);
    
    std::cout << "Market data pipeline started" << std::endl;
    std::cout << "Listening on " << config_.data_source_ip << ":" << config_.data_source_port << std::endl;
}

void MarketDataPipeline::stop() {
    if (!running_) return;
    
    running_ = false;
    
    // Stop UDP receiver
    udp_receiver_.stop();
    
    // Wait for threads to finish
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
    
    if (stats_thread_.joinable()) {
        stats_thread_.join();
    }
    
    std::cout << "Market data pipeline stopped" << std::endl;
}

void MarketDataPipeline::processingLoop() {
    const size_t buffer_size = 8192;
    std::vector<uint8_t> buffer(buffer_size);
    
    while (running_) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Receive data from UDP
        ssize_t bytes_received = udp_receiver_.receive(buffer.data(), buffer.size());
        
        if (bytes_received > 0) {
            // Process the received data
            tick_processor_.processBuffer(buffer.data(), bytes_received);
            
            // Write to disk if needed
            if (config_.enable_logging) {
                market_data_writer_.writeBuffer(buffer.data(), bytes_received);
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto processing_time = end_time - start_time;
        
        // Update statistics
        updateStats(processing_time);
        
        // Yield to other threads if no data
        if (bytes_received <= 0) {
            std::this_thread::yield();
        }
    }
}

void MarketDataPipeline::statsLoop() {
    const auto stats_interval = std::chrono::seconds(5);
    auto last_stats_time = std::chrono::high_resolution_clock::now();
    
    while (running_) {
        auto current_time = std::chrono::high_resolution_clock::now();
        
        if (current_time - last_stats_time >= stats_interval) {
            printStats();
            last_stats_time = current_time;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void MarketDataPipeline::updateStats(const Nanoseconds& processing_time) {
    MarketDataStats current_stats = stats_.load();
    current_stats.messages_processed++;
    current_stats.total_processing_time += processing_time;
    current_stats.min_processing_time = std::min(current_stats.min_processing_time, processing_time);
    current_stats.max_processing_time = std::max(current_stats.max_processing_time, processing_time);
    current_stats.avg_processing_time = current_stats.total_processing_time / current_stats.messages_processed;
    stats_.store(current_stats);
}

void MarketDataPipeline::printStats() {
    MarketDataStats current_stats = stats_.load();
    MarketDataStats order_book_stats = order_book_manager_.getTotalStats();
    
    std::cout << "\n=== Market Data Pipeline Statistics ===" << std::endl;
    std::cout << "Messages Processed: " << current_stats.messages_processed << std::endl;
    std::cout << "Trades Processed: " << order_book_stats.trades_processed << std::endl;
    std::cout << "Quotes Processed: " << order_book_stats.quotes_processed << std::endl;
    std::cout << "Processing Time (avg): " 
              << std::chrono::duration_cast<std::chrono::nanoseconds>(current_stats.avg_processing_time).count() 
              << " ns" << std::endl;
    std::cout << "Processing Time (min): " 
              << std::chrono::duration_cast<std::chrono::nanoseconds>(current_stats.min_processing_time).count() 
              << " ns" << std::endl;
    std::cout << "Processing Time (max): " 
              << std::chrono::duration_cast<std::chrono::nanoseconds>(current_stats.max_processing_time).count() 
              << " ns" << std::endl;
    
    // Print order book statistics
    auto snapshots = order_book_manager_.getAllSnapshots();
    uint32_t active_symbols = 0;
    for (const auto& snapshot : snapshots) {
        if (snapshot.bid_levels > 0 || snapshot.ask_levels > 0) {
            active_symbols++;
        }
    }
    std::cout << "Active Symbols: " << active_symbols << std::endl;
    std::cout << "=====================================" << std::endl;
}

MarketDataStats MarketDataPipeline::getStats() const {
    return stats_.load();
}

std::vector<OrderBookSnapshot> MarketDataPipeline::getOrderBookSnapshots() const {
    return order_book_manager_.getAllSnapshots();
}

// Global pipeline instance for signal handling
static MarketDataPipeline* g_pipeline = nullptr;

void signalHandler(int signal) {
    if (g_pipeline) {
        std::cout << "\nReceived signal " << signal << ", shutting down..." << std::endl;
        g_pipeline->stop();
    }
}

} // namespace hf_arbitrage

int main(int argc, char* argv[]) {
    using namespace hf_arbitrage;
    
    // Default configuration
    Config config;
    config.data_source_ip = "127.0.0.1";
    config.data_source_port = 8888;
    config.output_directory = "./data";
    config.max_symbols = 1000;
    config.ring_buffer_size = 1000000;
    config.enable_logging = true;
    config.log_level = "INFO";
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--ip" && i + 1 < argc) {
            config.data_source_ip = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            config.data_source_port = static_cast<uint16_t>(std::stoi(argv[++i]));
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_directory = argv[++i];
        } else if (arg == "--symbols" && i + 1 < argc) {
            config.max_symbols = static_cast<uint32_t>(std::stoi(argv[++i]));
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --ip <ip>        Data source IP address (default: 127.0.0.1)" << std::endl;
            std::cout << "  --port <port>    Data source port (default: 8888)" << std::endl;
            std::cout << "  --output <dir>   Output directory (default: ./data)" << std::endl;
            std::cout << "  --symbols <num>  Maximum number of symbols (default: 1000)" << std::endl;
            std::cout << "  --help           Show this help message" << std::endl;
            return 0;
        }
    }
    
    // Create and start pipeline
    MarketDataPipeline pipeline(config);
    g_pipeline = &pipeline;
    
    // Set up signal handlers
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    try {
        pipeline.start();
        
        // Keep main thread alive
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 