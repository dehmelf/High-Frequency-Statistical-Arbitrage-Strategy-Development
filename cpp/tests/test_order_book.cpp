#include <gtest/gtest.h>
#include "../include/order_book.h"
#include <chrono>

using namespace hf_arbitrage;

class OrderBookTest : public ::testing::Test {
protected:
    void SetUp() override {
        order_book = std::make_unique<OrderBook>(1); // Symbol ID 1
    }
    
    std::unique_ptr<OrderBook> order_book;
};

TEST_F(OrderBookTest, Initialization) {
    EXPECT_EQ(order_book->getBidLevels(), 0);
    EXPECT_EQ(order_book->getAskLevels(), 0);
    EXPECT_EQ(order_book->getBestBid(), 0.0);
    EXPECT_EQ(order_book->getBestAsk(), 0.0);
    EXPECT_EQ(order_book->getSpread(), 0.0);
}

TEST_F(OrderBookTest, AddBidOrder) {
    order_book->addOrder(Side::BUY, 100.0, 100, 1);
    
    EXPECT_EQ(order_book->getBidLevels(), 1);
    EXPECT_EQ(order_book->getBestBid(), 100.0);
    EXPECT_EQ(order_book->getBestBidQuantity(), 100);
}

TEST_F(OrderBookTest, AddAskOrder) {
    order_book->addOrder(Side::SELL, 101.0, 50, 2);
    
    EXPECT_EQ(order_book->getAskLevels(), 1);
    EXPECT_EQ(order_book->getBestAsk(), 101.0);
    EXPECT_EQ(order_book->getBestAskQuantity(), 50);
}

TEST_F(OrderBookTest, SpreadCalculation) {
    order_book->addOrder(Side::BUY, 100.0, 100, 1);
    order_book->addOrder(Side::SELL, 101.0, 50, 2);
    
    EXPECT_EQ(order_book->getSpread(), 1.0);
    EXPECT_EQ(order_book->getMidPrice(), 100.5);
}

TEST_F(OrderBookTest, ProcessTrade) {
    order_book->addOrder(Side::BUY, 100.0, 100, 1);
    order_book->addOrder(Side::SELL, 101.0, 50, 2);
    
    // Process a trade at 100.5
    order_book->processTrade(100.5, 25, Side::SELL);
    
    // Check that bid quantity was reduced
    EXPECT_EQ(order_book->getBestBidQuantity(), 75);
}

TEST_F(OrderBookTest, GetSnapshot) {
    order_book->addOrder(Side::BUY, 100.0, 100, 1);
    order_book->addOrder(Side::SELL, 101.0, 50, 2);
    
    auto snapshot = order_book->getSnapshot();
    
    EXPECT_EQ(snapshot.symbol_id, 1);
    EXPECT_EQ(snapshot.bid_levels, 1);
    EXPECT_EQ(snapshot.ask_levels, 1);
    EXPECT_EQ(snapshot.bids[0].price, 100.0);
    EXPECT_EQ(snapshot.asks[0].price, 101.0);
}

TEST_F(OrderBookTest, ClearOrderBook) {
    order_book->addOrder(Side::BUY, 100.0, 100, 1);
    order_book->addOrder(Side::SELL, 101.0, 50, 2);
    
    order_book->clear();
    
    EXPECT_EQ(order_book->getBidLevels(), 0);
    EXPECT_EQ(order_book->getAskLevels(), 0);
    EXPECT_EQ(order_book->getBestBid(), 0.0);
    EXPECT_EQ(order_book->getBestAsk(), 0.0);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 