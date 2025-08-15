#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <chrono>

namespace leadlag {

// Nanosecond timestamp type
using timestamp_t = std::chrono::nanoseconds;
using price_t = double;
using size_t = std::uint64_t;

struct Quote {
    timestamp_t timestamp;
    price_t bid_price;
    price_t ask_price;
    size_t bid_size;
    size_t ask_size;
    std::string exchange;
    std::string symbol;
    
    // For sorted container operations
    bool operator<(const Quote& other) const {
        return timestamp < other.timestamp;
    }
};

struct LeadLagResult {
    std::string symbol1;
    std::string symbol2;
    std::string exchange1;
    std::string exchange2;
    double correlation;
    int64_t lag_nanoseconds;
    double information_ratio;
    double sharpe_ratio;
};

struct Config {
    std::vector<std::string> symbols;
    std::vector<std::string> exchanges;
    int64_t window_size_ns;
    int64_t max_lag_ns;
    int lag_step_ns;
    int num_gpus;
    bool use_simd;
    std::string polygon_api_key;
    std::string data_path;
};

} // namespace leadlag
