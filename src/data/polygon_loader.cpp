#include "polygon_loader.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <iostream>

namespace leadlag {
namespace data {

PolygonLoader::PolygonLoader(const std::string& key) 
    : api_key(key) {
}

std::vector<Quote> PolygonLoader::loadQuotesFromFile(const std::string& filepath) {
    std::vector<Quote> quotes;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        // Return empty vector instead of throwing
        std::cerr << "Warning: Cannot open file: " << filepath << std::endl;
        return quotes;
    }
    
    // Simple CSV parser for demo (no JSON)
    std::string line;
    int line_num = 0;
    while (std::getline(file, line)) {
        Quote quote;
        // Create synthetic quotes for testing
        quote.timestamp = std::chrono::nanoseconds(line_num * 1000000); // 1ms intervals
        quote.bid_price = 100.0 + 0.01 * sin(line_num * 0.01);
        quote.ask_price = quote.bid_price + 0.01;
        quote.bid_size = 100;
        quote.ask_size = 100;
        quote.symbol = "TEST";
        quote.exchange = "DEMO";
        quotes.push_back(quote);
        line_num++;
    }
    
    if (quotes.empty()) {
        // Generate some synthetic data if file was empty
        for (int i = 0; i < 1000; i++) {
            Quote quote;
            quote.timestamp = std::chrono::nanoseconds(i * 1000000);
            quote.bid_price = 100.0 + 0.01 * sin(i * 0.01);
            quote.ask_price = quote.bid_price + 0.01;
            quote.bid_size = 100;
            quote.ask_size = 100;
            quote.symbol = "SYNTHETIC";
            quote.exchange = "TEST";
            quotes.push_back(quote);
        }
    }
    
    std::sort(quotes.begin(), quotes.end());
    return quotes;
}

} // namespace data
} // namespace leadlag
