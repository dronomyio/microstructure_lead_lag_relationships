#pragma once
#include <string>
#include <vector>
#include "../include/types.hpp"

namespace leadlag {
namespace data {

class PolygonLoader {
public:
    PolygonLoader(const std::string& api_key);
    std::vector<Quote> loadQuotesFromFile(const std::string& filepath);
    
private:
    std::string api_key;
};

} // namespace data
} // namespace leadlag
