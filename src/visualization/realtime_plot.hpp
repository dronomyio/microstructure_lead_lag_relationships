#pragma once
#include <vector>
#include "../include/types.hpp"
#include "../include/config.hpp"

namespace leadlag {
namespace visualization {

class RealTimePlot {
public:
    void start(const std::vector<LeadLagResult>& results, const Config& config);
};

} // namespace visualization
} // namespace leadlag
