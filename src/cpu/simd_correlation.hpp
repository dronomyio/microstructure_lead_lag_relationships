#pragma once
#include <vector>
#include "../include/types.hpp"
#include "../include/config.hpp"

namespace leadlag {
namespace cpu {

class SIMDCorrelation {
public:
    void compute(
        const std::vector<float>& prices1,
        const std::vector<float>& prices2,
        std::vector<LeadLagResult>& results,
        const Config& config
    );
    
private:
    float computeCorrelationAtLag(
        const float* prices1,
        const float* prices2,
        int n_samples,
        int lag
    );
    
    float calculateInformationRatio(const std::vector<float>& correlations);
};

} // namespace cpu
} // namespace leadlag
