#pragma once
#include <vector>
#include "../include/types.hpp"
#include "../include/config.hpp"

namespace leadlag {
namespace cuda {

class MultiGPULeadLag {
public:
    MultiGPULeadLag(int n_gpus);
    ~MultiGPULeadLag();
    
    void computeLeadLag(
        const std::vector<float>& prices1,
        const std::vector<float>& prices2,
        std::vector<LeadLagResult>& results,
        const Config& config
    );
    
private:
    int num_gpus;
    float calculateInformationRatio(const std::vector<float>& correlations);
};

} // namespace cuda
} // namespace leadlag
