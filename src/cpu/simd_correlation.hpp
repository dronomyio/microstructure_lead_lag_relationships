// src/cpu/simd_correlation.hpp
#pragma once
#include <vector>
#include "../../include/types.hpp"

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
    // Main correlation computation dispatcher
    float computeCorrelationAtLagSIMD(
        const float* prices1,
        const float* prices2,
        int n_samples,
        int lag_samples
    );
    
    // Legacy scalar version (for compatibility)
    float computeCorrelationAtLag(
        const float* prices1,
        const float* prices2,
        int n_samples,
        int lag_samples
    );
    
    // SIMD-specific implementations
    #ifdef __AVX512F__
    float computeCorrelationAVX512(
        const float* prices1,
        const float* prices2,
        int n_samples,
        int lag_samples
    );
    #endif
    
    #ifdef __AVX2__
    float computeCorrelationAVX2(
        const float* prices1,
        const float* prices2,
        int n_samples,
        int lag_samples
    );
    #endif
    
    #ifdef __SSE4_1__
    float computeCorrelationSSE(
        const float* prices1,
        const float* prices2,
        int n_samples,
        int lag_samples
    );
    #endif
    
    // Scalar fallback
    float computeCorrelationScalar(
        const float* prices1,
        const float* prices2,
        int n_samples,
        int lag_samples
    );
    
    // Helper to compute correlation from accumulated sums
    float computeCorrelationFromSums(
        float sum_xy, float sum_x, float sum_y, 
        float sum_x2, float sum_y2, int count
    );
    
    // Information ratio calculation (also SIMD-optimized)
    float calculateInformationRatio(const std::vector<float>& correlations);
};

} // namespace cpu
} // namespace leadlag
