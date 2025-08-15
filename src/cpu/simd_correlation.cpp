#include "simd_correlation.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#ifdef __AVX2__
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace leadlag {
namespace cpu {

void SIMDCorrelation::compute(
    const std::vector<float>& prices1,
    const std::vector<float>& prices2,
    std::vector<LeadLagResult>& results,
    const Config& config
) {
    int n_samples = std::min(prices1.size(), prices2.size());
    int num_lags = (2 * config.max_lag_ns / config.lag_step_ns) + 1;
    std::vector<float> correlations(num_lags);
    
    std::cout << "CPU: Testing " << num_lags << " different lags from -" 
              << config.max_lag_ns << " to +" << config.max_lag_ns << " ns" << std::endl;
    
    #pragma omp parallel for
    for (int lag_idx = 0; lag_idx < num_lags; lag_idx++) {
        int lag_ns = -config.max_lag_ns + lag_idx * config.lag_step_ns;
        int lag_samples = lag_ns / 100;  // Convert ns to samples (100ns per sample)
        
        float correlation = computeCorrelationAtLag(prices1.data(), prices2.data(), 
                                                   n_samples, lag_samples);
        correlations[lag_idx] = correlation;
    }
    
    // Find maximum correlation
    auto max_it = std::max_element(correlations.begin(), correlations.end());
    int max_idx = std::distance(correlations.begin(), max_it);
    int optimal_lag_ns = -config.max_lag_ns + max_idx * config.lag_step_ns;
    
    std::cout << "CPU: Maximum correlation = " << *max_it 
              << " at lag = " << optimal_lag_ns << " ns" << std::endl;
    
    LeadLagResult result;
    result.correlation = *max_it;
    result.lag_nanoseconds = optimal_lag_ns;
    result.information_ratio = calculateInformationRatio(correlations);
    result.sharpe_ratio = result.information_ratio * sqrtf(252.0f);
    
    results.push_back(result);
}

float SIMDCorrelation::computeCorrelationAtLag(
    const float* prices1,
    const float* prices2,
    int n_samples,
    int lag_samples  // lag in samples, not nanoseconds
) {
    double sum_xy = 0.0;
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_x2 = 0.0;
    double sum_y2 = 0.0;
    int count = 0;
    
    // Calculate correlation with proper lag
    for (int i = 0; i < n_samples; i++) {
        int j = i - lag_samples;  // If lag is positive, series1 leads series2
        if (j >= 0 && j < n_samples) {
            double x = prices1[i];
            double y = prices2[j];
            sum_xy += x * y;
            sum_x += x;
            sum_y += y;
            sum_x2 += x * x;
            sum_y2 += y * y;
            count++;
        }
    }
    
    if (count > 1) {
        double n = count;
        double numerator = n * sum_xy - sum_x * sum_y;
        double denominator = sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));
        
        if (denominator > 1e-10) {
            double correlation = numerator / denominator;
            return std::max(-1.0, std::min(1.0, correlation));
        }
    }
    
    return 0.0f;
}

float SIMDCorrelation::calculateInformationRatio(const std::vector<float>& correlations) {
    if (correlations.empty()) return 0.0f;
    
    float mean = std::accumulate(correlations.begin(), correlations.end(), 0.0f) / correlations.size();
    float variance = 0.0f;
    for (float c : correlations) {
        float diff = c - mean;
        variance += diff * diff;
    }
    variance /= correlations.size();
    float std_dev = sqrtf(variance);
    return std_dev > 0 ? mean / std_dev : 0.0f;
}

} // namespace cpu
} // namespace leadlag
