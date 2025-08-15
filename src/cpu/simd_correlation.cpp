// src/cpu/simd_correlation.cpp - PROPER SIMD IMPLEMENTATION
#include "simd_correlation.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cstring>
#ifdef __AVX512F__
#include <immintrin.h>
#endif
#ifdef __AVX2__
#include <immintrin.h>
#endif
#ifdef __SSE4_1__
#include <smmintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace leadlag {
namespace cpu {

// Aligned memory allocation for SIMD
template<typename T>
T* aligned_alloc(size_t n, size_t alignment = 64) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, n * sizeof(T)) != 0) {
        throw std::bad_alloc();
    }
    return static_cast<T*>(ptr);
}

void SIMDCorrelation::compute(
    const std::vector<float>& prices1,
    const std::vector<float>& prices2,
    std::vector<LeadLagResult>& results,
    const Config& config
) {
    int n_samples = std::min(prices1.size(), prices2.size());
    int num_lags = (2 * config.max_lag_ns / config.lag_step_ns) + 1;
    std::vector<float> correlations(num_lags);
    
    std::cout << "CPU SIMD: Testing " << num_lags << " different lags from -" 
              << config.max_lag_ns << " to +" << config.max_lag_ns << " ns" << std::endl;
    
    // Detect SIMD capability
    #ifdef __AVX512F__
    std::cout << "Using AVX-512 SIMD instructions (16 floats per operation)" << std::endl;
    #elif defined(__AVX2__)
    std::cout << "Using AVX2 SIMD instructions (8 floats per operation)" << std::endl;
    #elif defined(__SSE4_1__)
    std::cout << "Using SSE4.1 SIMD instructions (4 floats per operation)" << std::endl;
    #else
    std::cout << "Warning: No SIMD support detected, using scalar fallback" << std::endl;
    #endif
    
    #pragma omp parallel for schedule(dynamic)
    for (int lag_idx = 0; lag_idx < num_lags; lag_idx++) {
        int lag_ns = -config.max_lag_ns + lag_idx * config.lag_step_ns;
        int lag_samples = lag_ns / 100;
        
        float correlation = computeCorrelationAtLagSIMD(prices1.data(), prices2.data(), 
                                                        n_samples, lag_samples);
        correlations[lag_idx] = correlation;
    }
    
    auto max_it = std::max_element(correlations.begin(), correlations.end());
    int max_idx = std::distance(correlations.begin(), max_it);
    int optimal_lag_ns = -config.max_lag_ns + max_idx * config.lag_step_ns;
    
    std::cout << "CPU SIMD: Maximum correlation = " << *max_it 
              << " at lag = " << optimal_lag_ns << " ns" << std::endl;
    
    LeadLagResult result;
    result.correlation = *max_it;
    result.lag_nanoseconds = optimal_lag_ns;
    result.information_ratio = calculateInformationRatio(correlations);
    result.sharpe_ratio = result.information_ratio * sqrtf(252.0f);
    
    results.push_back(result);
}

float SIMDCorrelation::computeCorrelationAtLagSIMD(
    const float* prices1,
    const float* prices2,
    int n_samples,
    int lag_samples
) {
#ifdef __AVX512F__
    return computeCorrelationAVX512(prices1, prices2, n_samples, lag_samples);
#elif defined(__AVX2__)
    return computeCorrelationAVX2(prices1, prices2, n_samples, lag_samples);
#elif defined(__SSE4_1__)
    return computeCorrelationSSE(prices1, prices2, n_samples, lag_samples);
#else
    return computeCorrelationScalar(prices1, prices2, n_samples, lag_samples);
#endif
}

#ifdef __AVX512F__
float SIMDCorrelation::computeCorrelationAVX512(
    const float* prices1,
    const float* prices2,
    int n_samples,
    int lag_samples
) {
    __m512 vsum_xy = _mm512_setzero_ps();
    __m512 vsum_x = _mm512_setzero_ps();
    __m512 vsum_y = _mm512_setzero_ps();
    __m512 vsum_x2 = _mm512_setzero_ps();
    __m512 vsum_y2 = _mm512_setzero_ps();
    
    int count = 0;
    int start_idx = std::max(0, -lag_samples);
    int end_idx = std::min(n_samples, n_samples - lag_samples);
    
    // Process 16 floats at a time with AVX-512
    int i;
    for (i = start_idx; i + 15 < end_idx; i += 16) {
        int j = i + lag_samples;
        
        // Load 16 floats at once
        __m512 vx = _mm512_loadu_ps(&prices1[i]);
        __m512 vy = _mm512_loadu_ps(&prices2[j]);
        
        // Accumulate sums using FMA (Fused Multiply-Add)
        vsum_xy = _mm512_fmadd_ps(vx, vy, vsum_xy);
        vsum_x = _mm512_add_ps(vsum_x, vx);
        vsum_y = _mm512_add_ps(vsum_y, vy);
        vsum_x2 = _mm512_fmadd_ps(vx, vx, vsum_x2);
        vsum_y2 = _mm512_fmadd_ps(vy, vy, vsum_y2);
        
        count += 16;
    }
    
    // Horizontal reduction for AVX-512
    float sum_xy = _mm512_reduce_add_ps(vsum_xy);
    float sum_x = _mm512_reduce_add_ps(vsum_x);
    float sum_y = _mm512_reduce_add_ps(vsum_y);
    float sum_x2 = _mm512_reduce_add_ps(vsum_x2);
    float sum_y2 = _mm512_reduce_add_ps(vsum_y2);
    
    // Handle remaining elements
    for (; i < end_idx; i++) {
        int j = i + lag_samples;
        float x = prices1[i];
        float y = prices2[j];
        sum_xy += x * y;
        sum_x += x;
        sum_y += y;
        sum_x2 += x * x;
        sum_y2 += y * y;
        count++;
    }
    
    return computeCorrelationFromSums(sum_xy, sum_x, sum_y, sum_x2, sum_y2, count);
}
#endif

#ifdef __AVX2__
float SIMDCorrelation::computeCorrelationAVX2(
    const float* prices1,
    const float* prices2,
    int n_samples,
    int lag_samples
) {
    __m256 vsum_xy = _mm256_setzero_ps();
    __m256 vsum_x = _mm256_setzero_ps();
    __m256 vsum_y = _mm256_setzero_ps();
    __m256 vsum_x2 = _mm256_setzero_ps();
    __m256 vsum_y2 = _mm256_setzero_ps();
    
    int count = 0;
    int start_idx = std::max(0, -lag_samples);
    int end_idx = std::min(n_samples, n_samples - lag_samples);
    
    // Process 8 floats at a time with AVX2
    int i;
    for (i = start_idx; i + 7 < end_idx; i += 8) {
        int j = i + lag_samples;
        
        // Load 8 floats at once
        __m256 vx = _mm256_loadu_ps(&prices1[i]);
        __m256 vy = _mm256_loadu_ps(&prices2[j]);
        
        // Use FMA instructions for better performance
        vsum_xy = _mm256_fmadd_ps(vx, vy, vsum_xy);
        vsum_x = _mm256_add_ps(vsum_x, vx);
        vsum_y = _mm256_add_ps(vsum_y, vy);
        vsum_x2 = _mm256_fmadd_ps(vx, vx, vsum_x2);
        vsum_y2 = _mm256_fmadd_ps(vy, vy, vsum_y2);
        
        count += 8;
    }
    
    // Horizontal reduction for AVX2
    // Extract high and low 128-bit lanes
    __m128 low_xy = _mm256_extractf128_ps(vsum_xy, 0);
    __m128 high_xy = _mm256_extractf128_ps(vsum_xy, 1);
    __m128 sum_xy_128 = _mm_add_ps(low_xy, high_xy);
    
    __m128 low_x = _mm256_extractf128_ps(vsum_x, 0);
    __m128 high_x = _mm256_extractf128_ps(vsum_x, 1);
    __m128 sum_x_128 = _mm_add_ps(low_x, high_x);
    
    __m128 low_y = _mm256_extractf128_ps(vsum_y, 0);
    __m128 high_y = _mm256_extractf128_ps(vsum_y, 1);
    __m128 sum_y_128 = _mm_add_ps(low_y, high_y);
    
    __m128 low_x2 = _mm256_extractf128_ps(vsum_x2, 0);
    __m128 high_x2 = _mm256_extractf128_ps(vsum_x2, 1);
    __m128 sum_x2_128 = _mm_add_ps(low_x2, high_x2);
    
    __m128 low_y2 = _mm256_extractf128_ps(vsum_y2, 0);
    __m128 high_y2 = _mm256_extractf128_ps(vsum_y2, 1);
    __m128 sum_y2_128 = _mm_add_ps(low_y2, high_y2);
    
    // Horizontal sum of 128-bit vectors
    sum_xy_128 = _mm_hadd_ps(sum_xy_128, sum_xy_128);
    sum_xy_128 = _mm_hadd_ps(sum_xy_128, sum_xy_128);
    float sum_xy = _mm_cvtss_f32(sum_xy_128);
    
    sum_x_128 = _mm_hadd_ps(sum_x_128, sum_x_128);
    sum_x_128 = _mm_hadd_ps(sum_x_128, sum_x_128);
    float sum_x = _mm_cvtss_f32(sum_x_128);
    
    sum_y_128 = _mm_hadd_ps(sum_y_128, sum_y_128);
    sum_y_128 = _mm_hadd_ps(sum_y_128, sum_y_128);
    float sum_y = _mm_cvtss_f32(sum_y_128);
    
    sum_x2_128 = _mm_hadd_ps(sum_x2_128, sum_x2_128);
    sum_x2_128 = _mm_hadd_ps(sum_x2_128, sum_x2_128);
    float sum_x2 = _mm_cvtss_f32(sum_x2_128);
    
    sum_y2_128 = _mm_hadd_ps(sum_y2_128, sum_y2_128);
    sum_y2_128 = _mm_hadd_ps(sum_y2_128, sum_y2_128);
    float sum_y2 = _mm_cvtss_f32(sum_y2_128);
    
    // Handle remaining elements
    for (; i < end_idx; i++) {
        int j = i + lag_samples;
        float x = prices1[i];
        float y = prices2[j];
        sum_xy += x * y;
        sum_x += x;
        sum_y += y;
        sum_x2 += x * x;
        sum_y2 += y * y;
        count++;
    }
    
    return computeCorrelationFromSums(sum_xy, sum_x, sum_y, sum_x2, sum_y2, count);
}
#endif

#ifdef __SSE4_1__
float SIMDCorrelation::computeCorrelationSSE(
    const float* prices1,
    const float* prices2,
    int n_samples,
    int lag_samples
) {
    __m128 vsum_xy = _mm_setzero_ps();
    __m128 vsum_x = _mm_setzero_ps();
    __m128 vsum_y = _mm_setzero_ps();
    __m128 vsum_x2 = _mm_setzero_ps();
    __m128 vsum_y2 = _mm_setzero_ps();
    
    int count = 0;
    int start_idx = std::max(0, -lag_samples);
    int end_idx = std::min(n_samples, n_samples - lag_samples);
    
    // Process 4 floats at a time with SSE
    int i;
    for (i = start_idx; i + 3 < end_idx; i += 4) {
        int j = i + lag_samples;
        
        __m128 vx = _mm_loadu_ps(&prices1[i]);
        __m128 vy = _mm_loadu_ps(&prices2[j]);
        
        vsum_xy = _mm_add_ps(vsum_xy, _mm_mul_ps(vx, vy));
        vsum_x = _mm_add_ps(vsum_x, vx);
        vsum_y = _mm_add_ps(vsum_y, vy);
        vsum_x2 = _mm_add_ps(vsum_x2, _mm_mul_ps(vx, vx));
        vsum_y2 = _mm_add_ps(vsum_y2, _mm_mul_ps(vy, vy));
        
        count += 4;
    }
    
    // Horizontal sum for SSE
    vsum_xy = _mm_hadd_ps(vsum_xy, vsum_xy);
    vsum_xy = _mm_hadd_ps(vsum_xy, vsum_xy);
    float sum_xy = _mm_cvtss_f32(vsum_xy);
    
    vsum_x = _mm_hadd_ps(vsum_x, vsum_x);
    vsum_x = _mm_hadd_ps(vsum_x, vsum_x);
    float sum_x = _mm_cvtss_f32(vsum_x);
    
    vsum_y = _mm_hadd_ps(vsum_y, vsum_y);
    vsum_y = _mm_hadd_ps(vsum_y, vsum_y);
    float sum_y = _mm_cvtss_f32(vsum_y);
    
    vsum_x2 = _mm_hadd_ps(vsum_x2, vsum_x2);
    vsum_x2 = _mm_hadd_ps(vsum_x2, vsum_x2);
    float sum_x2 = _mm_cvtss_f32(vsum_x2);
    
    vsum_y2 = _mm_hadd_ps(vsum_y2, vsum_y2);
    vsum_y2 = _mm_hadd_ps(vsum_y2, vsum_y2);
    float sum_y2 = _mm_cvtss_f32(vsum_y2);
    
    // Handle remaining elements
    for (; i < end_idx; i++) {
        int j = i + lag_samples;
        float x = prices1[i];
        float y = prices2[j];
        sum_xy += x * y;
        sum_x += x;
        sum_y += y;
        sum_x2 += x * x;
        sum_y2 += y * y;
        count++;
    }
    
    return computeCorrelationFromSums(sum_xy, sum_x, sum_y, sum_x2, sum_y2, count);
}
#endif

float SIMDCorrelation::computeCorrelationScalar(
    const float* prices1,
    const float* prices2,
    int n_samples,
    int lag_samples
) {
    double sum_xy = 0.0;
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_x2 = 0.0;
    double sum_y2 = 0.0;
    int count = 0;
    
    int start_idx = std::max(0, -lag_samples);
    int end_idx = std::min(n_samples, n_samples - lag_samples);
    
    for (int i = start_idx; i < end_idx; i++) {
        int j = i + lag_samples;
        double x = prices1[i];
        double y = prices2[j];
        sum_xy += x * y;
        sum_x += x;
        sum_y += y;
        sum_x2 += x * x;
        sum_y2 += y * y;
        count++;
    }
    
    return computeCorrelationFromSums(sum_xy, sum_x, sum_y, sum_x2, sum_y2, count);
}

float SIMDCorrelation::computeCorrelationFromSums(
    float sum_xy, float sum_x, float sum_y, 
    float sum_x2, float sum_y2, int count
) {
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

float SIMDCorrelation::computeCorrelationAtLag(
    const float* prices1,
    const float* prices2,
    int n_samples,
    int lag_samples
) {
    // This is kept for compatibility but calls the SIMD version
    return computeCorrelationAtLagSIMD(prices1, prices2, n_samples, lag_samples);
}

float SIMDCorrelation::calculateInformationRatio(const std::vector<float>& correlations) {
    if (correlations.empty()) return 0.0f;
    
    // Use SIMD for mean calculation
    float mean = 0.0f;
    
#ifdef __AVX2__
    __m256 vsum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 7 < correlations.size(); i += 8) {
        __m256 v = _mm256_loadu_ps(&correlations[i]);
        vsum = _mm256_add_ps(vsum, v);
    }
    // Horizontal sum
    __m128 low = _mm256_extractf128_ps(vsum, 0);
    __m128 high = _mm256_extractf128_ps(vsum, 1);
    __m128 sum128 = _mm_add_ps(low, high);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    mean = _mm_cvtss_f32(sum128);
    
    // Handle remaining
    for (; i < correlations.size(); i++) {
        mean += correlations[i];
    }
#else
    mean = std::accumulate(correlations.begin(), correlations.end(), 0.0f);
#endif
    
    mean /= correlations.size();
    
    // Calculate variance with SIMD
    float variance = 0.0f;
    
#ifdef __AVX2__
    __m256 vmean = _mm256_set1_ps(mean);
    __m256 vvar = _mm256_setzero_ps();
    
    i = 0;
    for (; i + 7 < correlations.size(); i += 8) {
        __m256 v = _mm256_loadu_ps(&correlations[i]);
        __m256 diff = _mm256_sub_ps(v, vmean);
        vvar = _mm256_fmadd_ps(diff, diff, vvar);
    }
    
    // Horizontal sum
    low = _mm256_extractf128_ps(vvar, 0);
    high = _mm256_extractf128_ps(vvar, 1);
    sum128 = _mm_add_ps(low, high);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    variance = _mm_cvtss_f32(sum128);
    
    // Handle remaining
    for (; i < correlations.size(); i++) {
        float diff = correlations[i] - mean;
        variance += diff * diff;
    }
#else
    for (float c : correlations) {
        float diff = c - mean;
        variance += diff * diff;
    }
#endif
    
    variance /= correlations.size();
    float std_dev = sqrtf(variance);
    return std_dev > 0 ? mean / std_dev : 0.0f;
}

} // namespace cpu
} // namespace leadlag
