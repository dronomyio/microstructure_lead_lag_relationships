#include "lead_lag_kernel.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <stdexcept>

namespace leadlag {
namespace cuda {

__global__ void computeCorrelationKernel(
    const float* prices1,
    const float* prices2,
    float* correlations,
    int n_samples,
    int max_lag,
    int lag_step
) {
    int lag_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_lags = (2 * max_lag / lag_step) + 1;
    
    if (lag_idx >= num_lags) return;
    
    int lag = -max_lag + lag_idx * lag_step;
    int lag_samples = lag / 100;  // Convert nanoseconds to samples
    
    double sum_xy = 0.0;
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_x2 = 0.0;
    double sum_y2 = 0.0;
    int count = 0;
    
    for (int i = 0; i < n_samples; i++) {
        int j = i + lag_samples;
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
        
        if (denominator > 0) {
            double correlation = numerator / denominator;
            // Clamp to [-1, 1]
            correlations[lag_idx] = fmaxf(-1.0f, fminf(1.0f, (float)correlation));
        } else {
            correlations[lag_idx] = 0.0f;
        }
    } else {
        correlations[lag_idx] = 0.0f;
    }
}

MultiGPULeadLag::MultiGPULeadLag(int n_gpus) : num_gpus(n_gpus) {
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    
    if (err != cudaSuccess || device_count == 0) {
        throw std::runtime_error("No CUDA devices available");
    }
    
    num_gpus = std::min(num_gpus, device_count);
    std::cout << "CUDA: Using " << num_gpus << " out of " << device_count << " available GPUs" << std::endl;
}

MultiGPULeadLag::~MultiGPULeadLag() {
}

void MultiGPULeadLag::computeLeadLag(
    const std::vector<float>& prices1,
    const std::vector<float>& prices2,
    std::vector<LeadLagResult>& results,
    const Config& config
) {
    int n_samples = std::min(prices1.size(), prices2.size());
    int num_lags = (2 * config.max_lag_ns / config.lag_step_ns) + 1;
    
    std::cout << "CUDA: Processing " << n_samples << " samples with " << num_lags << " lags" << std::endl;
    
    float *d_prices1 = nullptr, *d_prices2 = nullptr, *d_correlations = nullptr;
    
    // Allocate device memory
    cudaMalloc(&d_prices1, n_samples * sizeof(float));
    cudaMalloc(&d_prices2, n_samples * sizeof(float));
    cudaMalloc(&d_correlations, num_lags * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_prices1, prices1.data(), n_samples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prices2, prices2.data(), n_samples * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threads_per_block = 256;
    int blocks = (num_lags + threads_per_block - 1) / threads_per_block;
    
    computeCorrelationKernel<<<blocks, threads_per_block>>>(
        d_prices1, d_prices2, d_correlations,
        n_samples, config.max_lag_ns, config.lag_step_ns
    );
    
    cudaDeviceSynchronize();
    
    // Copy results back
    std::vector<float> correlations(num_lags);
    cudaMemcpy(correlations.data(), d_correlations, num_lags * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Find maximum correlation
    auto max_it = std::max_element(correlations.begin(), correlations.end());
    int max_idx = std::distance(correlations.begin(), max_it);
    
    // Debug output
    std::cout << "CUDA: Max correlation = " << *max_it << " at lag index " << max_idx << std::endl;
    
    LeadLagResult result;
    result.correlation = *max_it;
    result.lag_nanoseconds = -config.max_lag_ns + max_idx * config.lag_step_ns;
    result.information_ratio = calculateInformationRatio(correlations);
    result.sharpe_ratio = result.information_ratio * sqrtf(252.0f);
    
    results.push_back(result);
    
    // Cleanup
    cudaFree(d_prices1);
    cudaFree(d_prices2);
    cudaFree(d_correlations);
    
    std::cout << "CUDA: Computation complete" << std::endl;
}

float MultiGPULeadLag::calculateInformationRatio(const std::vector<float>& correlations) {
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

} // namespace cuda
} // namespace leadlag
