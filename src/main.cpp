#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <memory>
#include <cmath>
#include <sys/stat.h>
#include <sys/types.h>
#include "include/types.hpp"
#include "src/cpu/simd_correlation.hpp"
#include "src/data/polygon_loader.hpp"
#ifdef __CUDACC__
#include "src/cuda/lead_lag_kernel.cuh"
#endif
#include "src/visualization/realtime_plot.hpp"

using namespace leadlag;

Config loadConfig(const std::string& config_path) {
    Config config;
    config.symbols = {"AAPL", "MSFT"};
    config.exchanges = {"XNGS", "XNYS"};
    config.window_size_ns = 1000000000;
    config.max_lag_ns = 10000;  // Reduced to 10 microseconds for better resolution
    config.lag_step_ns = 100;   // 100ns steps
    config.num_gpus = 1;
    config.use_simd = true;
    config.data_path = "./data";
    config.polygon_api_key = "";
    
    std::cout << "Using configuration with max lag: " << config.max_lag_ns << " ns" << std::endl;
    return config;
}

void ensureDirectoryExists(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        mkdir(path.c_str(), 0755);
        std::cout << "Created directory: " << path << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Lead-Lag Analysis System" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    Config config = loadConfig("./config/config.json");
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Max lag: ±" << config.max_lag_ns << " ns" << std::endl;
    std::cout << "  Lag step: " << config.lag_step_ns << " ns" << std::endl;
    std::cout << "  Number of lags to test: " << (2 * config.max_lag_ns / config.lag_step_ns + 1) << std::endl;
    
    // Ensure output directory exists (in parent directory)
    ensureDirectoryExists("../output");
    
    // Generate synthetic data with known lag
    std::vector<float> prices1, prices2;
    int known_lag_samples = 50;  // 50 samples lag
    int known_lag_ns = known_lag_samples * 100;  // 5000 ns (assuming 100ns per sample)
    
    std::cout << "\nGenerating synthetic data:" << std::endl;
    std::cout << "  Known lag: " << known_lag_samples << " samples = " << known_lag_ns << " ns" << std::endl;
    std::cout << "  Generating 10000 samples..." << std::endl;
    
    for (int i = 0; i < 10000; i++) {
        prices1.push_back(100.0 + 0.01 * sin(i * 0.01));
    }
    for (int i = 0; i < 10000; i++) {
        // Create lagged series
        int source_idx = i - known_lag_samples;
        if (source_idx >= 0 && source_idx < 10000) {
            prices2.push_back(prices1[source_idx]);
        } else {
            prices2.push_back(100.0);  // Default value for out of range
        }
    }
    
    std::cout << "  Series 1: " << prices1.size() << " samples" << std::endl;
    std::cout << "  Series 2: " << prices2.size() << " samples (lagged copy of Series 1)" << std::endl;
    
    std::vector<LeadLagResult> results;
    
    #ifdef __CUDACC__
    if (config.num_gpus > 0) {
        try {
            std::cout << "\n--- Running CUDA Analysis ---" << std::endl;
            auto cuda_analyzer = std::make_unique<cuda::MultiGPULeadLag>(config.num_gpus);
            cuda_analyzer->computeLeadLag(prices1, prices2, results, config);
        } catch (const std::exception& e) {
            std::cout << "CUDA error: " << e.what() << std::endl;
        }
    }
    #endif
    
    if (config.use_simd) {
        std::cout << "\n--- Running CPU SIMD Analysis ---" << std::endl;
        auto cpu_analyzer = std::make_unique<cpu::SIMDCorrelation>();
        cpu_analyzer->compute(prices1, prices2, results, config);
    }
    
    if (!results.empty()) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "RESULTS" << std::endl;
        std::cout << "========================================" << std::endl;
        
        int result_num = 1;
        for (const auto& result : results) {
            std::cout << "\nResult #" << result_num++ << ":" << std::endl;
            std::cout << "  Correlation coefficient: " << result.correlation << std::endl;
            std::cout << "  Detected lag: " << result.lag_nanoseconds << " ns" << std::endl;
            std::cout << "  Expected lag: " << known_lag_ns << " ns" << std::endl;
            std::cout << "  Error: " << abs(result.lag_nanoseconds - known_lag_ns) << " ns" << std::endl;
            
            if (result.lag_nanoseconds > 0) {
                std::cout << "  Interpretation: Series 1 LEADS Series 2 by " << result.lag_nanoseconds << " ns" << std::endl;
            } else if (result.lag_nanoseconds < 0) {
                std::cout << "  Interpretation: Series 2 LEADS Series 1 by " << -result.lag_nanoseconds << " ns" << std::endl;
            }
        }
        
        // Save results to parent output directory
        std::string output_file = "../output/lead_lag_results.txt";
        std::ofstream file(output_file);
        if (file.is_open()) {
            file << "Lead-Lag Analysis Results\n";
            file << "==========================\n\n";
            file << "Test Configuration:\n";
            file << "  Known lag: " << known_lag_ns << " ns\n";
            file << "  Max search range: ±" << config.max_lag_ns << " ns\n";
            file << "  Step size: " << config.lag_step_ns << " ns\n\n";
            
            result_num = 1;
            for (const auto& result : results) {
                file << "Result #" << result_num++ << ":\n";
                file << "  Correlation: " << result.correlation << "\n";
                file << "  Detected lag: " << result.lag_nanoseconds << " ns\n";
                file << "  Error from known lag: " << abs(result.lag_nanoseconds - known_lag_ns) << " ns\n\n";
            }
            file.close();
            std::cout << "\n✓ Results saved to: " << output_file << std::endl;
        } else {
            std::cout << "\n✗ Could not save results to: " << output_file << std::endl;
        }
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Analysis Complete" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    return 0;
}
