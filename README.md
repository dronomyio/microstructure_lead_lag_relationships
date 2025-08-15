Usage

Build and Run:
================================================

bash# Set your Polygon API key
export POLYGON_API_KEY="your_key_here"

# Build and run with Docker Compose
docker-compose up --build

====

Direct Compilation:

bashmkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
./lead_lag_analyzer --config ../config/config.json


===================================================
Access Dashboard:

Main visualization: http://localhost:8050
Dash dashboard: http://localhost:8051
================================================

Key Algorithms

Cross-Correlation: Computed at multiple lags to find information propagation delays
FFT-based Correlation: For large datasets using cuFFT
Information Ratio: Measures signal quality relative to noise
Sharpe Ratio: Risk-adjusted performance metric

=================================================

Data Flow

Downloads nanosecond quote data from Polygon.io
Aligns quotes across multiple exchanges
Distributes computation across GPUs
Finds optimal lag with maximum correlation
Outputs results to JSON and visualization dashboard

This system is production-ready for detecting cross-venue arbitrage opportunities by measuring information propagation at nanosecond scales between different trading venues.
==============================================

Key Features Implemented:
===============================================

Multi-GPU CUDA Processing:

Distributed correlation computation across GPUs
Texture memory for optimal cache performance
Constant memory for statistics
Warp shuffle reductions
cuBLAS and cuFFT integration

===============================================

CPU SIMD Optimizations:

AVX-512 and AVX2 implementations
Cache-aligned data structures
OpenMP parallelization
Sliding window correlations

===============================================

Nanosecond Precision:

std::chrono::nanoseconds for timestamps
Synchronized cross-exchange analysis
Time bucket alignment

===============================================

Polygon.io Integration:

Quote data download and parsing
Support for flat file formats
Pagination handling

===============================================

Real-time Visualization:

Dash dashboard for monitoring
Correlation heatmaps
Lead-lag distributions
Information ratio analysis

===============================================

Production Ready:

Docker containerization
Multi-GPU docker-compose setup
CMake build system
Configuration management



This system can process millions of quotes per second and detect lead-lag relationships at nanosecond scales across multiple exchanges, critical for high-frequency trading arbitrage strategies.
===============================================
