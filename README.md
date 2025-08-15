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
Multi-GPU compose setup
CMake build system
Configuration management



This system can process millions of quotes per second and detect lead-lag relationships at nanosecond scales across multiple exchanges, critical for high-frequency trading arbitrage strategies.
===============================================


## ===============INTUITION===============================

# Graphical Intuition of Lead-Lag Correlation Analysis
## 1. What We're Actually Measuring - Lead-Lag Relationship
PRICE SERIES OVER TIME:
════════════════════════

No Lag (Correlation = 0.3):
Exchange A: ──╱╲──╱╲────╱╲──
Exchange B: ────╱╲──╱╲──╱───
            Movements don't align well

Perfect Lag of 500ns (Correlation = 0.99):
Exchange A: ──╱╲──╱╲────╱╲──╱╲──
Exchange B:    ──╱╲──╱╲────╱╲──╱╲
            ↑────────────────────↑
            B follows A perfectly with 500ns delay!
## 2. Correlation Calculation - Visual Intuition

PEARSON CORRELATION FORMULA VISUALLY:
══════════════════════════════════════

Step 1: Center the data (remove means)
    
    Price A                         Centered A
    104 ●                           +2 ●
    103 ●       ●                   +1 ●   ●
    102 ●   ●       ●                0 ●       ●
    101     ●   ●   ●   → Mean=102  -1     ●   ●
    100         ●                   -2         ●
        t1  t2  t3  t4  t5              t1  t2  t3  t4  t5

Step 2: Multiply corresponding points
    
    Centered A × Centered B = Products
    (+2) × (+1.8) = +3.6  ← Same direction, positive product
    (+1) × (+0.9) = +0.9  ← Same direction, positive product
    (-1) × (-0.8) = +0.8  ← Same direction, positive product
    (-2) × (+0.5) = -1.0  ← Opposite direction, negative product
    
    Sum of products → Positive = Positive correlation!

Step 3: Normalize by standard deviations
    
    Correlation = Sum(products) / (StdDev_A × StdDev_B × n)
                = Covariance / (StdDev_A × StdDev_B)

## 3. Testing Multiple Lags - Finding the Peak
CORRELATION AT DIFFERENT LAGS:
═══════════════════════════════

We slide Series B relative to Series A and compute correlation:

Lag = -1000ns (B leads A by 1000ns):
    A: ████████████████
    B: ██████████████████
       Poor alignment → Correlation = 0.2

Lag = -500ns:
    A: ████████████████
    B:   ████████████████
       Better alignment → Correlation = 0.5

Lag = 0ns (No lag):
    A: ████████████████
    B: ████████████████
       Good alignment → Correlation = 0.7

Lag = +500ns (A leads B by 500ns):
    A: ████████████████
    B:     ████████████████
       BEST alignment → Correlation = 0.99 ← PEAK!

Lag = +1000ns:
    A: ████████████████
    B:       ████████████████
       Alignment degrading → Correlation = 0.6

CORRELATION vs LAG GRAPH:
    1.0 │      ╱╲
        │     ╱  ╲ ← Peak at +500ns
    0.8 │    ╱    ╲   (A leads B)
Corr    │   ╱      ╲
    0.6 │  ╱        ╲
        │ ╱          ╲
    0.4 │╱            ╲
        │              ╲
    0.2 │_______________╲_____
        -1000  -500  0  +500  +1000
                   Lag (ns)

  ## 4. The SIMD Parallel Computation

  SIMD PROCESSES 8 CORRELATIONS AT ONCE:
════════════════════════════════════════

Traditional (Sequential):
Lag₁: ●────────────────> Corr₁
Lag₂:   ●────────────────> Corr₂
Lag₃:     ●────────────────> Corr₃
...
Time: ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓

AVX2 SIMD (8 Parallel):
┌─────────────────────┐
│ Lag₁ Lag₂ ... Lag₈  │ ──> [Corr₁|Corr₂|...|Corr₈]
└─────────────────────┘
All 8 computed simultaneously!
Time: ▓▓▓ (8x faster!)

With 48 Cores + OpenMP:
Core 0:  [Lag₁-₈]   ──> [Corr₁-₈]
Core 1:  [Lag₉-₁₆]  ──> [Corr₉-₁₆]
...
Core 47: [Lag₃₇₇-₃₈₄] ──> [Corr₃₇₇-₃₈₄]
Time: ▓ (384x faster!)

## 5. Information Ratio - Signal Quality Visualization
INFORMATION RATIO INTUITION:
════════════════════════════

High IR (Good Signal):          Low IR (Noisy Signal):
Correlations across lags:       Correlations across lags:
    │  ●●●                          │    ●  ●
0.9 │ ●   ●                      0.9│  ●  ●●  ●
    │●     ●● Consistent!           │ ● ●●  ●● ● Scattered!
0.8 │        ●                   0.8│●  ●  ●  ● ●
    │                               │  ●  ●  ●●
0.7 │                            0.7│ ●  ●●  ●  ●
    └────────────                   └────────────
    Mean: 0.85                      Mean: 0.80
    StdDev: 0.05 (tight!)          StdDev: 0.15 (spread!)
    IR = 0.85/0.05 = 17.0          IR = 0.80/0.15 = 5.3
    
    RELIABLE for trading!          RISKY for trading!

## 6. The Complete Pipeline Visualization

FULL LEAD-LAG DETECTION PIPELINE:
══════════════════════════════════

1. INPUT: Raw Price Streams
   NASDAQ: ═══╱╲═══╱╲═══╱═══╲═══
   NYSE:   ═══╱╲═══╱╲═══╱═══╲═══
           Time →

2. SIMD CORRELATION COMPUTATION
   ┌─────────────────────────────┐
   │  For each lag (-1μs to +1μs)│
   │  ┌───────────────────┐      │
   │  │ Load 8 prices (A) │      │
   │  │ Load 8 prices (B) │      │
   │  │ vsum_xy += A × B  │      │
   │  │ vsum_x += A       │      │
   │  │ vsum_y += B       │      │
   │  │ vsum_x2 += A × A  │      │
   │  │ vsum_y2 += B × B  │      │
   │  └───────────────────┘      │
   │  Pearson Formula → Corr[lag]│
   └─────────────────────────────┘

3. CORRELATION CURVE
        1.0┤     ●●●●●
           │   ●●     ●●  ← Peak = 0.99
        0.8│  ●         ●    at +500ns
           │ ●           ●
        0.6│●             ●
           └────────────────
           -1μs    0    +1μs

4. FIND MAXIMUM
   max_correlation = 0.99
   optimal_lag = +500ns
   
5. CALCULATE METRICS
   Information Ratio = mean/stddev = 8.5
   Sharpe Ratio = IR × √252 = 135
   
6. TRADING SIGNAL
    NASDAQ leads NYSE by 500ns
    Signal quality: EXCELLENT (IR > 2)
    Action: Buy NYSE when NASDAQ rises
## 7. Why This Matters for Trading

ARBITRAGE OPPORTUNITY WINDOW:
══════════════════════════════

Price Movement Detection:
                                    
NASDAQ  │     ╱╲    Your algo detects
price   │    ╱  ╲   rise at T+0
        │___╱____╲_______________
        
NYSE    │            ╱╲   NYSE updates
price   │           ╱  ╲  at T+500ns
        │__________╱____╲________
                   ↑      ↑
                   │      │
                   │      Too late!
                   │
                500ns PROFIT WINDOW
                   │
         You have 500 nanoseconds to:
         1. Detect NASDAQ price rise
         2. Send buy order to NYSE
         3. Execute before NYSE updates
         
         Profit = NYSE_new - NYSE_old
                = $100.02 - $100.00
                = $0.02 per share
                × 10,000 shares
                = $200 per trade

## 8. Visual Summary - The Edge
WITHOUT LEAD-LAG DETECTION:
═══════════════════════════
Trader sees prices:
NASDAQ: $100.00 → $100.02
NYSE:   $100.00 → $100.02
Action: None (no opportunity visible)
Profit: $0

WITH LEAD-LAG DETECTION:
════════════════════════
System detects 500ns lead:
T+0ns:   NASDAQ: $100.00 → $100.02 ← Detected!
T+50ns:  Send order to NYSE at $100.00
T+200ns: Order executes at NYSE $100.00
T+500ns: NYSE updates to $100.02
T+600ns: Sell at $100.02
Profit: $0.02 × volume

The 500ns edge = Millions in profit over thousands of trades!
