Usage

Build and Run:
================================================

bash# Set your Polygon API key
export POLYGON_API_KEY="your_key_here"

# Build and run with Docker Compose
docker-compose up --build

====

Direct Compilation:
```
bash$ mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
./lead_lag_analyzer --config ../config/config.json
```

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

In modern markets, the same asset trades on multiple exchanges simultaneously (e.g., AAPL on NASDAQ, NYSE, BATS). Due to technological differences, geographic distances, and market microstructure, price discovery often happens first on one exchange and then propagates to others with a delay measured in microseconds or nanoseconds.

### The Arbitrage Opportunity
If you can detect that Exchange A consistently leads Exchange B by 500 nanoseconds, you can:

1. See a price move on Exchange A
2. Immediately trade on Exchange B before its price adjusts
3. Capture the spread as risk-free profit

### 1. Cross-Correlation at Multiple Lags
Intuition
```
Cross-correlation measures how similar two signals are when one is shifted in time relative to the other.
Exchange A: --[Price Move]------------------------->
Exchange B: -----------[Same Price Move]------------>
              <---lag--->
```
### Mathematical Insight
```
# For each possible lag τ:
correlation(τ) = Σ(price_A[t] * price_B[t + τ]) / normalization

# Peak correlation occurs at τ = true_lag
```
### Why Multiple Lags?

We don't know the delay a priori
Test lags from -1ms to +1ms in 100ns increments
The lag with maximum correlation reveals the information propagation delay
Positive lag: A leads B
Negative lag: B leads A
```
Real-World Example
Lag (ns)    Correlation
-1000       0.12
-500        0.31
0           0.67
+500        0.94  <-- Peak! Exchange A leads by 500ns
+1000       0.45
```
### 2. FFT-Based Correlation
Intuition
Computing correlation at thousands of lags is computationally expensive (O(n²)). FFT transforms this into frequency domain where correlation becomes multiplication (O(n log n)).
#### The Trick
Time Domain:                     Frequency Domain:
correlation = convolution   -->  multiplication
O(n²) operations            -->  O(n log n) operations

#### Process

1. FFT both price series → frequency domain
2. Multiply (conjugate of one with the other)
3. Inverse FFT → all correlations at once

#### Practical Impact

For 1 million quotes with 10,000 lag tests:

- Direct method: 10 billion operations
- FFT method: ~20 million operations
- 500x speedup enables real-time analysis

## 3. Information Ratio (IR)
#### Intuition
Not all lead-lag relationships are tradeable. IR measures the quality and consistency of the signal.
IR = (Mean Return of Strategy) / (Standard Deviation of Returns)
   = Signal Strength / Noise Level
### Why It Matters
High correlation doesn't guarantee profitability if:

The signal is inconsistent (high variance)
Transaction costs exceed the spread
The relationship breaks down frequently

### Interpretation
```
- IR < 0.5: Too noisy, likely unprofitable after costs
- IR 0.5-1.0: Potentially profitable with good execution
- IR > 1.0: Strong, consistent signal worth trading
- IR > 2.0: Exceptional opportunity (rare)
```
### Example Scenario
python
#### Two detected relationships:
```
Relationship A: correlation=0.8, occurs 90% of time → IR=1.8 ✓
Relationship B: correlation=0.9, occurs 20% of time → IR=0.4 ✗
```
## 4. Sharpe Ratio
#### Intuition
Sharpe extends IR by considering the risk-free rate and annualizing returns. It answers: "Is this strategy worth the capital allocation?"
```
Sharpe = (Return - Risk_Free_Rate) / Volatility
       = Risk-Adjusted Return (annualized)
Key Differences from IR

Annualized: Scales with √252 (trading days)
Risk-free benchmark: Compares to T-bills (~5% currently)
Capital efficiency: Helps size positions
```
#### Trading Decisions
```
Sharpe < 0.5: Not worth the capital
Sharpe 0.5-1.0: Acceptable for diversification
Sharpe 1.0-2.0: Good strategy, allocate capital
Sharpe > 2.0: Excellent, maximize position (within risk limits)
```
### Putting It All Together: A Real Example
```
Imagine detecting this pattern:
python
# AAPL quotes arriving at different exchanges
NASDAQ (14:30:00.000000000): Bid $150.00, Ask $150.01

IEX    (14:30:00.000000750): Bid $149.99, Ask $150.00  # 750ns delay

NYSE   (14:30:00.000001200): Bid $149.99, Ask $150.00  # 1200ns delay
```
### Analysis results:
```
NASDAQ→IEX:  Lag=750ns,  Correlation=0.92, IR=1.4, Sharpe=1.8
NASDAQ→NYSE: Lag=1200ns, Correlation=0.88, IR=0.9, Sharpe=1.1
```
#### Trading Strategy
```
Monitor NASDAQ (the leader)
When NASDAQ bid rises to $150.01:
Immediately buy on IEX at $150.00 (750ns window)
Sell when IEX catches up to $150.01
Profit: $0.01 per share, minus costs
```


# Graphical Intuition of Lead-Lag Correlation Analysis
## 1. What We're Actually Measuring - Lead-Lag Relationship
PRICE SERIES OVER TIME:
════════════════════════
```
No Lag (Correlation = 0.3):
Exchange A: ──╱╲──╱╲────╱╲──
Exchange B: ────╱╲──╱╲──╱───
            Movements don't align well

Perfect Lag of 500ns (Correlation = 0.99):
Exchange A: ──╱╲──╱╲────╱╲──╱╲──
Exchange B:    ──╱╲──╱╲────╱╲──╱╲
            ↑────────────────────↑
            B follows A perfectly with 500ns delay!
```
## 2. Correlation Calculation - Visual Intuition

PEARSON CORRELATION FORMULA VISUALLY:
══════════════════════════════════════
```
Step 1: Center the data (remove means)
    
    Price A                         Centered A
    104 ●                           +2 ●
    103 ●       ●                   +1 ●   ●
    102 ●   ●       ●                0 ●       ●
    101     ●   ●   ●   → Mean=102  -1     ●   ●
    100         ●                   -2         ●
        t1  t2  t3  t4  t5              t1  t2  t3  t4  t5
```

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
```
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
```

  ## 4. The SIMD Parallel Computation

  SIMD PROCESSES 8 CORRELATIONS AT ONCE:
════════════════════════════════════════
```
Traditional (Sequential):

Lag₁: ●────────────────> Corr₁


Lag₂:   ●────────────────> Corr₂


Lag₃:     ●────────────────> Corr₃


```
```

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




Core 47: [Lag₃₇₇-₃₈₄] ──> [Corr₃₇₇-₃₈₄]


Time: ▓ (384x faster!)
```

## 5. Information Ratio - Signal Quality Visualization


INFORMATION RATIO INTUITION:


════════════════════════════

```

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
```
## 6. The Complete Pipeline Visualization

FULL LEAD-LAG DETECTION PIPELINE:
══════════════════════════════════
```
1. INPUT: Raw Price Streams
   
   NASDAQ: ═══╱╲═══╱╲═══╱═══╲═══
   
   NYSE:   ═══╱╲═══╱╲═══╱═══╲═══
           Time →

3. SIMD CORRELATION COMPUTATION
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

4. CORRELATION CURVE
        1.0┤     ●●●●●
           │   ●●     ●●  ← Peak = 0.99
        0.8│  ●         ●    at +500ns
           │ ●           ●
        0.6│●             ●
           └────────────────
           -1μs    0    +1μs
```
5. FIND MAXIMUM
   max_correlation = 0.99
   optimal_lag = +500ns
   
6. CALCULATE METRICS
   Information Ratio = mean/stddev = 8.5
   Sharpe Ratio = IR × √252 = 135
   
7. TRADING SIGNAL
    NASDAQ leads NYSE by 500ns
    Signal quality: EXCELLENT (IR > 2)
    Action: Buy NYSE when NASDAQ rises
## 7. Why This Matters for Trading

ARBITRAGE OPPORTUNITY WINDOW:
══════════════════════════════
```
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
```
## 8. Visual Summary - The Edge
```
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
```

# Theory

Got it—here’s a tight, publish-ready rewrite of your explainer first, then a deeper dive with concrete examples, math, and GPU/FFT implementation tips.

# Drop-in Rewrite (clean, concise)

**Lead–Lag Intuition (HFT context)**
When the same instrument trades on multiple venues, price discovery often happens on one first (“leader”) and then propagates to others (“laggards”) after a small delay (µs–ns). If you can identify a stable delay, you can read the move on the leader and act on the laggard before it updates.

**1) Cross-Correlation across lags**
Cross-correlation asks: *how similar is series B to a time-shifted version of series A?*
Scan lags $\tau$ over a window (e.g., $[-1\text{ ms}, +1\text{ ms}]$ at 100 ns steps). The lag $\tau^\*$ with the highest (normalized) correlation is your estimated propagation delay:

* $\tau^\*>0$: A leads B by $\tau^\*$.
* $\tau^\*<0$: B leads A by $|\tau^\*|$.

**2) FFT-based correlation (cuFFT)**
Brute-force correlation at many lags is $O(n^2)$. Using FFT:
correlation ≈ IFFT( FFT(A) · conj(FFT(B)) ) → $O(n\log n)$.
This gives *all lags at once* and is fast enough for million-tick windows on GPU.

**3) Information Ratio (IR)**
High correlation isn’t automatically tradable. IR measures *signal quality*:

$$
\text{IR} = \frac{\mathbb{E}[r]}{\sigma(r)}
$$

Use IR to tell persistent edges from noisy coincidences (and to see if an edge survives fees/slippage).

**4) Sharpe Ratio**
Sharpe adds a risk-free baseline and annualization:

$$
\text{Sharpe} = \frac{\mathbb{E}[r]-r_f}{\sigma(r)} \times \sqrt{\text{periods per year}}
$$

It answers: *is this worth capital vs. T-bills and other strategies?* Use Sharpe for allocation; use IR for signal vetting.

**Putting it together**

1. Find $\tau^\*$ via cross-corr (FFT).
2. Convert the lag edge into a *predictive trade rule* (e.g., buy B when A upticks and B hasn’t yet).
3. Measure realized returns → compute IR/Sharpe after costs.
4. Trade only if the microsecond window exceeds your total latency budget and the IR/Sharpe are robust out-of-sample.

---

# Deeper Dive (intuition, math, examples, GPU code patterns)

## 1) What “lead–lag” really captures

At microsecond scales: feed latencies, venue distance, matching-engine queues, and router paths create tiny but systematic delays. You’re not forecasting *direction* from scratch—you’re *relaying* discovered information from A to B before most of the world sees it on B.

**ASCII timeline**

```
A (leader):  ---- price jump ----->│
B (laggard): --------- same jump ------->│
                     <----- τ ----->
Trade on B within τ, before it updates.
```

## 2) Cross-correlation that actually works in production

**Prep matters more than the formula:**

* Align clocks (PTP/GPS), or use exchange sequence numbers if provided.
* Use a common sampling grid (e.g., 100 ns or 1 µs bars of mid-price deltas) or event-sync (last-tick-carry-forward).
* Detrend and standardize: $x_t \leftarrow (x_t - \bar{x})/\sigma$.
* Winsorize extreme deltas to tame outliers.

**Definition (normalized):**

$$
\rho_{AB}(\tau)=\frac{\sum_t x_A(t)\,x_B(t+\tau)}{\sqrt{\sum_t x_A(t)^2}\sqrt{\sum_t x_B(t)^2}}
$$

The peak $\tau^\*=\arg\max_\tau \rho_{AB}(\tau)$ is your estimated delay.

**Sign convention:** if the *best* correlation occurs at $+\;500\text{ ns}$, A leads B by 500 ns.

**Multiple lags:** scanning a fine grid is crucial; tiny steps (50–200 ns) often change the winner.

## 3) FFT-based correlation on GPU (cuFFT/CuPy pattern)

**Key points**

* Zero-pad to at least $n+m-1$ to avoid circular wrap.
* Use real-to-complex (R2C) FFTs for speed.
* Multiply $\mathcal{F}(A) \cdot \overline{\mathcal{F}(B)}$, then IFFT.
* Normalize by $\|A\|\|B\|$ to get correlation, not raw convolution.

**Pseudocode (CuPy, mirrors cuFFT logic)**

```python
import cupy as cp

def xcorr_normalized(A, B):
    n = len(A) + len(B) - 1
    N = 1 << (n-1).bit_length()  # next pow2
    A0 = (A - A.mean()) / (A.std() + 1e-12)
    B0 = (B - B.mean()) / (B.std() + 1e-12)
    FA = cp.fft.rfft(cp.pad(A0, (0, N-len(A0))))
    FB = cp.fft.rfft(cp.pad(B0, (0, N-len(B0))))
    corr = cp.fft.irfft(FA * cp.conj(FB), n=N)
    corr = corr[:n]  # valid part
    # shift so that lags run from -(len(B)-1) .. +(len(A)-1)
    corr = cp.concatenate([corr[-(len(B)-1):], corr[:len(A)]])
    # already normalized by std of A,B; optional scale to [-1,1] if needed
    return corr
```

**Lag grid mapping:** the index at maximum gives $\tau^\*$ via `lag_index * dt - (len(B)-1)*dt`.

**Complexity:** $O(n\log n)$ vs $O(n^2)$ brute force → orders-of-magnitude faster at 10⁶ points.

## 4) From correlation to *tradable* edge

Correlation peak says “A’s moves show up in B after $\tau^\*$.” To monetize, turn it into a *predictive regression*:

$$
\Delta p_B(t) = \beta\,\Delta p_A(t-\tau^\*) + \varepsilon_t
$$

If $\beta>0$, an uptick on A predicts an uptick on B after $\tau^\*$.

**Expected fill-to-fill edge (toy):**

$$
\mathbb{E}[\text{Edge}] \approx \beta\cdot \mathbb{E}[|\Delta p_A|] - \text{fees} - \text{slippage} - \text{adverse selection}
$$

Only trade when $\mathbb{E}[\text{Edge}] > 0$ by a safety margin.

**Latency budget (must be < lag):**

$$
L_{\text{total}} = L_{\text{feed}}+L_{\text{decode}}+L_{\text{compute}}+L_{\text{route}}+L_{\text{venue}}+\text{jitter}
$$

Require $L_{\text{total}} \ll \tau^\*$ (and remember your competitors are racing you).

**Concrete micro example**

* Found $\tau^\*=750\text{ ns}$, $\beta=0.65$.
* Median $|\Delta p_A|$ (1-tick) = \$0.01.
* Fees+slippage+AS ≈ \$0.006/share.
* Edge/share ≈ $0.65\times 0.01 - 0.006 = 0.0005$ (\$0.0005).
  Scale by fills/second and fill probability to estimate P\&L, then compute IR/Sharpe.

## 5) Information Ratio vs. Sharpe (and how to use them)

**IR (signal quality):**

$$
\text{IR}=\frac{\bar{r}}{\sigma_r}
$$

* Use on *strategy return series* (after costs) to judge persistence/cleanliness.
* Good quick filter: IR < 0.5 (usually not worth it); IR > 1 (promising); IR > 2 (rare/great).

**Sharpe (capital allocation):**

$$
\text{Sharpe}=\frac{\bar{r}-r_f}{\sigma_r}\times \sqrt{\text{periods/yr}}
$$

* Compare across strategies and vs. cash; size risk budgets with it.
* A strategy with modest IR but high capacity can beat a high-IR, low-capacity one.

**Mini example (per-trade returns, then annualized)**

* Mean per-trade $= 2.5$ bps, stdev $= 2.0$ bps → IR $= 1.25$.
* With 200k trades/day and weak correlation across trades, daily Sharpe can be estimated by aggregating to daily returns and annualizing (don’t annualize tick-level Sharpe directly).

## 6) Statistical hygiene (avoid chasing ghosts)

* **Multiple-testing control:** scanning thousands of lags/venues → apply Bonferroni/Holm or, better, *block bootstrap* to get a distribution of peak correlations.
* **Stability checks:** rolling windows by time-of-day; split by volatility regimes; holdout days.
* **Asynchronous trading:** if not on a common grid, Hayashi–Yoshida-style estimators help (Epps effect can depress correlation if you oversample).
* **Zero-padding + windowing:** avoid circular convolution; optional taper to reduce edge artifacts.
* **Clock sanity:** even 1–2 µs skew can flip “leader” labels.

## 7) Frequency-domain angle (phase → delay)

Coherence pinpoints *which frequencies* carry the lead–lag; the *phase slope* gives delay:

$$
\tau \approx -\frac{1}{2\pi}\frac{d\phi(f)}{df}
$$

If delay is frequency-dependent (microstructure!), consider a band-limited trade trigger (e.g., emphasize 1–10 kHz components that actually move quotes).

## 8) Practical pipeline you can ship

1. **Preprocess**: align, resample (or last-tick carry), standardize, winsorize.
2. **Scan lags**: FFT-xcorr on GPU; get $\tau^\*$, peak value, width.
3. **Validate**: bootstrapped p-values, rolling stability, venue-pair matrix.
4. **Translate to rule**: $\Delta p_A(t-\tau^\*) \to$ action on B; enforce *latency budget*.
5. **Simulate with costs**: queue position, partial fills, cancel/replace.
6. **Score**: IR/Sharpe on out-of-sample; capacity test; stress in news spikes.
7. **Deploy**: monitors for lag drift, auto-disable on regime breaks; daily re-estimation.

## 9) “Why not just trade the biggest correlation?”

Because the *biggest* peak might:

* Come from an illiquid venue (no fills),
* Vanish during volatility (edge decays just when you want it), or
* Be a clock artifact (mis-sync).
  IR/Sharpe on realistic fills (not midpoint fantasy) is the final arbiter.

---

## Quick reference thresholds (rule-of-thumb)

* **Peak xcorr** > 0.6 and stable across days/time-buckets → investigate.
* **Lag window** $>$ 3× your *95th-percentile* end-to-end latency → executable.
* **IR** > 1.0 out-of-sample (post-cost) → candidates.
* **Sharpe (annualized)** > 1.5 with real capacity → allocate.

If you want, I can turn this into a small **GPU notebook** (CuPy) that: (a) estimates $\tau^\*$, (b) builds the predictive rule, and (c) reports IR/Sharpe with a cost model using your own nanosecond A/B streams.

# Examples
Here you go—a tiny visual demo:

Lead–Lag Example (zoomed)

<img width="1697" height="1101" alt="output-3" src="https://github.com/user-attachments/assets/c3e9af91-318c-488f-b62a-628a94518a22" />

Cross-Correlation vs Lag (µs)
<img width="1718" height="1139" alt="output-4" src="https://github.com/user-attachments/assets/5add97eb-210a-4532-88ee-d8da3903e610" />

Toy Strategy: Cumulative P&L (no costs)
<img width="1693" height="1101" alt="output-5" src="https://github.com/user-attachments/assets/b2564833-d155-4daa-a3e6-7da51baa043b" />


* First chart: two series where **B lags A by \~700µs** (you can see B “copy” A a hair later).
* Second chart: **cross-correlation vs. lag**; the peak lands exactly at **+700µs**, confirming “A leads B.”
* Third chart: a toy **cumulative P\&L** from a simple rule (trade B on big moves in A, exit after the estimated lag). I also print the estimated lag, number of trades, and per-trade IR/Sharpe-like numbers right under the second plot.

If you want, I can tweak:

* the true lag,
* sampling period (e.g., 10µs),
* noise level (to show how IR collapses with microstructure noise), or
* swap in an FFT-based xcorr version that scales to 1M+ ticks.

# With **Polygon’s stock Quotes and Trades flat files**, you can see *exactly* where lead–lag is introduced because each record carries multiple clocks:

* **participant/exchange timestamp (`participant_timestamp`)** – when the event was born at the exchange’s matching engine/gateway. ([Polygon][1])
* **SIP timestamp (`sip_timestamp`)** – when the **SIP** (CTA for Tapes A/B, UTP for Tape C) received and disseminated that event. ([Polygon][1])
* **TRF timestamp (`trf_timestamp`)** – for off-exchange prints, when the FINRA Trade Reporting Facility received the trade. ([Polygon][2])
* Plus **exchange ID / tape / sequence\_number / condition codes** for filtering and ordering. ([Polygon][1])

Below is where the lag actually comes from in your data—and how to measure each piece.

---

## 1) Exchange → SIP latency (feed/aggregation delay)

**Where it’s generated:** network + processing time for an exchange’s message to reach the SIP and be consolidated.
**How you see it:**

$$
\Delta_{\text{ex→SIP}} = \texttt{sip\_timestamp} - \texttt{participant\_timestamp} \quad (\text{per message})
$$

Group by **exchange id** and **tape** to get distributions; CTA (A/B) is operated from **Mahwah, NJ** and UTP (C) from **Carteret, NJ**, so geography alone creates systematic microsecond differences. ([Nasdaq][3], [Traders Magazine][4])

**Use cases**

* Build histograms of $\Delta_{\text{ex→SIP}}$ by venue to quantify typical and tail latencies.
* Monitor drift: if an exchange’s $\Delta$ shifts, your execution windows change.

---

## 2) Inter-venue propagation (A leads B)

**Where it’s generated:** a price discovery event on one venue is echoed on others after routing/response delays (smart order routing, copy-cat liquidity, market makers repricing).
**How you see it:** work **entirely in participant time** to avoid SIP jitter:

1. From Quotes flat files, create per-exchange signals (e.g., mid-price delta, best bid change).
2. Cross-correlate Exchange A vs Exchange B over lags $\tau$. Peak at $\tau^\!>\!0$ ⇒ **A leads B by $\tau$**.
   This isolates *true* venue-to-venue delays using the **at-exchange clock** Polygon provides. (Participant vs SIP clocks and their role in TAQ are discussed in microstructure literature.) ([Polygon][1], [The Microstructure Exchange][5])

**Tip:** also check the **width** and **stability** of the correlation peak by time-of-day; open/close behave differently.

---

## 3) Trade → Quote reaction (price update after prints)

**Where it’s generated:** after a trade consumes liquidity, makers/venues update quotes.
**How you see it:**

* Pair **Trades** and **Quotes** by *venue* (same exchange id) and use **participant timestamps**:

$$
\tau_{\text{T→Q}} = t^{\text{quote}}_{\text{participant}} - t^{\text{trade}}_{\text{participant}}
$$

* Alternatively, relate **Trades (participant)** to **NBBO changes (SIP)** if you want the consolidated view.
  Filter on **regular trade/quote condition codes** to avoid specials. ([Polygon][6])

---

## 4) Off-exchange (TRF) path

**Where it’s generated:** dark/wholesale trades are reported to **FINRA TRF** first, then make their way into SIP views.
**How you see it:**

$$
\Delta_{\text{TRF path}} = \texttt{trf\_timestamp} - \texttt{participant\_timestamp}
$$

…and compare to the SIP time for when that print appears (or when NBBO responds). TRF has its **own** path and can exhibit different lags vs lit venues. ([Polygon][2])

---

## 5) SIP location asymmetry (Tape A/B vs C)

**Where it’s generated:** the two equity SIPs are in **different data centers** (CTA in **Mahwah**, UTP in **Carteret**). Messages may traverse Mahwah↔Carteret↔Secaucus, adding geographic micro-latency and asymmetries by tape—visible in your $\Delta_{\text{ex→SIP}}$ distributions. ([Nasdaq][3], [Federal Register][7])

---

## 6) “Not in your timestamps” (what you can ignore)

Polygon’s flat files don’t record “Polygon-received” times—only **participant/SIP/TRF**. Your **download/upload** steps, CSV decompression, etc., don’t affect those event times, so they won’t pollute your lead–lag estimates.

---

## Concrete “how-to” with your files

**A) Venue → SIP latency by exchange**

* Use **Quotes** and **Trades** CSVs for the day.
* Compute `sip_timestamp - participant_timestamp` per row.
* Aggregate by `exchange_id` and `tape` → median / p95 / p99 plots.
* Expect Tape-specific patterns (A/B vs C). ([Polygon][1], [Nasdaq][3])

**B) Venue-to-venue lead–lag**

* From **Quotes**, build per-exchange mid-price change series keyed by **participant time**.
* Cross-correlate A vs B over $\pm$ a few ms; peak $\Rightarrow$ leader & delay.

**C) Trade → NBBO response**

* From **Trades** (participant time) and **NBBO** (SIP; via WS/REST if you need NBBO), cross-correlate trade signs with NBBO mid changes. ([Polygon][8])

**D) Cleanliness**

* Sort by **sequence\_number** within ticker/venue as a tie-breaker.
* Drop/flag unusual **condition codes** (auction, odd-lot, derivatively priced). ([Polygon][6])

---

## Quick mapping: “what lag am I measuring?”

| Measurement you take                                  | Field(s) used                            | What it captures                                            |
| ----------------------------------------------------- | ---------------------------------------- | ----------------------------------------------------------- |
| `sip_timestamp − participant_timestamp`               | both                                     | **Exchange→SIP** transmission/processing delay              |
| X-corr of Exchange A vs Exchange B (participant time) | participant timestamps only              | **True venue-to-venue propagation** (information discovery) |
| Trade→Quote delay on same venue                       | both participant timestamps              | **Post-trade quote reaction** (inventory/queue updates)     |
| TRF trade reporting delay                             | `trf_timestamp`, `participant_timestamp` | **Off-exchange reporting path**                             |
| NBBO update timing vs venue                           | NBBO SIP vs venue participant            | **SIP consolidation latency** + **venue→SIP path**          |

---

### Why this matters for your strategy

* Use **participant time** to discover *economic* lead–lags (who truly leads whom).
* Use **SIP time** to understand *what a SIP-only consumer sees* (and to size latency-arb windows). SIP latency differences of \~1 ms are documented in the literature and vary by path/location. ([UC Berkeley Law][9])

If you want, I can whip up a small script that ingests one day of Polygon **Quotes** + **Trades**, produces (1) venue→SIP latency histograms and (2) a matrix of venue-to-venue lead–lags using participant time, then (3) overlays trade→quote reaction times—ready to swap in your tickers.

[1]: https://polygon.io/docs/rest/stocks/trades-quotes/quotes?utm_source=chatgpt.com "Quotes | Stocks REST API"
[2]: https://polygon.io/knowledge-base/article/which-timestamps-are-returned-for-polygons-stock-trades-and-nbbo-quotes?utm_source=chatgpt.com "Which timestamps are returned for Polygon's stock trades ..."
[3]: https://www.nasdaq.com/articles/time-is-relativity-what-physics-has-to-say-about-market-infrastructure-2020-04-09?utm_source=chatgpt.com "Time is Relativity: What Physics Has to Say About Market ..."
[4]: https://www.tradersmagazine.com/tech-tuesday/tech-tuesday-how-trades-speed-between-venues/?utm_source=chatgpt.com "TECH TUESDAY: How Trades Speed Between Venues"
[5]: https://microstructure.exchange/slides/TAQ_Participant_Timestamp_Sander_Schwenk-Nebbe.pdf?utm_source=chatgpt.com "The Participant Timestamp: Get The Most Out Of TAQ Data*"
[6]: https://polygon.io/docs/rest/stocks/market-operations/condition-codes?utm_source=chatgpt.com "Condition Codes | Stocks REST API"
[7]: https://www.federalregister.gov/documents/2020/01/14/2020-00360/notice-of-proposed-order-directing-the-exchanges-and-the-financial-industry-regulatory-authority-to?utm_source=chatgpt.com "Notice of Proposed Order Directing the Exchanges and ..."
[8]: https://polygon.io/docs/websocket/stocks/quotes?utm_source=chatgpt.com "Quotes | Stocks WebSocket"
[9]: https://www.law.berkeley.edu/wp-content/uploads/2019/10/bartlett_mccrary_latency2017.pdf?utm_source=chatgpt.com "[PDF] How Rigged Are Stock Markets? Evidence from Microsecond ..."

