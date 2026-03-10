# Irreversible Dynamics in Financial Markets  
### Scaling Parameters as Structural State Variables

This repository contains the full research pipeline investigating whether scaling-law parameters derived from long historical financial time series encode economically meaningful information about market stress and structural regime shifts.

The study is based on over a century of daily DJIA data and combines:

- rolling-window estimation  
- stochastic scaling theory  
- Monte Carlo calibration  
- multi-method stress-dependence testing  

The objective is not descriptive modeling, but identification of **structural state variables** capable of capturing regime reconfiguration under stress.

---

## Research Question

Do scaling parameters — typically treated as static calibration constants — behave as dynamic variables that systematically respond to financial stress?

If so, they may provide:

- Early-warning information  
- Regime identification signals  
- Structural fragility diagnostics  
- Nonlinear stress transmission insight  

---

## Methodological Overview

The empirical pipeline follows:

1. **Historical data ingestion** (DJIA daily closes, 1896–present)  
2. Log-return transformation  
3. Rolling-window parameter estimation  
4. Stress proxy alignment (realized volatility)  
5. Multi-framework dependence testing  
6. Impulse response and multi-scale diagnostics  

No manual tuning or post-hoc adjustments are applied.

---

## Core Quantitative Components

### 1. Rolling Structural Estimation

Parameters estimated dynamically:

- **β** — Volatility autocorrelation decay exponent  
- **Dₑ** — Effective anomalous diffusion exponent  
- **corr(β, Dₑ)** — Rolling structural coupling  

These parameters are extracted via:

- log-binned ACF regression  
- HAC (Newey–West) corrected inference  
- spline inversion of simulated calibration curves  

---

### 2. Monte Carlo Calibration

A truncated Lévy autoregressive (TL-AR) simulation framework is implemented to:

- generate theoretical β(D) mappings  
- calibrate empirical inversion  
- control estimation uncertainty  

This step bridges stochastic theory and empirical measurement.

---

### 3. Stress-Dependence Testing

To avoid reliance on a single econometric specification, five complementary frameworks are implemented:

- Toda–Yamamoto augmented VAR causality  
- Quantile predictability  
- Local Projections (Jordà)  
- Distance correlation (nonlinear dependence)  
- Wavelet coherence (time-frequency structure)  

Inference accounts for:

- overlapping windows  
- serial dependence  
- nonlinear effects  
- multi-scale structure  

---

The separation ensures:

- clean data lineage  
- reproducible computation  
- modular extensibility  

---

## Results (Selected Examples)

The `results/` folder includes representative outputs such as:

- Rolling structural parameter trajectories  
- Stress–parameter cross-sectional relationships  
- Local projection impulse responses  
- Wavelet coherence heatmaps  

These illustrate:

- Short-horizon adaptation in β  
- Slower structural reconfiguration in Dₑ  
- Delayed decoupling under stress  

The full estimation framework produces additional robustness specifications not all displayed here.

---

## Reproducibility & Extension

All results are fully script-driven and reproducible.

The framework can be extended to:

- Alternative indices (S&P 500, NASDAQ, Nikkei)  
- Alternative stress measures  
- Different scaling distributions  
- Higher-frequency datasets  

The architecture is research-oriented rather than production-trading optimized.

---

## Why This Matters for Quant Research

Most volatility models focus on conditional variance forecasting.  
This project instead investigates **structural parameter dynamics**.

It emphasizes:

- multi-method validation  
- nonlinear dependence detection  
- regime-dependent behavior  
- calibration uncertainty awareness  

The goal is not backtest performance, but structural signal extraction and interpretability.

---

## Technical Stack

Python ecosystem:

- numpy, pandas  
- statsmodels  
- scipy  
- matplotlib  
- wavelet analysis (pywt)  

Emphasis on modularity, reproducibility, and transparent statistical inference.

---

## Author

Giacomo Grassi  
MSc Quantitative Finance — Bocconi University  
BSc Physics — University of Padua  

Research interests:

- Scaling laws in finance  
- Volatility dynamics  
- Structural regime shifts  
- Complex adaptive systems  
