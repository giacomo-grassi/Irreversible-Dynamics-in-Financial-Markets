# src/ — Core Research Engine

This directory contains the computational backbone of the project.

All empirical results, parameter estimates, simulations, and statistical tests are generated through the modular Python code implemented here. The goal of this folder is not exploratory analysis (handled in `/notebooks`), but **structured, reproducible, research-grade computation**.

---

## 🎯 Design Philosophy

The codebase is built around three principles:

1. **Modularity** — Each conceptual block (estimation, simulation, testing) is isolated.
2. **Reproducibility** — Deterministic seeds and explicit parameter control.
3. **Research Transparency** — Methods mirror the structure described in the paper.

The architecture reflects the logical flow of the research:

Data → Scaling Estimation → Monte Carlo Mapping → Stress Testing → Output Generation

---

## 🧠 Functional Components

### 1️⃣ Data Processing

Handles:

- Log-return computation
- Rolling-window construction
- Realized volatility calculation
- Alignment of stress and scaling series

Care is taken to preserve time indexing integrity and avoid look-ahead bias.

---

### 2️⃣ Scaling Parameter Estimation

Implements estimation of:

- **β** — Volatility autocorrelation decay exponent  
- **Dₑ** — Effective anomalous diffusion exponent  

Key procedures include:

- Log-binned ACF regression
- HAC (Newey–West) corrected inference
- Spline inversion of the simulated β(D) mapping
- Rolling-window estimation framework

This module transforms raw returns into economically interpretable state variables.

---

### 3️⃣ Monte Carlo Simulation (TL-AR Framework)

Implements the Truncated Lévy Autoregressive simulation used to generate the theoretical β(D) curve.

Core features:

- AR order m = 100
- Reset horizon τc
- Multiple replications for stability
- Deterministic seed control

The simulation stage bridges theory and empirical calibration.

---

### 4️⃣ Stress-Dependence Testing

Implements five complementary statistical methodologies:

- Toda–Yamamoto Granger causality
- Quantile predictability analysis
- Local projections (Jordà)
- Distance correlation (nonlinear dependence)
- Wavelet coherence (time-frequency structure)

Each test is implemented with:

- Robust inference procedures
- HAC corrections or permutation-based significance
- Explicit lag structure control

This module formalizes the core research hypothesis:  
**Do scaling parameters systematically respond to financial stress?**

---

### 5️⃣ Visualization & Output Utilities

Functions to generate:

- Impulse response plots
- Lag-dependence diagrams
- Wavelet heatmaps
- Rolling parameter trajectories

All figures are reproducible directly from code without manual intervention.

---

## 🔬 Methodological Rigor

The implementation explicitly addresses:

- Overlapping window dependence
- Effective sample size correction
- Multiple testing concerns
- Nonlinear dependence detection
- Parameter inversion uncertainty

This ensures that statistical significance is not an artifact of estimation design.

---

## 🧩 Extensibility

The code is structured so it can be extended to:

- Alternative indices (S&P 500, NASDAQ, Nikkei)
- Alternative stress proxies
- Different scaling distributions (Student-t, GH, etc.)
- High-frequency data applications

---

## 📌 Intended Audience

This codebase is designed for:

- Quantitative researchers
- Financial econometricians
- Econophysics practitioners
- Advanced students in quantitative finance

It is not optimized for production trading systems, but for **structural financial research and methodological clarity**.

