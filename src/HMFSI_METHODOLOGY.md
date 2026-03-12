# Historical Macro-Financial Stress Index (HMFSI)
## Methodology, Academic Justification, and Implementation Guide

---

### 1. The Endogeneity Problem

The current analysis uses **Realized Volatility (RV)** computed from DJI log-returns as the stress proxy. Since β and D_e are also estimated from DJI returns, the stress measure and the targets share the same underlying data. Any statistical dependence found could partly reflect a **mechanical** link rather than an economically meaningful relationship.

The memo of criticisms (item #12, HIGH priority) explicitly states:

> "The results support a dependency between volatility amplitude and memory/efficiency, not exogenous macro causation."

The HMFSI is designed to break this link by constructing a stress measure entirely from **non-equity financial variables**.

---

### 2. Design Principles

| Principle | Implementation |
|-----------|---------------|
| **Exogeneity** | No component uses equity prices or equity-derived quantities |
| **Historical coverage** | At least two components available from 1871 (Shiller data) |
| **No look-ahead** | Expanding-window z-scores use only past information |
| **Transparency** | Equal-weight combination, no opaque PCA rotation |
| **Regime-aware splicing** | As more data become available, components are added seamlessly |
| **Replicability** | All data sources are public (Shiller website, FRED) |

---

### 3. Components

#### C1: Interest-Rate Volatility (IRV)
- **Definition**: 12-month rolling standard deviation of monthly changes in the 10-year government bond yield
- **Formula**: IRV_t = σ_{12}(ΔGS10_t)
- **Source**: Shiller dataset (1871–present)
- **Economic rationale**: Bond market uncertainty captures monetary policy surprises, flight-to-quality episodes, and macroeconomic uncertainty. Elevated IRV coincides with the Panic of 1907, both World Wars, the Volcker shock (1979–82), and the 2008–09 crisis.
- **Literature**: Gilchrist & Zakrajšek (2012) use Treasury yield volatility as a control in credit spread regressions.

#### C2: Inflation Surprise (IS)
- **Definition**: 12-month rolling standard deviation of monthly CPI log-changes
- **Formula**: IS_t = σ_{12}(Δlog CPI_t)
- **Source**: Shiller dataset (CPI-U from 1913; Warren-Pearson index spliced pre-1913)
- **Economic rationale**: Price instability erodes real returns and signals macroeconomic dislocations. The component captures WWI/WWII inflation, the 1970s stagflation, and the 2021–23 inflation surge.
- **Literature**: Jurado, Ludvigson & Ng (2015) include price uncertainty in their macro uncertainty index.

#### C3: Credit Spread (CS)
- **Definition**: Moody's Baa yield minus Moody's Aaa yield
- **Formula**: CS_t = BAA_t − AAA_t
- **Source**: FRED (1919-01–present)
- **Economic rationale**: The Baa–Aaa spread is the single most widely used indicator of credit stress. It widens sharply during recessions, banking crises, and corporate distress episodes. It spikes during the Great Depression (1930–33), the S&L crisis (1990), and the Global Financial Crisis (2008–09).
- **Literature**: Gilchrist & Zakrajšek (2012); Krishnamurthy & Muir (2020); Hakkio & Keeton (2009).

#### C4: Term Spread Stress (TSS)
- **Definition**: Negative of the yield curve slope (inverted so that positive = stress)
- **Formula**: TSS_t = −(GS10_t − TB3MS_t)
- **Source**: FRED TB3MS (1934-01–present); GS10 from Shiller
- **Economic rationale**: An inverted or flat yield curve is a canonical predictor of recessions and financial stress. The negative sign ensures that curve inversion → positive TSS → elevated stress.
- **Literature**: Estrella & Hardouvelis (1991); Ang, Piazzesi & Wei (2006).

#### C5: Real Rate Deviation (RRD)
- **Definition**: Deviation of the ex-post real long rate from its 24-month trailing mean
- **Formula**: RRD_t = (GS10_t − π_{12,t}) − MA_{24}(GS10 − π_{12})
- **Source**: Derived from Shiller (1871–present)
- **Economic rationale**: Sudden tightening of real monetary conditions (e.g., the Volcker shock) generates stress throughout the financial system. The deviation from trend captures *surprises* in real rates rather than their level.
- **Literature**: Inspired by the monetary conditions component of the KCFSI (Hakkio & Keeton, 2009).

---

### 4. Regime Timeline

| Period | Components | # | Notes |
|--------|-----------|---|-------|
| 1885–1919 | IRV + IS + RRD | 3 | Pre-Moody's era; bond and inflation data only |
| 1919–1934 | IRV + IS + CS + RRD | 4 | Credit spread added (Moody's begins) |
| 1934–present | IRV + IS + CS + TSS + RRD | 5 | Full model (T-Bill rate available) |

The regime transitions are smooth because:
1. Each component is z-scored independently
2. The index is the **average** of available z-scores
3. Adding a new component centered at zero does not create a level shift

---

### 5. Standardisation Procedure

For each component x_t:

1. **Expanding z-score** (no look-ahead):
   ```
   z_t = (x_t − μ̂_{1:t}) / σ̂_{1:t}
   ```
   where μ̂_{1:t} and σ̂_{1:t} are the sample mean and std using all data from the start up to date t.

2. **Equal-weight combination**:
   ```
   HMFSI_raw_t = (1/K_t) Σ_{k available} z_{k,t}
   ```
   where K_t is the number of non-missing components at date t.

3. **Full-sample re-standardisation** (for interpretability):
   ```
   HMFSI_t = (HMFSI_raw_t − mean(HMFSI_raw)) / std(HMFSI_raw)
   ```

Convention: HMFSI > 0 indicates above-average stress; HMFSI < 0 below-average.

---

### 6. Why Not PCA?

PCA would be technically superior for variance extraction, but:
- It requires a **fixed** set of variables across the full sample (incompatible with regime-aware splicing)
- The loadings are not interpretable across regimes
- Equal weighting is standard in the STLFSI, KCFSI, and other established indices
- With only 3–5 components, PCA offers marginal gains over equal weighting

PCA on the overlap period (1934+) can be used as a **robustness check** in the SI Appendix.

---

### 7. Validation Strategy

The HMFSI should be validated against:

1. **Known stress episodes**: Does it spike during the Panic of 1907, WWI (1914–18), the Great Depression (1929–33), WWII (1941–45), the Oil Crisis (1973–74), the Volcker Shock (1979–82), Black Monday (1987), the Asian/LTCM crisis (1997–98), the GFC (2007–09), and COVID-19 (2020)?

2. **Correlation with STLFSI4**: On the overlap period (1993+), what is the correlation between HMFSI and STLFSI4? A correlation of 0.5–0.8 would indicate that HMFSI captures genuine financial stress while being distinct from the equity-heavy STLFSI.

3. **Correlation with RV**: The HMFSI–RV correlation should be moderate (0.3–0.6) — high enough to confirm that both capture stress, low enough to confirm that HMFSI adds genuinely new information.

4. **Granger causality test**: Does HMFSI Granger-cause DJI-RV? If yes, this establishes that macro-financial conditions *lead* equity volatility, strengthening the exogeneity argument.

---

### 8. Integration with the Analysis Pipeline

```
# Step 1: Build HMFSI (one-time)
python hmfsi.py --output-dir hmfsi_output

# Step 2: Run the 5-test battery with HMFSI
python stress_master_all_tests.py \
  --rolling rigid_windows_results_125.csv \
  --prices PRICE_DEF.txt \
  --target col:beta_acf \
  --stress-source hmfsi:hmfsi_output/HMFSI.csv \
  --out results_hmfsi_beta

# Step 3: Repeat for D_e
python stress_master_all_tests.py \
  --rolling rigid_windows_results_125.csv \
  --prices PRICE_DEF.txt \
  --target col:De_acf_TL_AR \
  --stress-source hmfsi:hmfsi_output/HMFSI.csv \
  --out results_hmfsi_De
```

The alignment between the monthly HMFSI and the rolling-window dates (approximately monthly, end-of-window) is handled by forward-fill, which is conservative (uses only past stress information).

---

### 9. Expected Impact on the Paper

| Scenario | Implication |
|----------|------------|
| HMFSI replicates RV results (TY p < 0.01) | **Strong**: Exogenous stress genuinely predicts scaling parameters. This upgrades the claim from "internal feedback" to "macro-financial causation". |
| HMFSI shows weaker but significant results (p < 0.05) | **Moderate**: Macro stress has some predictive power, but the RV channel is partly mechanical. The paper reports both and discusses the decomposition. |
| HMFSI shows no significant results | **Informative null**: The RV results were indeed driven by endogeneity. The paper honestly reports this and reframes the contribution as characterising an internal feedback loop. |

All three scenarios produce a publishable paper. The HMFSI transforms the weakness (#12 in the memo) into a strength by allowing a clean **decomposition** of endogenous vs. exogenous stress channels.

---

### 10. References

- Estrella, A. & Hardouvelis, G.A. (1991). The term structure as a predictor of real economic activity. *Journal of Finance*, 46(2), 555–576.
- Gilchrist, S. & Zakrajšek, E. (2012). Credit spreads and business cycle fluctuations. *American Economic Review*, 102(4), 1692–1720.
- Hakkio, C.S. & Keeton, W.R. (2009). Financial stress: What is it, how can it be measured, and why does it matter? *FRBKC Economic Review*, 94(2), 5–50.
- Jurado, K., Ludvigson, S.C. & Ng, S. (2015). Measuring uncertainty. *American Economic Review*, 105(3), 1177–1216.
- Kliesen, K.L., Owyang, M.T. & Vermann, E.K. (2012). Disentangling diverse measures: A survey of financial stress indexes. *FRBSL Review*, 94(5), 369–397.
- Krishnamurthy, A. & Muir, T. (2020). How credit cycles across a financial crisis. Working paper, Stanford GSB.
- Shiller, R.J. (2000). *Irrational Exuberance*. Princeton University Press.
