#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baldovin_stella_analysis.py
===========================

Full estimation and analysis pipeline for the Baldovin-Stella (BS) model
applied to financial return time series.

This module implements:
  - Estimation of β (ACF-decay exponent) via five methods: ACF bootstrap,
    HAC Newey-West, DFA, Bayesian conjugate, and Whittle/GPH.
  - Simulation of three calibration curves β(D): TL-Joint (block mixing),
    TL-AR (autoregressive, PNAS procedure), and Student-t benchmark.
  - PCHIP spline inversion to recover the anomalous-diffusion exponent D_e.
  - Rolling-window estimation of β, D, and D_e over the full price series.
  - Comprehensive diagnostic and publication-quality plots.

References:
  - Baldovin & Stella, PNAS 104(50), 19741–19746 (2007)
  - Baldovin & Stella, Phys. Rev. E 75, 020101(R) (2007)
  - Sokolov et al., Physica A 336, 245 (2004)
"""

import os
import sys
import argparse
import json
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    from scipy import stats
    from scipy.interpolate import PchipInterpolator
    from scipy.optimize import minimize_scalar
    from scipy.ndimage import uniform_filter1d

    SCIPY_AVAILABLE = True
except ImportError:
    stats = None
    PchipInterpolator = None
    minimize_scalar = None
    uniform_filter1d = None
    SCIPY_AVAILABLE = False

CI: float = 1.96
_AR_COEFF_CACHE: Dict = {}

# =============================================================================
# TRUNCATED LÉVY CHARACTERISTIC FUNCTION PARAMETERS (Sokolov et al. 2004)
# =============================================================================
# CF: ẽg(k) = exp(-B*k² / (1 + C_α|k|^{2-α}))
# Full-sample DJIA estimates from Baldovin & Stella PNAS SI Text
ALPHA_TL = 0.7845  # Tail exponent
B_TL = 9.09e-5   # Scale parameter
C_TL = 1.90e-3   # Truncation parameter

# PNAS simulation parameters (SI Text)
TAU_C = 500    # Block reset period τ_c (SI Text)
M_AR  = 100    # AR memory length m (SI Text)
TMAX  = 33000  # Simulation length (~DJIA series length)

# Gaver-Stehfest Laplace inversion for the mixing distribution
GS_N  = 14     # Gaver-Stehfest order
SIG2_XMIN  = 1e-12   # σ² grid minimum
SIG2_XMAX  = 1e5     # σ² grid maximum
SIG2_NGRID = 1200    # σ² grid points

# Student benchmark
NU_STUDENT = 3.2     # Student-t degrees of freedom (benchmark)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _savefig(path: str, dpi: int = 150) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def _format_time_axis(ax: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=4, maxticks=12)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def read_prices_auto(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    try:
        return pd.read_csv(path, engine="python", sep=None)
    except:
        for sep in [",", ";", "\t", "|"]:
            try:
                return pd.read_csv(path, sep=sep)
            except:
                continue
    raise ValueError("Unable to read the file.")


def extract_prices_and_dates(df, price_col, date_col=None):
    if price_col not in df.columns:
        raise ValueError(f"Price column not found: {price_col}")
    p = pd.to_numeric(df[price_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(p)
    prices = p[mask]
    dates = None
    if date_col and date_col in df.columns:
        d = pd.to_datetime(df[date_col], errors="coerce").to_numpy()
        d = d[mask]
        if pd.Series(d).notna().mean() >= 0.8:
            dates = d
    return prices, dates


def detrend_log_prices(prices):
    prices = np.asarray(prices, float)
    prices = prices[~np.isnan(prices)]
    lp = np.log(prices)
    n = lp.size
    if n < 10:
        return lp
    x = np.arange(n, dtype=float)
    A = np.vstack([x, np.ones_like(x)]).T
    m, q = np.linalg.lstsq(A, lp, rcond=None)[0]
    return lp - (m * x + q)


def returns_from_logp(log_prices):
    return np.diff(np.asarray(log_prices, float))


def aggregated_returns(log_prices, T):
    lp = np.asarray(log_prices, float)
    if T <= 0 or lp.size <= T:
        return np.empty(0, float)
    return lp[T:] - lp[:-T]


# =============================================================================
# ACF COMPUTATION AND LOG-BINNING
# =============================================================================


def acf_fft(x, max_lag):
    """ACF via FFT — identical procedure for empirical and simulated series."""
    x = np.asarray(x, float)
    x = x - x.mean()
    n = x.size
    if n <= 1:
        return np.ones(1, float)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    X = np.fft.rfft(x, nfft)
    S = X * np.conjugate(X)
    acf = np.fft.irfft(S, nfft)[:n]
    acf /= acf[0] + 1e-18
    return acf[: max_lag + 1]


@dataclass
class BinningResult:
    x_binned: np.ndarray
    y_binned: np.ndarray
    n_per_bin: np.ndarray
    edges: np.ndarray
    mask_valid: np.ndarray
    nbins_original: int
    nbins_valid: int
    n_acf_above_threshold: int
    n_acf_total: int


def robust_logbin(
    lags,
    acf_values,
    nbins=16,
    lag_min=1,
    lag_max=100,
    min_acf_threshold=0.001,
    aggregation="median",
):
    """
    Log-binning with robust (median) aggregation.
    IMPORTANT: min_acf_threshold must be identical between empirical and simulated series.
    """
    lags = np.asarray(lags, float)
    acf_values = np.asarray(acf_values, float)
    in_range = (lags >= lag_min) & (lags <= lag_max)
    lags_r, acf_r = lags[in_range], acf_values[in_range]

    # Fraction of ACF values above threshold
    n_acf_total = len(acf_r)
    n_acf_above_threshold = int(np.sum(acf_r > min_acf_threshold))

    if len(lags_r) == 0:
        return BinningResult(
            np.array([]),
            np.array([]),
            np.array([], dtype=int),
            np.array([]),
            np.array([], dtype=bool),
            0,
            0,
            0,
            0,
        )

    log_min = np.log10(max(lag_min, 1))
    log_max = np.log10(lag_max)
    edges = np.logspace(log_min, log_max, nbins + 1)

    x_binned = np.full(nbins, np.nan)
    y_binned = np.full(nbins, np.nan)
    n_per_bin = np.zeros(nbins, dtype=int)

    for i in range(nbins):
        mask_bin = (lags_r >= edges[i]) & (lags_r < edges[i + 1])
        n_in_bin = np.sum(mask_bin)
        if n_in_bin == 0:
            continue
        lags_in_bin = lags_r[mask_bin]
        acf_in_bin = acf_r[mask_bin]
        x_binned[i] = np.exp(np.mean(np.log(lags_in_bin)))
        y_binned[i] = (
            np.median(acf_in_bin) if aggregation == "median" else np.mean(acf_in_bin)
        )
        n_per_bin[i] = n_in_bin

    # mask_valid: bins with valid data AND above ACF threshold
    mask_valid = (
        np.isfinite(y_binned) & (y_binned > min_acf_threshold) & np.isfinite(x_binned)
    )

    return BinningResult(
        x_binned=x_binned,
        y_binned=y_binned,
        n_per_bin=n_per_bin,
        edges=edges,
        mask_valid=mask_valid,
        nbins_original=nbins,
        nbins_valid=int(np.sum(mask_valid)),
        n_acf_above_threshold=n_acf_above_threshold,
        n_acf_total=n_acf_total,
    )


def apply_binning_with_fixed_edges(
    lags, acf_values, edges, min_acf_threshold=0.001, aggregation="median"
):
    """Apply binning with fixed edges (used in bootstrap resampling)."""
    lags = np.asarray(lags, float)
    acf_values = np.asarray(acf_values, float)
    nbins = len(edges) - 1
    x_binned = np.full(nbins, np.nan)
    y_binned = np.full(nbins, np.nan)
    for i in range(nbins):
        mask_bin = (lags >= edges[i]) & (lags < edges[i + 1])
        if np.sum(mask_bin) == 0:
            continue
        lags_in_bin = lags[mask_bin]
        acf_in_bin = acf_values[mask_bin]
        x_binned[i] = np.exp(np.mean(np.log(lags_in_bin)))
        y_binned[i] = (
            np.median(acf_in_bin) if aggregation == "median" else np.mean(acf_in_bin)
        )
    return x_binned, y_binned


# =============================================================================
# REGRESSION FIT METHODS
# =============================================================================


def theil_sen_slope(x, y):
    """Fit robusto Theil-Sen."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = x.size
    if n < 3:
        return np.nan, np.nan, np.nan
    slopes = []
    for i in range(n - 1):
        dx = x[i + 1 :] - x[i]
        dy = y[i + 1 :] - y[i]
        mask = np.abs(dx) > 1e-18
        if np.any(mask):
            slopes.extend(dy[mask] / dx[mask])
    if not slopes:
        return np.nan, np.nan, np.nan
    slope = float(np.median(slopes))
    intercept = float(np.median(y - slope * x))
    # R²
    y_pred = intercept + slope * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-18
    r2 = max(0.0, 1.0 - ss_res / ss_tot)
    return slope, intercept, r2


def wls_fit(x, y, weights=None):
    """Fit WLS con pesi opzionali (es. sqrt(n_per_bin))."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    if weights is None:
        weights = np.ones_like(x)
    else:
        weights = np.asarray(weights, float)
        if len(weights) > len(x):
            weights = weights[mask]
    weights = weights / (np.sum(weights) + 1e-18)
    X = np.vstack([np.ones_like(x), x]).T
    W = np.diag(weights)
    try:
        XtW = X.T @ W
        coef = np.linalg.solve(XtW @ X + 1e-10 * np.eye(2), XtW @ y)
    except:
        return np.nan, np.nan, np.nan
    intercept, slope = float(coef[0]), float(coef[1])
    yhat = X @ coef
    ss_res = float(np.sum(weights * (y - yhat) ** 2))
    ss_tot = float(np.sum(weights * (y - np.average(y, weights=weights)) ** 2)) + 1e-18
    r2 = max(0.0, 1.0 - ss_res / ss_tot)
    return slope, intercept, r2


def ols_fit(x, y):
    """Fit OLS standard."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    X = np.vstack([np.ones_like(x), x]).T
    try:
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
    except:
        return np.nan, np.nan, np.nan
    intercept, slope = float(coef[0]), float(coef[1])
    y_pred = X @ coef
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-18
    r2 = max(0.0, 1.0 - ss_res / ss_tot)
    return slope, intercept, r2


def fit_loglog_slope(log_x, log_y, fit_method="theil_sen", weights=None):
    """
    Dispatch centralizzato per fit slope in scala log-log.
    Usato IDENTICAMENTE per empirico e simulato.
    """
    if fit_method == "wls" and weights is not None:
        slope, intercept, r2 = wls_fit(log_x, log_y, weights)
    elif fit_method == "ols":
        slope, intercept, r2 = ols_fit(log_x, log_y)
    else:  # default: theil_sen
        slope, intercept, r2 = theil_sen_slope(log_x, log_y)
    return slope, intercept, r2


def block_resample(series, block_len, rng):
    """Block bootstrap resampling (Künsch 1989)."""
    n = len(series)
    if block_len >= n:
        return series.copy()
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, max(1, n - block_len + 1), size=n_blocks)
    resampled = []
    for s in starts:
        resampled.append(series[s : s + block_len])
    return np.concatenate(resampled)[:n]


# =============================================================================
# DATACLASSES
# =============================================================================


@dataclass
class BetaCoreResult:
    """
    Output of the unified β core estimator.
    Used identically for empirical and simulated series.
    """

    beta: float
    r2: float
    nbins_valid: int
    intercept: float
    success: bool
    fail_reason: str
    n_acf_above_threshold: int
    n_acf_total: int
    residuals: Optional[np.ndarray] = None


@dataclass
class BetaEstimate:
    beta: float
    stderr: float
    ci: Tuple[float, float]
    details: Dict


@dataclass
class DEstimate:
    D: float
    stderr: float
    ci: Tuple[float, float]
    details: Dict


@dataclass
class DeEstimate:
    De: float
    stderr: float
    ci: Tuple[float, float]
    details: Dict


@dataclass
class DePointDiagnostics:
    """Diagnostiche per singolo punto D_e nella curva simulata."""

    De: float
    pass_rate: float
    n_total: int
    n_pass: int
    n_fit_fail: int
    n_bins_insufficient: int
    n_beta_invalid: int
    median_nbins_valid: float
    mean_frac_acf_above_threshold: float
    beta_values: np.ndarray  # all replicates (including NaN)
    sel_bias_warn: bool


@dataclass
class BetaSimCurve:
    """Curva di calibrazione β_sim(D_e) con diagnostiche complete."""

    De_grid: np.ndarray
    beta_mean: np.ndarray
    beta_std: np.ndarray
    beta_std_smooth: np.ndarray
    spline: Optional[Any]
    derivative: Optional[np.ndarray]
    diagnostics: List[DePointDiagnostics]
    params: Dict  # parametri usati (per verifica coerenza)
    details: Dict


# =============================================================================
# UNIFIED CORE ESTIMATOR FOR β
# =============================================================================


def estimate_beta_acf_core(
    series: np.ndarray,
    lag_min: int,
    lag_max: int,
    nbins: int,
    min_acf_threshold: float,
    fit_method: str,
    min_bins_required: int,
    aggregation: str = "median",
) -> BetaCoreResult:
    """
    CORE ESTIMATOR UNIFICATO per β via ACF.

    Questa funzione DEVE essere usata sia per la stima empirica sia per
    every simulated replicate when building the β_sim(D_e) curve.

    Parameters
    ----------
    series : array
        Input time series (typically |r_t|)
    lag_min, lag_max : int
        ACF lag range.
    nbins : int
        Number of log-spaced bins.
    min_acf_threshold : float
        Minimum ACF value for a bin to be considered valid (must be identical
        between empirical and simulated series for coherent inversion).
    fit_method : str
        Slope estimator: "theil_sen", "wls", or "ols".
    min_bins_required : int
        Minimum number of valid bins required for a successful fit.
    aggregation : str
        Within-bin aggregation: "median" (default) or "mean".

    Returns
    -------
    BetaCoreResult with β, R², diagnostics, and success/fail_reason flags.
    """
    series = np.asarray(series, float)
    series = series[np.isfinite(series)]
    n = series.size

    # Minimum length check
    if n < max(100, lag_max + 10):
        return BetaCoreResult(
            beta=np.nan,
            r2=np.nan,
            nbins_valid=0,
            intercept=np.nan,
            success=False,
            fail_reason="too_few_points",
            n_acf_above_threshold=0,
            n_acf_total=0,
        )

    # Adjust lag_max if needed
    lag_max_eff = min(lag_max, n // 4)
    if lag_max_eff <= lag_min:
        return BetaCoreResult(
            beta=np.nan,
            r2=np.nan,
            nbins_valid=0,
            intercept=np.nan,
            success=False,
            fail_reason="lag_range_invalid",
            n_acf_above_threshold=0,
            n_acf_total=0,
        )

    # Compute ACF
    acf = acf_fft(series, lag_max_eff)
    lags = np.arange(1, lag_max_eff + 1, dtype=float)
    acf_values = acf[1:]

    # Log-binning with threshold
    bin_result = robust_logbin(
        lags,
        acf_values,
        nbins=nbins,
        lag_min=lag_min,
        lag_max=lag_max_eff,
        min_acf_threshold=min_acf_threshold,  # CRITICO!
        aggregation=aggregation,
    )

    # Check sufficient bins
    if bin_result.nbins_valid < min_bins_required:
        return BetaCoreResult(
            beta=np.nan,
            r2=np.nan,
            nbins_valid=bin_result.nbins_valid,
            intercept=np.nan,
            success=False,
            fail_reason="too_few_valid_bins",
            n_acf_above_threshold=bin_result.n_acf_above_threshold,
            n_acf_total=bin_result.n_acf_total,
        )

    # Extract valid bins
    mask = bin_result.mask_valid
    x = bin_result.x_binned[mask]
    y = bin_result.y_binned[mask]
    n_per_bin = bin_result.n_per_bin[mask]

    # Log-transform for power-law regression
    log_x, log_y = np.log(x), np.log(y)

    # Fit with specified method
    weights = np.sqrt(n_per_bin) if fit_method == "wls" else None
    slope, intercept, r2 = fit_loglog_slope(log_x, log_y, fit_method, weights)

    # β = -slope in log-log space
    beta = -slope

    # Compute residuals
    y_pred = intercept + slope * log_x
    residuals = log_y - y_pred

    # Validate β estimate
    if not np.isfinite(beta):
        return BetaCoreResult(
            beta=np.nan,
            r2=r2,
            nbins_valid=bin_result.nbins_valid,
            intercept=intercept,
            success=False,
            fail_reason="fit_returned_nan",
            n_acf_above_threshold=bin_result.n_acf_above_threshold,
            n_acf_total=bin_result.n_acf_total,
            residuals=residuals,
        )

    # Do not silently discard β ≤ 0; record as a diagnostic failure
    # Flag via fail_reason but still return the value
    if beta <= 0:
        return BetaCoreResult(
            beta=beta,
            r2=r2,
            nbins_valid=bin_result.nbins_valid,
            intercept=intercept,
            success=False,
            fail_reason="beta_not_positive",
            n_acf_above_threshold=bin_result.n_acf_above_threshold,
            n_acf_total=bin_result.n_acf_total,
            residuals=residuals,
        )

    # Success
    return BetaCoreResult(
        beta=beta,
        r2=r2,
        nbins_valid=bin_result.nbins_valid,
        intercept=intercept,
        success=True,
        fail_reason="",
        n_acf_above_threshold=bin_result.n_acf_above_threshold,
        n_acf_total=bin_result.n_acf_total,
        residuals=residuals,
    )


# =============================================================================
# EMPIRICAL β ESTIMATION (via unified core estimator)
# =============================================================================


def estimate_beta_acf_robust(
    abs_returns,
    lag_min=1,
    lag_max=100,
    nbins=16,
    n_boot=200,
    block_len=None,
    random_state=1234,
    min_acf_threshold=0.001,
    fit_method="theil_sen",
    aggregation="median",
    min_bins_required=6,
):
    """
    Stima robusta β via ACF con bootstrap.
    USA IL CORE ESTIMATOR UNIFICATO.
    """
    rng = np.random.default_rng(random_state)
    abs_returns = np.asarray(abs_returns, float)
    abs_returns = abs_returns[np.isfinite(abs_returns)]
    n = abs_returns.size

    # Point estimate via core estimator
    core_result = estimate_beta_acf_core(
        abs_returns,
        lag_min,
        lag_max,
        nbins,
        min_acf_threshold,
        fit_method,
        min_bins_required,
        aggregation,
    )

    if not core_result.success:
        return BetaEstimate(
            core_result.beta,
            np.nan,
            (np.nan, np.nan),
            {
                "reason": core_result.fail_reason,
                "nbins_valid": core_result.nbins_valid,
                "method": f"ACF_core_{fit_method}",
            },
        )

    beta = core_result.beta
    r2 = core_result.r2

    # Block bootstrap for standard error
    if block_len is None:
        block_len = max(20, min(lag_max, n // 20))

    # Compute bin edges from the full series (fixed for bootstrap)
    lag_max_eff = min(lag_max, n // 4)
    acf = acf_fft(abs_returns, lag_max_eff)
    lags = np.arange(1, lag_max_eff + 1, dtype=float)
    acf_values = acf[1:]
    bin_result = robust_logbin(
        lags,
        acf_values,
        nbins=nbins,
        lag_min=lag_min,
        lag_max=lag_max_eff,
        min_acf_threshold=min_acf_threshold,
    )
    edges_fixed = bin_result.edges
    mask_fixed = bin_result.mask_valid

    betas_boot = []
    n_boot_fail = 0
    for _ in range(n_boot):
        boot_series = block_resample(abs_returns, block_len, rng)

        # Apply core estimator to each bootstrap replicate
        core_b = estimate_beta_acf_core(
            boot_series,
            lag_min,
            lag_max,
            nbins,
            min_acf_threshold,
            fit_method,
            min_bins_required,
            aggregation,
        )

        if core_b.success and np.isfinite(core_b.beta) and core_b.beta > 0:
            betas_boot.append(core_b.beta)
        else:
            n_boot_fail += 1

    if len(betas_boot) >= 20:
        stderr = float(np.std(betas_boot, ddof=1))
        ci_low = float(np.quantile(betas_boot, 0.025))
        ci_high = float(np.quantile(betas_boot, 0.975))
    else:
        stderr, ci_low, ci_high = np.nan, np.nan, np.nan

    details = {
        "method": f"ACF_core_{fit_method}",
        "lag_range": (lag_min, min(lag_max, n // 4)),
        "nbins_valid": core_result.nbins_valid,
        "n_boot_successful": len(betas_boot),
        "n_boot_failed": n_boot_fail,
        "block_len": block_len,
        "r2": float(r2),
        "intercept": float(core_result.intercept),
        "min_acf_threshold": min_acf_threshold,
        "fit_method": fit_method,
        "min_bins_required": min_bins_required,
        "residuals": (
            core_result.residuals.tolist() if core_result.residuals is not None else []
        ),
    }

    return BetaEstimate(float(beta), stderr, (ci_low, ci_high), details)


def estimate_beta_acf_hac(
    abs_returns,
    lag_min=1,
    lag_max=100,
    nbins=16,
    min_acf_threshold=0.001,
    fit_method="ols",
):
    """Estimate β from log-binned ACF with HAC (Newey-West) standard errors."""
    abs_returns = np.asarray(abs_returns, float)
    abs_returns = abs_returns[np.isfinite(abs_returns)]
    n = abs_returns.size
    lag_max = min(lag_max, n // 4)
    if n < 100 or lag_max <= lag_min:
        return BetaEstimate(
            np.nan, np.nan, (np.nan, np.nan), {"reason": "insufficient_data"}
        )

    acf = acf_fft(abs_returns, lag_max)
    lags = np.arange(1, lag_max + 1, dtype=float)
    acf_values = acf[1:]
    bin_result = robust_logbin(
        lags,
        acf_values,
        nbins=nbins,
        lag_min=lag_min,
        lag_max=lag_max,
        min_acf_threshold=min_acf_threshold,
    )
    if bin_result.nbins_valid < 6:
        return BetaEstimate(
            np.nan, np.nan, (np.nan, np.nan), {"reason": "too_few_bins"}
        )

    mask = bin_result.mask_valid
    x = np.log(bin_result.x_binned[mask])
    y = np.log(bin_result.y_binned[mask])
    n_pts = len(x)
    X = np.vstack([np.ones(n_pts), x]).T

    try:
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
    except:
        return BetaEstimate(
            np.nan, np.nan, (np.nan, np.nan), {"reason": "lstsq_failed"}
        )

    beta = -coef[1]
    residuals = y - X @ coef
    bandwidth = max(1, min(int(np.floor(4 * (n_pts / 100) ** (2 / 9))), n_pts - 1))

    # HAC covariance via Newey-West
    u = residuals.reshape(-1, 1)
    S = np.zeros((2, 2))
    for j in range(bandwidth + 1):
        weight = 1.0 if j == 0 else 1.0 - j / (bandwidth + 1)
        for t in range(j, n_pts):
            Xt = X[t : t + 1].T
            Xtj = X[t - j : t - j + 1].T
            ut = u[t : t + 1]
            utj = u[t - j : t - j + 1]
            S += weight * (Xt @ ut) @ (utj.T @ Xtj.T)
            if j > 0:
                S += weight * (Xtj @ utj) @ (ut.T @ Xt.T)
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        V_hac = XtX_inv @ S @ XtX_inv
        stderr = float(np.sqrt(max(0, V_hac[1, 1])))
    except:
        stderr = np.nan

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-18
    r2 = max(0.0, 1.0 - ss_res / ss_tot)
    ci = (
        (beta - CI * stderr, beta + CI * stderr)
        if np.isfinite(stderr)
        else (np.nan, np.nan)
    )

    details = {
        "method": "ACF_HAC",
        "lag_range": (lag_min, lag_max),
        "nbins_valid": bin_result.nbins_valid,
        "bandwidth": bandwidth,
        "r2": float(r2),
        "min_acf_threshold": min_acf_threshold,
    }
    return BetaEstimate(float(beta), stderr, ci, details)


# =============================================================================
# β ESTIMATION: DETRENDED FLUCTUATION ANALYSIS (DFA)
# =============================================================================


def dfa_fluctuations(x, order=2, scales=None):
    """Detrended Fluctuation Analysis."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 256:
        raise ValueError("Series too short for DFA (minimum ~256 observations)")
    y = np.cumsum(x - np.mean(x))
    if scales is None:
        s_min, s_max = 16, max(32, n // 4)
        scales = np.unique(
            np.floor(np.logspace(np.log10(s_min), np.log10(s_max), 24)).astype(int)
        )
    scales = np.asarray(scales, int)
    scales = scales[(scales >= max(8, order + 2)) & (scales <= n // 2)]
    Fs, used = [], []
    for s in scales:
        nseg = n // s
        if nseg < 2:
            continue
        rms2 = []
        for start in (0, n - nseg * s):
            for i in range(nseg):
                seg = y[start + i * s : start + (i + 1) * s]
                t = np.arange(s, dtype=float)
                coef = np.polyfit(t, seg, deg=order)
                fit = np.polyval(coef, t)
                rms2.append(np.mean((seg - fit) ** 2))
        Fs.append(np.sqrt(np.mean(rms2)))
        used.append(s)
    used = np.asarray(used, int)
    F = np.asarray(Fs, float)
    m = np.isfinite(F) & (F > 0) & np.isfinite(used) & (used > 0)
    return used[m], F[m]


def dfa_local_slopes(scales, F, window=5):
    """Compute local DFA slopes to detect cross-over scales."""
    log_s, log_F = np.log(scales.astype(float)), np.log(F)
    n = len(log_s)
    local_H, centers = [], []
    half = window // 2
    for i in range(half, n - half):
        seg_s = log_s[i - half : i + half + 1]
        seg_F = log_F[i - half : i + half + 1]
        if len(seg_s) >= 3:
            slope, _, _ = theil_sen_slope(seg_s, seg_F)
            local_H.append(slope)
            centers.append(scales[i])
    return np.array(centers), np.array(local_H)


def estimate_beta_dfa(
    abs_returns, order=2, n_boot=200, block_len=None, random_state=1234
):
    """Estimate β via Detrended Fluctuation Analysis: β = 2 − 2H."""
    rng = np.random.default_rng(random_state)
    x = np.asarray(abs_returns, float)
    x = x[np.isfinite(x)]
    if x.size < 256:
        return BetaEstimate(
            np.nan, np.nan, (np.nan, np.nan), {"reason": "too_few_points_dfa"}
        )
    try:
        scales, F = dfa_fluctuations(x, order=order)
    except Exception as e:
        return BetaEstimate(np.nan, np.nan, (np.nan, np.nan), {"reason": str(e)})
    if scales.size < 6:
        return BetaEstimate(
            np.nan, np.nan, (np.nan, np.nan), {"reason": "too_few_scales"}
        )

    log_s, log_F = np.log(scales.astype(float)), np.log(F)
    H, intercept, r2 = theil_sen_slope(log_s, log_F)
    beta = float(2.0 - 2.0 * H)
    if not np.isfinite(beta):
        return BetaEstimate(np.nan, np.nan, (np.nan, np.nan), {"reason": "fit_failed"})

    centers, local_H = dfa_local_slopes(scales, F)

    if block_len is None:
        block_len = max(32, min(int(round(x.size ** (1 / 3))), x.size // 5))

    betas_boot = []
    for _ in range(n_boot):
        xb = block_resample(x, block_len, rng)
        try:
            scb, Fb = dfa_fluctuations(xb, order=order, scales=scales)
            if scb.size < 5:
                continue
            Hb, _, _ = theil_sen_slope(np.log(scb.astype(float)), np.log(Fb))
            beta_b = 2.0 - 2.0 * Hb
            if np.isfinite(beta_b):
                betas_boot.append(beta_b)
        except:
            continue

    if len(betas_boot) >= 20:
        stderr = float(np.std(betas_boot, ddof=1))
        ci_low = float(np.quantile(betas_boot, 0.025))
        ci_high = float(np.quantile(betas_boot, 0.975))
    else:
        stderr, ci_low, ci_high = np.nan, np.nan, np.nan

    details = {
        "method": f"DFA-{order}",
        "H": float(H),
        "r2": float(r2),
        "n_scales": len(scales),
        "scale_range": (int(scales[0]), int(scales[-1])),
        "n_boot_successful": len(betas_boot),
        "local_H_centers": centers.tolist() if len(centers) > 0 else [],
        "local_H_values": local_H.tolist() if len(local_H) > 0 else [],
    }
    return BetaEstimate(float(beta), stderr, (ci_low, ci_high), details)


# =============================================================================
# β ESTIMATION: WHITTLE / GPH SPECTRAL METHOD
# =============================================================================


def whittle_stability_curve(abs_returns, m_values=None):
    """Curva di stabilità d(m) per selezione automatica bandwidth."""
    x = np.asarray(abs_returns, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 256:
        return np.array([]), np.array([]), np.array([])
    x_centered = x - np.mean(x)
    periodogram = np.abs(np.fft.fft(x_centered)[: n // 2 + 1]) ** 2 / n
    freqs = np.arange(n // 2 + 1) / n
    freqs, periodogram = freqs[1:], periodogram[1:]
    if m_values is None:
        m_min = max(10, int(n**0.4))
        m_max = min(len(freqs) // 2, int(n**0.8))
        m_values = np.unique(np.linspace(m_min, m_max, 30).astype(int))
    d_estimates = []
    for m in m_values:
        if m > len(freqs) - 1:
            d_estimates.append(np.nan)
            continue
        log_f = np.log(freqs[:m])
        log_I = np.log(periodogram[:m] + 1e-18)
        slope, _, _ = theil_sen_slope(log_f, log_I)
        d_estimates.append(-slope / 2)
    d_estimates = np.array(d_estimates)
    beta_estimates = 1 - 2 * d_estimates
    return m_values, d_estimates, beta_estimates


def select_m_by_plateau(m_values, d_estimates):
    """Selezione automatica bandwidth via plateau."""
    valid = np.isfinite(d_estimates)
    if not np.any(valid):
        return int(m_values[len(m_values) // 2]) if len(m_values) > 0 else 50
    d_valid, m_valid = d_estimates[valid], m_values[valid]
    window = max(3, len(d_valid) // 5)
    min_var, best_idx = np.inf, len(d_valid) // 2
    for i in range(len(d_valid) - window):
        local_var = np.var(d_valid[i : i + window])
        if local_var < min_var:
            min_var, best_idx = local_var, i + window // 2
    return int(m_valid[best_idx])


def estimate_beta_whittle(
    abs_returns, bandwidth=None, random_state=1234, auto_select_m=True
):
    """Estimate β via Whittle/GPH with automatic bandwidth selection."""
    x = np.asarray(abs_returns, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 256:
        return BetaEstimate(
            np.nan, np.nan, (np.nan, np.nan), {"reason": "too_few_points"}
        )

    m_values, d_estimates, beta_estimates = whittle_stability_curve(x)
    if len(m_values) == 0:
        return BetaEstimate(
            np.nan, np.nan, (np.nan, np.nan), {"reason": "stability_curve_failed"}
        )

    if auto_select_m:
        m = select_m_by_plateau(m_values, d_estimates)
    elif bandwidth is not None:
        m = int(bandwidth * n)
    else:
        m = int(n**0.5)
    m = max(10, min(m, n // 4))

    x_centered = x - np.mean(x)
    periodogram = np.abs(np.fft.fft(x_centered)[: n // 2 + 1]) ** 2 / n
    freqs = np.arange(n // 2 + 1) / n
    freqs, periodogram = freqs[1:], periodogram[1:]
    if m > len(freqs):
        m = len(freqs)

    log_f = np.log(freqs[:m])
    log_I = np.log(periodogram[:m] + 1e-18)
    slope, intercept, _ = theil_sen_slope(log_f, log_I)
    d_hat = -slope / 2
    beta = 1 - 2 * d_hat

    stderr_d = np.pi / (2 * np.sqrt(6 * m)) if m > 0 else np.nan
    stderr_beta = 2 * stderr_d

    if not np.isfinite(beta) or beta < -0.5 or beta > 1.5:
        return BetaEstimate(
            np.nan, np.nan, (np.nan, np.nan), {"reason": "invalid_estimate"}
        )

    ci = (beta - CI * stderr_beta, beta + CI * stderr_beta)
    details = {
        "method": "GPH_Whittle",
        "d_hat": float(d_hat),
        "bandwidth_m": int(m),
        "auto_select_m": auto_select_m,
        "m_stability_curve": (
            m_values.tolist() if len(m_values) < 50 else m_values[::2].tolist()
        ),
        "d_stability_curve": (
            d_estimates.tolist() if len(d_estimates) < 50 else d_estimates[::2].tolist()
        ),
    }
    return BetaEstimate(float(beta), float(stderr_beta), ci, details)


# =============================================================================
# β ESTIMATION: BAYESIAN CONJUGATE REGRESSION
# =============================================================================


def estimate_beta_bayes(
    abs_returns,
    lag_min=1,
    lag_max=100,
    nbins=16,
    min_acf_threshold=0.001,
    random_state=1234,
    nsamp=6000,
):
    """Bayesian estimate of β with a Normal-Inverse-Gamma conjugate prior."""
    if stats is None:
        return BetaEstimate(
            np.nan, np.nan, (np.nan, np.nan), {"reason": "scipy_missing"}
        )
    abs_returns = np.asarray(abs_returns, float)
    abs_returns = abs_returns[np.isfinite(abs_returns)]
    n = abs_returns.size
    lag_max = min(lag_max, n // 4)
    if n < 100 or lag_max <= lag_min:
        return BetaEstimate(
            np.nan, np.nan, (np.nan, np.nan), {"reason": "insufficient_data"}
        )

    acf = acf_fft(abs_returns, lag_max)
    lags = np.arange(1, lag_max + 1, dtype=float)
    acf_values = acf[1:]
    bin_result = robust_logbin(
        lags,
        acf_values,
        nbins=nbins,
        lag_min=lag_min,
        lag_max=lag_max,
        min_acf_threshold=min_acf_threshold,
    )
    if bin_result.nbins_valid < 8:
        return BetaEstimate(
            np.nan, np.nan, (np.nan, np.nan), {"reason": "too_few_bins"}
        )

    mask = bin_result.mask_valid
    log_x = np.log(bin_result.x_binned[mask])
    log_y = np.log(bin_result.y_binned[mask])
    n_pts = len(log_x)
    X = np.vstack([np.ones(n_pts), log_x]).T

    # Weakly informative (non-informative) prior
    a0, b0 = 2.0, 1.0
    w0 = np.zeros(2)
    V0 = np.eye(2) * 1e6
    V0_inv = np.linalg.inv(V0)

    try:
        Vn_inv = V0_inv + X.T @ X
        Vn = np.linalg.inv(Vn_inv)
        wn = Vn @ (V0_inv @ w0 + X.T @ log_y)
        an = a0 + n_pts / 2
        bn = b0 + 0.5 * (log_y @ log_y + w0.T @ V0_inv @ w0 - wn.T @ Vn_inv @ wn)
        bn = max(bn, 0.01)
    except:
        return BetaEstimate(
            np.nan, np.nan, (np.nan, np.nan), {"reason": "posterior_failed"}
        )

    try:
        rng = np.random.default_rng(random_state)
        sigma2 = stats.invgamma(a=an, scale=bn).rvs(
            size=nsamp, random_state=random_state
        )
        w_samp = np.zeros((nsamp, 2))
        for i in range(nsamp):
            w_samp[i] = rng.multivariate_normal(mean=wn, cov=Vn * sigma2[i])
    except:
        return BetaEstimate(
            np.nan, np.nan, (np.nan, np.nan), {"reason": "sampling_failed"}
        )

    beta_samp = -w_samp[:, 1]
    valid = (beta_samp > 0) & (beta_samp < 2) & np.isfinite(beta_samp)
    beta_samp = beta_samp[valid]
    if len(beta_samp) < 100:
        return BetaEstimate(
            np.nan, np.nan, (np.nan, np.nan), {"reason": "too_few_valid_samples"}
        )

    beta_mean = float(np.mean(beta_samp))
    beta_sd = float(np.std(beta_samp, ddof=1))
    ci = (float(np.quantile(beta_samp, 0.025)), float(np.quantile(beta_samp, 0.975)))

    details = {
        "method": "Bayes_conjugate",
        "lag_range": (lag_min, lag_max),
        "nbins_valid": bin_result.nbins_valid,
        "nsamp": nsamp,
        "n_valid_samples": len(beta_samp),
        "min_acf_threshold": min_acf_threshold,
    }
    return BetaEstimate(beta_mean, beta_sd, ci, details)


# =============================================================================
# SCALING EXPONENT D: ESTIMATION
# =============================================================================


def estimate_D_quantiles(
    log_prices, T_values, ps=(0.6, 0.7, 0.8, 0.9, 0.95), min_count_per_T=200
):
    """Estimate the scaling exponent D via quantile scaling."""
    R = {T: aggregated_returns(log_prices, T) for T in T_values}
    counts = {T: len(r) for T, r in R.items()}
    Ts = [T for T in T_values if counts[T] >= min_count_per_T]
    if len(Ts) < 3:
        return DEstimate(np.nan, np.nan, (np.nan, np.nan), {"reason": "insufficient_T"})
    Ts = sorted(Ts)
    Q = np.array([np.quantile(np.abs(R[T]), ps) for T in Ts], float)
    logT = np.log(np.array(Ts, float))
    wT = np.sqrt(np.array([counts[T] for T in Ts], float))
    slopes = []
    for j, p in enumerate(ps):
        y = np.log(Q[:, j] + 1e-18)
        slope, _, _ = wls_fit(logT, y, weights=wT)
        slopes.append(slope)
    slopes = np.array(slopes, float)
    finite = np.isfinite(slopes)
    if not np.any(finite):
        return DEstimate(np.nan, np.nan, (np.nan, np.nan), {"reason": "numeric_error"})
    D_hat = float(np.median(slopes[finite]))
    M = int(np.sum(finite))
    stderr = float(np.std(slopes[finite], ddof=1) / math.sqrt(M)) if M > 1 else np.nan
    ci = (
        (D_hat - CI * stderr, D_hat + CI * stderr)
        if np.isfinite(stderr)
        else (np.nan, np.nan)
    )
    details = {"T_used": Ts, "ps": list(ps), "slopes": slopes.tolist()}
    return DEstimate(D_hat, stderr, ci, details)


# =============================================================================
# PNAS SIMULATION: WEIGHTS, TRUNCATED-LÉVY CF, LAPLACE INVERSION
# =============================================================================


def weights_w_pnas(D: float, tau_c: int) -> np.ndarray:
    """
    Pesi w_i = a_i^{2D} = i^{2D} - (i-1)^{2D}

    """
    i = np.arange(1, tau_c + 1, dtype=np.float64)
    w = np.power(i, 2.0 * D) - np.power(i - 1.0, 2.0 * D)
    
    if np.any(~np.isfinite(w)) or np.any(w <= 0):
        w = np.ones(tau_c, dtype=np.float64)
    return w


def eg_truncated_levy(
    k: np.ndarray, B: float = B_TL, C: float = C_TL, alpha: float = ALPHA_TL
) -> np.ndarray:
    """
    Truncated-Lévy CF (PNAS SI Text Eq. 35):
    ẽg(k) = exp(−Bk²/(1 + C_α|k|^{2−α}))

    Sokolov et al. (Physica A 336, 2004) show this is a valid CF admitting
    a variance-mixture (subordination) representation.
    """
    k = np.asarray(k, dtype=np.float64)
    denom = 1.0 + C * np.power(np.abs(k) + 1e-18, 2.0 - alpha)
    return np.exp(-(B * k * k) / denom)


def phi_sigma2_tl(
    s: np.ndarray, B: float = B_TL, C: float = C_TL, alpha: float = ALPHA_TL
) -> np.ndarray:
    """
    Trasformata di Laplace di σ²: φ(s) = E[e^{-s σ²}] = ẽg(√(2s))
    Dalla rappresentazione subordinazione (Sokolov et al. Physica A 336, 2004).
    """
    s = np.asarray(s, dtype=np.float64)
    k = np.sqrt(2.0 * np.maximum(s, 0))
    return eg_truncated_levy(k, B=B, C=C, alpha=alpha)


def gaver_stehfest_weights(N: int) -> np.ndarray:
    """Compute Gaver-Stehfest weights for numerical Laplace inversion."""
    if N % 2 != 0 or N < 2:
        raise ValueError("GS_N must be even and >= 2")
    V = np.zeros(N + 1, dtype=np.float64)
    n2 = N // 2
    for k in range(1, N + 1):
        s = 0.0
        jmin = (k + 1) // 2
        jmax = min(k, n2)
        for j in range(jmin, jmax + 1):
            num = (j**n2) * math.comb(2 * j, j) * math.comb(j, k - j)
            den = math.factorial(n2)
            s += num / den
        V[k] = s * ((-1) ** (k + n2))
    return V


def inv_laplace_pdf_gs(phi_func, x: np.ndarray, N: int, V: np.ndarray) -> np.ndarray:
    """Numerical Laplace inversion via the Gaver-Stehfest algorithm."""
    x = np.asarray(x, dtype=np.float64)
    ln2 = math.log(2.0)
    fx = np.zeros_like(x)
    for k in range(1, N + 1):
        fx += V[k] * phi_func((k * ln2) / x)
    fx *= ln2 / x
    fx = np.where(np.isfinite(fx) & (fx > 0), fx, 0.0)
    return fx


def build_sigma2_prior_tl(
    B: float = B_TL,
    C: float = C_TL,
    alpha: float = ALPHA_TL,
    xmin: float = SIG2_XMIN,
    xmax: float = SIG2_XMAX,
    ngrid: int = SIG2_NGRID,
    gs_n: int = GS_N,
) -> dict:
    """
    Build σ² grid and prior distribution from the truncated-Lévy CF.
    Uses Gaver-Stehfest numerical Laplace inversion.
    """
    V = gaver_stehfest_weights(gs_n)

    def _phi(s):
        return phi_sigma2_tl(s, B=B, C=C, alpha=alpha)

    # Log-spaced grid for σ²
    x = np.exp(np.linspace(np.log(xmin), np.log(xmax), ngrid)).astype(np.float64)
    pdf = inv_laplace_pdf_gs(_phi, x, N=gs_n, V=V)

    area = float(np.trapz(pdf, x))
    if area <= 0:
        # Fallback: uniform prior (if Laplace inversion fails)
        pdf = np.ones_like(x) / (xmax - xmin)
        area = 1.0
    pdf /= area

    # Quadrature weights for discrete integration
    dx = np.diff(x)
    w_node = np.empty_like(x)
    w_node[0] = dx[0] * 0.5
    w_node[-1] = dx[-1] * 0.5
    w_node[1:-1] = 0.5 * (dx[:-1] + dx[1:])

    mean_sig2 = float(np.trapz(pdf * x, x))

    return {
        "x": x,
        "pdf": pdf,
        "w_node": w_node,
        "log_pdf": np.log(np.maximum(pdf, 1e-300)),
        "log_x": np.log(x),
        "inv_x": 1.0 / x,
        "mean": max(mean_sig2, 1e-10),
    }


# =============================================================================
# EMPIRICAL CF PARAMETER ESTIMATION (α, B, C)
# =============================================================================


@dataclass
class CFParams:
    """Parametri stimati della CF truncated Lévy."""

    alpha: float
    B: float
    C: float
    alpha_stderr: float
    B_stderr: float
    C_stderr: float
    r2: float
    details: Dict


def compute_empirical_cf(returns: np.ndarray, k_grid: np.ndarray) -> np.ndarray:
    """Compute the empirical characteristic function CF(k) = E[exp(ikr)]."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = len(r)
    if n == 0:
        return np.zeros_like(k_grid, dtype=np.complex128)

    # CF empirica (parte reale per simmetria)
    cf = np.zeros(len(k_grid), dtype=np.complex128)
    for i, k in enumerate(k_grid):
        cf[i] = np.mean(np.exp(1j * k * r))

    return cf


def truncated_levy_cf_model(
    k: np.ndarray, alpha: float, B: float, C: float
) -> np.ndarray:
    """Truncated-Lévy characteristic function: ẽg(k) = exp(−Bk²/(1 + C|k|^{2−α}))."""
    k = np.asarray(k, dtype=np.float64)
    abs_k = np.abs(k) + 1e-18
    denom = 1.0 + C * np.power(abs_k, 2.0 - alpha)
    return np.exp(-(B * k * k) / denom)


def estimate_cf_params(
    returns: np.ndarray,
    k_max: float = 50.0,
    n_k: int = 200,
    alpha_init: float = 0.8,
    B_init: float = 6.5e-5,
    C_init: float = 1e-3,
) -> CFParams:
    """
    Estimate truncated-Lévy CF parameters (α, B, C) from the empirical CF.

    Method: nonlinear least squares minimisation of |CF_emp(k) − CF_model(k)|².

    Returns:
        CFParams with estimates and uncertainties.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]

    if len(r) < 100:
        return CFParams(
            alpha=np.nan,
            B=np.nan,
            C=np.nan,
            alpha_stderr=np.nan,
            B_stderr=np.nan,
            C_stderr=np.nan,
            r2=np.nan,
            details={"error": "insufficient_data"},
        )

    # Normalizza rendimenti per stabilità numerica
    r_std = np.std(r)
    if r_std <= 0:
        r_std = 1.0
    r_norm = r / r_std

    # k-grid (log-spacing captures both low and high frequency behaviour)
    k_grid = np.concatenate(
        [np.linspace(0.1, 5, n_k // 2), np.linspace(5, k_max, n_k // 2)]
    )

    # CF empirica
    cf_emp = compute_empirical_cf(r_norm, k_grid)
    cf_emp_real = np.real(cf_emp)

    # Pesi: più peso alle basse frequenze (più informative)
    weights = 1.0 / (1.0 + k_grid)

    # Funzione obiettivo
    def residuals(params):
        alpha, log_B, log_C = params
        B = np.exp(log_B)
        C = np.exp(log_C)

        # Vincoli
        if alpha <= 0 or alpha >= 2:
            return np.full(len(k_grid), 1e10)
        if B <= 0 or C <= 0:
            return np.full(len(k_grid), 1e10)

        cf_model = truncated_levy_cf_model(k_grid, alpha, B, C)
        return weights * (cf_emp_real - cf_model)

    # Initial guess (in log-space per B, C)
    x0 = [alpha_init, np.log(B_init), np.log(C_init)]

    result = None
    if SCIPY_AVAILABLE:
        try:
            from scipy.optimize import least_squares

            result = least_squares(
                residuals,
                x0,
                bounds=([0.01, -20, -20], [1.99, 5, 5]),
                method="trf",
                max_nfev=1000,
            )
        except Exception as e:
            _log(f"Warning: CF fit failed ({e})")

    if result is None or not result.success:
        # Fallback: usa valori PNAS
        return CFParams(
            alpha=ALPHA_TL,
            B=B_TL,
            C=C_TL,
            alpha_stderr=np.nan,
            B_stderr=np.nan,
            C_stderr=np.nan,
            r2=np.nan,
            details={"method": "fallback_pnas", "reason": "optimization_failed"},
        )

    alpha_fit = result.x[0]
    B_fit = np.exp(result.x[1]) * r_std**2  # De-normalizza
    C_fit = np.exp(result.x[2]) * r_std ** (2 - alpha_fit)

    # Compute R²
    cf_model_fit = truncated_levy_cf_model(
        k_grid, alpha_fit, np.exp(result.x[1]), np.exp(result.x[2])
    )
    ss_res = np.sum(weights**2 * (cf_emp_real - cf_model_fit) ** 2)
    ss_tot = np.sum(weights**2 * (cf_emp_real - np.mean(cf_emp_real)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + 1e-18)

    # Stima incertezze via Jacobiano (se disponibile)
    try:
        J = result.jac
        cov = np.linalg.inv(J.T @ J) * (ss_res / (len(k_grid) - 3))
        se = np.sqrt(np.diag(cov))
        alpha_se = se[0]
        B_se = B_fit * se[1]  # Delta method per log-transform
        C_se = C_fit * se[2]
    except:
        alpha_se = B_se = C_se = np.nan

    return CFParams(
        alpha=alpha_fit,
        B=B_fit,
        C=C_fit,
        alpha_stderr=alpha_se,
        B_stderr=B_se,
        C_stderr=C_se,
        r2=r2,
        details={
            "method": "least_squares",
            "n_obs": len(r),
            "r_std": r_std,
            "k_max": k_max,
            "n_k": n_k,
            "pnas_reference": {"alpha": ALPHA_TL, "B": B_TL, "C": C_TL},
        },
    )


def plot_cf_fit(outdir, returns: np.ndarray, cf_params: CFParams):
    """Plot empirical vs. fitted truncated-Lévy characteristic function."""
    figdir = os.path.join(outdir, "fig")
    _ensure_dir(figdir)

    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    r_std = np.std(r) if np.std(r) > 0 else 1.0
    r_norm = r / r_std

    # Griglia k
    k_grid = np.linspace(0.1, 50, 300)

    # CF empirica
    cf_emp = compute_empirical_cf(r_norm, k_grid)
    cf_emp_real = np.real(cf_emp)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: empirical vs model CF (linear scale)
    ax = axes[0, 0]
    ax.plot(k_grid, cf_emp_real, "b-", linewidth=1.5, alpha=0.7, label="CF empirica")

    if np.isfinite(cf_params.alpha):
        # Modello con parametri stimati
        cf_fit = truncated_levy_cf_model(
            k_grid,
            cf_params.alpha,
            cf_params.B / r_std**2,
            cf_params.C / r_std ** (2 - cf_params.alpha),
        )
        ax.plot(
            k_grid,
            cf_fit,
            "r--",
            linewidth=2,
            label=f"Fit: α={cf_params.alpha:.3f}, B={cf_params.B:.2e}, C={cf_params.C:.2e}",
        )

    # Modello PNAS reference
    cf_pnas = truncated_levy_cf_model(
        k_grid, ALPHA_TL, B_TL / r_std**2, C_TL / r_std ** (2 - ALPHA_TL)
    )
    ax.plot(
        k_grid,
        cf_pnas,
        "g:",
        linewidth=1.5,
        alpha=0.7,
        label=f"PNAS: α={ALPHA_TL}, B={B_TL:.2e}, C={C_TL:.2e}",
    )

    ax.set_xlabel("k", fontsize=12)
    ax.set_ylabel("Re[CF(k)]", fontsize=12)
    ax.set_title("Funzione Caratteristica: Empirica vs Modello", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 50)

    # Panel 2: CF in log-log scale (tail structure)
    ax = axes[0, 1]
    log_cf_emp = -np.log(np.maximum(cf_emp_real, 1e-10))
    ax.loglog(
        k_grid, log_cf_emp, "b-", linewidth=1.5, alpha=0.7, label="−log(CF) empirica"
    )

    if np.isfinite(cf_params.alpha):
        log_cf_fit = -np.log(np.maximum(cf_fit, 1e-10))
        ax.loglog(k_grid, log_cf_fit, "r--", linewidth=2, label="Fit")

    ax.set_xlabel("k", fontsize=12)
    ax.set_ylabel("−log(CF(k))", fontsize=12)
    ax.set_title("CF in scala log-log", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    # Panel 3: residuals
    ax = axes[1, 0]
    if np.isfinite(cf_params.alpha):
        residuals = cf_emp_real - cf_fit
        ax.plot(k_grid, residuals, "k-", linewidth=1, alpha=0.7)
        ax.axhline(0, color="r", linestyle="--", alpha=0.5)
        ax.fill_between(k_grid, -0.1, 0.1, alpha=0.1, color="green")
    ax.set_xlabel("k", fontsize=12)
    ax.set_ylabel("Residui", fontsize=12)
    ax.set_title("Residui del fit", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel 4: parameter summary
    ax = axes[1, 1]
    ax.axis("off")

    info_text = "PARAMETRI CF TRUNCATED LÉVY\n"
    info_text += "=" * 40 + "\n\n"
    info_text += f"Modello: ẽg(k) = exp(-B·k² / (1 + C·|k|^{{2-α}}))\n\n"

    info_text += "STIMA EMPIRICA:\n"
    if np.isfinite(cf_params.alpha):
        info_text += f"  α = {cf_params.alpha:.4f}"
        if np.isfinite(cf_params.alpha_stderr):
            info_text += f" ± {cf_params.alpha_stderr:.4f}"
        info_text += "\n"

        info_text += f"  B = {cf_params.B:.2e}"
        if np.isfinite(cf_params.B_stderr):
            info_text += f" ± {cf_params.B_stderr:.2e}"
        info_text += "\n"

        info_text += f"  C = {cf_params.C:.2e}"
        if np.isfinite(cf_params.C_stderr):
            info_text += f" ± {cf_params.C_stderr:.2e}"
        info_text += "\n"

        info_text += f"\n  R² = {cf_params.r2:.4f}\n"
    else:
        info_text += "  (fit non riuscito)\n"

    info_text += "\nRIFERIMENTO PNAS (DJI 1900-2005):\n"
    info_text += f"  α = {ALPHA_TL}\n"
    info_text += f"  B = {B_TL:.2e}\n"
    info_text += f"  C = {C_TL:.2e}\n"

    ax.text(
        0.1,
        0.9,
        info_text,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    _savefig(os.path.join(figdir, "cf_truncated_levy_fit.png"))


def sample_sigma2_from_prior_tl(
    rng: np.random.Generator, pre: dict, size: int
) -> np.ndarray:
    """Sample σ² from the truncated-Lévy mixing prior."""
    x = pre["x"]
    pdf = pre["pdf"]
    w_node = pre["w_node"]
    pmf = pdf * w_node
    cdf = np.cumsum(pmf)
    cdf /= cdf[-1]
    u = rng.random(size=size, dtype=np.float64)
    return np.interp(u, cdf, x)


def sample_sigma2_posterior_tl(
    rng: np.random.Generator, pre: dict, p: int, Q: float
) -> float:
    """
    Sample σ² from the posterior conditioned on a window of p observations.

    Likelihood for p points:
    ∏ N(r_j; 0, σ² w_j) → logL(σ²) = −p/2 · log(σ²) − Q/(2σ²)
    where Q = Σ r_j²/w_j

    Posterior(σ²) ∝ prior(σ²) · (σ²)^{−p/2} · exp(−Q/(2σ²))
    """
    x = pre["x"]
    log_pdf = pre["log_pdf"]
    log_x = pre["log_x"]
    inv_x = pre["inv_x"]
    w_node = pre["w_node"]

    # Unnormalized log-posterior
    logpost = log_pdf - 0.5 * p * log_x - 0.5 * Q * inv_x

    m = np.max(logpost)
    post = np.exp(logpost - m) * w_node

    cdf = np.cumsum(post)
    tot = cdf[-1]
    if not np.isfinite(tot) or tot <= 0:
        return float(sample_sigma2_from_prior_tl(rng, pre, size=1)[0])

    cdf /= tot
    u = float(rng.random())
    return float(np.interp(u, cdf, x))


# =============================================================================
# THREE PNAS-FAITHFUL SIMULATION METHODS
# =============================================================================


def simulate_blocks_tl_joint(
    D: float, rng: np.random.Generator, pre: dict, tau_c: int = TAU_C, tmax: int = TMAX
) -> np.ndarray:
    """
    (1) TL-JOINT: σ² campionato una volta per blocco, poi Gaussiane indipendenti.

    Variance mixture con σ² dalla distribuzione di mixing ottenuta per
    Gaver-Stehfest Laplace inversion of the truncated-Lévy CF.
    """
    w = weights_w_pnas(D, tau_c)
    n_blocks = tmax // tau_c

    sig2 = sample_sigma2_from_prior_tl(rng, pre, size=n_blocks).reshape(n_blocks, 1)

    # Normalise by prior mean
    if pre["mean"] > 0:
        sig2 = sig2 / pre["mean"]

    z = rng.standard_normal(size=(n_blocks, tau_c)).astype(np.float64)
    r = z * np.sqrt(sig2) * np.sqrt(w)[None, :]
    return r


def simulate_blocks_tl_ar(
    D: float,
    rng: np.random.Generator,
    pre: dict,
    m: int = M_AR,
    tau_c: int = TAU_C,
    tmax: int = TMAX,
) -> np.ndarray:
    """
    (2) TL-AR: Procedura autoregressiva con VARIANZA CONDIZIONALE.

    Dal paper e201230134y.pdf, la varianza condizionale nel framework Ba-St è:

    var(r_n | r_0,...,r_{n-1}) = w_n × [ν + Σᵢ r²ᵢ/wᵢ] / (ν + n - 1)

    Dove:
    - w_n = (n+1)^{2D} − n^{2D} is the PNAS weight
    - ν  is an effective degrees-of-freedom parameter (proxied by NU_STUDENT)
    - Σᵢ rᵢ²/wᵢ  is the normalised sum of squares

    DIFFERENZA CHIAVE vs TL-JOINT:
    - TL-JOINT: σ² constant within block → dependence only from w_i
    - TL-AR:    σ² evolves within block based on history → volatility clustering

    Questo produce una curva β(D) simile a TL-JOINT ma con effetti di
    "reinforcement" della volatilità.
    """
    w = weights_w_pnas(D, tau_c)
    n_blocks = tmax // tau_c
    r_blocks = np.empty((n_blocks, tau_c), dtype=np.float64)

    # Effective degrees-of-freedom parameter
    nu_eff = NU_STUDENT

    # Base scale from the truncated-Lévy prior
    sig2_prior_mean = pre["mean"] if pre["mean"] > 0 else 1.0

    for b in range(n_blocks):
        # Draw a block-level σ²_base from the prior
        sig2_base = (
            float(sample_sigma2_from_prior_tl(rng, pre, size=1)[0]) / sig2_prior_mean
        )

        # Somma quadrati normalizzati (accumulata dentro il blocco)
        Q = 0.0

        for t in range(tau_c):
            # Number of returns already generated in this block
            n_past = t

            # Conditional variance: Baldovin-Stella formula
            # var_t = w_t × sig2_base × [nu + Q] / [nu + n_past]
            # Ma per evitare divisione per zero e instabilità:
            if n_past < 5:
                # Primi passi: usa solo σ²_base × w_t
                var_t = w[t] * sig2_base
            else:
                # Formula condizionale completa
                # Numerator (ν + Q) grows with history; denominator (ν + n) normalises
                var_t = w[t] * sig2_base * (nu_eff + Q) / (nu_eff + n_past - 1)

            # Draw return
            rt = math.sqrt(max(var_t, 1e-20)) * float(rng.standard_normal())
            r_blocks[b, t] = rt

            # Update normalised sum of squares
            # Normalised squared contribution
            if w[t] > 0 and sig2_base > 0:
                Q += (rt * rt) / (w[t] * sig2_base)

    return r_blocks


def simulate_blocks_student(
    D: float,
    rng: np.random.Generator,
    nu: float = NU_STUDENT,
    tau_c: int = TAU_C,
    tmax: int = TMAX,
) -> np.ndarray:
    """
    (3) Student benchmark: variance-mixture con σ² = ν/χ²(ν).

    Benchmark semplice, non usa la CF truncated Lévy.
    """
    w = weights_w_pnas(D, tau_c)
    n_blocks = tmax // tau_c

    u = rng.chisquare(df=nu, size=(n_blocks, 1)).astype(np.float64)
    sig2 = nu / u

    z = rng.standard_normal(size=(n_blocks, tau_c)).astype(np.float64)
    r = z * np.sqrt(sig2) * np.sqrt(w)[None, :]
    return r


def block_corr_abs_pnas(r_blocks: np.ndarray, max_lag: int) -> np.ndarray:
    """
    ACF di |r| calcolata dentro-blocco e mediata (coerente con PNAS/SI).
    """
    abs_r = np.abs(r_blocks).astype(np.float64)
    mean_abs = abs_r.mean()
    var_abs = abs_r.var(ddof=0)
    if var_abs <= 0:
        return np.zeros(max_lag + 1, dtype=np.float64)

    c = np.empty(max_lag + 1, dtype=np.float64)
    c[0] = 1.0
    for tau in range(1, max_lag + 1):
        prod = abs_r[:, :-tau] * abs_r[:, tau:]
        cov = prod.mean() - mean_abs * mean_abs
        c[tau] = cov / var_abs
    return c


def fit_beta_from_acf_pnas(
    acf: np.ndarray, lag_min: int = 5, lag_max: int = 100
) -> float:
    """Estimate β from simulated ACF via simple log-log regression."""
    taus = np.arange(1, acf.size, dtype=np.float64)
    y = acf[1:]
    m = (taus >= lag_min) & (taus <= lag_max) & np.isfinite(y) & (y > 0)
    if m.sum() < 10:
        return np.nan
    x = np.log(taus[m])
    yy = np.log(y[m])
    slope, _ = np.polyfit(x, yy, 1)
    return float(-slope)


def summarize_betas_pnas(betas: list) -> Tuple[float, float, float, int]:
    """Compute mean, std, sem, and count from a list of β values."""
    arr = np.array([b for b in betas if np.isfinite(b)], dtype=np.float64)
    n = int(arr.size)
    if n == 0:
        return (np.nan, np.nan, np.nan, 0)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    sem = float(std / math.sqrt(n)) if n > 1 else 0.0
    return (mean, std, sem, n)


# =============================================================================
# AR SIMULATION AND CALIBRATION (with complete diagnostics)
# =============================================================================


def _ar_coeffs_from_De(De, L):
    """Compute AR coefficients for the BS autoregressive model, with caching."""
    cache_key = (round(De, 6), L)
    if cache_key in _AR_COEFF_CACHE:
        return _AR_COEFF_CACHE[cache_key]
    i = np.arange(1, L + 1, dtype=np.float64)
    if De <= 0:
        a = np.ones(L, dtype=np.float64)
    else:
        a = (np.power(i, 2.0 * De) - np.power(i - 1.0, 2.0 * De)) ** (1.0 / (2.0 * De))
    norm = np.sqrt(np.sum(a * a))
    result = a / norm if norm > 0 else a
    _AR_COEFF_CACHE[cache_key] = result
    return result


def simulate_ar_returns_from_eps(De, eps, L=128, burn_in=500):
    """Simulate an AR series with burn-in (see Baldovin & Stella SI Text)."""
    a = _ar_coeffs_from_De(De, L)
    out = np.convolve(np.abs(eps), a, mode="full")
    start_idx = L + burn_in
    return out[start_idx:] if len(out) > start_idx else out[L:] if len(out) > L else out


def smooth_sigma(sigma, window=5):
    """Moving-average smoothing of the simulated sigma series."""
    if uniform_filter1d is not None:
        return uniform_filter1d(sigma, size=window, mode="nearest")
    n = len(sigma)
    smoothed = np.copy(sigma)
    half = window // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        vals = sigma[lo:hi]
        vals = vals[np.isfinite(vals)]
        if len(vals) > 0:
            smoothed[i] = np.mean(vals)
    return smoothed


def prepare_beta_sim_curve_coherent(
    n_series: int,
    lag_min: int,
    lag_max: int,
    nbins: int,
    min_acf_threshold: float,
    fit_method: str,
    min_bins_required: int,
    aggregation: str,
    L: int = 128,
    De_grid: Optional[np.ndarray] = None,
    reps: int = 50,
    seed: int = 1234,
    burn_in: int = 500,
    pass_rate_threshold: float = 0.8,
) -> BetaSimCurve:
    """
    Build the β_sim(D_e) calibration curve using the identical estimator as the empirical fit.

    Uses the same core estimator as the empirical β fit for full coherence.

    Parameters that must be identical to the empirical estimation for coherent inversion:
    - min_acf_threshold
    - min_bins_required
    - fit_method
    - aggregation
    - lag_min, lag_max, nbins

    Diagnostics for each D_e grid point:
    - pass_rate: fraction of replicates that satisfy the minimum quality criteria
    - median_nbins_valid: mediana bin validi
    - fraction_acf_above_threshold: frazione ACF sopra soglia
    - fraction_fit_success: frazione fit riusciti
    - breakdown motivi fallimento

    Selection bias handling:
    - β ≤ 0 is not silently discarded
    - It is recorded as a diagnostic failure
    - sel_bias_warn is raised when pass_rate < threshold
    """
    rng = np.random.default_rng(seed)

    if De_grid is None:
        De_grid = np.linspace(0.05, 0.48, 80)

    n_De = len(De_grid)
    total_len = n_series + L + burn_in + 100

    # Common random numbers: same innovations for all D_e values
    eps_matrix = rng.standard_normal((reps, total_len))

    # Storage arrays
    beta_all = np.full((n_De, reps), np.nan)
    diagnostics_list = []

    # Parameters saved for coherence verification
    params = {
        "n_series": n_series,
        "lag_min": lag_min,
        "lag_max": lag_max,
        "nbins": nbins,
        "min_acf_threshold": min_acf_threshold,
        "fit_method": fit_method,
        "min_bins_required": min_bins_required,
        "aggregation": aggregation,
        "L": L,
        "burn_in": burn_in,
        "reps": reps,
        "seed": seed,
    }

    for i, De in enumerate(De_grid):
        n_fit_fail = 0
        n_bins_insufficient = 0
        n_beta_invalid = 0
        n_pass = 0
        nbins_valid_list = []
        frac_acf_above_list = []

        for r in range(reps):
            # Simula serie
            series = simulate_ar_returns_from_eps(De, eps_matrix[r], L, burn_in)
            if len(series) > n_series:
                series = series[:n_series]

            # USA IL CORE ESTIMATOR IDENTICO ALL'EMPIRICO
            core_result = estimate_beta_acf_core(
                series,
                lag_min,
                lag_max,
                nbins,
                min_acf_threshold,
                fit_method,
                min_bins_required,
                aggregation,
            )

            # Record diagnostics
            nbins_valid_list.append(core_result.nbins_valid)
            if core_result.n_acf_total > 0:
                frac_acf_above_list.append(
                    core_result.n_acf_above_threshold / core_result.n_acf_total
                )

            # Analizza risultato
            if core_result.success:
                beta_all[i, r] = core_result.beta
                n_pass += 1
            else:
                # Record failure reason (no silent discarding)
                if core_result.fail_reason == "too_few_valid_bins":
                    n_bins_insufficient += 1
                elif core_result.fail_reason == "beta_not_positive":
                    n_beta_invalid += 1
                    beta_all[i, r] = core_result.beta  # Store the value anyway for diagnostics
                else:
                    n_fit_fail += 1

        # Compute diagnostics for this D_e grid point
        pass_rate = n_pass / reps if reps > 0 else 0.0
        median_nbins = float(np.median(nbins_valid_list)) if nbins_valid_list else 0.0
        mean_frac_acf = (
            float(np.mean(frac_acf_above_list)) if frac_acf_above_list else 0.0
        )
        sel_bias_warn = pass_rate < pass_rate_threshold

        diag = DePointDiagnostics(
            De=De,
            pass_rate=pass_rate,
            n_total=reps,
            n_pass=n_pass,
            n_fit_fail=n_fit_fail,
            n_bins_insufficient=n_bins_insufficient,
            n_beta_invalid=n_beta_invalid,
            median_nbins_valid=median_nbins,
            mean_frac_acf_above_threshold=mean_frac_acf,
            beta_values=beta_all[i, :].copy(),
            sel_bias_warn=sel_bias_warn,
        )
        diagnostics_list.append(diag)

    # Compute mean and std (using all valid values, not only successes)
    beta_mean = np.full(n_De, np.nan)
    beta_std = np.full(n_De, np.nan)

    for i in range(n_De):
        # Use only β > 0 for the mean (negatives already recorded in diagnostics)
        valid = beta_all[i, :]
        valid = valid[np.isfinite(valid) & (valid > 0)]
        if len(valid) >= 5:
            # Trimmed mean per robustezza
            valid_sorted = np.sort(valid)
            trim = max(1, len(valid) // 10)
            trimmed = (
                valid_sorted[trim:-trim] if 2 * trim < len(valid) else valid_sorted
            )
            beta_mean[i] = np.mean(trimmed)
            beta_std[i] = np.std(trimmed, ddof=1)

    beta_std_smooth = smooth_sigma(beta_std, window=5)

    # PCHIP spline
    spline = None
    if SCIPY_AVAILABLE and PchipInterpolator is not None:
        valid_idx = np.isfinite(beta_mean)
        if np.sum(valid_idx) >= 4:
            try:
                spline = PchipInterpolator(De_grid[valid_idx], beta_mean[valid_idx])
            except:
                pass

    # Numerical derivative
    derivative = np.full(n_De, np.nan)
    for i in range(1, n_De - 1):
        if np.isfinite(beta_mean[i - 1]) and np.isfinite(beta_mean[i + 1]):
            derivative[i] = (beta_mean[i + 1] - beta_mean[i - 1]) / (
                De_grid[i + 1] - De_grid[i - 1]
            )

    # Summary details for output
    details = {
        "method": f"ACF_core_{fit_method}",
        "crn": True,
        "n_unreliable_points": sum(1 for d in diagnostics_list if d.sel_bias_warn),
        "overall_pass_rate": float(np.mean([d.pass_rate for d in diagnostics_list])),
    }

    return BetaSimCurve(
        De_grid=De_grid,
        beta_mean=beta_mean,
        beta_std=beta_std,
        beta_std_smooth=beta_std_smooth,
        spline=spline,
        derivative=derivative,
        diagnostics=diagnostics_list,
        params=params,
        details=details,
    )


def verify_coherence(
    empirical_params: Dict, curve: BetaSimCurve
) -> Tuple[bool, List[str]]:
    """
    Verify that empirical and simulated estimator parameters are identical.
    """
    mismatches = []
    curve_params = curve.params

    for key in [
        "lag_min",
        "lag_max",
        "nbins",
        "min_acf_threshold",
        "fit_method",
        "min_bins_required",
        "aggregation",
    ]:
        if key in empirical_params and key in curve_params:
            if empirical_params[key] != curve_params[key]:
                mismatches.append(
                    f"{key}: emp={empirical_params[key]} vs sim={curve_params[key]}"
                )

    return len(mismatches) == 0, mismatches


# =============================================================================
# THREE PNAS-FAITHFUL CALIBRATION CURVES β(D)
# =============================================================================


@dataclass
class MultiCurveResult:
    """Risultato con tutte e tre le curve di calibrazione PNAS."""

    tl_joint: BetaSimCurve
    tl_ar: BetaSimCurve
    student: BetaSimCurve
    params: Dict


def prepare_multi_beta_curves_pnas(
    De_grid: np.ndarray,
    n_repl_joint: int = 30,
    n_repl_ar: int = 15,
    n_repl_student: int = 25,
    max_lag: int = 120,
    lag_min: int = 5,
    lag_max_fit: int = 100,
    seed: int = 123456,
    tau_c: int = TAU_C,
    tmax: int = TMAX,
    m_ar: int = M_AR,
) -> MultiCurveResult:
    """
    Build three PNAS-faithful β(D) calibration curves:
    1) TL-JOINT: mixing distribution (σ² per blocco)
    2) TL-AR(m=100): procedura autoregressiva PNAS con posterior
    3) Student benchmark: variance-mixture Student-t

    Riferimenti:
    - Baldovin & Stella, PNAS 104(50), 19741–19746 (2007)
    - SI Text: "Scaling and efficiency determine..."
    - Sokolov et al., Physica A 336, 245-251 (2004)
    """
    _log("Precomputo prior mixing distribution σ² (truncated Lévy)...")
    try:
        pre = build_sigma2_prior_tl(B=B_TL, C=C_TL, alpha=ALPHA_TL)
    except Exception as e:
        _log(f"WARNING: Inversione Laplace fallita ({e}), uso prior uniforme")
        pre = None

    n_De = len(De_grid)

    # Storage for three simulation methods
    beta_joint = np.full(n_De, np.nan)
    std_joint = np.full(n_De, np.nan)
    sem_joint = np.full(n_De, np.nan)

    beta_ar = np.full(n_De, np.nan)
    std_ar = np.full(n_De, np.nan)
    sem_ar = np.full(n_De, np.nan)

    beta_student = np.full(n_De, np.nan)
    std_student = np.full(n_De, np.nan)
    sem_student = np.full(n_De, np.nan)

    _log(f"Calcolo curve β(D) PNAS su {n_De} punti...")

    for i, D in enumerate(De_grid):
        if (i + 1) % 10 == 0 or i == 0:
            _log(f"  D = {D:.3f} ({i+1}/{n_De})")

        rng = np.random.default_rng(seed + 1000 * i)

        betas_j, betas_a, betas_s = [], [], []

        # TL-JOINT
        if pre is not None:
            for _ in range(n_repl_joint):
                try:
                    rb = simulate_blocks_tl_joint(D, rng, pre, tau_c=tau_c, tmax=tmax)
                    acf = block_corr_abs_pnas(rb, max_lag)
                    b = fit_beta_from_acf_pnas(acf, lag_min, lag_max_fit)
                    betas_j.append(b)
                except:
                    pass

        # TL-AR(m)
        if pre is not None:
            for _ in range(n_repl_ar):
                try:
                    rb = simulate_blocks_tl_ar(
                        D, rng, pre, m=m_ar, tau_c=tau_c, tmax=tmax
                    )
                    acf = block_corr_abs_pnas(rb, max_lag)
                    b = fit_beta_from_acf_pnas(acf, lag_min, lag_max_fit)
                    betas_a.append(b)
                except:
                    pass

        # Student benchmark
        for _ in range(n_repl_student):
            try:
                rb = simulate_blocks_student(
                    D, rng, nu=NU_STUDENT, tau_c=tau_c, tmax=tmax
                )
                acf = block_corr_abs_pnas(rb, max_lag)
                b = fit_beta_from_acf_pnas(acf, lag_min, lag_max_fit)
                betas_s.append(b)
            except:
                pass

        # Summarise
        mj, sj, ej, nj = summarize_betas_pnas(betas_j)
        ma, sa, ea, na = summarize_betas_pnas(betas_a)
        ms, ss, es, ns = summarize_betas_pnas(betas_s)

        beta_joint[i], std_joint[i], sem_joint[i] = mj, sj, ej
        beta_ar[i], std_ar[i], sem_ar[i] = ma, sa, ea
        beta_student[i], std_student[i], sem_student[i] = ms, ss, es

    # Build PCHIP spline for each curve
    def make_spline(De, beta):
        if SCIPY_AVAILABLE and PchipInterpolator is not None:
            valid = np.isfinite(beta)
            if np.sum(valid) >= 4:
                try:
                    return PchipInterpolator(De[valid], beta[valid])
                except:
                    pass
        return None

    # Derivate numeriche
    def compute_derivative(De, beta):
        deriv = np.full_like(beta, np.nan)
        for i in range(1, len(De) - 1):
            if np.isfinite(beta[i - 1]) and np.isfinite(beta[i + 1]):
                deriv[i] = (beta[i + 1] - beta[i - 1]) / (De[i + 1] - De[i - 1])
        return deriv

    params = {
        "alpha": ALPHA_TL,
        "B": B_TL,
        "C": C_TL,
        "tau_c": tau_c,
        "tmax": tmax,
        "m_ar": m_ar,
        "max_lag": max_lag,
        "lag_min": lag_min,
        "lag_max_fit": lag_max_fit,
        "n_repl_joint": n_repl_joint,
        "n_repl_ar": n_repl_ar,
        "n_repl_student": n_repl_student,
        "seed": seed,
        "nu_student": NU_STUDENT,
    }

    # Curve TL-JOINT
    curve_joint = BetaSimCurve(
        De_grid=De_grid.copy(),
        beta_mean=beta_joint,
        beta_std=std_joint,
        beta_std_smooth=smooth_sigma(std_joint, window=5),
        spline=make_spline(De_grid, beta_joint),
        derivative=compute_derivative(De_grid, beta_joint),
        diagnostics=[],
        params=params.copy(),
        details={
            "method": "TL_JOINT",
            "description": "Truncated Lévy mixing per blocco",
        },
    )

    # Curve TL-AR
    curve_ar = BetaSimCurve(
        De_grid=De_grid.copy(),
        beta_mean=beta_ar,
        beta_std=std_ar,
        beta_std_smooth=smooth_sigma(std_ar, window=5),
        spline=make_spline(De_grid, beta_ar),
        derivative=compute_derivative(De_grid, beta_ar),
        diagnostics=[],
        params=params.copy(),
        details={
            "method": "TL_AR",
            "description": f"Truncated Lévy autoregressivo PNAS (m={m_ar})",
        },
    )

    # Curve Student
    curve_student = BetaSimCurve(
        De_grid=De_grid.copy(),
        beta_mean=beta_student,
        beta_std=std_student,
        beta_std_smooth=smooth_sigma(std_student, window=5),
        spline=make_spline(De_grid, beta_student),
        derivative=compute_derivative(De_grid, beta_student),
        diagnostics=[],
        params=params.copy(),
        details={
            "method": "Student",
            "description": f"Student benchmark (ν={NU_STUDENT})",
        },
    )

    return MultiCurveResult(
        tl_joint=curve_joint, tl_ar=curve_ar, student=curve_student, params=params
    )


# =============================================================================
# D_e INVERSION
# =============================================================================


def calibrate_De_via_spline(beta_emp, curve):
    """Invert the calibration spline to obtain D_e from an empirical β."""
    if not np.isfinite(beta_emp):
        return np.nan, {"reason": "beta_emp_nan"}
    if curve.spline is None:
        valid = np.isfinite(curve.beta_mean)
        if not np.any(valid):
            return np.nan, {"reason": "no_valid_curve"}
        De_v, beta_v = curve.De_grid[valid], curve.beta_mean[valid]
        err = np.abs(beta_v - beta_emp)
        i0 = np.argmin(err)
        return float(De_v[i0]), {"method": "nearest", "error": float(err[i0])}
    if minimize_scalar is not None:
        try:
            result = minimize_scalar(
                lambda De: (curve.spline(De) - beta_emp) ** 2,
                bounds=(curve.De_grid.min(), curve.De_grid.max()),
                method="bounded",
            )
            if result.success:
                return float(result.x), {
                    "method": "spline_inversion",
                    "error": float(np.sqrt(result.fun)),
                }
        except:
            pass
    beta_smooth = curve.spline(curve.De_grid)
    err = np.abs(beta_smooth - beta_emp)
    i0 = np.argmin(err)
    return float(curve.De_grid[i0]), {
        "method": "spline_nearest",
        "error": float(err[i0]),
    }


def bayesian_De_posterior(
    beta_emp, beta_stderr, curve, prior_mean=None, prior_std=None
):
    """Bayesian inversion: compute posterior p(D_e | β_emp)."""
    De_grid = curve.De_grid
    m_De = curve.beta_mean
    s_sim = curve.beta_std_smooth
    valid = np.isfinite(m_De) & np.isfinite(s_sim)
    if not np.any(valid) or not np.isfinite(beta_emp):
        return np.nan, (np.nan, np.nan), np.array([]), np.array([]), np.array([])
    s_emp = beta_stderr if np.isfinite(beta_stderr) else 0.05

    if prior_mean is not None and prior_std is not None and stats is not None:
        prior = stats.norm(loc=prior_mean, scale=prior_std).pdf(De_grid)
    else:
        prior = np.ones_like(De_grid)
    prior = prior / (np.sum(prior) + 1e-18)

    var_total = s_emp**2 + s_sim**2 + 1e-10
    log_lik = np.full_like(De_grid, -np.inf)
    log_lik[valid] = -0.5 * (beta_emp - m_De[valid]) ** 2 / var_total[
        valid
    ] - 0.5 * np.log(var_total[valid])
    log_p = log_lik + np.log(prior + 1e-18)
    log_p -= np.max(log_p[np.isfinite(log_p)])
    p = np.exp(log_p)
    p /= np.sum(p) + 1e-18

    De_hat = float(np.sum(De_grid * p))
    cumsum = np.cumsum(p)
    try:
        ci_low = float(De_grid[np.searchsorted(cumsum, 0.025)])
        ci_high = float(De_grid[np.searchsorted(cumsum, 0.975)])
    except:
        ci_low, ci_high = np.nan, np.nan
    return De_hat, (ci_low, ci_high), De_grid, p, prior


def estimate_De_from_beta(beta_est, curve, use_bayesian=True):
    """Estimate D_e from an empirical β estimate."""
    if not np.isfinite(beta_est.beta):
        return DeEstimate(np.nan, np.nan, (np.nan, np.nan), {"reason": "beta_nan"})
    if use_bayesian and np.isfinite(beta_est.stderr):
        De_hat, ci, De_grid, posterior, prior = bayesian_De_posterior(
            beta_est.beta, beta_est.stderr, curve
        )
        if np.isfinite(De_hat):
            stderr = float(np.sqrt(np.sum(De_grid**2 * posterior) - De_hat**2))
            return DeEstimate(
                De_hat,
                stderr,
                ci,
                {"method": "bayesian_inversion", "beta_emp": beta_est.beta},
            )
    De_star, det = calibrate_De_via_spline(beta_est.beta, curve)
    if not np.isfinite(De_star):
        return DeEstimate(
            np.nan, np.nan, (np.nan, np.nan), {"reason": "calibration_failed"}
        )
    err = np.abs(curve.beta_mean - beta_est.beta)
    err = np.where(np.isfinite(err), err, np.inf)
    i0 = int(np.argmin(err))
    d_beta_d_De = curve.derivative[i0] if np.isfinite(curve.derivative[i0]) else np.nan
    stderr = (
        abs(beta_est.stderr / (d_beta_d_De + 1e-18))
        if np.isfinite(beta_est.stderr) and np.isfinite(d_beta_d_De)
        else np.nan
    )
    ci = (
        (De_star - CI * stderr, De_star + CI * stderr)
        if np.isfinite(stderr)
        else (np.nan, np.nan)
    )
    return DeEstimate(
        De_star, stderr, ci, {"method": "spline_inversion", "d_beta_d_De": d_beta_d_De}
    )


# =============================================================================
# MAPPING D_e MULTIPLO SU TUTTE LE CURVE (V3.5)
# =============================================================================


@dataclass
class MultiDeResult:
    """D_e stimato su tutte e tre le curve per un singolo metodo β."""

    beta_method: str
    beta_value: float
    beta_stderr: float
    De_tl_joint: DeEstimate
    De_tl_ar: DeEstimate
    De_student: DeEstimate


def estimate_De_multi_curves(
    beta_est: BetaEstimate, multi_curves: MultiCurveResult, beta_method_name: str
) -> MultiDeResult:
    """
    Compute D_e on all three calibration curves for a given β method.
    """
    De_joint = estimate_De_from_beta(beta_est, multi_curves.tl_joint)
    De_ar = estimate_De_from_beta(beta_est, multi_curves.tl_ar)
    De_student = estimate_De_from_beta(beta_est, multi_curves.student)

    return MultiDeResult(
        beta_method=beta_method_name,
        beta_value=beta_est.beta,
        beta_stderr=beta_est.stderr,
        De_tl_joint=De_joint,
        De_tl_ar=De_ar,
        De_student=De_student,
    )


def estimate_all_De_combinations(
    beta_estimates: Dict[str, BetaEstimate], multi_curves: MultiCurveResult
) -> Dict[str, MultiDeResult]:
    """
    Compute D_e for every (β-method, calibration curve) combination.
    """
    results = {}
    for name, beta_est in beta_estimates.items():
        results[name] = estimate_De_multi_curves(beta_est, multi_curves, name)
    return results


# =============================================================================
# STIMA FINALE PESATA (V3.5)
# =============================================================================


@dataclass
class WeightedEstimate:
    """Stima pesata con metadati."""

    value: float
    stderr: float
    ci: Tuple[float, float]
    methods_used: List[str]
    weights: Dict[str, float]
    individual_values: Dict[str, float]


def compute_weighted_mean(
    values: Dict[str, float], stderrs: Dict[str, float], selected_keys: List[str] = None
) -> WeightedEstimate:
    """
    Compute inverse-variance weighted mean of β and D_e estimates.

    Args:
        values: dict mapping name → value
        stderrs: dizionario {nome: stderr}
        selected_keys: list of keys to include (None = all)
    """
    if selected_keys is None:
        selected_keys = list(values.keys())

    # Filter to finite values
    valid_keys = []
    for k in selected_keys:
        if k in values and k in stderrs:
            if np.isfinite(values[k]) and np.isfinite(stderrs[k]) and stderrs[k] > 0:
                valid_keys.append(k)

    if len(valid_keys) == 0:
        return WeightedEstimate(
            value=np.nan,
            stderr=np.nan,
            ci=(np.nan, np.nan),
            methods_used=[],
            weights={},
            individual_values={},
        )

    # Inverse-variance weights
    weights = {}
    for k in valid_keys:
        weights[k] = 1.0 / (stderrs[k] ** 2)

    # Normalizza pesi
    w_sum = sum(weights.values())
    for k in weights:
        weights[k] /= w_sum

    # Media pesata
    mean = sum(weights[k] * values[k] for k in valid_keys)

    # Varianza: σ²_combined = 1 / Σ(1/σ²_i)
    var_combined = 1.0 / sum(1.0 / (stderrs[k] ** 2) for k in valid_keys)
    stderr = math.sqrt(var_combined)

    ci = (mean - CI * stderr, mean + CI * stderr)

    return WeightedEstimate(
        value=mean,
        stderr=stderr,
        ci=ci,
        methods_used=valid_keys,
        weights=weights,
        individual_values={k: values[k] for k in valid_keys},
    )


def compute_final_beta_estimate(
    beta_estimates: Dict[str, BetaEstimate], selected_methods: List[str] = None
) -> WeightedEstimate:
    """
Inverse-variance weighted combination of β estimates across selected methods.
    """
    values = {k: est.beta for k, est in beta_estimates.items()}
    stderrs = {k: est.stderr for k, est in beta_estimates.items()}
    return compute_weighted_mean(values, stderrs, selected_methods)


def compute_final_De_estimate(
    De_multi_results: Dict[str, MultiDeResult],
    selected_beta_methods: List[str] = None,
    selected_curves: List[str] = None,
) -> WeightedEstimate:
    """
Inverse-variance weighted combination of D_e estimates across selected methods and curves.

    Args:
        De_multi_results: risultati da estimate_all_De_combinations
        selected_beta_methods: β methods to include (None = all)
        selected_curves: calibration curves to use ('TL_JOINT', 'TL_AR', 'Student') (None = all)
    """
    if selected_beta_methods is None:
        selected_beta_methods = list(De_multi_results.keys())
    if selected_curves is None:
        selected_curves = ["TL_JOINT", "TL_AR", "Student"]

    values, stderrs = {}, {}

    for beta_method in selected_beta_methods:
        if beta_method not in De_multi_results:
            continue
        res = De_multi_results[beta_method]

        if "TL_JOINT" in selected_curves:
            key = f"{beta_method}_TL_JOINT"
            values[key] = res.De_tl_joint.De
            stderrs[key] = res.De_tl_joint.stderr

        if "TL_AR" in selected_curves:
            key = f"{beta_method}_TL_AR"
            values[key] = res.De_tl_ar.De
            stderrs[key] = res.De_tl_ar.stderr

        if "Student" in selected_curves:
            key = f"{beta_method}_Student"
            values[key] = res.De_student.De
            stderrs[key] = res.De_student.stderr

    return compute_weighted_mean(values, stderrs)


# =============================================================================
# ANALISI FINESTRE MOBILI
# =============================================================================


def estimate_D_local(log_prices, T_values, center, half_width):
    """Stima D locale in finestra."""
    n = len(log_prices)
    i_start, i_end = max(0, center - half_width), min(n, center + half_width)
    if i_end - i_start < max(T_values) + 50:
        return DEstimate(
            np.nan, np.nan, (np.nan, np.nan), {"reason": "window_too_small"}
        )
    return estimate_D_quantiles(log_prices[i_start:i_end], T_values)


def rigid_window_analysis(
    abs_returns,
    log_prices,
    dates,
    win_size,
    win_shift,
    lag_min,
    lag_max,
    nbins,
    min_acf_threshold,
    fit_method,
    min_bins_required,
    aggregation,
    multi_curves: MultiCurveResult,
    T_values,
    dfa_order=2,
    use_bayes=True,
    use_whittle=True,
    n_boot_beta=100,
    seed=1234,
):
    """
    Analisi finestre mobili con TUTTI i metodi e TUTTE le curve PNAS.

    Uses the three simulated curves (TL-JOINT, TL-AR, Student) for D_e inversion.
    For each β method, invert D_e on all three calibration curves.
    """
    n = len(abs_returns)
    if n < win_size:
        return pd.DataFrame()

    # Riferimenti alle tre curve
    curve_tl_joint = multi_curves.tl_joint
    curve_tl_ar = multi_curves.tl_ar
    curve_student = multi_curves.student

    rows = []
    k = 0

    for start in range(0, n - win_size + 1, win_shift):
        seg = abs_returns[start : start + win_size]
        end = start + win_size
        center_idx = start + win_size // 2

        # Date
        if dates is not None and center_idx < len(dates):
            window_date = pd.to_datetime(dates[center_idx])
            window_start_date = (
                pd.to_datetime(dates[start]) if start < len(dates) else None
            )
            window_end_date = pd.to_datetime(dates[min(end - 1, len(dates) - 1)])
        else:
            window_date = window_start_date = window_end_date = None

        # D locale
        D_local = estimate_D_local(log_prices, T_values, center_idx + 1, win_size // 2)

        # β ACF robusto
        beta_acf = estimate_beta_acf_robust(
            seg,
            lag_min=lag_min,
            lag_max=lag_max,
            nbins=nbins,
            n_boot=n_boot_beta,
            random_state=seed + 17 * k,
            min_acf_threshold=min_acf_threshold,
            fit_method=fit_method,
            min_bins_required=min_bins_required,
            aggregation=aggregation,
        )
        # D_e inversion on all three curves
        De_acf_joint = estimate_De_from_beta(beta_acf, curve_tl_joint)
        De_acf_ar = estimate_De_from_beta(beta_acf, curve_tl_ar)
        De_acf_student = estimate_De_from_beta(beta_acf, curve_student)

        # β HAC
        beta_hac = estimate_beta_acf_hac(
            seg,
            lag_min=lag_min,
            lag_max=lag_max,
            nbins=nbins,
            min_acf_threshold=min_acf_threshold,
        )
        De_hac_joint = estimate_De_from_beta(beta_hac, curve_tl_joint)
        De_hac_ar = estimate_De_from_beta(beta_hac, curve_tl_ar)
        De_hac_student = estimate_De_from_beta(beta_hac, curve_student)

        # β DFA
        beta_dfa = estimate_beta_dfa(
            seg,
            order=dfa_order,
            n_boot=max(30, n_boot_beta // 3),
            random_state=seed + 23 * k,
        )
        De_dfa_joint = estimate_De_from_beta(beta_dfa, curve_tl_joint)
        De_dfa_ar = estimate_De_from_beta(beta_dfa, curve_tl_ar)
        De_dfa_student = estimate_De_from_beta(beta_dfa, curve_student)

        # β Bayes
        if use_bayes and SCIPY_AVAILABLE:
            beta_bayes = estimate_beta_bayes(
                seg,
                lag_min=lag_min,
                lag_max=lag_max,
                nbins=nbins,
                min_acf_threshold=min_acf_threshold,
                random_state=seed + 31 * k,
            )
        else:
            beta_bayes = BetaEstimate(np.nan, np.nan, (np.nan, np.nan), {})
        De_bayes_joint = estimate_De_from_beta(beta_bayes, curve_tl_joint)
        De_bayes_ar = estimate_De_from_beta(beta_bayes, curve_tl_ar)
        De_bayes_student = estimate_De_from_beta(beta_bayes, curve_student)

        # β Whittle
        if use_whittle:
            beta_whittle = estimate_beta_whittle(seg, random_state=seed + 37 * k)
        else:
            beta_whittle = BetaEstimate(np.nan, np.nan, (np.nan, np.nan), {})
        De_whittle_joint = estimate_De_from_beta(beta_whittle, curve_tl_joint)
        De_whittle_ar = estimate_De_from_beta(beta_whittle, curve_tl_ar)
        De_whittle_student = estimate_De_from_beta(beta_whittle, curve_student)

        row = {
            "win_start": int(start),
            "win_end": int(end),
            "center_index": int(center_idx),
            "window_date": window_date.isoformat() if window_date else None,
            "window_start_date": (
                window_start_date.isoformat() if window_start_date else None
            ),
            "window_end_date": window_end_date.isoformat() if window_end_date else None,
            # D
            "D": D_local.D,
            "D_stderr": D_local.stderr,
            "D_ci_low": D_local.ci[0],
            "D_ci_high": D_local.ci[1],
            # β ACF
            "beta_acf": beta_acf.beta,
            "beta_acf_stderr": beta_acf.stderr,
            "beta_acf_ci_low": beta_acf.ci[0],
            "beta_acf_ci_high": beta_acf.ci[1],
            "beta_acf_r2": beta_acf.details.get("r2", np.nan),
            "De_acf_TL_JOINT": De_acf_joint.De,
            "De_acf_TL_JOINT_stderr": De_acf_joint.stderr,
            "De_acf_TL_AR": De_acf_ar.De,
            "De_acf_TL_AR_stderr": De_acf_ar.stderr,
            "De_acf_Student": De_acf_student.De,
            "De_acf_Student_stderr": De_acf_student.stderr,
            # β HAC
            "beta_hac": beta_hac.beta,
            "beta_hac_stderr": beta_hac.stderr,
            "beta_hac_r2": beta_hac.details.get("r2", np.nan),
            "De_hac_TL_JOINT": De_hac_joint.De,
            "De_hac_TL_JOINT_stderr": De_hac_joint.stderr,
            "De_hac_TL_AR": De_hac_ar.De,
            "De_hac_TL_AR_stderr": De_hac_ar.stderr,
            "De_hac_Student": De_hac_student.De,
            "De_hac_Student_stderr": De_hac_student.stderr,
            # β DFA
            "beta_dfa": beta_dfa.beta,
            "beta_dfa_stderr": beta_dfa.stderr,
            "beta_dfa_ci_low": beta_dfa.ci[0],
            "beta_dfa_ci_high": beta_dfa.ci[1],
            "beta_dfa_H": beta_dfa.details.get("H", np.nan),
            "beta_dfa_r2": beta_dfa.details.get("r2", np.nan),
            "De_dfa_TL_JOINT": De_dfa_joint.De,
            "De_dfa_TL_JOINT_stderr": De_dfa_joint.stderr,
            "De_dfa_TL_AR": De_dfa_ar.De,
            "De_dfa_TL_AR_stderr": De_dfa_ar.stderr,
            "De_dfa_Student": De_dfa_student.De,
            "De_dfa_Student_stderr": De_dfa_student.stderr,
            # β Bayes
            "beta_bayes": beta_bayes.beta,
            "beta_bayes_stderr": beta_bayes.stderr,
            "De_bayes_TL_JOINT": De_bayes_joint.De,
            "De_bayes_TL_JOINT_stderr": De_bayes_joint.stderr,
            "De_bayes_TL_AR": De_bayes_ar.De,
            "De_bayes_TL_AR_stderr": De_bayes_ar.stderr,
            "De_bayes_Student": De_bayes_student.De,
            "De_bayes_Student_stderr": De_bayes_student.stderr,
            # β Whittle
            "beta_whittle": beta_whittle.beta,
            "beta_whittle_stderr": beta_whittle.stderr,
            "beta_whittle_m": beta_whittle.details.get("bandwidth_m", np.nan),
            "De_whittle_TL_JOINT": De_whittle_joint.De,
            "De_whittle_TL_JOINT_stderr": De_whittle_joint.stderr,
            "De_whittle_TL_AR": De_whittle_ar.De,
            "De_whittle_TL_AR_stderr": De_whittle_ar.stderr,
            "De_whittle_Student": De_whittle_student.De,
            "De_whittle_Student_stderr": De_whittle_student.stderr,
        }
        rows.append(row)
        k += 1

        if k % 10 == 0:
            _log(f"Finestra {k}...")

    return pd.DataFrame(rows)


def rolling_corr_cov(x, y, window_size, step, dates=None):
    """Correlazione e covarianza rolling tra due serie."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < window_size:
        return pd.DataFrame()

    rows = []
    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        xs, ys = x[start:end], y[start:end]
        m = np.isfinite(xs) & np.isfinite(ys)
        n_eff = int(m.sum())
        corr = cov = corr_lo = corr_hi = np.nan

        if n_eff >= 3:
            xv, yv = xs[m], ys[m]
            cov = float(np.cov(xv, yv, ddof=1)[0, 1])
            sx, sy = float(np.std(xv, ddof=1)), float(np.std(yv, ddof=1))
            if sx > 0 and sy > 0:
                corr = float(cov / (sx * sy))
                if n_eff >= 4 and abs(corr) < 1.0:
                    z = np.arctanh(corr)
                    se_z = 1.0 / math.sqrt(max(1.0, n_eff - 3.0))
                    corr_lo = float(np.tanh(z - CI * se_z))
                    corr_hi = float(np.tanh(z + CI * se_z))

        center_idx = start + window_size // 2
        date_val = None
        if dates is not None and center_idx < len(dates):
            try:
                date_val = pd.to_datetime(dates[center_idx]).isoformat()
            except:
                pass

        rows.append(
            {
                "window_center_index": center_idx,
                "window_date": date_val,
                "n_pairs": n_eff,
                "corr": corr,
                "corr_ci_lo": corr_lo,
                "corr_ci_hi": corr_hi,
                "cov": cov,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Z-SCORE STANDARDISATION
# =============================================================================


def compute_zscore(series):
    """Standardize a series to zero mean and unit variance (z-score)."""
    series = np.asarray(series, float)
    valid = np.isfinite(series)
    if not np.any(valid):
        return np.full_like(series, np.nan)
    mu = np.nanmean(series)
    sigma = np.nanstd(series, ddof=1)
    if sigma < 1e-18:
        return np.full_like(series, np.nan)
    return (series - mu) / sigma


# =============================================================================
# GRAFICI INDIVIDUALI PER METODO β vs D
# =============================================================================


def plot_beta_method_analysis(
    outdir,
    df,
    dates,
    method_name,
    beta_col,
    D_col="D",
    corr_win_size=20,
    corr_win_step=5,
):
    """
    Genera grafici per un singolo metodo β:
    1. Serie grezza β e D
    2. Z-score di β e D
    3. Rolling correlation between z(β) and z(D)
    4. Rolling covariance between z(β) and z(D)
    """
    figdir = os.path.join(outdir, "fig", "beta_analysis")
    _ensure_dir(figdir)

    if beta_col not in df.columns or D_col not in df.columns:
        return

    beta = pd.to_numeric(df[beta_col], errors="coerce").values
    D = pd.to_numeric(df[D_col], errors="coerce").values

    if not np.any(np.isfinite(beta)):
        return

    # Determine x-axis: prefer dates when available
    if "window_date" in df.columns and df["window_date"].notna().any():
        x_dates = pd.to_datetime(df["window_date"])
        is_time = True
    else:
        x_dates = None
        is_time = False

    # Numerical indices for internal plotting
    n_windows = len(df)
    x_idx = np.arange(n_windows)

    # Compute z-score
    z_beta = compute_zscore(beta)
    z_D = compute_zscore(D)

    # Rolling correlation and covariance of z-scored series
    # Use window dates as the rolling x-axis
    corr_df = rolling_corr_cov(z_beta, z_D, corr_win_size, corr_win_step, dates=None)

    # 4-panel figure (no sharex to avoid alignment issues with mixed axes)
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))

    # Panel 1: raw series
    ax = axes[0]
    ax2 = ax.twinx()
    if is_time:
        (l1,) = ax.plot(x_dates, beta, "b-", linewidth=1.5, label=f"β ({method_name})")
        (l2,) = ax2.plot(x_dates, D, "r-", linewidth=1.5, alpha=0.7, label="D")
        _format_time_axis(ax)
    else:
        (l1,) = ax.plot(x_idx, beta, "b-", linewidth=1.5, label=f"β ({method_name})")
        (l2,) = ax2.plot(x_idx, D, "r-", linewidth=1.5, alpha=0.7, label="D")
    ax.set_ylabel(f"β ({method_name})", color="blue")
    ax2.set_ylabel("D", color="red")
    ax.tick_params(axis="y", labelcolor="blue")
    ax2.tick_params(axis="y", labelcolor="red")
    ax.axhline(0.2, color="blue", linestyle="--", alpha=0.3)
    ax2.axhline(0.5, color="red", linestyle="--", alpha=0.3)
    ax.legend(handles=[l1, l2], loc="upper right")
    ax.set_title(f"Raw series: β_{method_name} and D")
    ax.grid(True, alpha=0.3)

    # Panel 2: z-score
    ax = axes[1]
    if is_time:
        ax.plot(x_dates, z_beta, "b-", linewidth=1.5, label=f"z(β_{method_name})")
        ax.plot(x_dates, z_D, "r-", linewidth=1.5, alpha=0.7, label="z(D)")
        _format_time_axis(ax)
    else:
        ax.plot(x_idx, z_beta, "b-", linewidth=1.5, label=f"z(β_{method_name})")
        ax.plot(x_idx, z_D, "r-", linewidth=1.5, alpha=0.7, label="z(D)")
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.axhline(2, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(-2, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylabel("Z-score")
    ax.set_title(f"Z-score: β_{method_name} e D")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: rolling correlation
    ax = axes[2]
    if not corr_df.empty and len(corr_df) > 0:
        # Map rolling indices back to original window dates
        corr_center_indices = corr_df["window_center_index"].values.astype(int)
        if is_time:
            # Use dates at the rolling window centres
            xc = x_dates.iloc[corr_center_indices].values
            ax.plot(xc, corr_df["corr"].values, "g-", linewidth=1.5)
            ax.fill_between(
                xc,
                corr_df["corr_ci_lo"].values,
                corr_df["corr_ci_hi"].values,
                alpha=0.2,
                color="green",
            )
            _format_time_axis(ax)
        else:
            ax.plot(corr_center_indices, corr_df["corr"].values, "g-", linewidth=1.5)
            ax.fill_between(
                corr_center_indices,
                corr_df["corr_ci_lo"].values,
                corr_df["corr_ci_hi"].values,
                alpha=0.2,
                color="green",
            )
        ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_ylabel("Correlazione")
    ax.set_title(
        f"Correlazione rolling: z(β_{method_name}) vs z(D) [win={corr_win_size}, step={corr_win_step}]"
    )
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)

    # Panel 4: rolling covariance
    ax = axes[3]
    if not corr_df.empty and len(corr_df) > 0:
        if is_time:
            ax.plot(xc, corr_df["cov"].values, "purple", linewidth=1.5)
            _format_time_axis(ax)
        else:
            ax.plot(corr_center_indices, corr_df["cov"].values, "purple", linewidth=1.5)
        ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Date" if is_time else "Window index")
    ax.set_ylabel("Covarianza")
    ax.set_title(f"Covarianza rolling: z(β_{method_name}) vs z(D)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.suptitle(f"Analisi β_{method_name} vs D", fontsize=14, y=1.02)
    _savefig(os.path.join(figdir, f"beta_{method_name.lower()}_vs_D_analysis.png"))


# =============================================================================
# GRAFICI INDIVIDUALI PER D_e vs D
# =============================================================================


def plot_De_method_analysis(
    outdir, df, dates, method_name, De_col, D_col="D", corr_win_size=20, corr_win_step=5
):
    """
    Genera grafici per un singolo metodo D_e:
    1. Serie grezza D_e e D
    2. Z-score di D_e e D
    3. Rolling correlation between z(D_e) and z(D)
    4. Rolling covariance between z(D_e) and z(D)
    """
    figdir = os.path.join(outdir, "fig", "De_analysis")
    _ensure_dir(figdir)

    if De_col not in df.columns or D_col not in df.columns:
        return

    De = pd.to_numeric(df[De_col], errors="coerce").values
    D = pd.to_numeric(df[D_col], errors="coerce").values

    if not np.any(np.isfinite(De)):
        return

    # Determine x-axis: prefer dates when available
    if "window_date" in df.columns and df["window_date"].notna().any():
        x_dates = pd.to_datetime(df["window_date"])
        is_time = True
    else:
        x_dates = None
        is_time = False

    # Indici per plotting
    n_windows = len(df)
    x_idx = np.arange(n_windows)

    # Compute z-score
    z_De = compute_zscore(De)
    z_D = compute_zscore(D)

    # Rolling correlation and covariance of z-scored series
    corr_df = rolling_corr_cov(z_De, z_D, corr_win_size, corr_win_step, dates=None)

    # 4-panel figure (no sharex to avoid alignment issues with mixed axes)
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))

    # Panel 1: raw series
    ax = axes[0]
    ax2 = ax.twinx()
    if is_time:
        (l1,) = ax.plot(x_dates, De, "b-", linewidth=1.5, label=f"D_e ({method_name})")
        (l2,) = ax2.plot(x_dates, D, "r-", linewidth=1.5, alpha=0.7, label="D")
        _format_time_axis(ax)
    else:
        (l1,) = ax.plot(x_idx, De, "b-", linewidth=1.5, label=f"D_e ({method_name})")
        (l2,) = ax2.plot(x_idx, D, "r-", linewidth=1.5, alpha=0.7, label="D")
    ax.set_ylabel(f"D_e ({method_name})", color="blue")
    ax2.set_ylabel("D", color="red")
    ax.tick_params(axis="y", labelcolor="blue")
    ax2.tick_params(axis="y", labelcolor="red")
    ax.axhline(0.25, color="blue", linestyle="--", alpha=0.3)
    ax2.axhline(0.5, color="red", linestyle="--", alpha=0.3)
    ax.legend(handles=[l1, l2], loc="upper right")
    ax.set_title(f"Raw series: D_e ({method_name}) and D")
    ax.grid(True, alpha=0.3)

    # Panel 2: z-score
    ax = axes[1]
    if is_time:
        ax.plot(x_dates, z_De, "b-", linewidth=1.5, label=f"z(D_e_{method_name})")
        ax.plot(x_dates, z_D, "r-", linewidth=1.5, alpha=0.7, label="z(D)")
        _format_time_axis(ax)
    else:
        ax.plot(x_idx, z_De, "b-", linewidth=1.5, label=f"z(D_e_{method_name})")
        ax.plot(x_idx, z_D, "r-", linewidth=1.5, alpha=0.7, label="z(D)")
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.axhline(2, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(-2, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylabel("Z-score")
    ax.set_title(f"Z-score: D_e ({method_name}) e D")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: rolling correlation
    ax = axes[2]
    if not corr_df.empty and len(corr_df) > 0:
        # Map rolling indices back to original window dates
        corr_center_indices = corr_df["window_center_index"].values.astype(int)
        if is_time:
            # Use dates at the rolling window centres
            xc = x_dates.iloc[corr_center_indices].values
            ax.plot(xc, corr_df["corr"].values, "g-", linewidth=1.5)
            ax.fill_between(
                xc,
                corr_df["corr_ci_lo"].values,
                corr_df["corr_ci_hi"].values,
                alpha=0.2,
                color="green",
            )
            _format_time_axis(ax)
        else:
            ax.plot(corr_center_indices, corr_df["corr"].values, "g-", linewidth=1.5)
            ax.fill_between(
                corr_center_indices,
                corr_df["corr_ci_lo"].values,
                corr_df["corr_ci_hi"].values,
                alpha=0.2,
                color="green",
            )
        ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_ylabel("Correlazione")
    ax.set_title(
        f"Correlazione rolling: z(D_e_{method_name}) vs z(D) [win={corr_win_size}, step={corr_win_step}]"
    )
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)

    # Panel 4: rolling covariance
    ax = axes[3]
    if not corr_df.empty and len(corr_df) > 0:
        if is_time:
            ax.plot(xc, corr_df["cov"].values, "purple", linewidth=1.5)
            _format_time_axis(ax)
        else:
            ax.plot(corr_center_indices, corr_df["cov"].values, "purple", linewidth=1.5)
        ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Date" if is_time else "Window index")
    ax.set_ylabel("Covarianza")
    ax.set_title(f"Covarianza rolling: z(D_e_{method_name}) vs z(D)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.suptitle(f"Analisi D_e ({method_name}) vs D", fontsize=14, y=1.02)
    _savefig(os.path.join(figdir, f"De_{method_name.lower()}_vs_D_analysis.png"))


# =============================================================================
# GRAFICI CURVA CALIBRAZIONE CON INVERSIONE ESPLICITA
# =============================================================================


def plot_calibration_inversion_single(outdir, curve, beta_est, method_name):
    """
    Grafico che mostra esplicitamente come β_emp viene mappato a D_e.
    """
    figdir = os.path.join(outdir, "fig", "calibration_inversion")
    _ensure_dir(figdir)

    if not np.isfinite(beta_est.beta):
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    # Curva simulata
    valid = np.isfinite(curve.beta_mean)
    ax.fill_between(
        curve.De_grid,
        curve.beta_mean - curve.beta_std_smooth,
        curve.beta_mean + curve.beta_std_smooth,
        alpha=0.2,
        color="blue",
        label="±1σ simulato",
    )

    if curve.spline is not None:
        De_smooth = np.linspace(curve.De_grid.min(), curve.De_grid.max(), 200)
        beta_smooth = curve.spline(De_smooth)
        ax.plot(De_smooth, beta_smooth, "b-", linewidth=2, label="Curva β_sim(D_e)")
    else:
        ax.plot(curve.De_grid[valid], curve.beta_mean[valid], "b-", linewidth=2)

    # β empirico
    beta_emp = beta_est.beta
    beta_stderr = beta_est.stderr if np.isfinite(beta_est.stderr) else 0

    # Linea orizzontale da β_emp
    ax.axhline(
        beta_emp,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"β_emp ({method_name}) = {beta_emp:.4f}",
    )

    # Banda incertezza β
    if beta_stderr > 0:
        ax.axhspan(
            beta_emp - CI * beta_stderr,
            beta_emp + CI * beta_stderr,
            color="red",
            alpha=0.1,
            label=f"±{CI:.2f}σ_β",
        )

    # Trova intersezione (D_e*)
    De_star, details = calibrate_De_via_spline(beta_emp, curve)

    if np.isfinite(De_star):
        # Linea verticale a D_e*
        ax.axvline(
            De_star,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"D_e* = {De_star:.4f}",
        )

        # Freccia che mostra il mapping
        ax.annotate(
            "",
            xy=(De_star, beta_emp),
            xytext=(De_star, ax.get_ylim()[1] * 0.95),
            arrowprops=dict(arrowstyle="->", color="green", lw=2),
        )
        ax.annotate(
            "",
            xy=(De_star, beta_emp),
            xytext=(ax.get_xlim()[0] + 0.01, beta_emp),
            arrowprops=dict(arrowstyle="->", color="red", lw=2),
        )

        # Punto di intersezione
        ax.scatter(
            [De_star],
            [beta_emp],
            s=200,
            c="yellow",
            edgecolors="black",
            zorder=10,
            marker="*",
            label="Intersezione",
        )

    ax.set_xlabel("D_e (parametro endogeno)", fontsize=12)
    ax.set_ylabel("β (esponente decadimento ACF)", fontsize=12)
    ax.set_title(
        (
            f"Inversione β → D_e: Metodo {method_name}\n"
            f"β_emp = {beta_emp:.4f} → D_e* = {De_star:.4f}"
            if np.isfinite(De_star)
            else f"Inversione β → D_e: Metodo {method_name}"
        ),
        fontsize=14,
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Aggiungi spiegazione testuale
    text = (
        f"INTERPRETAZIONE:\n"
        f"• La curva blu mostra β_sim(D_e) dalla simulazione\n"
        f"• La linea rossa orizzontale è il β empirico misurato\n"
        f"• L'intersezione determina D_e*, il valore di D_e\n"
        f"  compatibile con l'osservazione empirica"
    )
    ax.text(
        0.98,
        0.02,
        text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    _savefig(os.path.join(figdir, f"calibration_inversion_{method_name.lower()}.png"))


# =============================================================================
# GRAFICO POSTERIOR BAYESIANO ESPLICATIVO
# =============================================================================


def plot_bayesian_posterior_explained(outdir, curve, beta_est, method_name="ACF"):
    """
    Grafico dettagliato della inversione Bayesiana con spiegazione.
    """
    figdir = os.path.join(outdir, "fig", "bayesian")
    _ensure_dir(figdir)

    if not np.isfinite(beta_est.beta) or not np.isfinite(beta_est.stderr):
        return

    beta_emp = beta_est.beta
    beta_stderr = beta_est.stderr

    De_hat, ci, De_grid, posterior, prior = bayesian_De_posterior(
        beta_emp, beta_stderr, curve
    )

    if len(posterior) == 0 or not np.isfinite(De_hat):
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: likelihood
    ax = axes[0, 0]
    m_De = curve.beta_mean
    s_sim = curve.beta_std_smooth
    valid = np.isfinite(m_De) & np.isfinite(s_sim)

    # Proportional likelihood
    var_total = beta_stderr**2 + s_sim**2 + 1e-10
    log_lik = np.full_like(curve.De_grid, -np.inf)
    log_lik[valid] = -0.5 * (beta_emp - m_De[valid]) ** 2 / var_total[valid]
    log_lik -= np.max(log_lik[np.isfinite(log_lik)])
    lik = np.exp(log_lik)
    lik /= np.sum(lik) + 1e-18

    ax.plot(curve.De_grid, lik, "b-", linewidth=2, label="Likelihood L(D_e | β_emp)")
    ax.fill_between(curve.De_grid, 0, lik, alpha=0.3, color="blue")
    ax.set_xlabel("D_e")
    ax.set_ylabel("Verosimiglianza (normalizzata)")
    ax.set_title("Verosimiglianza: quanto è compatibile ciascun D_e con β_emp")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: prior
    ax = axes[0, 1]
    ax.plot(curve.De_grid, prior, "g-", linewidth=2, label="Prior p(D_e)")
    ax.fill_between(curve.De_grid, 0, prior, alpha=0.3, color="green")
    ax.set_xlabel("D_e")
    ax.set_ylabel("Prior (normalizzato)")
    ax.set_title("Prior: credenze a priori su D_e (uniforme = non informativo)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: posterior
    ax = axes[1, 0]
    ax.plot(
        curve.De_grid, posterior, "r-", linewidth=2, label="Posterior p(D_e | β_emp)"
    )
    ax.fill_between(curve.De_grid, 0, posterior, alpha=0.3, color="red")
    ax.axvline(
        De_hat, color="k", linestyle="--", linewidth=2, label=f"E[D_e] = {De_hat:.4f}"
    )
    if np.isfinite(ci[0]) and np.isfinite(ci[1]):
        ax.axvline(
            ci[0],
            color="gray",
            linestyle=":",
            label=f"CI 95%: [{ci[0]:.3f}, {ci[1]:.3f}]",
        )
        ax.axvline(ci[1], color="gray", linestyle=":")
        ax.axvspan(ci[0], ci[1], alpha=0.1, color="gray")
    ax.set_xlabel("D_e")
    ax.set_ylabel("Posterior (normalizzato)")
    ax.set_title("Posterior: distribuzione finale di D_e dato β_emp")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 4: interpretive text
    ax = axes[1, 1]
    ax.axis("off")

    std_post = np.sqrt(np.sum(curve.De_grid**2 * posterior) - De_hat**2)
    explanation = f"""
INVERSIONE BAYESIANA: β_emp → D_e

FORMULA:
  p(D_e | β_emp) ∝ L(β_emp | D_e) × p(D_e)
  
  • L(β_emp | D_e): Likelihood - quanto è probabile osservare β_emp 
    the true D_e were that value. Computed from the simulated calibration curve.
    
  • p(D_e): Prior - credenze a priori su D_e. Qui usiamo un prior
    uniforme (non informativo): tutti i D_e sono equiprobabili.
    
  • p(D_e | β_emp): Posterior - la distribuzione aggiornata di D_e
    dopo aver osservato β_emp. Combina simulazione + osservazione.

RISULTATI PER {method_name}:
  • β empirico: {beta_emp:.4f} ± {beta_stderr:.4f}
  • D_e stimato (media posterior): {De_hat:.4f}
  • Intervallo credibilità 95%: [{ci[0]:.4f}, {ci[1]:.4f}]
  • Deviazione standard posterior: {std_post:.4f}

INTERPRETAZIONE:
  • La curva simulata β_sim(D_e) definisce la relazione teorica
  • L'inversione trova quale D_e produce β ≈ β_emp
  • L'incertezza su β_emp e la variabilità simulata determinano 
    l'ampiezza dell'intervallo di credibilità su D_e
"""
    ax.text(
        0.05,
        0.95,
        explanation,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    fig.suptitle(f"Inversione Bayesiana: {method_name}", fontsize=14)
    _savefig(os.path.join(figdir, f"bayesian_posterior_{method_name.lower()}.png"))


# =============================================================================
# GRAFICI DIAGNOSTICI (inclusi nuovi grafici da PDF)
# =============================================================================


def plot_acf_diagnostic(
    outdir, abs_returns, lag_range, nbins, min_acf_threshold, beta_est
):
    """Grafico diagnostico ACF (4 pannelli)."""
    figdir = os.path.join(outdir, "fig")
    _ensure_dir(figdir)
    lag_min, lag_max = int(lag_range[0]), min(int(lag_range[1]), len(abs_returns) // 4)
    if lag_max <= lag_min:
        return

    acf = acf_fft(abs_returns, lag_max)
    lags = np.arange(1, lag_max + 1, dtype=float)
    acf_values = acf[1:]
    bin_result = robust_logbin(
        lags,
        acf_values,
        nbins=nbins,
        lag_min=lag_min,
        lag_max=lag_max,
        min_acf_threshold=min_acf_threshold,
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: ACF (linear scale)
    ax = axes[0, 0]
    ax.plot(lags, acf_values, "b-", alpha=0.4, linewidth=0.5, label="ACF raw")
    mask = bin_result.mask_valid
    if np.any(mask):
        ax.scatter(
            bin_result.x_binned[mask],
            bin_result.y_binned[mask],
            c="red",
            s=40,
            zorder=5,
            label="Binned (validi)",
        )
    ax.axhline(
        min_acf_threshold,
        color="orange",
        linestyle="--",
        alpha=0.5,
        label=f"Soglia = {min_acf_threshold}",
    )
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF(|r|)")
    ax.set_title("ACF rendimenti assoluti (scala lineare)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: ACF in log-log with power-law fit
    ax = axes[0, 1]
    if np.any(mask):
        x_valid, y_valid = bin_result.x_binned[mask], bin_result.y_binned[mask]
        ax.loglog(x_valid, y_valid, "ro", markersize=8, label="Binned")
        if np.isfinite(beta_est.beta):
            log_x = np.log(x_valid)
            intercept = np.mean(np.log(y_valid)) + beta_est.beta * np.mean(log_x)
            xx = np.linspace(log_x.min(), log_x.max(), 100)
            yy = intercept - beta_est.beta * xx
            ax.loglog(
                np.exp(xx),
                np.exp(yy),
                "b--",
                linewidth=2,
                label=f"β = {beta_est.beta:.3f} ± {beta_est.stderr:.3f}",
            )
    ax.set_xlabel("Lag (log)")
    ax.set_ylabel("ACF(|r|) (log)")
    ax.set_title("ACF log-log con fit potenza")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    # Panel 3: residuals
    ax = axes[1, 0]
    residuals = beta_est.details.get("residuals", [])
    if len(residuals) > 0 and np.any(mask):
        x_valid = bin_result.x_binned[mask]
        ax.scatter(x_valid[: len(residuals)], residuals, c="blue", s=30, alpha=0.7)
        ax.axhline(0, color="r", linestyle="--")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Residui (log-log)")
        ax.set_title("Residui del fit log-log")
        ax.grid(True, alpha=0.3)

    # Panel 4: data points per bin
    ax = axes[1, 1]
    n_per_bin = bin_result.n_per_bin
    valid_idx = np.where(np.isfinite(bin_result.x_binned))[0]
    if len(valid_idx) > 0:
        colors = ["green" if mask[i] else "gray" for i in valid_idx]
        ax.bar(range(len(valid_idx)), n_per_bin[valid_idx], color=colors, alpha=0.7)
        ax.set_xlabel("Indice bin")
        ax.set_ylabel("Points per bin")
        ax.set_title(
            f"Points per bin ({bin_result.nbins_valid}/{bin_result.nbins_original} valid)"
        )
    ax.grid(True, alpha=0.3)

    r2 = beta_est.details.get("r2", np.nan)
    method = beta_est.details.get("fit_method", "theil_sen")
    fig.suptitle(f"Diagnostica ACF: R² = {r2:.4f}, metodo = {method}", fontsize=11)
    _savefig(os.path.join(figdir, "acf_diagnostic.png"))


def plot_dfa_diagnostic(outdir, abs_returns, beta_dfa, order=2):
    """Grafico diagnostico DFA (2 pannelli)."""
    figdir = os.path.join(outdir, "fig")
    _ensure_dir(figdir)
    try:
        scales, F = dfa_fluctuations(abs_returns, order=order)
    except:
        return
    if len(scales) < 3:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.loglog(scales, F, "bo-", markersize=6, label="F(s)")
    if np.isfinite(beta_dfa.beta):
        H = beta_dfa.details.get("H", (2 - beta_dfa.beta) / 2)
        log_s, log_F = np.log(scales.astype(float)), np.log(F)
        intercept = np.mean(log_F) - H * np.mean(log_s)
        ss = np.linspace(log_s.min(), log_s.max(), 100)
        ff = intercept + H * ss
        ax.loglog(
            np.exp(ss),
            np.exp(ff),
            "r--",
            linewidth=2,
            label=f"H = {H:.3f}, β = {beta_dfa.beta:.3f}",
        )
    ax.set_xlabel("Scale s")
    ax.set_ylabel("F(s)")
    ax.set_title(f"DFA-{order}: F(s) ~ s^H")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    r2 = beta_dfa.details.get("r2", np.nan)
    if np.isfinite(r2):
        ax.text(0.05, 0.05, f"R² = {r2:.4f}", transform=ax.transAxes, fontsize=10)

    ax = axes[1]
    local_centers = beta_dfa.details.get("local_H_centers", [])
    local_H = beta_dfa.details.get("local_H_values", [])
    if len(local_centers) > 0 and len(local_H) > 0:
        ax.plot(local_centers, local_H, "go-", markersize=6, label="H locale")
        if np.isfinite(beta_dfa.beta):
            H_global = beta_dfa.details.get("H", (2 - beta_dfa.beta) / 2)
            ax.axhline(
                H_global, color="r", linestyle="--", label=f"H globale = {H_global:.3f}"
            )
        ax.set_xlabel("Scale s")
        ax.set_ylabel("H locale")
        ax.set_title("Slope locale (per cross-over)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    _savefig(os.path.join(figdir, "dfa_diagnostic.png"))


def plot_whittle_stability(outdir, beta_whittle):
    """Grafico stabilità Whittle (2 pannelli)."""
    figdir = os.path.join(outdir, "fig")
    _ensure_dir(figdir)
    m_curve = beta_whittle.details.get("m_stability_curve", [])
    d_curve = beta_whittle.details.get("d_stability_curve", [])
    if len(m_curve) == 0 or len(d_curve) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(m_curve, d_curve, "b-", linewidth=1.5, label="d(m)")
    m_selected = beta_whittle.details.get("bandwidth_m", np.nan)
    d_hat = beta_whittle.details.get("d_hat", np.nan)
    if np.isfinite(m_selected):
        ax.axvline(m_selected, color="r", linestyle="--", label=f"m = {m_selected}")
    if np.isfinite(d_hat):
        ax.axhline(d_hat, color="g", linestyle="--", label=f"d = {d_hat:.3f}")
    ax.set_xlabel("Bandwidth m")
    ax.set_ylabel("d")
    ax.set_title("Curva stabilità Whittle/GPH: d(m)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    beta_curve = 1 - 2 * np.array(d_curve)
    ax.plot(m_curve, beta_curve, "b-", linewidth=1.5, label="β(m)")
    if np.isfinite(beta_whittle.beta):
        ax.axhline(
            beta_whittle.beta,
            color="g",
            linestyle="--",
            label=f"β = {beta_whittle.beta:.3f}",
        )
    ax.axhline(0.2, color="r", linestyle=":", alpha=0.5, label="β ≈ 0.2")
    ax.set_xlabel("Bandwidth m")
    ax.set_ylabel("β")
    ax.set_title("β(m) = 1 - 2d(m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _savefig(os.path.join(figdir, "whittle_stability.png"))


def plot_calibration_curve_diagnostic(outdir, curve, beta_acf, De_acf):
    """
    Grafico diagnostico curva calibrazione (6 pannelli).
    Include nuovi grafici dal PDF: pass_rate, scatter repliche, fallimenti.
    """
    figdir = os.path.join(outdir, "fig")
    _ensure_dir(figdir)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Panel 1: β_sim(D_e) curve with uncertainty bands
    ax = axes[0, 0]
    valid = np.isfinite(curve.beta_mean)
    ax.scatter(
        curve.De_grid[valid],
        curve.beta_mean[valid],
        c="gray",
        s=20,
        alpha=0.5,
        label="Simulated points",
    )
    lo = curve.beta_mean - curve.beta_std_smooth
    hi = curve.beta_mean + curve.beta_std_smooth
    ax.fill_between(
        curve.De_grid, lo, hi, alpha=0.2, color="blue", label="±1σ (smooth)"
    )
    if curve.spline is not None:
        De_smooth = np.linspace(curve.De_grid.min(), curve.De_grid.max(), 200)
        beta_smooth = curve.spline(De_smooth)
        ax.plot(De_smooth, beta_smooth, "b-", linewidth=2, label="Spline (PCHIP)")
    if np.isfinite(beta_acf.beta):
        ax.axhline(
            beta_acf.beta,
            color="r",
            linestyle="--",
            label=f"β_emp = {beta_acf.beta:.3f}",
        )
    if np.isfinite(De_acf.De):
        ax.axvline(
            De_acf.De, color="g", linestyle="--", label=f"D_e* = {De_acf.De:.3f}"
        )
    ax.set_xlabel("D_e")
    ax.set_ylabel("β_sim")
    ax.set_title("Curva calibrazione β_sim(D_e)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: pass rate vs D_e
    ax = axes[0, 1]
    De_vals = [d.De for d in curve.diagnostics]
    pass_rates = [d.pass_rate for d in curve.diagnostics]
    ax.plot(De_vals, pass_rates, "b-o", markersize=4, label="Pass rate")
    ax.axhline(0.8, color="r", linestyle="--", alpha=0.5, label="Threshold = 0.8")
    # Colora punti unreliable
    for d in curve.diagnostics:
        if d.sel_bias_warn:
            ax.scatter([d.De], [d.pass_rate], c="red", s=50, marker="x", zorder=5)
    ax.set_xlabel("D_e")
    ax.set_ylabel("Pass rate")
    ax.set_title("Pass-rate per D_e (punti rossi = unreliable)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Panel 3: derivative dβ/dD_e
    ax = axes[0, 2]
    deriv = curve.derivative
    valid_d = np.isfinite(deriv)
    if np.any(valid_d):
        ax.plot(curve.De_grid[valid_d], deriv[valid_d], "b-", linewidth=1.5)
        ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("D_e")
    ax.set_ylabel("dβ/dD_e")
    ax.set_title("Derivata numerica (identificabilità)")
    ax.grid(True, alpha=0.3)

    # Panel 4: median valid bins vs D_e
    ax = axes[1, 0]
    median_nbins = [d.median_nbins_valid for d in curve.diagnostics]
    ax.plot(De_vals, median_nbins, "g-o", markersize=4)
    ax.axhline(
        curve.params.get("min_bins_required", 6),
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"min_bins = {curve.params.get('min_bins_required', 6)}",
    )
    ax.set_xlabel("D_e")
    ax.set_ylabel("Median nbins_valid")
    ax.set_title("Mediana bin validi per D_e")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 5: scatter of β replicates
    ax = axes[1, 1]
    for d in curve.diagnostics[
        :: max(1, len(curve.diagnostics) // 20)
    ]:  # Mostra subset
        betas = d.beta_values[np.isfinite(d.beta_values)]
        if len(betas) > 0:
            ax.scatter([d.De] * len(betas), betas, alpha=0.3, s=10, c="blue")
    ax.plot(curve.De_grid, curve.beta_mean, "r-", linewidth=2, label="Media")
    ax.set_xlabel("D_e")
    ax.set_ylabel("β (repliche)")
    ax.set_title("Scatter β per tutte le repliche")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 6: failure mode histogram
    ax = axes[1, 2]
    fail_types = {
        "fit_fail": sum(d.n_fit_fail for d in curve.diagnostics),
        "bins_insuff": sum(d.n_bins_insufficient for d in curve.diagnostics),
        "beta_invalid": sum(d.n_beta_invalid for d in curve.diagnostics),
    }
    total_reps = sum(d.n_total for d in curve.diagnostics)
    total_pass = sum(d.n_pass for d in curve.diagnostics)
    fail_types["success"] = total_pass

    labels = list(fail_types.keys())
    values = list(fail_types.values())
    colors = ["red", "orange", "yellow", "green"]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Conteggio")
    ax.set_title(f"Breakdown fallimenti (tot repliche = {total_reps})")
    for i, v in enumerate(values):
        ax.text(
            i,
            v + total_reps * 0.01,
            f"{v/total_reps*100:.1f}%",
            ha="center",
            fontsize=9,
        )

    fig.suptitle(
        f"Diagnostica curva calibrazione - metodo: {curve.params.get('fit_method', '?')}",
        fontsize=12,
    )
    _savefig(os.path.join(figdir, "calibration_diagnostic.png"))


def plot_calibration_curve_simple(outdir, curve, beta_acf, De_acf):
    """Grafico semplificato curva calibrazione con posterior."""
    figdir = os.path.join(outdir, "fig")
    _ensure_dir(figdir)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: β_sim(D_e) curve
    ax = axes[0, 0]
    valid = np.isfinite(curve.beta_mean)
    lo = curve.beta_mean - curve.beta_std_smooth
    hi = curve.beta_mean + curve.beta_std_smooth
    ax.fill_between(curve.De_grid, lo, hi, alpha=0.2, color="blue", label="±1σ")
    if curve.spline is not None:
        De_smooth = np.linspace(curve.De_grid.min(), curve.De_grid.max(), 200)
        ax.plot(De_smooth, curve.spline(De_smooth), "b-", linewidth=2, label="Spline")
    if np.isfinite(beta_acf.beta):
        ax.axhline(
            beta_acf.beta,
            color="r",
            linestyle="--",
            label=f"β_emp = {beta_acf.beta:.3f}",
        )
    if np.isfinite(De_acf.De):
        ax.axvline(
            De_acf.De, color="g", linestyle="--", label=f"D_e* = {De_acf.De:.3f}"
        )
    ax.set_xlabel("D_e")
    ax.set_ylabel("β_sim")
    ax.set_title("Curva β_sim(D_e)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: simulation uncertainty σ_sim
    ax = axes[0, 1]
    ax.plot(curve.De_grid, curve.beta_std, "b-", alpha=0.5, label="σ_sim raw")
    ax.plot(
        curve.De_grid, curve.beta_std_smooth, "r-", linewidth=2, label="σ_sim smooth"
    )
    ax.set_xlabel("D_e")
    ax.set_ylabel("σ_sim")
    ax.set_title("Varianza simulazione")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: derivative
    ax = axes[1, 0]
    deriv = curve.derivative
    valid_d = np.isfinite(deriv)
    if np.any(valid_d):
        ax.plot(curve.De_grid[valid_d], deriv[valid_d], "b-", linewidth=1.5)
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("D_e")
    ax.set_ylabel("dβ/dD_e")
    ax.set_title("Derivata (identificabilità)")
    ax.grid(True, alpha=0.3)

    # Panel 4: Bayesian posterior
    ax = axes[1, 1]
    if np.isfinite(beta_acf.beta) and np.isfinite(beta_acf.stderr):
        De_hat, ci, De_grid, posterior, prior = bayesian_De_posterior(
            beta_acf.beta, beta_acf.stderr, curve
        )
        if len(posterior) > 0:
            ax.plot(
                De_grid,
                prior / np.max(prior) * np.max(posterior) * 0.3,
                "g--",
                alpha=0.5,
                linewidth=1,
                label="Prior (scalato)",
            )
            ax.plot(De_grid, posterior, "b-", linewidth=2, label="Posterior")
            ax.fill_between(De_grid, 0, posterior, alpha=0.3)
            if np.isfinite(De_hat):
                ax.axvline(
                    De_hat, color="r", linestyle="--", label=f"E[D_e] = {De_hat:.3f}"
                )
    ax.set_xlabel("D_e")
    ax.set_ylabel("Densità")
    ax.set_title("Inversione Bayesiana: p(D_e | β_emp)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    _savefig(os.path.join(figdir, "calibration_curve.png"))


def plot_multi_curves_pnas(
    outdir, multi_curves: MultiCurveResult, beta_emp: float = None
):
    """
    Grafico combinato delle TRE curve β(D) PNAS:
    1) TL-JOINT
    2) TL-AR(m=100)
    3) Student benchmark
    """
    figdir = os.path.join(outdir, "fig")
    _ensure_dir(figdir)

    fig, ax = plt.subplots(figsize=(12, 8))

    De = multi_curves.tl_joint.De_grid

    # TL-JOINT
    valid_j = np.isfinite(multi_curves.tl_joint.beta_mean)
    if np.any(valid_j):
        sem_j = multi_curves.tl_joint.beta_std / np.sqrt(
            multi_curves.params.get("n_repl_joint", 30)
        )
        ax.errorbar(
            De[valid_j],
            multi_curves.tl_joint.beta_mean[valid_j],
            yerr=sem_j[valid_j],
            fmt="^-",
            linewidth=2,
            capsize=3,
            markersize=6,
            label="TL-JOINT (mixing per blocco)",
            color="blue",
        )

    # TL-AR
    valid_a = np.isfinite(multi_curves.tl_ar.beta_mean)
    if np.any(valid_a):
        sem_a = multi_curves.tl_ar.beta_std / np.sqrt(
            multi_curves.params.get("n_repl_ar", 15)
        )
        ax.errorbar(
            De[valid_a],
            multi_curves.tl_ar.beta_mean[valid_a],
            yerr=sem_a[valid_a],
            fmt="s--",
            linewidth=2,
            capsize=3,
            markersize=6,
            label=f"TL-AR (replica PNAS, m={M_AR})",
            color="orange",
        )

    # Student
    valid_s = np.isfinite(multi_curves.student.beta_mean)
    if np.any(valid_s):
        sem_s = multi_curves.student.beta_std / np.sqrt(
            multi_curves.params.get("n_repl_student", 25)
        )
        ax.errorbar(
            De[valid_s],
            multi_curves.student.beta_mean[valid_s],
            yerr=sem_s[valid_s],
            fmt="o:",
            linewidth=1.8,
            capsize=3,
            markersize=5,
            label=f"Student benchmark (ν={NU_STUDENT})",
            color="green",
        )

    # β empirico
    if beta_emp is not None and np.isfinite(beta_emp):
        ax.axhline(
            beta_emp,
            color="red",
            linestyle="-",
            linewidth=2.5,
            alpha=0.7,
            label=f"β empirico = {beta_emp:.4f}",
        )

    ax.set_xlabel(r"$D$", fontsize=14)
    ax.set_ylabel(r"$\beta$", fontsize=14)
    ax.set_title(
        r"Curva teorica $\beta(D)$ — Baldovin–Stella PNAS: TL-JOINT vs TL-AR vs Student",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xlim(De.min() - 0.02, De.max() + 0.02)

    _savefig(os.path.join(figdir, "beta_curves_pnas_combined.png"))

    # Grafici individuali
    for curve, name, color in [
        (multi_curves.tl_joint, "TL_JOINT", "blue"),
        (multi_curves.tl_ar, "TL_AR", "orange"),
        (multi_curves.student, "Student", "green"),
    ]:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        valid = np.isfinite(curve.beta_mean)
        if np.any(valid):
            ax2.plot(
                De[valid],
                curve.beta_mean[valid],
                "-",
                linewidth=2,
                color=color,
                label=name,
            )
            ax2.fill_between(
                De[valid],
                curve.beta_mean[valid] - curve.beta_std[valid],
                curve.beta_mean[valid] + curve.beta_std[valid],
                alpha=0.2,
                color=color,
            )
        if beta_emp is not None and np.isfinite(beta_emp):
            ax2.axhline(
                beta_emp,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"β emp = {beta_emp:.4f}",
            )
        ax2.set_xlabel(r"$D$", fontsize=14)
        ax2.set_ylabel(r"$\beta$", fontsize=14)
        ax2.set_title(
            f'Curva β(D) — {name}\n{curve.details.get("description", "")}', fontsize=12
        )
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        _savefig(os.path.join(figdir, f"beta_curve_{name}.png"))


# =============================================================================
# GRAFICI MAPPING D_e MULTIPLO (V3.5)
# =============================================================================


def plot_De_calibration_single(
    outdir,
    curve: BetaSimCurve,
    beta_est: BetaEstimate,
    De_est: DeEstimate,
    beta_method: str,
    curve_name: str,
):
    """
    Grafico calibrazione D_e per singola combinazione (metodo_β, curva).
    """
    figdir = os.path.join(outdir, "fig", "calibration")
    _ensure_dir(figdir)

    fig, ax = plt.subplots(figsize=(10, 7))

    De = curve.De_grid
    valid = np.isfinite(curve.beta_mean)

    if np.any(valid):
        # Curva β(D)
        ax.plot(De[valid], curve.beta_mean[valid], "b-", linewidth=2, label="β_sim(D)")
        ax.fill_between(
            De[valid],
            curve.beta_mean[valid] - curve.beta_std_smooth[valid],
            curve.beta_mean[valid] + curve.beta_std_smooth[valid],
            alpha=0.2,
            color="blue",
        )

    # β empirico con incertezza
    if np.isfinite(beta_est.beta):
        ax.axhline(
            beta_est.beta,
            color="red",
            linestyle="-",
            linewidth=2.5,
            label=f"β_{{{beta_method}}} = {beta_est.beta:.4f}",
        )
        if np.isfinite(beta_est.stderr):
            ax.axhspan(
                beta_est.beta - beta_est.stderr,
                beta_est.beta + beta_est.stderr,
                alpha=0.15,
                color="red",
            )

    # D_e stimato con incertezza
    if np.isfinite(De_est.De):
        ax.axvline(
            De_est.De,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"D_e = {De_est.De:.4f} ± {De_est.stderr:.4f}",
        )
        if np.isfinite(De_est.stderr):
            ax.axvspan(
                De_est.De - De_est.stderr,
                De_est.De + De_est.stderr,
                alpha=0.15,
                color="green",
            )

    ax.set_xlabel(r"$D$", fontsize=14)
    ax.set_ylabel(r"$\beta$", fontsize=14)
    ax.set_title(
        f"Calibrazione: β_{{{beta_method}}} → D_e su curva {curve_name}", fontsize=12
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Annotazione
    info_text = f"β = {beta_est.beta:.4f} ± {beta_est.stderr:.4f}\n"
    info_text += f"D_e = {De_est.De:.4f} ± {De_est.stderr:.4f}"
    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    _savefig(os.path.join(figdir, f"calibration_{beta_method}_{curve_name}.png"))


def plot_all_De_calibrations(
    outdir,
    multi_curves: MultiCurveResult,
    beta_estimates: Dict[str, BetaEstimate],
    De_multi_results: Dict[str, MultiDeResult],
):
    """
    Genera grafici di calibrazione per TUTTE le combinazioni (metodo_β, curva).
    """
    curves_map = {
        "TL_JOINT": multi_curves.tl_joint,
        "TL_AR": multi_curves.tl_ar,
        "Student": multi_curves.student,
    }

    for beta_method, multi_de in De_multi_results.items():
        if beta_method not in beta_estimates:
            continue
        beta_est = beta_estimates[beta_method]

        for curve_name, De_est in [
            ("TL_JOINT", multi_de.De_tl_joint),
            ("TL_AR", multi_de.De_tl_ar),
            ("Student", multi_de.De_student),
        ]:
            if curve_name in curves_map:
                plot_De_calibration_single(
                    outdir,
                    curves_map[curve_name],
                    beta_est,
                    De_est,
                    beta_method,
                    curve_name,
                )


def plot_De_summary_all_combinations(
    outdir, De_multi_results: Dict[str, MultiDeResult]
):
    """
    Grafico riassuntivo D_e per tutte le combinazioni (metodo_β, curva).
    """
    figdir = os.path.join(outdir, "fig")
    _ensure_dir(figdir)

    fig, ax = plt.subplots(figsize=(14, 8))

    # Prepara dati
    labels, values, errors, colors = [], [], [], []
    color_map = {"TL_JOINT": "blue", "TL_AR": "orange", "Student": "green"}

    x_pos = 0
    x_positions = []
    group_starts = []

    for beta_method in De_multi_results.keys():
        group_starts.append(x_pos)
        res = De_multi_results[beta_method]

        for curve_name, De_est in [
            ("TL_JOINT", res.De_tl_joint),
            ("TL_AR", res.De_tl_ar),
            ("Student", res.De_student),
        ]:
            if np.isfinite(De_est.De):
                labels.append(f"{beta_method}\n{curve_name}")
                values.append(De_est.De)
                errors.append(De_est.stderr if np.isfinite(De_est.stderr) else 0)
                colors.append(color_map.get(curve_name, "gray"))
                x_positions.append(x_pos)
            x_pos += 1
        x_pos += 0.5  # Gap tra gruppi

    if len(values) > 0:
        ax.bar(
            x_positions,
            values,
            yerr=errors,
            capsize=4,
            color=colors,
            alpha=0.7,
            edgecolor="black",
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.axhline(
            0.25, color="red", linestyle="--", alpha=0.5, label="D_e = 0.25 (PNAS DJI)"
        )
        ax.set_ylabel(r"$D_e$", fontsize=14)
        ax.set_title(
            "Stima D_e: tutte le combinazioni (metodo β × curva simulata)", fontsize=12
        )
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    _savefig(os.path.join(figdir, "De_all_combinations_summary.png"))


# =============================================================================
# GRAFICI STIMA FINALE PESATA (V3.5)
# =============================================================================


def plot_final_weighted_estimates(
    outdir, final_beta: WeightedEstimate, final_De: WeightedEstimate, D_est: DEstimate
):
    """
    Grafico riassuntivo delle stime finali pesate.
    """
    figdir = os.path.join(outdir, "fig")
    _ensure_dir(figdir)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: final β with per-method contributions
    ax = axes[0]
    if len(final_beta.individual_values) > 0:
        methods = list(final_beta.individual_values.keys())
        vals = [final_beta.individual_values[m] for m in methods]
        weights = [final_beta.weights.get(m, 0) for m in methods]
        colors = plt.cm.Blues(np.array(weights) / max(weights) * 0.7 + 0.3)

        bars = ax.barh(methods, vals, color=colors, edgecolor="black", alpha=0.8)
        ax.axvline(
            final_beta.value,
            color="red",
            linewidth=3,
            linestyle="-",
            label=f"β finale = {final_beta.value:.4f}",
        )
        ax.axvspan(final_beta.ci[0], final_beta.ci[1], alpha=0.2, color="red")

        # Annotazione pesi
        for i, (bar, w) in enumerate(zip(bars, weights)):
            ax.text(
                bar.get_width() + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"w={w:.2f}",
                va="center",
                fontsize=8,
            )

    ax.set_xlabel(r"$\beta$", fontsize=12)
    ax.set_title(
        f"Stima finale β\n(media pesata, σ = {final_beta.stderr:.4f})", fontsize=11
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")

    # Panel 2: final D_e with per-method contributions
    ax = axes[1]
    if len(final_De.individual_values) > 0:
        methods = list(final_De.individual_values.keys())
        vals = [final_De.individual_values[m] for m in methods]
        weights = [final_De.weights.get(m, 0) for m in methods]
        colors = plt.cm.Greens(np.array(weights) / max(weights) * 0.7 + 0.3)

        bars = ax.barh(methods, vals, color=colors, edgecolor="black", alpha=0.8)
        ax.axvline(
            final_De.value,
            color="red",
            linewidth=3,
            linestyle="-",
            label=f"D_e finale = {final_De.value:.4f}",
        )
        ax.axvspan(final_De.ci[0], final_De.ci[1], alpha=0.2, color="red")

    ax.set_xlabel(r"$D_e$", fontsize=12)
    ax.set_title(
        f"Stima finale D_e\n(media pesata, σ = {final_De.stderr:.4f})", fontsize=11
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")

    # Panel 3: D vs D_e comparison
    ax = axes[2]
    estimates = {
        "D (quantili)": (D_est.D, D_est.stderr),
        "D_e (finale)": (final_De.value, final_De.stderr),
    }

    y_pos = list(range(len(estimates)))
    labels = list(estimates.keys())
    vals = [estimates[k][0] for k in labels]
    errs = [estimates[k][1] if np.isfinite(estimates[k][1]) else 0 for k in labels]

    ax.barh(
        y_pos,
        vals,
        xerr=errs,
        capsize=5,
        color=["purple", "green"],
        alpha=0.7,
        edgecolor="black",
    )
    ax.axvline(0.5, color="blue", linestyle="--", alpha=0.5, label="D = 0.5 (EMH)")
    ax.axvline(
        0.25, color="orange", linestyle="--", alpha=0.5, label="D_e ≈ 0.24 (PNAS)"
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Value", fontsize=12)
    ax.set_title("Confronto D vs D_e", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    _savefig(os.path.join(figdir, "final_weighted_estimates.png"))


def plot_weights_breakdown(
    outdir, final_beta: WeightedEstimate, final_De: WeightedEstimate
):
    """
    Grafico dettagliato dei pesi usati nelle stime finali.
    """
    figdir = os.path.join(outdir, "fig")
    _ensure_dir(figdir)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pesi β
    ax = axes[0]
    if len(final_beta.weights) > 0:
        methods = list(final_beta.weights.keys())
        weights = [final_beta.weights[m] for m in methods]
        ax.pie(weights, labels=methods, autopct="%1.1f%%", startangle=90)
        ax.set_title(
            f"Pesi per stima β finale\nMetodi usati: {len(methods)}", fontsize=11
        )

    # Pesi D_e
    ax = axes[1]
    if len(final_De.weights) > 0:
        methods = list(final_De.weights.keys())
        weights = [final_De.weights[m] for m in methods]

        # Se troppi metodi, mostra solo top 10
        if len(methods) > 10:
            sorted_idx = np.argsort(weights)[::-1][:10]
            methods = [methods[i] for i in sorted_idx]
            weights = [weights[i] for i in sorted_idx]
            ax.pie(weights, labels=methods, autopct="%1.1f%%", startangle=90)
            ax.set_title(
                f"Pesi per stima D_e finale (top 10)\nCombinazioni totali: {len(final_De.weights)}",
                fontsize=11,
            )
        else:
            ax.pie(weights, labels=methods, autopct="%1.1f%%", startangle=90)
            ax.set_title(
                f"Pesi per stima D_e finale\nCombinazioni usate: {len(methods)}",
                fontsize=11,
            )

    plt.tight_layout()
    _savefig(os.path.join(figdir, "weights_breakdown.png"))


# =============================================================================
# V3.5: GRAFICI SERIE STORICHE β E D_e PER TUTTI I METODI E CURVE
# =============================================================================


def plot_beta_all_methods_timeseries(outdir, df, dates=None):
    """
    Rolling β time series for all estimation methods (ACF, HAC, DFA, Bayes, Whittle).
    """
    figdir = os.path.join(outdir, "fig", "timeseries")
    _ensure_dir(figdir)

    if df.empty:
        return

    methods = ["ACF", "HAC", "DFA", "Bayes", "Whittle"]
    colors = ["blue", "orange", "green", "red", "purple"]

    # x-axis
    if "window_date" in df.columns:
        x = pd.to_datetime(df["window_date"])
    else:
        x = df["center_index"].values

    fig, ax = plt.subplots(figsize=(14, 7))

    for method, color in zip(methods, colors):
        col = f"beta_{method.lower()}"
        stderr_col = f"beta_{method.lower()}_stderr"

        if col in df.columns:
            y = df[col].values
            valid = np.isfinite(y)
            if np.sum(valid) > 0:
                ax.plot(
                    np.array(x)[valid],
                    y[valid],
                    "-",
                    color=color,
                    linewidth=1.5,
                    alpha=0.8,
                    label=f"β_{method}",
                )

                if stderr_col in df.columns:
                    stderr = df[stderr_col].values
                    stderr_valid = np.isfinite(stderr) & valid
                    if np.sum(stderr_valid) > 0:
                        ax.fill_between(
                            np.array(x)[stderr_valid],
                            (y - stderr)[stderr_valid],
                            (y + stderr)[stderr_valid],
                            alpha=0.15,
                            color=color,
                        )

    ax.set_xlabel(
        "Date" if "window_date" in df.columns else "Window index", fontsize=12
    )
    ax.set_ylabel(r"$\beta$", fontsize=14)
    ax.set_title(r"Rolling $\beta$ — all methods", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _savefig(os.path.join(figdir, "beta_all_methods_timeseries.png"))


def plot_De_all_combinations_timeseries(outdir, df, dates=None):
    """
    Rolling D_e time series for all (β-method × calibration curve) combinations.
    """
    figdir = os.path.join(outdir, "fig", "timeseries")
    _ensure_dir(figdir)

    if df.empty:
        return

    methods = ["acf", "hac", "dfa", "bayes", "whittle"]
    curves = ["TL_JOINT", "TL_AR", "Student"]
    curve_styles = {"TL_JOINT": "-", "TL_AR": "--", "Student": ":"}
    method_colors = {
        "acf": "blue",
        "hac": "orange",
        "dfa": "green",
        "bayes": "red",
        "whittle": "purple",
    }

    # x-axis
    if "window_date" in df.columns:
        x = pd.to_datetime(df["window_date"])
    else:
        x = df["center_index"].values

    # Un grafico per curva
    for curve in curves:
        fig, ax = plt.subplots(figsize=(14, 7))

        for method in methods:
            col = f"De_{method}_{curve}"

            if col in df.columns:
                y = df[col].values
                valid = np.isfinite(y)
                if np.sum(valid) > 0:
                    ax.plot(
                        np.array(x)[valid],
                        y[valid],
                        "-",
                        color=method_colors[method],
                        linewidth=1.5,
                        alpha=0.8,
                        label=f"D_e ({method.upper()})",
                    )

        ax.set_xlabel(
            "Date" if "window_date" in df.columns else "Window index", fontsize=12
        )
        ax.set_ylabel(r"$D_e$", fontsize=14)
        ax.set_title(f"Rolling $D_e$ — {curve} curve", fontsize=12)
        ax.axhline(
            0.25, color="gray", linestyle="--", alpha=0.5, label="D_e ≈ 0.25 (PNAS)"
        )
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        _savefig(os.path.join(figdir, f"De_{curve}_all_methods_timeseries.png"))





def plot_zscore_beta_timeseries(outdir, df, dates=None):
    """
    Z-score di β per tutti i metodi.
    """
    figdir = os.path.join(outdir, "fig", "zscore")
    _ensure_dir(figdir)

    if df.empty:
        return

    methods = ["ACF", "HAC", "DFA", "Bayes", "Whittle"]
    colors = ["blue", "orange", "green", "red", "purple"]

    # x-axis
    if "window_date" in df.columns:
        x = pd.to_datetime(df["window_date"])
    else:
        x = df["center_index"].values

    fig, ax = plt.subplots(figsize=(14, 7))

    for method, color in zip(methods, colors):
        col = f"beta_{method.lower()}"

        if col in df.columns:
            z = compute_zscore(df[col].values)
            valid = np.isfinite(z)
            if np.sum(valid) > 0:
                ax.plot(
                    np.array(x)[valid],
                    z[valid],
                    "-",
                    color=color,
                    linewidth=1.5,
                    alpha=0.8,
                    label=f"z(β_{method})",
                )

    ax.axhline(0, color="black", linestyle="-", linewidth=1)
    ax.axhline(2, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(-2, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel(
        "Date" if "window_date" in df.columns else "Window index", fontsize=12
    )
    ax.set_ylabel(r"Z-score", fontsize=14)
    ax.set_title(r"Z-score di $\beta$ — Tutti i metodi", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _savefig(os.path.join(figdir, "zscore_beta_all_methods.png"))


def plot_zscore_De_timeseries(outdir, df, curves=["TL_JOINT", "TL_AR", "Student"]):
    """
    Z-score di D_e per tutte le combinazioni.
    """
    figdir = os.path.join(outdir, "fig", "zscore")
    _ensure_dir(figdir)

    if df.empty:
        return

    methods = ["acf", "hac", "dfa", "bayes", "whittle"]
    method_colors = {
        "acf": "blue",
        "hac": "orange",
        "dfa": "green",
        "bayes": "red",
        "whittle": "purple",
    }

    # x-axis
    if "window_date" in df.columns:
        x = pd.to_datetime(df["window_date"])
    else:
        x = df["center_index"].values

    for curve in curves:
        fig, ax = plt.subplots(figsize=(14, 7))

        for method in methods:
            col = f"De_{method}_{curve}"

            if col in df.columns:
                z = compute_zscore(df[col].values)
                valid = np.isfinite(z)
                if np.sum(valid) > 0:
                    ax.plot(
                        np.array(x)[valid],
                        z[valid],
                        "-",
                        color=method_colors[method],
                        linewidth=1.5,
                        alpha=0.8,
                        label=f"z(D_e_{method.upper()})",
                    )

        ax.axhline(0, color="black", linestyle="-", linewidth=1)
        ax.axhline(2, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(-2, color="gray", linestyle="--", alpha=0.5)

        ax.set_xlabel(
            "Date" if "window_date" in df.columns else "Window index", fontsize=12
        )
        ax.set_ylabel(r"Z-score", fontsize=14)
        ax.set_title(f"Z-score di $D_e$ — Curva {curve}", fontsize=12)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        _savefig(os.path.join(figdir, f"zscore_De_{curve}_all_methods.png"))


def plot_zscore_corr_cov_vs_D(outdir, df, window_size=20, step=5):
    """
    Correlazione e covarianza rolling tra z-score di (β, D_e) e D.
    Crea grafici per TUTTE le combinazioni (metodo × curva).
    In sovrimpressione: eventi esogeni principali (S&P500) come linee verticali tratteggiate,
    solo se l'asse x è temporale (colonna 'window_date').
    """
    figdir = os.path.join(outdir, "fig", "zscore_correlation")
    _ensure_dir(figdir)

    if df.empty or len(df) < window_size:
        return

    methods = ["acf", "hac", "dfa", "bayes", "whittle"]
    curves = ["TL_JOINT", "TL_AR", "Student"]

    # --- Eventi esogeni (date ancora) ---
    exog_events = [
        ("1907 Panic", "1907-10-01"),
        ("1929 Crash", "1929-10-24"),
        ("WWII", "1939-09-01"),
        ("Oil Shock", "1973-10-01"),
        ("Black Monday", "1987-10-19"),
        ("Dot-com", "2000-03-10"),
        ("GFC", "2008-09-15"),
        ("COVID", "2020-03-01"),
    ]

    def _overlay_events(ax, x_is_datetime: bool, label_first: bool = True):
        """
        Disegna linee verticali tratteggiate per eventi esogeni.
        Se x non è datetime (asse indice), non disegna nulla.
        """
        if not x_is_datetime:
            return
        first = True
        for _, d in exog_events:
            dt = pd.to_datetime(d)
            if first and label_first:
                ax.axvline(
                    dt,
                    color="black",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.7,
                    label="Eventi esogeni",
                )
                first = False
            else:
                ax.axvline(dt, color="black", linestyle="--", linewidth=1.2, alpha=0.7)

    # --- z(D) ---
    D = df["D"].values
    z_D = compute_zscore(D)

    # x-axis
    x_is_datetime = "window_date" in df.columns
    if x_is_datetime:
        x_all = pd.to_datetime(df["window_date"])
    else:
        x_all = df["center_index"].values

    # ===== (β, D) correlation/covariance =====
    fig_beta, axes_beta = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    for method in methods:
        beta_col = f"beta_{method}"
        if beta_col not in df.columns:
            continue

        z_beta = compute_zscore(df[beta_col].values)

        corrs, covs, x_pts = [], [], []
        for start in range(0, len(df) - window_size + 1, step):
            end = start + window_size
            center = start + window_size // 2

            zb = z_beta[start:end]
            zd = z_D[start:end]
            valid = np.isfinite(zb) & np.isfinite(zd)

            if np.sum(valid) >= 5:
                corr = np.corrcoef(zb[valid], zd[valid])[0, 1]
                cov = np.cov(zb[valid], zd[valid], ddof=1)[0, 1]
                corrs.append(corr)
                covs.append(cov)
                x_pts.append(x_all[center] if center < len(x_all) else center)

        if len(corrs) > 0:
            axes_beta[0].plot(
                x_pts, corrs, "-", linewidth=1.5, alpha=0.8, label=f"β_{method.upper()}"
            )
            axes_beta[1].plot(
                x_pts, covs, "-", linewidth=1.5, alpha=0.8, label=f"β_{method.upper()}"
            )

    # Eventi esogeni (una sola entry in legenda sul primo subplot)
    _overlay_events(axes_beta[0], x_is_datetime, label_first=True)
    _overlay_events(axes_beta[1], x_is_datetime, label_first=False)

    axes_beta[0].axhline(0, color="black", linestyle="-", linewidth=1)
    axes_beta[0].set_ylabel("Correlazione", fontsize=12)
    axes_beta[0].set_title(r"Correlazione rolling tra z($\beta$) e z(D)", fontsize=12)
    axes_beta[0].legend(loc="upper right", fontsize=9)
    axes_beta[0].grid(True, alpha=0.3)

    axes_beta[1].axhline(0, color="black", linestyle="-", linewidth=1)
    axes_beta[1].set_xlabel("Date" if x_is_datetime else "Window index", fontsize=12)
    axes_beta[1].set_ylabel("Covarianza", fontsize=12)
    axes_beta[1].set_title(r"Covarianza rolling tra z($\beta$) e z(D)", fontsize=12)
    axes_beta[1].legend(loc="upper right", fontsize=9)
    axes_beta[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _savefig(os.path.join(figdir, "zscore_beta_D_corr_cov.png"))
    plt.close(fig_beta)

    # ===== (D_e, D) correlation/covariance per ogni curva =====
    for curve in curves:
        fig_de, axes_de = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        for method in methods:
            De_col = f"De_{method}_{curve}"
            if De_col not in df.columns:
                continue

            z_De = compute_zscore(df[De_col].values)

            corrs, covs, x_pts = [], [], []
            for start in range(0, len(df) - window_size + 1, step):
                end = start + window_size
                center = start + window_size // 2

                zde = z_De[start:end]
                zd = z_D[start:end]
                valid = np.isfinite(zde) & np.isfinite(zd)

                if np.sum(valid) >= 5:
                    corr = np.corrcoef(zde[valid], zd[valid])[0, 1]
                    cov = np.cov(zde[valid], zd[valid], ddof=1)[0, 1]
                    corrs.append(corr)
                    covs.append(cov)
                    x_pts.append(x_all[center] if center < len(x_all) else center)

            if len(corrs) > 0:
                axes_de[0].plot(
                    x_pts,
                    corrs,
                    "-",
                    linewidth=1.5,
                    alpha=0.8,
                    label=f"D_e ({method.upper()})",
                )
                axes_de[1].plot(
                    x_pts,
                    covs,
                    "-",
                    linewidth=1.5,
                    alpha=0.8,
                    label=f"D_e ({method.upper()})",
                )

        # Eventi esogeni (una sola entry in legenda sul primo subplot)
        _overlay_events(axes_de[0], x_is_datetime, label_first=True)
        _overlay_events(axes_de[1], x_is_datetime, label_first=False)

        axes_de[0].axhline(0, color="black", linestyle="-", linewidth=1)
        axes_de[0].set_ylabel("Correlazione", fontsize=12)
        axes_de[0].set_title(
            f"Correlazione rolling tra z($D_e$) e z(D) — Curva {curve}", fontsize=12
        )
        axes_de[0].legend(loc="upper right", fontsize=9)
        axes_de[0].grid(True, alpha=0.3)

        axes_de[1].axhline(0, color="black", linestyle="-", linewidth=1)
        axes_de[1].set_xlabel("Date" if x_is_datetime else "Window index", fontsize=12)
        axes_de[1].set_ylabel("Covarianza", fontsize=12)
        axes_de[1].set_title(
            f"Covarianza rolling tra z($D_e$) e z(D) — Curva {curve}", fontsize=12
        )
        axes_de[1].legend(loc="upper right", fontsize=9)
        axes_de[1].grid(True, alpha=0.3)

        plt.tight_layout()
        _savefig(os.path.join(figdir, f"zscore_De_{curve}_D_corr_cov.png"))
        plt.close(fig_de)


def plot_summary_all_De_timeseries(outdir, df):
    """
    Grafico riassuntivo con D_e per tutte le curve in sottopannelli.
    """
    figdir = os.path.join(outdir, "fig", "timeseries")
    _ensure_dir(figdir)

    if df.empty:
        return

    curves = ["TL_JOINT", "TL_AR", "Student"]
    methods = ["acf", "hac", "dfa", "bayes", "whittle"]
    method_colors = {
        "acf": "blue",
        "hac": "orange",
        "dfa": "green",
        "bayes": "red",
        "whittle": "purple",
    }

    # x-axis
    if "window_date" in df.columns:
        x = pd.to_datetime(df["window_date"])
    else:
        x = df["center_index"].values

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    for i, curve in enumerate(curves):
        ax = axes[i]

        for method in methods:
            col = f"De_{method}_{curve}"

            if col in df.columns:
                y = df[col].values
                valid = np.isfinite(y)
                if np.sum(valid) > 0:
                    ax.plot(
                        np.array(x)[valid],
                        y[valid],
                        "-",
                        color=method_colors[method],
                        linewidth=1.5,
                        alpha=0.8,
                        label=f"{method.upper()}",
                    )

        ax.axhline(0.25, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel(r"$D_e$", fontsize=12)
        ax.set_title(f"Curva {curve}", fontsize=11)
        ax.legend(loc="upper right", fontsize=9, ncol=5)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(
        "Date" if "window_date" in df.columns else "Window index", fontsize=12
    )
    fig.suptitle("Rolling $D_e$ — all curves and methods", fontsize=14)

    plt.tight_layout()
    _savefig(os.path.join(figdir, "De_all_curves_all_methods_summary.png"))


# =============================================================================
# GRAFICI SINGOLI CORRELAZIONE/COVARIANZA (V3.5-3)
# =============================================================================


def plot_single_correlation_series(outdir, df, window_size=20, step=5):
    """
    Genera grafici SINGOLI per ogni combinazione di:
    - (beta_metodo, D)
    - (De_metodo_curva, D)

    Ogni grafico mostra:
    - Rolling correlation between z-scored series with bootstrap confidence band
    - Rolling covariance between z-scored series
    - Asse x formattato in date

    Plots are saved in the 'single_correlation' subdirectory.
    """
    figdir = os.path.join(outdir, "single_correlation")
    _ensure_dir(figdir)

    if df.empty or len(df) < window_size:
        _log("DataFrame troppo corto per analisi correlazione singola")
        return

    # Definizioni
    beta_methods = ["acf", "hac", "dfa", "bayes", "whittle"]
    curves = ["TL_JOINT", "TL_AR", "Student"]

    # Z-score di D
    D = df["D"].values
    z_D = compute_zscore(D)

    # x-axis
    if "window_date" in df.columns and df["window_date"].notna().any():
        x_all = pd.to_datetime(df["window_date"])
        use_dates = True
    else:
        x_all = df["center_index"].values
        use_dates = False

    def compute_rolling_stats(z_series, z_D, x_all, window_size, step):
        """Compute rolling correlation and covariance with bootstrap confidence bands."""
        corrs, covs, x_pts = [], [], []
        corr_ci_lo, corr_ci_hi = [], []
        cov_ci_lo, cov_ci_hi = [], []

        for start in range(0, len(df) - window_size + 1, step):
            end = start + window_size
            center = start + window_size // 2

            zs = z_series[start:end]
            zd = z_D[start:end]
            valid = np.isfinite(zs) & np.isfinite(zd)
            n_valid = int(np.sum(valid))

            if n_valid >= 5:
                zs_v, zd_v = zs[valid], zd[valid]

                # Correlazione
                corr = np.corrcoef(zs_v, zd_v)[0, 1]
                corrs.append(corr)

                # CI per correlazione (Fisher z-transform)
                if n_valid >= 4 and abs(corr) < 0.9999:
                    z_fisher = np.arctanh(corr)
                    se_z = 1.0 / np.sqrt(max(1.0, n_valid - 3.0))
                    ci_lo = float(np.tanh(z_fisher - CI * se_z))
                    ci_hi = float(np.tanh(z_fisher + CI * se_z))
                else:
                    ci_lo, ci_hi = corr, corr
                corr_ci_lo.append(ci_lo)
                corr_ci_hi.append(ci_hi)

                # Covarianza
                cov = np.cov(zs_v, zd_v, ddof=1)[0, 1]
                covs.append(cov)

                # CI approssimativo per covarianza (bootstrap-like usando SE)
                # SE(cov) ≈ (1 + r²) / sqrt(n-1) * std(z1) * std(z2)
                std_zs = np.std(zs_v, ddof=1)
                std_zd = np.std(zd_v, ddof=1)
                if n_valid > 3:
                    se_cov = (1 + corr**2) / np.sqrt(n_valid - 1) * std_zs * std_zd
                    cov_ci_lo.append(cov - CI * se_cov)
                    cov_ci_hi.append(cov + CI * se_cov)
                else:
                    cov_ci_lo.append(cov)
                    cov_ci_hi.append(cov)

                x_pts.append(x_all[center] if center < len(x_all) else center)
            else:
                corrs.append(np.nan)
                covs.append(np.nan)
                corr_ci_lo.append(np.nan)
                corr_ci_hi.append(np.nan)
                cov_ci_lo.append(np.nan)
                cov_ci_hi.append(np.nan)
                x_pts.append(x_all[center] if center < len(x_all) else center)

        return {
            "x": x_pts,
            "corr": np.array(corrs),
            "corr_ci_lo": np.array(corr_ci_lo),
            "corr_ci_hi": np.array(corr_ci_hi),
            "cov": np.array(covs),
            "cov_ci_lo": np.array(cov_ci_lo),
            "cov_ci_hi": np.array(cov_ci_hi),
        }

    def create_single_plot(stats, title, filename, var_name, use_dates):
        """Crea un singolo grafico con correlazione e covarianza."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        x = np.array(stats["x"])
        corr = stats["corr"]
        corr_lo = stats["corr_ci_lo"]
        corr_hi = stats["corr_ci_hi"]
        cov = stats["cov"]
        cov_lo = stats["cov_ci_lo"]
        cov_hi = stats["cov_ci_hi"]

        # Maschera per valori validi
        valid = np.isfinite(corr)

        if np.sum(valid) < 2:
            plt.close(fig)
            return

        # Panel 1: rolling correlation with confidence band
        ax = axes[0]
        ax.plot(
            x[valid], corr[valid], "b-", linewidth=2, label="Correlazione", zorder=3
        )
        ax.fill_between(
            x[valid],
            corr_lo[valid],
            corr_hi[valid],
            alpha=0.3,
            color="blue",
            label="IC 95%",
            zorder=2,
        )
        ax.axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.7)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
        ax.axhline(-0.5, color="gray", linestyle="--", alpha=0.4)
        ax.set_ylabel("Correlazione", fontsize=12)
        ax.set_title(f"Correlazione rolling tra z({var_name}) e z(D)", fontsize=12)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.1, 1.1)

        if use_dates:
            _format_time_axis(ax)

        # Panel 2: rolling covariance with confidence band
        ax = axes[1]
        ax.plot(x[valid], cov[valid], "r-", linewidth=2, label="Covarianza", zorder=3)
        ax.fill_between(
            x[valid],
            cov_lo[valid],
            cov_hi[valid],
            alpha=0.3,
            color="red",
            label="IC 95%",
            zorder=2,
        )
        ax.axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.7)
        ax.set_xlabel("Date" if use_dates else "Window index", fontsize=12)
        ax.set_ylabel("Covarianza", fontsize=12)
        ax.set_title(f"Covarianza rolling tra z({var_name}) e z(D)", fontsize=12)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)

        if use_dates:
            _format_time_axis(ax)

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        _savefig(os.path.join(figdir, filename))

    # ========== GRAFICI PER BETA ==========
    _log("Generating individual β vs D correlation plots...")

    for method in beta_methods:
        beta_col = f"beta_{method}"

        if beta_col not in df.columns:
            continue

        z_beta = compute_zscore(df[beta_col].values)

        if not np.any(np.isfinite(z_beta)):
            continue

        stats = compute_rolling_stats(z_beta, z_D, x_all, window_size, step)

        if len(stats["x"]) > 0:
            title = f"Correlazione e Covarianza: β_{method.upper()} vs D"
            filename = f"single_corr_beta_{method}_vs_D.png"
            create_single_plot(stats, title, filename, f"β_{method.upper()}", use_dates)

    # ========== GRAFICI PER D_e ==========
    _log("Generating individual D_e vs D correlation plots...")

    for method in beta_methods:
        for curve in curves:
            de_col = f"De_{method}_{curve}"

            if de_col not in df.columns:
                continue

            z_De = compute_zscore(df[de_col].values)

            if not np.any(np.isfinite(z_De)):
                continue

            stats = compute_rolling_stats(z_De, z_D, x_all, window_size, step)

            if len(stats["x"]) > 0:
                title = (
                    f"Correlazione e Covarianza: D_e ({method.upper()}, {curve}) vs D"
                )
                filename = f"single_corr_De_{method}_{curve}_vs_D.png"
                create_single_plot(
                    stats,
                    title,
                    filename,
                    f"D_e_{{{method.upper()},{curve}}}",
                    use_dates,
                )

    _log(f"Grafici singoli correlazione salvati in: {figdir}")


def plot_global_comparison(outdir, D_est, beta_estimates, De_estimates):
    """Grafico confronto globale β, D_e, D."""
    figdir = os.path.join(outdir, "fig")
    _ensure_dir(figdir)
    cl = 100.0 * math.erf(CI / math.sqrt(2.0))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    methods, betas, errs = [], [], []
    for name, est in beta_estimates.items():
        if np.isfinite(est.beta):
            methods.append(name)
            betas.append(est.beta)
            errs.append(CI * est.stderr if np.isfinite(est.stderr) else 0)
    if methods:
        y = np.arange(len(methods))
        ax.errorbar(betas, y, xerr=errs, fmt="o", capsize=5, markersize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(methods)
        ax.axvline(0.2, color="r", linestyle="--", alpha=0.5, label="β ≈ 0.2")
        ax.legend()
    ax.set_xlabel("β")
    ax.set_title(f"Confronto stime β (CI ≈ {cl:.0f}%)")
    ax.grid(True, axis="x", alpha=0.3)

    ax = axes[1]
    methods, des, errs = [], [], []
    for name, est in De_estimates.items():
        if np.isfinite(est.De):
            methods.append(name)
            des.append(est.De)
            errs.append(CI * est.stderr if np.isfinite(est.stderr) else 0)
    if methods:
        y = np.arange(len(methods))
        ax.errorbar(des, y, xerr=errs, fmt="o", capsize=5, markersize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(methods)
        ax.axvline(0.25, color="r", linestyle="--", alpha=0.5, label="D_e ≈ 0.25")
        ax.legend()
    ax.set_xlabel("D_e")
    ax.set_title(f"Confronto stime D_e (CI ≈ {cl:.0f}%)")
    ax.grid(True, axis="x", alpha=0.3)

    ax = axes[2]
    err = CI * D_est.stderr if np.isfinite(D_est.stderr) else 0
    ax.bar(["D"], [D_est.D], yerr=[err], capsize=5, color="steelblue", alpha=0.7)
    ax.axhline(0.5, color="r", linestyle="--", alpha=0.5, label="D = 0.5")
    ax.set_ylabel("D")
    ax.set_title(f"D = {D_est.D:.4f} ± {err:.4f}")
    ax.legend()

    _savefig(os.path.join(figdir, "global_comparison.png"))


def plot_historical_evolution(outdir, df, dates=None):
    """Grafico evoluzione storica combinato (3 pannelli: β, D, D_e)."""
    figdir = os.path.join(outdir, "fig")
    _ensure_dir(figdir)
    if df.empty:
        return

    # Determina asse x
    if "window_date" in df.columns and df["window_date"].notna().any():
        x = pd.to_datetime(df["window_date"])
        is_time = True
    elif dates is not None and "center_index" in df.columns:
        try:
            idx = df["center_index"].astype(int).values
            idx = np.clip(idx, 0, len(dates) - 1)
            x = pd.to_datetime(dates[idx])
            is_time = True
        except:
            x = df["center_index"].values
            is_time = False
    else:
        x = (
            df["center_index"].values
            if "center_index" in df.columns
            else np.arange(len(df))
        )
        is_time = False

    cl = 100.0 * math.erf(CI / math.sqrt(2.0))
    colors = {
        "ACF": "blue",
        "HAC": "cyan",
        "DFA": "green",
        "Bayes": "red",
        "Whittle": "orange",
    }

    def _plot_with_ci(ax, x, y, se, label, color):
        y, se = np.asarray(y, float), np.asarray(se, float)
        mask = np.isfinite(y)
        if not np.any(mask):
            return
        ax.plot(x, y, "-", color=color, label=label, linewidth=1.5)
        if np.any(np.isfinite(se)):
            lo = np.where(np.isfinite(se), y - CI * se, y)
            hi = np.where(np.isfinite(se), y + CI * se, y)
            ax.fill_between(x, lo, hi, alpha=0.15, color=color)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Pannello β
    ax = axes[0]
    for col, label in [
        ("beta_acf", "ACF"),
        ("beta_hac", "HAC"),
        ("beta_dfa", "DFA"),
        ("beta_bayes", "Bayes"),
        ("beta_whittle", "Whittle"),
    ]:
        if col in df.columns:
            se = df.get(col + "_stderr", pd.Series([np.nan] * len(df))).values
            _plot_with_ci(ax, x, df[col].values, se, label, colors[label])
    ax.axhline(0.2, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("β")
    ax.set_title(f"Evoluzione β (CI ≈ {cl:.0f}%)")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)

    # Pannello D
    ax = axes[1]
    _plot_with_ci(
        ax,
        x,
        df["D"].values,
        df.get("D_stderr", pd.Series([np.nan] * len(df))).values,
        "D",
        "purple",
    )
    ax.axhline(0.5, color="r", linestyle="--", alpha=0.5)
    ax.set_ylabel("D")
    ax.set_title(f"Evoluzione D (CI ≈ {cl:.0f}%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Pannello D_e
    ax = axes[2]
    for col, label in [("De_acf", "ACF"), ("De_dfa", "DFA"), ("De_bayes", "Bayes")]:
        if col in df.columns:
            se = df.get(col + "_stderr", pd.Series([np.nan] * len(df))).values
            _plot_with_ci(ax, x, df[col].values, se, label, colors[label])
    ax.axhline(0.25, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Date" if is_time else "Window index")
    ax.set_ylabel("D_e")
    ax.set_title(f"Evoluzione D_e (CI ≈ {cl:.0f}%)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if is_time:
        _format_time_axis(axes[2])

    fig.suptitle("Evoluzione storica: β, D, D_e", fontsize=14)
    _savefig(os.path.join(figdir, "historical_evolution.png"))


def plot_beta_separate(outdir, df, dates=None):
    """Grafici separati evoluzione β per metodo."""
    figdir = os.path.join(outdir, "fig")
    _ensure_dir(figdir)
    if df.empty:
        return

    if "window_date" in df.columns and df["window_date"].notna().any():
        x = pd.to_datetime(df["window_date"])
        is_time = True
    else:
        x = (
            df["center_index"].values
            if "center_index" in df.columns
            else np.arange(len(df))
        )
        is_time = False

    cl = 100.0 * math.erf(CI / math.sqrt(2.0))

    for method, col, color in [
        ("ACF", "beta_acf", "blue"),
        ("HAC", "beta_hac", "cyan"),
        ("DFA", "beta_dfa", "green"),
        ("Bayes", "beta_bayes", "red"),
        ("Whittle", "beta_whittle", "orange"),
    ]:
        if col not in df.columns:
            continue
        y = pd.to_numeric(df[col], errors="coerce").values
        if not np.any(np.isfinite(y)):
            continue

        fig, ax = plt.subplots(figsize=(12, 5))
        se = df.get(col + "_stderr", pd.Series([np.nan] * len(df))).values
        ax.plot(x, y, "-", color=color, linewidth=1.5)
        if np.any(np.isfinite(se)):
            lo = np.where(np.isfinite(se), y - CI * se, y)
            hi = np.where(np.isfinite(se), y + CI * se, y)
            ax.fill_between(x, lo, hi, alpha=0.2, color=color)
        ax.axhline(0.2, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Date" if is_time else "Window index")
        ax.set_ylabel(f"β ({method})")
        ax.set_title(f"Evoluzione β - Metodo {method} (CI ≈ {cl:.0f}%)")
        ax.grid(True, alpha=0.3)
        if is_time:
            _format_time_axis(ax)
        _savefig(os.path.join(figdir, f"beta_{method.lower()}_evolution.png"))


def plot_correlations_rolling(outdir, df, dates=None, window_size=20, step=5):
    """Grafici correlazioni rolling tra metodi β."""
    figdir = os.path.join(outdir, "fig")
    _ensure_dir(figdir)
    if df.empty or len(df) < window_size:
        return

    # Determina se abbiamo date
    if "window_date" in df.columns and df["window_date"].notna().any():
        x_dates = pd.to_datetime(df["window_date"])
        is_time = True
    else:
        x_dates = None
        is_time = False

    pairs = [
        ("beta_acf", "beta_dfa", "ACF vs DFA"),
        ("beta_acf", "beta_whittle", "ACF vs Whittle"),
        ("beta_acf", "beta_bayes", "ACF vs Bayes"),
        ("beta_dfa", "beta_whittle", "DFA vs Whittle"),
        ("beta_acf", "beta_hac", "ACF vs HAC"),
    ]

    fig, axes = plt.subplots(len(pairs), 1, figsize=(14, 3 * len(pairs)))
    if len(pairs) == 1:
        axes = [axes]

    for ax, (col1, col2, label) in zip(axes, pairs):
        if col1 not in df.columns or col2 not in df.columns:
            ax.text(
                0.5,
                0.5,
                f"{label}: dati non disponibili",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue
        x1 = pd.to_numeric(df[col1], errors="coerce").values
        x2 = pd.to_numeric(df[col2], errors="coerce").values
        corr_df = rolling_corr_cov(x1, x2, window_size, step, dates=None)
        if corr_df.empty:
            continue

        # Map rolling indices back to original window dates
        corr_center_indices = corr_df["window_center_index"].values.astype(int)
        if is_time:
            xc = x_dates.iloc[corr_center_indices].values
        else:
            xc = corr_center_indices

        ax.plot(xc, corr_df["corr"].values, "b-", linewidth=1.5, label=label)
        ax.fill_between(
            xc,
            corr_df["corr_ci_lo"].values,
            corr_df["corr_ci_hi"].values,
            alpha=0.2,
            color="blue",
        )
        ax.axhline(0, color="k", linestyle="--", alpha=0.3)
        ax.set_ylabel("Correlazione")
        ax.set_title(f"Corr rolling: {label}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.1, 1.1)
        if is_time:
            _format_time_axis(ax)

    axes[-1].set_xlabel("Date" if is_time else "Window index")

    fig.suptitle("Correlazioni rolling tra metodi β", fontsize=14)
    plt.tight_layout()
    _savefig(os.path.join(figdir, "beta_correlations_rolling.png"))


def plot_D_De_correlation(outdir, df, dates=None, window_size=20, step=5):
    """
    Grafico correlazione rolling D vs D_e.

    NOTA: D_e è calcolato usando la curva AR originale (prepare_beta_sim_curve_coherent),
    NON le curve PNAS (TL-JOINT, TL-AR, Student).
    """
    figdir = os.path.join(outdir, "fig")
    _ensure_dir(figdir)
    if df.empty or len(df) < window_size:
        return

    # Determina se abbiamo date
    if "window_date" in df.columns and df["window_date"].notna().any():
        x_dates = pd.to_datetime(df["window_date"])
        is_time = True
    else:
        x_dates = None
        is_time = False

    if "D" not in df.columns or "De_acf" not in df.columns:
        return

    D = pd.to_numeric(df["D"], errors="coerce").values
    De = pd.to_numeric(df["De_acf"], errors="coerce").values

    # Z-score usando compute_zscore per consistenza
    D_z = compute_zscore(D)
    De_z = compute_zscore(De)

    corr_df = rolling_corr_cov(D_z, De_z, window_size, step, dates=None)
    if corr_df.empty:
        return

    # Mappa gli indici di rolling alle date delle finestre originali
    corr_center_indices = corr_df["window_center_index"].values.astype(int)
    if is_time:
        xc = x_dates.iloc[corr_center_indices].values
    else:
        xc = corr_center_indices

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    ax = axes[0]
    ax.plot(xc, corr_df["corr"].values, "b-", linewidth=1.5, label="Corr(z(D), z(D_e))")
    ax.fill_between(
        xc,
        corr_df["corr_ci_lo"].values,
        corr_df["corr_ci_hi"].values,
        alpha=0.2,
        color="blue",
    )
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_ylabel("Correlazione")
    ax.set_title("Correlazione rolling z(D) vs z(D_e) — Curva AR originale, β_ACF")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.1, 1.1)
    if is_time:
        _format_time_axis(ax)

    ax = axes[1]
    ax.plot(xc, corr_df["cov"].values, "g-", linewidth=1.5, label="Cov(z(D), z(D_e))")
    ax.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("Date" if is_time else "Window index")
    ax.set_ylabel("Covarianza")
    ax.set_title("Covarianza rolling z(D) vs z(D_e) — Curva AR originale, β_ACF")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if is_time:
        _format_time_axis(ax)

    fig.suptitle(
        "NOTA: D_e calcolato con curva AR originale (non PNAS)",
        fontsize=10,
        color="gray",
    )
    plt.tight_layout()
    _savefig(os.path.join(figdir, "D_De_correlation_rolling.png"))


def plot_histograms(outdir, df):
    """Istogrammi e boxplot delle stime."""
    figdir = os.path.join(outdir, "fig")
    _ensure_dir(figdir)
    if df.empty:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Istogramma β
    ax = axes[0, 0]
    for col, color, label in [
        ("beta_acf", "blue", "ACF"),
        ("beta_dfa", "green", "DFA"),
        ("beta_bayes", "red", "Bayes"),
        ("beta_whittle", "orange", "Whittle"),
    ]:
        if col in df.columns:
            y = pd.to_numeric(df[col], errors="coerce").values
            y = y[np.isfinite(y)]
            if len(y) > 0:
                ax.hist(y, bins=20, alpha=0.4, color=color, label=label)
    ax.axvline(0.2, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("β")
    ax.set_title("Distribuzione β")
    ax.legend()

    # Istogramma D
    ax = axes[0, 1]
    y = pd.to_numeric(df["D"], errors="coerce").values
    y = y[np.isfinite(y)]
    if len(y) > 0:
        ax.hist(y, bins=20, alpha=0.7, color="purple")
    ax.axvline(0.5, color="r", linestyle="--", alpha=0.5)
    ax.set_xlabel("D")
    ax.set_title("Distribuzione D")

    # Istogramma D_e
    ax = axes[0, 2]
    for col, color, label in [
        ("De_acf", "blue", "ACF"),
        ("De_dfa", "green", "DFA"),
        ("De_bayes", "red", "Bayes"),
    ]:
        if col in df.columns:
            y = pd.to_numeric(df[col], errors="coerce").values
            y = y[np.isfinite(y)]
            if len(y) > 0:
                ax.hist(y, bins=20, alpha=0.4, color=color, label=label)
    ax.axvline(0.25, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("D_e")
    ax.set_title("Distribuzione D_e")
    ax.legend()

    # Boxplot β
    ax = axes[1, 0]
    data, labels = [], []
    for col, label in [
        ("beta_acf", "ACF"),
        ("beta_dfa", "DFA"),
        ("beta_bayes", "Bayes"),
        ("beta_whittle", "Whittle"),
    ]:
        if col in df.columns:
            y = pd.to_numeric(df[col], errors="coerce").values
            y = y[np.isfinite(y)]
            if len(y) > 0:
                data.append(y)
                labels.append(label)
    if data:
        ax.boxplot(data, tick_labels=labels)
    ax.axhline(0.2, color="r", linestyle="--", alpha=0.5)
    ax.set_ylabel("β")
    ax.set_title("Box plot β")

    # Boxplot stderr
    ax = axes[1, 1]
    data, labels = [], []
    for col in [
        "beta_acf_stderr",
        "beta_dfa_stderr",
        "beta_bayes_stderr",
        "beta_whittle_stderr",
    ]:
        if col in df.columns:
            y = pd.to_numeric(df[col], errors="coerce").values
            y = y[np.isfinite(y)]
            if len(y) > 0:
                data.append(y)
                labels.append(col.replace("_stderr", "").replace("beta_", ""))
    if data:
        ax.boxplot(data, tick_labels=labels)
    ax.set_ylabel("stderr")
    ax.set_title("Confronto incertezze β")

    # Scatter D vs D_e
    ax = axes[1, 2]
    if "D" in df.columns and "De_acf" in df.columns:
        D = pd.to_numeric(df["D"], errors="coerce").values
        De = pd.to_numeric(df["De_acf"], errors="coerce").values
        mask = np.isfinite(D) & np.isfinite(De)
        if np.any(mask):
            ax.scatter(D[mask], De[mask], alpha=0.6, s=30)
            ax.set_xlabel("D")
            ax.set_ylabel("D_e (ACF)")
            ax.set_title("Scatter D vs D_e")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _savefig(os.path.join(figdir, "histograms.png"))


# =============================================================================
# CONFIGURAZIONE INTERATTIVA E MAIN
# =============================================================================


def _ask(prompt, default=""):
    txt = input(f"{prompt} [{default}]: ").strip()
    return txt if txt else default


def _ask_int(prompt, default):
    while True:
        txt = _ask(prompt, str(default))
        try:
            return int(txt)
        except:
            print("Invalid value.")


def _ask_float(prompt, default):
    while True:
        txt = _ask(prompt, str(default))
        try:
            return float(txt.replace(",", "."))
        except:
            print("Invalid value.")


def _ask_list_int(prompt, default):
    txt = _ask(prompt, " ".join(str(x) for x in default))
    if not txt.strip():
        return list(default)
    parts = []
    for tok in txt.replace(",", " ").split():
        try:
            parts.append(int(tok))
        except:
            pass
    return sorted(set(parts)) if parts else list(default)


def _ask_yes_no(prompt, default=False):
    d = "s" if default else "n"
    txt = _ask(f"{prompt} (s/n)", d).lower()
    return txt[0] == "s" if txt else default


def _ask_choice(prompt, choices, default):
    txt = _ask(f"{prompt} ({'/'.join(choices)})", default)
    return txt if txt in choices else default


def _interactive_config():
    print("\n" + "=" * 60)
    print("BALDOVIN-STELLA ANALYSIS - Configuration")
    print("=" * 60 + "\n")

    fpath = _ask(
        "Data file path", "PRICE_DEF.txt" if os.path.exists("PRICE_DEF.txt") else ""
    )
    while not os.path.exists(fpath):
        print("Invalid path.")
        fpath = _ask("Data file path", "")

    try:
        df0 = read_prices_auto(fpath)
        cols = list(df0.columns)
        print(f"Colonne: {cols}")
    except:
        cols = []

    price_default = (
        "Close" if "Close" in cols else (cols[1] if len(cols) >= 2 else "Close")
    )
    date_default = "Date" if "Date" in cols else (cols[0] if cols else "")
    price_col = _ask("Colonna prezzi", price_default)
    date_col = _ask("Colonna date (vuoto se assente)", date_default)

    lag_min = _ask_int("Lag minimo", 1)
    lag_max = _ask_int("Lag massimo", 100)
    nbins = _ask_int("Number of log-bins", 16)
    min_acf_threshold = _ask_float("Soglia ACF minima", 0.001)
    fit_method = _ask_choice("Metodo fit", ["theil_sen", "wls", "ols"], "theil_sen")
    min_bins_required = _ask_int("Bin minimi richiesti", 6)

    L_ar = _ask_int("L (coefficienti AR)", 128)
    grid_n = _ask_int("D_e grid points", 100)
    reps = _ask_int("Repliche simulazione", 50)
    burn_in = _ask_int("Burn-in", 500)

    rigid_win_size = _ask_int("Rolling window length", 1200)
    rigid_win_shift = _ask_int("Shift finestre analisi", 250)
    corr_win_size = _ask_int("Finestra correlazione rolling", 20)
    corr_win_step = _ask_int("Step correlazione rolling", 5)
    use_bayes = SCIPY_AVAILABLE and _ask_yes_no("Stima Bayesiana?", True)
    use_whittle = _ask_yes_no("Stima Whittle/GPH?", True)
    T_list = _ask_list_int("Scale T per D", [1, 5, 10, 20, 30])
    dfa_order = _ask_int("Ordine DFA", 2)
    ci_z = _ask_float("z per CI (1.96 → 95%)", 1.96)
    outdir = _ask("Output directory", "out_baldovin_stella_v3_2")

    return argparse.Namespace(
        file=fpath,
        price_col=price_col,
        date_col=date_col if date_col.strip() else None,
        lag_min=lag_min,
        lag_max=lag_max,
        nbins=nbins,
        min_acf_threshold=min_acf_threshold,
        fit_method=fit_method,
        min_bins_required=min_bins_required,
        aggregation="median",
        L_ar=L_ar,
        grid_n=grid_n,
        reps=reps,
        burn_in=burn_in,
        rigid_win_size=rigid_win_size,
        rigid_win_shift=rigid_win_shift,
        corr_win_size=corr_win_size,
        corr_win_step=corr_win_step,
        use_bayes=use_bayes,
        use_whittle=use_whittle,
        T=T_list,
        dfa_order=dfa_order,
        ci_z=ci_z,
        outdir=outdir,
        n_boot=200,
        # V3.5: Default per stime finali pesate
        final_beta_methods=None,
        final_De_curves=None,
        final_De_beta_methods=None,
    )


def main(argv=None):
    global CI
    start_time = time.time()

    if argv is None:
        argv = sys.argv[1:]

    ap = argparse.ArgumentParser(description="Baldovin-Stella Analysis")
    ap.add_argument("--file", "-f", help="Data file path")
    ap.add_argument("--price-col", help="Colonna prezzi")
    ap.add_argument("--date-col", help="Colonna date")
    ap.add_argument("--lag-min", type=int, default=1)
    ap.add_argument("--lag-max", type=int, default=100)
    ap.add_argument("--nbins", type=int, default=16)
    ap.add_argument("--min-acf-threshold", type=float, default=0.001)
    ap.add_argument(
        "--fit-method", choices=["theil_sen", "wls", "ols"], default="theil_sen"
    )
    ap.add_argument("--min-bins-required", type=int, default=6)
    ap.add_argument("--aggregation", choices=["median", "mean"], default="median")
    ap.add_argument("--L-ar", type=int, default=128)
    ap.add_argument("--grid-n", type=int, default=100)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--burn-in", type=int, default=500)
    ap.add_argument("--rigid-win-size", type=int, default=1200)
    ap.add_argument("--rigid-win-shift", type=int, default=250)
    ap.add_argument(
        "--corr-win-size",
        type=int,
        default=20,
        help="Finestra per correlazione rolling",
    )
    ap.add_argument(
        "--corr-win-step", type=int, default=5, help="Step per correlazione rolling"
    )
    ap.add_argument("--use-bayes", action="store_true", default=True)
    ap.add_argument("--use-whittle", action="store_true", default=True)
    ap.add_argument("--T", type=int, nargs="+", default=[1, 5, 10, 20, 30])
    ap.add_argument("--dfa-order", type=int, default=2)
    ap.add_argument("--ci-z", type=float, default=1.96)
    ap.add_argument("--outdir", default="out_baldovin_stella_v3_5")
    ap.add_argument("--n-boot", type=int, default=200)

    # V3.5: Selezione metodi/curve per stima finale pesata
    ap.add_argument(
        "--final-beta-methods",
        nargs="+",
        default=None,
        help="Metodi β per stima finale (ACF,HAC,DFA,Bayes,Whittle). Default: tutti",
    )
    ap.add_argument(
        "--final-De-curves",
        nargs="+",
        default=None,
        help="Curve per stima D_e finale (TL_JOINT,TL_AR,Student). Default: tutte",
    )
    ap.add_argument(
        "--final-De-beta-methods",
        nargs="+",
        default=None,
        help="Metodi β per stima D_e finale. Default: tutti",
    )

    if len(argv) == 0:
        args = _interactive_config()
    else:
        args = ap.parse_args(argv)

    CI = float(args.ci_z)
    outdir = args.outdir
    _ensure_dir(outdir)
    _ensure_dir(os.path.join(outdir, "fig"))

    print("\n" + "=" * 70)
    print("BALDOVIN-STELLA ANALYSIS")
    print("=" * 70)

    # ===== LETTURA DATI =====
    _log("Lettura dati...")
    df = read_prices_auto(args.file)
    if args.price_col is None:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        args.price_col = num_cols[-1] if num_cols else df.columns[-1]
        print(f"    Uso colonna prezzi: '{args.price_col}'")

    prices, dates = extract_prices_and_dates(df, args.price_col, args.date_col)
    _log(f"Prezzi: {len(prices)} osservazioni")
    if dates is not None:
        _log(f"Date: {dates[0]} → {dates[-1]}")

    logp = detrend_log_prices(prices)
    r1 = returns_from_logp(logp)
    abs_r1 = np.abs(r1)

    # ===== PARAMETRI CONDIVISI (per coerenza) =====
    empirical_params = {
        "lag_min": args.lag_min,
        "lag_max": args.lag_max,
        "nbins": args.nbins,
        "min_acf_threshold": args.min_acf_threshold,
        "fit_method": args.fit_method,
        "min_bins_required": args.min_bins_required,
        "aggregation": args.aggregation,
    }
    _log(f"Parametri coerenza: {empirical_params}")

    # ===== STIMA GLOBALE D =====
    _log("Stima globale D...")
    D_est = estimate_D_quantiles(logp, T_values=args.T)
    _log(f"D = {D_est.D:.4f} ± {D_est.stderr:.4f}")

    # ===== V3.5: STIMA PARAMETRI CF TRUNCATED LÉVY (α, B, C) =====
    _log("Stima parametri CF truncated Lévy (α, B, C)...")
    cf_params = estimate_cf_params(r1, k_max=50.0, n_k=200)
    if np.isfinite(cf_params.alpha):
        _log(f"  α = {cf_params.alpha:.4f} ± {cf_params.alpha_stderr:.4f}")
        _log(f"  B = {cf_params.B:.2e} ± {cf_params.B_stderr:.2e}")
        _log(f"  C = {cf_params.C:.2e} ± {cf_params.C_stderr:.2e}")
        _log(f"  R² = {cf_params.r2:.4f}")
    else:
        _log("  (stima non riuscita, uso valori PNAS)")

    # ===== CURVA CALIBRAZIONE COERENTE (metodo AR originale) =====
    _log("Preparazione curva calibrazione AR (metodo originale)...")
    De_grid = np.linspace(0.10, 0.48, args.grid_n)

    curve = prepare_beta_sim_curve_coherent(
        n_series=len(abs_r1),
        lag_min=args.lag_min,
        lag_max=args.lag_max,
        nbins=args.nbins,
        min_acf_threshold=args.min_acf_threshold,  # CRITICO!
        fit_method=args.fit_method,  # CRITICO!
        min_bins_required=args.min_bins_required,  # CRITICO!
        aggregation=args.aggregation,
        L=args.L_ar,
        De_grid=De_grid,
        reps=args.reps,
        seed=1234,
        burn_in=args.burn_in,
    )

    # Test 1: Verifica coerenza parametri
    is_coherent, mismatches = verify_coherence(empirical_params, curve)
    if is_coherent:
        _log("✓ Test coerenza PASS: parametri empirici = parametri simulati")
    else:
        _log(f"✗ Test coerenza FAIL: {mismatches}")

    # ===== TRE CURVE PNAS-FAITHFUL =====
    _log("Costruzione TRE curve β(D) PNAS-faithful (TL-JOINT, TL-AR, Student)...")
    multi_curves = prepare_multi_beta_curves_pnas(
        De_grid=De_grid,
        n_repl_joint=max(20, args.reps),
        n_repl_ar=max(10, args.reps // 2),
        n_repl_student=max(15, args.reps // 2),
        seed=54321,
    )
    _log("✓ Curve PNAS costruite")

    # Test 2: Verifica pass-rate (curva AR originale)
    n_reliable = sum(1 for d in curve.diagnostics if not d.sel_bias_warn)
    n_total = len(curve.diagnostics)
    pct_reliable = n_reliable / n_total * 100 if n_total > 0 else 0
    if pct_reliable >= 90:
        _log(f"✓ Test pass-rate PASS: {pct_reliable:.1f}% punti D_e affidabili")
    else:
        _log(
            f"⚠ Test pass-rate WARN: {pct_reliable:.1f}% punti D_e affidabili (target ≥90%)"
        )

    # ===== STIME β GLOBALI =====
    _log("Stima β con metodi multipli...")
    beta_estimates, De_estimates = {}, {}

    # ACF robusto (usa core estimator)
    beta_acf = estimate_beta_acf_robust(
        abs_r1,
        lag_min=args.lag_min,
        lag_max=args.lag_max,
        nbins=args.nbins,
        n_boot=args.n_boot,
        random_state=1234,
        min_acf_threshold=args.min_acf_threshold,
        fit_method=args.fit_method,
        min_bins_required=args.min_bins_required,
        aggregation=args.aggregation,
    )
    beta_estimates["ACF"] = beta_acf
    De_acf = estimate_De_from_beta(beta_acf, curve)
    De_estimates["ACF"] = De_acf
    _log(
        f"β_ACF = {beta_acf.beta:.4f} ± {beta_acf.stderr:.4f} (R²={beta_acf.details.get('r2', np.nan):.4f})"
    )

    # HAC
    beta_hac = estimate_beta_acf_hac(
        abs_r1,
        lag_min=args.lag_min,
        lag_max=args.lag_max,
        nbins=args.nbins,
        min_acf_threshold=args.min_acf_threshold,
    )
    beta_estimates["HAC"] = beta_hac
    De_hac = estimate_De_from_beta(beta_hac, curve)  # AGGIUNTO
    De_estimates["HAC"] = De_hac  # AGGIUNTO
    _log(f"β_HAC = {beta_hac.beta:.4f} ± {beta_hac.stderr:.4f}")

    # DFA
    beta_dfa = estimate_beta_dfa(
        abs_r1, order=args.dfa_order, n_boot=args.n_boot, random_state=4321
    )
    beta_estimates["DFA"] = beta_dfa
    De_dfa = estimate_De_from_beta(beta_dfa, curve)
    De_estimates["DFA"] = De_dfa
    _log(
        f"β_DFA = {beta_dfa.beta:.4f} ± {beta_dfa.stderr:.4f} (H={beta_dfa.details.get('H', np.nan):.4f})"
    )

    # Bayes
    if args.use_bayes and SCIPY_AVAILABLE:
        beta_bayes = estimate_beta_bayes(
            abs_r1,
            lag_min=args.lag_min,
            lag_max=args.lag_max,
            nbins=args.nbins,
            min_acf_threshold=args.min_acf_threshold,
            random_state=8765,
        )
        beta_estimates["Bayes"] = beta_bayes
        De_bayes = estimate_De_from_beta(beta_bayes, curve)
        De_estimates["Bayes"] = De_bayes
        _log(f"β_Bayes = {beta_bayes.beta:.4f} ± {beta_bayes.stderr:.4f}")

    # Whittle
    if args.use_whittle:
        beta_whittle = estimate_beta_whittle(abs_r1, random_state=9876)
        beta_estimates["Whittle"] = beta_whittle
        De_whittle = estimate_De_from_beta(beta_whittle, curve)  # AGGIUNTO
        De_estimates["Whittle"] = De_whittle  # AGGIUNTO
        _log(f"β_Whittle = {beta_whittle.beta:.4f} ± {beta_whittle.stderr:.4f}")

    # ===== V3.5: MAPPING D_e SU TUTTE LE CURVE =====
    _log("Calcolo D_e su TUTTE le curve (TL-JOINT, TL-AR, Student)...")
    De_multi_results = estimate_all_De_combinations(beta_estimates, multi_curves)

    # Stampa risultati
    for beta_method, multi_de in De_multi_results.items():
        _log(f"  {beta_method}:")
        for curve_name, De_est in [
            ("TL_JOINT", multi_de.De_tl_joint),
            ("TL_AR", multi_de.De_tl_ar),
            ("Student", multi_de.De_student),
        ]:
            if np.isfinite(De_est.De):
                _log(f"    {curve_name}: D_e = {De_est.De:.4f} ± {De_est.stderr:.4f}")

    # ===== V3.5: STIME FINALI PESATE =====
    _log("Calcolo stime finali pesate...")

    # Selezione metodi/curve (da CLI o default)
    selected_beta_methods = args.final_beta_methods
    selected_De_curves = args.final_De_curves
    selected_De_beta_methods = args.final_De_beta_methods

    # Stima finale β
    final_beta = compute_final_beta_estimate(beta_estimates, selected_beta_methods)
    _log(f"β FINALE (pesato): {final_beta.value:.4f} ± {final_beta.stderr:.4f}")
    _log(f"  Metodi usati: {', '.join(final_beta.methods_used)}")

    # Stima finale D_e
    final_De = compute_final_De_estimate(
        De_multi_results, selected_De_beta_methods, selected_De_curves
    )
    _log(f"D_e FINALE (pesato): {final_De.value:.4f} ± {final_De.stderr:.4f}")
    _log(f"  Combinazioni usate: {len(final_De.methods_used)}")

    # ===== GRAFICI GLOBALI =====
    _log("Generating global diagnostic plots...")
    plot_acf_diagnostic(
        outdir,
        abs_r1,
        (args.lag_min, args.lag_max),
        args.nbins,
        args.min_acf_threshold,
        beta_acf,
    )
    plot_dfa_diagnostic(outdir, abs_r1, beta_dfa, args.dfa_order)
    if args.use_whittle and "Whittle" in beta_estimates:
        plot_whittle_stability(outdir, beta_estimates["Whittle"])
    plot_calibration_curve_diagnostic(outdir, curve, beta_acf, De_acf)
    plot_calibration_curve_simple(outdir, curve, beta_acf, De_acf)

    # V3.5: Grafico CF truncated Lévy
    _log("Generating truncated-Lévy CF fit plot...")
    plot_cf_fit(outdir, r1, cf_params)

    # ===== GRAFICI CURVE PNAS (TL-JOINT, TL-AR, Student) =====
    _log("Generating PNAS β(D) calibration curve plots...")
    plot_multi_curves_pnas(outdir, multi_curves, beta_emp=beta_acf.beta)

    # ===== V3.5: GRAFICI CALIBRAZIONE D_e MULTIPLI =====
    _log(
        "Generating D_e calibration plots for all (β-method × curve) combinations..."
    )
    plot_all_De_calibrations(outdir, multi_curves, beta_estimates, De_multi_results)
    plot_De_summary_all_combinations(outdir, De_multi_results)

    # ===== V3.5: GRAFICI STIME FINALI PESATE =====
    _log("Generating weighted final estimate plots...")
    plot_final_weighted_estimates(outdir, final_beta, final_De, D_est)
    plot_weights_breakdown(outdir, final_beta, final_De)

    plot_global_comparison(outdir, D_est, beta_estimates, De_estimates)

    # Grafici inversione per ogni metodo β
    _log("Generating calibration inversion plots by method...")
    for name, est in beta_estimates.items():
        plot_calibration_inversion_single(outdir, curve, est, name)

    # Grafici posterior Bayesiano dettagliati
    _log("Generating Bayesian posterior plots...")
    for name, est in beta_estimates.items():
        plot_bayesian_posterior_explained(outdir, curve, est, name)

    # ===== ANALISI FINESTRE MOBILI =====
    _log("Analisi finestre mobili...")
    _log(f"Finestra: {args.rigid_win_size}, Shift: {args.rigid_win_shift}")

    rigid_df = rigid_window_analysis(
        abs_r1,
        logp,
        dates,
        win_size=args.rigid_win_size,
        win_shift=args.rigid_win_shift,
        lag_min=args.lag_min,
        lag_max=args.lag_max,
        nbins=args.nbins,
        min_acf_threshold=args.min_acf_threshold,
        fit_method=args.fit_method,
        min_bins_required=args.min_bins_required,
        aggregation=args.aggregation,
        multi_curves=multi_curves,  # V3.5: Usa le tre curve PNAS
        T_values=args.T,
        dfa_order=args.dfa_order,
        use_bayes=args.use_bayes and SCIPY_AVAILABLE,
        use_whittle=args.use_whittle,
        n_boot_beta=max(50, args.n_boot // 2),
        seed=24680,
    )

    if rigid_df.empty:
        _log("No valid windows found.")
    else:
        _log(f"{len(rigid_df)} finestre analizzate.")
        rigid_df.to_csv(os.path.join(outdir, "rigid_windows_results.csv"), index=False)

        _log("Generating historical evolution plots...")
        plot_historical_evolution(outdir, rigid_df, dates)
        plot_beta_separate(outdir, rigid_df, dates)

        # Usa parametri correlazione da args (con default se non CLI)
        corr_win = getattr(args, "corr_win_size", min(20, max(5, len(rigid_df) // 4)))
        corr_step = getattr(args, "corr_win_step", 5)

        _log("Generating rolling correlation plots...")
        plot_correlations_rolling(
            outdir, rigid_df, dates, window_size=corr_win, step=corr_step
        )

        # ===== V3.5: NUOVI GRAFICI SERIE STORICHE E Z-SCORE =====
        _log("Generating rolling β and D_e time-series plots for all methods and curves...")
        plot_beta_all_methods_timeseries(outdir, rigid_df, dates)
        plot_De_all_combinations_timeseries(outdir, rigid_df, dates)
        plot_summary_all_De_timeseries(outdir, rigid_df)

        _log("Generating z-score plots...")
        plot_zscore_beta_timeseries(outdir, rigid_df, dates)
        plot_zscore_De_timeseries(
            outdir, rigid_df, curves=["TL_JOINT", "TL_AR", "Student"]
        )

        _log("Generating z-score correlation/covariance plots vs D...")
        plot_zscore_corr_cov_vs_D(
            outdir, rigid_df, window_size=corr_win, step=corr_step
        )

        # ===== V3.5-3: GRAFICI SINGOLI CORRELAZIONE =====
        _log(
            "Generating individual correlation/covariance plots (single_correlation/)..."
        )
        plot_single_correlation_series(
            outdir, rigid_df, window_size=corr_win, step=corr_step
        )

        _log("Generating individual β vs D analysis plots...")
        for method, beta_col in [
            ("ACF", "beta_acf"),
            ("HAC", "beta_hac"),
            ("DFA", "beta_dfa"),
            ("Bayes", "beta_bayes"),
            ("Whittle", "beta_whittle"),
        ]:
            plot_beta_method_analysis(
                outdir, rigid_df, dates, method, beta_col, "D", corr_win, corr_step
            )

        _log("Generating individual D_e vs D analysis plots (per calibration curve)...")
        for curve_name in ["TL_JOINT", "TL_AR", "Student"]:
            for method in ["ACF", "HAC", "DFA", "Bayes", "Whittle"]:
                de_col = f"De_{method.lower()}_{curve_name}"
                if de_col in rigid_df.columns:
                    plot_De_method_analysis(
                        outdir,
                        rigid_df,
                        dates,
                        f"{method}_{curve_name}",
                        de_col,
                        "D",
                        corr_win,
                        corr_step,
                    )

        plot_histograms(outdir, rigid_df)

    # ===== SALVATAGGIO RISULTATI =====
    global_results = {
        "version": "1.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "coherence_test": {"passed": is_coherent, "mismatches": mismatches},
        "pass_rate_test": {"pct_reliable": pct_reliable, "threshold": 90.0},
        "empirical_params": empirical_params,
        "curve_params": curve.params,
        "settings": {
            "L_ar": args.L_ar,
            "burn_in": args.burn_in,
            "grid_n": args.grid_n,
            "reps": args.reps,
            "rigid_win_size": args.rigid_win_size,
            "rigid_win_shift": args.rigid_win_shift,
        },
        "D": {"value": D_est.D, "stderr": D_est.stderr, "ci": D_est.ci},
        # V3.5: Parametri CF truncated Lévy
        "cf_params": {
            "alpha": cf_params.alpha if np.isfinite(cf_params.alpha) else None,
            "alpha_stderr": (
                cf_params.alpha_stderr if np.isfinite(cf_params.alpha_stderr) else None
            ),
            "B": cf_params.B if np.isfinite(cf_params.B) else None,
            "B_stderr": cf_params.B_stderr if np.isfinite(cf_params.B_stderr) else None,
            "C": cf_params.C if np.isfinite(cf_params.C) else None,
            "C_stderr": cf_params.C_stderr if np.isfinite(cf_params.C_stderr) else None,
            "r2": cf_params.r2 if np.isfinite(cf_params.r2) else None,
            "pnas_reference": {"alpha": ALPHA_TL, "B": B_TL, "C": C_TL},
        },
    }

    for name, est in beta_estimates.items():
        global_results[f"beta_{name.lower()}"] = {
            "value": est.beta,
            "stderr": est.stderr,
            "ci": est.ci,
            "r2": est.details.get("r2"),
            "method": est.details.get("method", name),
        }

    for name, est in De_estimates.items():
        global_results[f"De_{name.lower()}"] = {
            "value": est.De,
            "stderr": est.stderr,
            "ci": est.ci,
        }

    # Diagnostiche curva
    global_results["curve_diagnostics"] = {
        "overall_pass_rate": curve.details.get("overall_pass_rate"),
        "n_unreliable_points": curve.details.get("n_unreliable_points"),
        "per_De": [
            {
                "De": d.De,
                "pass_rate": d.pass_rate,
                "median_nbins": d.median_nbins_valid,
                "sel_bias_warn": d.sel_bias_warn,
            }
            for d in curve.diagnostics[:: max(1, len(curve.diagnostics) // 20)]
        ],  # Subset
    }

    # ===== SALVATAGGIO CURVE PNAS =====
    _log("Salvataggio curve β(D) PNAS...")
    curve_df = pd.DataFrame(
        {
            "D": multi_curves.tl_joint.De_grid,
            "beta_TL_JOINT": multi_curves.tl_joint.beta_mean,
            "std_TL_JOINT": multi_curves.tl_joint.beta_std,
            "beta_TL_AR": multi_curves.tl_ar.beta_mean,
            "std_TL_AR": multi_curves.tl_ar.beta_std,
            "beta_Student": multi_curves.student.beta_mean,
            "std_Student": multi_curves.student.beta_std,
        }
    )
    curve_df.to_csv(os.path.join(outdir, "beta_curves_pnas.csv"), index=False)

    # V3.5: Salva D_e per tutte le combinazioni
    _log("Salvataggio D_e tutte le combinazioni...")
    De_multi_rows = []
    for beta_method, multi_de in De_multi_results.items():
        for curve_name, De_est in [
            ("TL_JOINT", multi_de.De_tl_joint),
            ("TL_AR", multi_de.De_tl_ar),
            ("Student", multi_de.De_student),
        ]:
            De_multi_rows.append(
                {
                    "beta_method": beta_method,
                    "curve": curve_name,
                    "beta_value": multi_de.beta_value,
                    "beta_stderr": multi_de.beta_stderr,
                    "De": De_est.De,
                    "De_stderr": De_est.stderr,
                    "De_ci_low": De_est.ci[0],
                    "De_ci_high": De_est.ci[1],
                }
            )
    De_multi_df = pd.DataFrame(De_multi_rows)
    De_multi_df.to_csv(os.path.join(outdir, "De_all_combinations.csv"), index=False)

    # Aggiungi info curve PNAS a global_results
    global_results["pnas_curves"] = {
        "params": multi_curves.params,
        "tl_joint_valid": int(np.sum(np.isfinite(multi_curves.tl_joint.beta_mean))),
        "tl_ar_valid": int(np.sum(np.isfinite(multi_curves.tl_ar.beta_mean))),
        "student_valid": int(np.sum(np.isfinite(multi_curves.student.beta_mean))),
    }

    # V3.5: Aggiungi D_e multi-curva
    global_results["De_multi_curve"] = {}
    for beta_method, multi_de in De_multi_results.items():
        global_results["De_multi_curve"][beta_method] = {
            "TL_JOINT": {
                "value": multi_de.De_tl_joint.De,
                "stderr": multi_de.De_tl_joint.stderr,
            },
            "TL_AR": {
                "value": multi_de.De_tl_ar.De,
                "stderr": multi_de.De_tl_ar.stderr,
            },
            "Student": {
                "value": multi_de.De_student.De,
                "stderr": multi_de.De_student.stderr,
            },
        }

    # V3.5: Aggiungi stime finali pesate
    global_results["final_estimates"] = {
        "beta": {
            "value": final_beta.value,
            "stderr": final_beta.stderr,
            "ci": final_beta.ci,
            "methods_used": final_beta.methods_used,
            "weights": final_beta.weights,
        },
        "De": {
            "value": final_De.value,
            "stderr": final_De.stderr,
            "ci": final_De.ci,
            "n_combinations": len(final_De.methods_used),
        },
    }

    runtime = time.time() - start_time
    global_results["runtime_seconds"] = runtime
    global_results["n_windows"] = len(rigid_df) if not rigid_df.empty else 0

    def json_ser(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        return str(obj)

    with open(os.path.join(outdir, "global_results.json"), "w") as f:
        json.dump(global_results, f, indent=2, default=json_ser)

    # Report testuale
    report_lines = [
        "BALDOVIN-STELLA ANALYSIS",
        f"{'=' * 50}",
        f"File: {args.file}",
        f"Osservazioni: {len(prices)}",
        f"Date: {dates[0] if dates is not None else 'N/A'} → {dates[-1] if dates is not None else 'N/A'}",
        "",
        "TEST COERENZA",
        f"{'=' * 50}",
        f"Parametri identici empirico/simulato: {'PASS' if is_coherent else 'FAIL'}",
        f"Pass-rate D_e affidabili: {pct_reliable:.1f}% (target ≥90%)",
        "",
        "PARAMETRI COERENZA",
        f"{'=' * 50}",
    ]
    for k, v in empirical_params.items():
        report_lines.append(f"  {k}: {v}")

    report_lines.extend(
        [
            "",
            "RISULTATI GLOBALI",
            f"{'=' * 50}",
            f"D = {D_est.D:.4f} ± {D_est.stderr:.4f}",
            "",
            "PARAMETRI CF TRUNCATED LÉVY (V3.5)",
            f"{'=' * 50}",
            "Modello: ẽg(k) = exp(-B·k² / (1 + C·|k|^{2-α}))",
            "",
        ]
    )
    if np.isfinite(cf_params.alpha):
        report_lines.append("Stima empirica:")
        alpha_str = f"  α = {cf_params.alpha:.4f}"
        if np.isfinite(cf_params.alpha_stderr):
            alpha_str += f" ± {cf_params.alpha_stderr:.4f}"
        report_lines.append(alpha_str)

        B_str = f"  B = {cf_params.B:.2e}"
        if np.isfinite(cf_params.B_stderr):
            B_str += f" ± {cf_params.B_stderr:.2e}"
        report_lines.append(B_str)

        C_str = f"  C = {cf_params.C:.2e}"
        if np.isfinite(cf_params.C_stderr):
            C_str += f" ± {cf_params.C_stderr:.2e}"
        report_lines.append(C_str)

        report_lines.append(f"  R² fit = {cf_params.r2:.4f}")
    else:
        report_lines.append("  (stima empirica non riuscita)")

    report_lines.extend(
        [
            "",
            "Riferimento PNAS (DJI 1900-2005):",
            f"  α = {ALPHA_TL}",
            f"  B = {B_TL:.2e}",
            f"  C = {C_TL:.2e}",
            "",
            "STIME β:",
        ]
    )
    for name, est in beta_estimates.items():
        r2 = est.details.get("r2", np.nan)
        report_lines.append(
            f"  β_{name} = {est.beta:.4f} ± {est.stderr:.4f} (R²={r2:.4f})"
        )

    report_lines.append("")
    report_lines.append("STIME D_e (curva AR originale):")
    for name, est in De_estimates.items():
        report_lines.append(f"  D_e_{name} = {est.De:.4f} ± {est.stderr:.4f}")

    # V3.5: D_e su tutte le curve
    report_lines.extend(
        [
            "",
            "D_e SU TUTTE LE CURVE (V3.5)",
            f"{'=' * 50}",
        ]
    )
    for beta_method, multi_de in De_multi_results.items():
        report_lines.append(f"  β_{beta_method}:")
        for curve_name, De_est in [
            ("TL_JOINT", multi_de.De_tl_joint),
            ("TL_AR", multi_de.De_tl_ar),
            ("Student", multi_de.De_student),
        ]:
            if np.isfinite(De_est.De):
                report_lines.append(
                    f"    → {curve_name}: D_e = {De_est.De:.4f} ± {De_est.stderr:.4f}"
                )

    # V3.5: Stime finali pesate
    report_lines.extend(
        [
            "",
            "STIME FINALI PESATE (V3.5)",
            f"{'=' * 50}",
            f"β FINALE = {final_beta.value:.4f} ± {final_beta.stderr:.4f}",
            f"  Metodi usati: {', '.join(final_beta.methods_used)}",
            f"  Pesi: {', '.join(f'{k}={v:.2f}' for k,v in final_beta.weights.items())}",
            "",
            f"D_e FINALE = {final_De.value:.4f} ± {final_De.stderr:.4f}",
            f"  Combinazioni usate: {len(final_De.methods_used)}",
        ]
    )
    if len(final_De.weights) <= 10:
        report_lines.append(
            f"  Pesi: {', '.join(f'{k}={v:.2f}' for k,v in final_De.weights.items())}"
        )
    else:
        top_weights = sorted(
            final_De.weights.items(), key=lambda x: x[1], reverse=True
        )[:5]
        report_lines.append(
            f"  Top 5 pesi: {', '.join(f'{k}={v:.2f}' for k,v in top_weights)}"
        )

    report_lines.extend(
        [
            "",
            "CURVE β(D) PNAS-FAITHFUL",
            f"{'=' * 50}",
            f"Parametri SI Text: α={ALPHA_TL}, B={B_TL:.2e}, C={C_TL:.2e}",
            f"τ_c={TAU_C}, m={M_AR}, TMAX={TMAX}",
            "",
            f"TL-JOINT (mixing per blocco): {int(np.sum(np.isfinite(multi_curves.tl_joint.beta_mean)))} punti validi",
            f"TL-AR (autoregressivo m={M_AR}): {int(np.sum(np.isfinite(multi_curves.tl_ar.beta_mean)))} punti validi",
            f"Student (ν={NU_STUDENT}): {int(np.sum(np.isfinite(multi_curves.student.beta_mean)))} punti validi",
            "",
            f"Finestre analizzate: {len(rigid_df) if not rigid_df.empty else 0}",
            f"Runtime: {runtime:.1f}s",
        ]
    )

    with open(os.path.join(outdir, "report.txt"), "w") as f:
        f.write("\n".join(report_lines))

    print("\n" + "=" * 70)
    _log(f"ANALISI COMPLETATA in {runtime:.1f}s")
    _log(f"Output in: {outdir}")
    print("=" * 70)

    print("\n--- RIEPILOGO ---")
    print(f"Coerenza: {'PASS' if is_coherent else 'FAIL'}")
    print(f"D = {D_est.D:.4f} ± {D_est.stderr:.4f}")
    print("\nStime β per metodo:")
    for name, est in beta_estimates.items():
        print(f"  β_{name} = {est.beta:.4f} ± {est.stderr:.4f}")
    print("\nStime D_e (curva AR):")
    for name, est in De_estimates.items():
        print(f"  D_e_{name} = {est.De:.4f} ± {est.stderr:.4f}")

    print("\n" + "=" * 50)
    print("STIME FINALI PESATE (V3.5)")
    print("=" * 50)
    print(f"β FINALE  = {final_beta.value:.4f} ± {final_beta.stderr:.4f}")
    print(f"D_e FINALE = {final_De.value:.4f} ± {final_De.stderr:.4f}")
    print(f"(basato su {len(final_De.methods_used)} combinazioni metodo×curva)")


if __name__ == "__main__":
    main()
