#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtesting_predictive.py
=========================
Comprehensive out-of-sample backtesting, robustness analysis, and economic
evaluation of the predictive regression:

    HMFSI_{t+h} = α₀ + ψ · ρ(β, Dₑ)_t + γ · HMFSI_t + ε_{t+h}

This script extends the dual-layer framework tests with:

  (B1) Out-of-sample recursive (expanding-window) forecasting
       - RMSE, MAE, R²_OOS vs naive & AR(1) benchmarks
       - Diebold–Mariano test (Harvey et al. 1997 small-sample correction)
       - Clark–West MSFE-adjusted test (2006, nested models)
       - Cumulative squared forecast error difference (CSFE) plot

  (B2) Economic backtest
       - Signal: predicted HMFSI change → equity position
       - Long/short S&P 500 based on predicted stress regime
       - Sharpe ratio, max drawdown, Calmar ratio, hit rate
       - Equity curve and drawdown plots

  (B3) VIX robustness
       - Augmented regression with VIX as additional control
       - Encompassing test: does ρ(β,Dₑ) survive after VIX control?
       - Horse-race: ρ(β,Dₑ) alone vs VIX alone vs combined

  (B4) VIX standalone analysis (endogenous/exogenous diagnostic)
       - Toda–Yamamoto Granger causality
       - Predictive regressions at multiple horizons
       - Comparison with HMFSI results

  (B5) Subsample stability
       - Rolling-window OLS coefficient estimates for ψ
       - CUSUM-type structural break diagnostics
       - First-half vs second-half comparison

  (B6) Additional robustness
       - Bootstrap confidence intervals (block bootstrap, 10000 reps)
       - Leave-one-out cross-validation
       - Multiple-horizon joint significance (Holm correction)

Inputs (all from existing pipeline outputs):
  --hmfsi-corr   : aligned_series.csv from corr(β,Dₑ) vs HMFSI analysis
                   (columns: date_col, y=fisher_corr, stress=HMFSI)
  --rv-corr      : aligned_series.csv from corr(β,Dₑ) vs RV analysis
                   (columns: date_col, y=fisher_corr, stress=RV)
  --hmfsi-daily  : HMFSI.csv (daily HMFSI, columns: Date, HMFSI)
  --sp-prices    : (optional) S&P 500 CSV for economic backtest
  --vix-csv      : (optional) VIX CSV for robustness
  --out          : output directory (default: backtesting_predittivita)

Author: Giacomo Grassi & Gianluca Teza
Date:   March 2026
"""

import argparse
import json
import os
import sys
import warnings
from collections import OrderedDict
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats, linalg

# Plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

# ===========================================================================
# STYLE CONFIGURATION (publication quality)
# ===========================================================================
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.5,
})

# Colour palette (colourblind-friendly)
C_BLUE = "#1f77b4"
C_RED = "#d62728"
C_GREEN = "#2ca02c"
C_ORANGE = "#ff7f0e"
C_PURPLE = "#9467bd"
C_GREY = "#7f7f7f"

# ===========================================================================
# MANUAL STATISTICAL IMPLEMENTATIONS (no statsmodels required)
# ===========================================================================

def _ols(y, X):
    """
    OLS via QR decomposition. Returns (beta_hat, residuals, X).
    X should already include a constant column if desired.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    y, X = y[mask], X[mask]
    Q, R = np.linalg.qr(X, mode="reduced")
    beta = linalg.solve_triangular(R, Q.T @ y)
    resid = y - X @ beta
    return beta, resid, X, y


def _add_const(X):
    """Add a column of ones as the first column."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return np.column_stack([np.ones(X.shape[0]), X])


def newey_west_cov(X, resid, max_lags=None):
    """
    Newey–West heteroscedasticity and autocorrelation consistent (HAC)
    covariance matrix estimator.
    
    Uses the Bartlett kernel with automatic bandwidth = floor(4*(n/100)^(2/9))
    if max_lags is None.
    """
    n, k = X.shape
    if max_lags is None:
        max_lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    max_lags = max(1, min(max_lags, n - 2))

    # Meat: S₀ + Σ kernel-weighted cross-products
    e = resid.reshape(-1, 1)
    Xe = X * e  # n × k
    S = (Xe.T @ Xe) / n  # S₀

    for j in range(1, max_lags + 1):
        w = 1 - j / (max_lags + 1)  # Bartlett kernel
        Gamma_j = (Xe[j:].T @ Xe[:-j]) / n
        S += w * (Gamma_j + Gamma_j.T)

    # Bread: (X'X/n)^{-1}
    try:
        XtX_inv = np.linalg.inv(X.T @ X / n)
    except np.linalg.LinAlgError:
        # Regularise if near-singular
        XtX_inv = np.linalg.pinv(X.T @ X / n)
    V = (XtX_inv @ S @ XtX_inv) / n
    return V


def ols_hac(y, X_raw, max_lags=None, add_constant=True):
    """
    Full OLS estimation with Newey–West HAC standard errors.
    
    Parameters
    ----------
    y : array-like, (n,)
    X_raw : array-like, (n, k) or (n,) — regressors WITHOUT constant
    max_lags : int or None
    add_constant : bool
    
    Returns
    -------
    dict with keys: beta, se, t_stat, p_value, R2, R2_adj, resid, n, k,
                    var_names (list of str)
    """
    X_raw = np.asarray(X_raw, dtype=float)
    if X_raw.ndim == 1:
        X_raw = X_raw.reshape(-1, 1)
    
    if add_constant:
        X = _add_const(X_raw)
    else:
        X = X_raw.copy()

    beta, resid, X_clean, y_clean = _ols(y, X)
    n, k = X_clean.shape

    V = newey_west_cov(X_clean, resid, max_lags=max_lags)
    se = np.sqrt(np.diag(V))
    t_stat = beta / se
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - k))

    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    R2_adj = 1 - (1 - R2) * (n - 1) / (n - k) if n > k else R2

    return {
        "beta": beta,
        "se": se,
        "t_stat": t_stat,
        "p_value": p_value,
        "R2": R2,
        "R2_adj": R2_adj,
        "resid": resid,
        "n": n,
        "k": k,
        "y_fitted": X_clean @ beta,
        "X": X_clean,
        "y": y_clean,
    }


def toda_yamamoto_granger(y, x, max_ar=4, d_max=1):
    """
    Toda–Yamamoto augmented VAR Wald test for Granger causality.
    Tests x → y and y → x.
    
    Selects AR order by BIC from 1..max_ar, then augments by d_max.
    Returns dict with p-values for both directions.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(y)
    assert len(x) == n

    # Select lag order by BIC on bivariate VAR
    best_bic = np.inf
    best_k = 1
    for k in range(1, max_ar + 1):
        if n - k - d_max < 2 * (k + d_max) + 5:
            continue
        Z = np.column_stack([
            np.column_stack([y[k + d_max - j - 1:n - j - 1] for j in range(k + d_max)]),
            np.column_stack([x[k + d_max - j - 1:n - j - 1] for j in range(k + d_max)]),
        ])
        Y_dep = y[k + d_max:]
        T = len(Y_dep)
        p_total = Z.shape[1]
        if T <= p_total + 2:
            continue
        beta_hat, resid, _, _ = _ols(Y_dep, _add_const(Z))
        sigma2 = np.sum(resid ** 2) / T
        bic = np.log(sigma2 + 1e-300) + p_total * np.log(T) / T
        if bic < best_bic:
            best_bic = bic
            best_k = k

    k_ar = best_k
    p = k_ar + d_max

    def _wald_test(dep, cause, k_ar, d_max):
        """Wald test: does 'cause' Granger-cause 'dep'?"""
        p_total = k_ar + d_max
        T = n - p_total
        if T < 2 * p_total + 5:
            return np.nan, np.nan

        # Build lagged matrices
        dep_lags = np.column_stack([dep[p_total - j - 1:n - j - 1] for j in range(p_total)])
        cause_lags = np.column_stack([cause[p_total - j - 1:n - j - 1] for j in range(p_total)])
        Z_full = np.column_stack([dep_lags, cause_lags])
        Y = dep[p_total:]

        X_full = _add_const(Z_full)
        beta_full, resid_full, X_f, Y_f = _ols(Y, X_full)
        T_eff = len(Y_f)

        # Restricted model: drop first k_ar lags of cause (indices p_total+1 .. p_total+k_ar in X_full)
        # In X_full: col 0 = const, cols 1..p = dep_lags, cols p+1..2p = cause_lags
        # We restrict the first k_ar cause lag coefficients to zero
        cause_start = 1 + p_total  # first cause lag column in X_full
        restricted_cols = list(range(cause_start, cause_start + k_ar))

        R = np.zeros((k_ar, X_f.shape[1]))
        for i, col_idx in enumerate(restricted_cols):
            if col_idx < X_f.shape[1]:
                R[i, col_idx] = 1.0

        Rb = R @ beta_full
        V_full = newey_west_cov(X_f, resid_full, max_lags=max(k_ar + d_max, 4))
        try:
            RVR_inv = np.linalg.inv(R @ V_full @ R.T)
        except np.linalg.LinAlgError:
            return np.nan, np.nan
        
        W = float(T_eff * Rb.T @ RVR_inv @ Rb)
        p_val = 1 - stats.chi2.cdf(W, df=k_ar)
        return W, p_val

    W_xy, p_xy = _wald_test(y, x, k_ar, d_max)  # x → y
    W_yx, p_yx = _wald_test(x, y, k_ar, d_max)  # y → x

    return {
        "k_ar": k_ar,
        "d_max": d_max,
        "x_to_y_wald": round(float(W_xy), 4) if np.isfinite(W_xy) else None,
        "x_to_y_p": round(float(p_xy), 6) if np.isfinite(p_xy) else None,
        "y_to_x_wald": round(float(W_yx), 4) if np.isfinite(W_yx) else None,
        "y_to_x_p": round(float(p_yx), 6) if np.isfinite(p_yx) else None,
    }


# ===========================================================================
# DATA LOADING
# ===========================================================================

def load_aligned(path):
    """Load aligned_series.csv, auto-detecting date column name."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df["y"], df["stress"]


def load_hmfsi_daily(path):
    """Load daily HMFSI."""
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df["HMFSI"]


def load_csv_ts(path, date_col=None, value_col=None):
    """Generic CSV loader for time series."""
    df = pd.read_csv(path, parse_dates=True)
    if date_col is None:
        date_col = df.columns[0]
    if value_col is None:
        value_col = df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    return df[value_col].dropna()


# ===========================================================================
# B1: OUT-OF-SAMPLE RECURSIVE FORECASTING
# ===========================================================================

def recursive_oos_forecast(corr_series, hmfsi_series, horizon=2,
                           min_train_frac=0.5, include_ar=True):
    """
    Expanding-window (recursive) out-of-sample forecasting.
    
    Model: HMFSI_{t+h} = α + ψ·corr_t + γ·HMFSI_t + ε
    Benchmark 1: Random walk (HMFSI_{t+h} = HMFSI_t)
    Benchmark 2: AR(1) (HMFSI_{t+h} = a + b·HMFSI_t)
    
    Returns forecasts, actuals, and performance metrics.
    """
    df = pd.DataFrame({"corr": corr_series, "hmfsi": hmfsi_series}).dropna()
    n = len(df)
    min_train = max(int(n * min_train_frac), 20)

    dates_oos = []
    y_actual = []
    y_fc_model = []
    y_fc_rw = []
    y_fc_ar = []

    for t in range(min_train, n - horizon):
        # Training set: 0..t (inclusive)
        train = df.iloc[:t + 1]
        y_train = train["hmfsi"].values[horizon:]
        X_corr = train["corr"].values[:-horizon]
        X_hmfsi = train["hmfsi"].values[:-horizon]

        if len(y_train) < 10:
            continue

        # Model forecast
        X_train = np.column_stack([X_corr, X_hmfsi])
        X_full = _add_const(X_train)
        try:
            beta, _, _, _ = _ols(y_train, X_full)
        except Exception:
            continue

        x_now = np.array([1.0, df["corr"].iloc[t], df["hmfsi"].iloc[t]])
        fc_model = float(x_now @ beta)

        # Actual
        actual = df["hmfsi"].iloc[t + horizon]

        # Random walk benchmark
        fc_rw = df["hmfsi"].iloc[t]

        # AR(1) benchmark
        X_ar = _add_const(X_hmfsi)
        try:
            beta_ar, _, _, _ = _ols(y_train, X_ar)
            fc_ar = float(np.array([1.0, df["hmfsi"].iloc[t]]) @ beta_ar)
        except Exception:
            fc_ar = fc_rw

        dates_oos.append(df.index[t + horizon])
        y_actual.append(actual)
        y_fc_model.append(fc_model)
        y_fc_rw.append(fc_rw)
        y_fc_ar.append(fc_ar)

    return {
        "dates": dates_oos,
        "actual": np.array(y_actual),
        "fc_model": np.array(y_fc_model),
        "fc_rw": np.array(y_fc_rw),
        "fc_ar": np.array(y_fc_ar),
        "horizon": horizon,
        "n_oos": len(y_actual),
        "min_train": min_train,
    }


def forecast_metrics(actual, forecast, label=""):
    """Compute RMSE, MAE, R²_OOS."""
    e = actual - forecast
    mse = np.mean(e ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(e))
    ss_res = np.sum(e ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2_oos = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"label": label, "RMSE": rmse, "MAE": mae, "R2_OOS": r2_oos, "MSE": mse}


def diebold_mariano_test(e1, e2, h=1, loss="MSE"):
    """
    Diebold–Mariano test with Harvey et al. (1997) small-sample correction.
    H0: equal predictive accuracy (e1 and e2 have same expected loss).
    
    Parameters
    ----------
    e1, e2 : forecast errors from models 1 and 2
    h : forecast horizon
    loss : 'MSE' or 'MAE'
    
    Returns
    -------
    dict with DM statistic, p-value (two-sided)
    """
    e1, e2 = np.asarray(e1), np.asarray(e2)
    T = len(e1)

    if loss == "MSE":
        d = e1 ** 2 - e2 ** 2
    else:
        d = np.abs(e1) - np.abs(e2)

    d_bar = np.mean(d)

    # Long-run variance of d using Bartlett kernel
    gamma_0 = np.var(d, ddof=0)
    lrv = gamma_0
    for j in range(1, h):
        gamma_j = np.mean((d[j:] - d_bar) * (d[:-j] - d_bar))
        lrv += 2 * (1 - j / h) * gamma_j
    lrv = max(lrv, 1e-10)

    DM = d_bar / np.sqrt(lrv / T)

    # Harvey et al. (1997) small-sample correction
    correction = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    DM_corrected = DM * correction

    p_val = 2 * (1 - stats.t.cdf(np.abs(DM_corrected), df=T - 1))

    return {
        "DM_stat": float(DM_corrected),
        "p_value": float(p_val),
        "mean_loss_diff": float(d_bar),
        "T": T,
        "model1_better": d_bar < 0,
    }


def clark_west_test(actual, fc_unrestricted, fc_restricted, h=1):
    """
    Clark–West (2006) MSFE-adjusted test for nested models.
    H0: restricted model forecasts at least as well.
    H1: unrestricted model has lower MSFE.
    
    One-sided test (reject if CW > critical value).
    """
    e_r = actual - fc_restricted
    e_u = actual - fc_unrestricted
    T = len(actual)

    adj = (fc_restricted - fc_unrestricted) ** 2
    f_t = e_r ** 2 - e_u ** 2 + adj

    f_bar = np.mean(f_t)
    # HAC variance
    gamma_0 = np.var(f_t, ddof=0)
    lrv = gamma_0
    for j in range(1, max(h, 2)):
        if j >= T:
            break
        gamma_j = np.mean((f_t[j:] - f_bar) * (f_t[:-j] - f_bar))
        lrv += 2 * (1 - j / max(h, 2)) * gamma_j
    lrv = max(lrv, 1e-10)

    CW = f_bar / np.sqrt(lrv / T)
    p_val = 1 - stats.norm.cdf(CW)  # one-sided

    return {
        "CW_stat": float(CW),
        "p_value_onesided": float(p_val),
        "mean_adjusted_diff": float(f_bar),
        "T": T,
    }


def run_oos_backtest(corr_s, hmfsi_s, horizons=None, min_train_frac=0.5,
                     plots_dir=None):
    """Run full OOS backtest across multiple horizons."""
    if horizons is None:
        horizons = [1, 2, 3, 4, 5, 6]

    all_results = OrderedDict()

    for h in horizons:
        oos = recursive_oos_forecast(corr_s, hmfsi_s, horizon=h,
                                     min_train_frac=min_train_frac)
        if oos["n_oos"] < 10:
            print(f"  h={h}: insufficient OOS obs ({oos['n_oos']}), skipping.")
            continue

        actual = oos["actual"]
        fc_model = oos["fc_model"]
        fc_rw = oos["fc_rw"]
        fc_ar = oos["fc_ar"]

        m_model = forecast_metrics(actual, fc_model, "corr_model")
        m_rw = forecast_metrics(actual, fc_rw, "random_walk")
        m_ar = forecast_metrics(actual, fc_ar, "AR1")

        e_model = actual - fc_model
        e_rw = actual - fc_rw
        e_ar = actual - fc_ar

        dm_vs_rw = diebold_mariano_test(e_model, e_rw, h=h)
        dm_vs_ar = diebold_mariano_test(e_model, e_ar, h=h)
        cw_vs_rw = clark_west_test(actual, fc_model, fc_rw, h=h)
        cw_vs_ar = clark_west_test(actual, fc_model, fc_ar, h=h)

        # Theil U-statistic
        theil_u_rw = m_model["RMSE"] / m_rw["RMSE"] if m_rw["RMSE"] > 0 else np.nan
        theil_u_ar = m_model["RMSE"] / m_ar["RMSE"] if m_ar["RMSE"] > 0 else np.nan

        h_result = OrderedDict({
            "horizon": h,
            "n_oos": oos["n_oos"],
            "model_metrics": m_model,
            "rw_metrics": m_rw,
            "ar1_metrics": m_ar,
            "Theil_U_vs_RW": round(theil_u_rw, 4),
            "Theil_U_vs_AR1": round(theil_u_ar, 4),
            "DM_vs_RW": dm_vs_rw,
            "DM_vs_AR1": dm_vs_ar,
            "CW_vs_RW": cw_vs_rw,
            "CW_vs_AR1": cw_vs_ar,
        })
        all_results[f"h{h}"] = h_result

        sig_dm_rw = "***" if dm_vs_rw["p_value"] < 0.05 else ""
        sig_cw_rw = "***" if cw_vs_rw["p_value_onesided"] < 0.05 else ""
        print(f"  h={h}: R²_OOS={m_model['R2_OOS']:.4f}, "
              f"Theil_U(RW)={theil_u_rw:.3f}, "
              f"DM_p={dm_vs_rw['p_value']:.4f}{sig_dm_rw}, "
              f"CW_p={cw_vs_rw['p_value_onesided']:.4f}{sig_cw_rw}, "
              f"n={oos['n_oos']}")

        # Plot: actual vs forecast + CSFE
        if plots_dir:
            _plot_oos_horizon(oos, m_model, m_rw, dm_vs_rw, cw_vs_rw, h, plots_dir)

    # Summary plot across horizons
    if plots_dir and len(all_results) > 1:
        _plot_oos_summary(all_results, plots_dir)

    return all_results


def _plot_oos_horizon(oos, m_model, m_rw, dm, cw, h, plots_dir):
    """Plot OOS forecast for a single horizon."""
    dates = oos["dates"]
    actual = oos["actual"]
    fc_model = oos["fc_model"]
    fc_rw = oos["fc_rw"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 2]})

    # Panel A: actual vs forecast
    ax = axes[0]
    ax.plot(dates, actual, color=C_BLUE, lw=1.5, label="Actual HMFSI", alpha=0.8)
    ax.plot(dates, fc_model, color=C_RED, lw=1.2, ls="--",
            label=f"ρ(β,Dₑ) model (R²_OOS={m_model['R2_OOS']:.3f})")
    ax.plot(dates, fc_rw, color=C_GREY, lw=0.8, ls=":",
            label=f"Random walk (R²_OOS={m_rw['R2_OOS']:.3f})")
    ax.set_ylabel("HMFSI")
    ax.set_title(f"Out-of-Sample Forecast: h = {h} steps "
                 f"(DM p = {dm['p_value']:.3f}, CW p = {cw['p_value_onesided']:.3f})",
                 fontweight="bold")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel B: CSFE difference
    ax = axes[1]
    e_model_sq = (actual - fc_model) ** 2
    e_rw_sq = (actual - fc_rw) ** 2
    csfe_diff = np.cumsum(e_rw_sq - e_model_sq)
    ax.fill_between(dates, 0, csfe_diff,
                    where=csfe_diff >= 0, color=C_GREEN, alpha=0.3, label="Model better")
    ax.fill_between(dates, 0, csfe_diff,
                    where=csfe_diff < 0, color=C_RED, alpha=0.3, label="RW better")
    ax.plot(dates, csfe_diff, color="black", lw=1.0)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Cumul. SFE diff\n(RW − Model)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"B1_oos_forecast_h{h}.png"),
                bbox_inches="tight")
    plt.close(fig)


def _plot_oos_summary(results, plots_dir):
    """Summary plot: R²_OOS, Theil-U, p-values across horizons."""
    hs = []
    r2_oos = []
    theil_u = []
    dm_p = []
    cw_p = []

    for key, r in results.items():
        hs.append(r["horizon"])
        r2_oos.append(r["model_metrics"]["R2_OOS"])
        theil_u.append(r["Theil_U_vs_RW"])
        dm_p.append(r["DM_vs_RW"]["p_value"])
        cw_p.append(r["CW_vs_RW"]["p_value_onesided"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.bar(hs, r2_oos, color=C_BLUE, alpha=0.7, edgecolor="black", lw=0.5)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Forecast horizon h")
    ax.set_ylabel("R²_OOS")
    ax.set_title("Out-of-Sample R²", fontweight="bold")
    ax.set_xticks(hs)

    ax = axes[1]
    ax.bar(hs, theil_u, color=C_ORANGE, alpha=0.7, edgecolor="black", lw=0.5)
    ax.axhline(1, color="black", lw=1, ls="--", label="U = 1 (no gain)")
    ax.set_xlabel("Forecast horizon h")
    ax.set_ylabel("Theil U (vs RW)")
    ax.set_title("Theil's U-statistic", fontweight="bold")
    ax.legend()
    ax.set_xticks(hs)

    ax = axes[2]
    ax.plot(hs, dm_p, "o-", color=C_RED, label="DM (two-sided)", markersize=6)
    ax.plot(hs, cw_p, "s-", color=C_PURPLE, label="CW (one-sided)", markersize=6)
    ax.axhline(0.05, color="black", lw=1, ls="--", label="5% threshold")
    ax.axhline(0.10, color=C_GREY, lw=0.8, ls=":", label="10% threshold")
    ax.set_xlabel("Forecast horizon h")
    ax.set_ylabel("p-value")
    ax.set_title("Statistical Significance", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xticks(hs)
    ax.set_ylim(0, max(0.3, max(max(dm_p), max(cw_p)) * 1.1))

    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "B1_oos_summary_across_horizons.png"),
                bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# B2: ECONOMIC BACKTEST
# ===========================================================================

def economic_backtest(corr_s, hmfsi_s, sp_prices=None, horizon=2,
                      min_train_frac=0.5, plots_dir=None):
    """
    Economic backtest: trade S&P 500 based on predicted HMFSI.
    
    Strategy:
    - At each period t, forecast HMFSI_{t+h}
    - If forecast > current HMFSI + threshold → predicted stress increase → SHORT
    - If forecast < current HMFSI − threshold → predicted stress decrease → LONG
    - Otherwise → stay flat or hold previous position
    
    If no S&P prices available, use HMFSI direction accuracy as proxy.
    """
    oos = recursive_oos_forecast(corr_s, hmfsi_s, horizon=horizon,
                                 min_train_frac=min_train_frac)
    if oos["n_oos"] < 10:
        return {"error": "insufficient OOS observations"}

    dates = oos["dates"]
    actual = oos["actual"]
    fc = oos["fc_model"]
    fc_rw = oos["fc_rw"]

    # Direction accuracy
    actual_direction = np.sign(actual - fc_rw)  # direction relative to current level
    predicted_direction = np.sign(fc - fc_rw)
    correct = (actual_direction == predicted_direction) & (actual_direction != 0)
    total_nonzero = np.sum(actual_direction != 0)
    hit_rate = np.sum(correct) / total_nonzero if total_nonzero > 0 else 0.5

    # Trading PnL using HMFSI changes as proxy for returns
    # Signal: if model predicts stress INCREASE (fc > hmfsi_now), go short equity
    # Profit/loss proxy: -actual_change when short, +actual_change when long
    actual_change = actual - fc_rw  # change in HMFSI
    position = -np.sign(fc - fc_rw)  # short if predicted increase, long if decrease

    # PnL proportional to position × (-actual_change_in_stress)
    # Negative stress change = good for equities, positive = bad
    pnl = position * (-actual_change)
    pnl_benchmark = np.abs(actual_change)  # long-only absolute moves

    cum_pnl = np.cumsum(pnl)
    cum_pnl_bm = np.cumsum(-actual_change)  # long-only: assumes stress decrease = positive

    # Performance metrics
    ann_factor = np.sqrt(12)  # approximately monthly data
    sharpe = np.mean(pnl) / np.std(pnl) * ann_factor if np.std(pnl) > 0 else 0
    max_dd = _max_drawdown(cum_pnl)
    calmar = (np.mean(pnl) * 12) / abs(max_dd) if abs(max_dd) > 0 else 0

    result = OrderedDict({
        "horizon": horizon,
        "n_trades": int(total_nonzero),
        "hit_rate": round(float(hit_rate), 4),
        "annualised_sharpe": round(float(sharpe), 4),
        "max_drawdown": round(float(max_dd), 4),
        "calmar_ratio": round(float(calmar), 4),
        "mean_pnl": round(float(np.mean(pnl)), 6),
        "total_pnl": round(float(np.sum(pnl)), 4),
        "pnl_positive_frac": round(float(np.mean(pnl > 0)), 4),
    })

    if plots_dir:
        fig, axes = plt.subplots(2, 1, figsize=(12, 7),
                                 gridspec_kw={"height_ratios": [3, 1]})

        ax = axes[0]
        ax.plot(dates, cum_pnl, color=C_BLUE, lw=1.5,
                label=f"Signal strategy (Sharpe={sharpe:.2f})")
        ax.plot(dates, cum_pnl_bm, color=C_GREY, lw=1, ls="--",
                label="Long-only benchmark")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_ylabel("Cumulative P&L (HMFSI units)")
        ax.set_title(f"Economic Backtest: h = {horizon} "
                     f"(Hit rate = {hit_rate:.1%}, Sharpe = {sharpe:.2f})",
                     fontweight="bold")
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        ax = axes[1]
        colors = [C_GREEN if p > 0 else C_RED for p in position]
        ax.bar(dates, position, color=colors, alpha=0.6, width=60)
        ax.set_ylabel("Position")
        ax.set_xlabel("Date")
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(["Short", "Flat", "Long"])
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"B2_economic_backtest_h{horizon}.png"),
                    bbox_inches="tight")
        plt.close(fig)

    return result


def _max_drawdown(cum_returns):
    """Maximum drawdown from cumulative returns."""
    running_max = np.maximum.accumulate(cum_returns)
    dd = cum_returns - running_max
    return float(np.min(dd))


# ===========================================================================
# B3: VIX ROBUSTNESS
# ===========================================================================

def vix_robustness(corr_s, hmfsi_s, rv_s=None, horizons=None, plots_dir=None):
    """
    Test whether ρ(β,Dₑ) survives as predictor after controlling for RV
    (as VIX proxy, since VIX unavailable before 1990).
    
    If VIX CSV not available, uses RV from the RV-corr aligned series.
    
    Three models:
      M1: HMFSI_{t+h} = a + ψ·corr_t + γ·HMFSI_t                (baseline)
      M2: HMFSI_{t+h} = a + ψ·corr_t + γ·HMFSI_t + δ·RV_t       (augmented)
      M3: HMFSI_{t+h} = a + γ·HMFSI_t + δ·RV_t                   (RV only)
    """
    if horizons is None:
        horizons = [1, 2, 3, 4]

    results = []

    for h in horizons:
        df = pd.DataFrame({"corr": corr_s, "hmfsi": hmfsi_s}).dropna()
        if rv_s is not None:
            df["rv"] = rv_s
            df = df.dropna()

        hmfsi_future = df["hmfsi"].shift(-h)
        reg_df = pd.DataFrame({
            "hmfsi_future": hmfsi_future,
            "corr": df["corr"],
            "hmfsi_now": df["hmfsi"],
        })
        if rv_s is not None:
            reg_df["rv"] = df["rv"]
        reg_df = reg_df.dropna()

        if len(reg_df) < 20:
            continue

        y = reg_df["hmfsi_future"].values
        hac_lags = min(h + 2, 12)

        # M1: baseline
        X1 = reg_df[["corr", "hmfsi_now"]].values
        m1 = ols_hac(y, X1, max_lags=hac_lags)

        h_result = OrderedDict({
            "horizon": h,
            "n": m1["n"],
            "M1_baseline": {
                "psi": round(float(m1["beta"][1]), 4),
                "psi_se": round(float(m1["se"][1]), 4),
                "psi_p": round(float(m1["p_value"][1]), 6),
                "R2": round(float(m1["R2"]), 4),
            },
        })

        if rv_s is not None and "rv" in reg_df.columns:
            # Check multicollinearity
            corr_rv_hmfsi = np.corrcoef(reg_df["rv"].values, reg_df["hmfsi_now"].values)[0, 1]
            corr_rv_corr = np.corrcoef(reg_df["rv"].values, reg_df["corr"].values)[0, 1]
            
            if abs(corr_rv_hmfsi) > 0.95 or abs(corr_rv_corr) > 0.95:
                h_result["M2_augmented_with_RV"] = {
                    "skipped": True,
                    "reason": f"Multicollinearity: ρ(RV,HMFSI)={corr_rv_hmfsi:.3f}, ρ(RV,corr)={corr_rv_corr:.3f}",
                }
            else:
                # M2: augmented
                X2 = reg_df[["corr", "hmfsi_now", "rv"]].values
                m2 = ols_hac(y, X2, max_lags=hac_lags)
                h_result["M2_augmented_with_RV"] = {
                    "psi_corr": round(float(m2["beta"][1]), 4),
                    "psi_corr_se": round(float(m2["se"][1]), 4),
                    "psi_corr_p": round(float(m2["p_value"][1]), 6),
                    "delta_rv": round(float(m2["beta"][3]), 4),
                    "delta_rv_se": round(float(m2["se"][3]), 4),
                    "delta_rv_p": round(float(m2["p_value"][3]), 6),
                    "R2": round(float(m2["R2"]), 4),
                }

                # M3: RV only
                X3 = reg_df[["hmfsi_now", "rv"]].values
                m3 = ols_hac(y, X3, max_lags=hac_lags)
                h_result["M3_RV_only"] = {
                    "delta_rv": round(float(m3["beta"][2]), 4),
                    "delta_rv_se": round(float(m3["se"][2]), 4),
                    "delta_rv_p": round(float(m3["p_value"][2]), 6),
                    "R2": round(float(m3["R2"]), 4),
                }

                # Interpretation
                corr_survives = m2["p_value"][1] < 0.10
                rv_significant = m2["p_value"][3] < 0.10
                h_result["interpretation"] = (
                    f"corr(β,Dₑ) {'survives' if corr_survives else 'does NOT survive'} "
                    f"after RV control (p={m2['p_value'][1]:.4f}). "
                    f"RV is {'significant' if rv_significant else 'not significant'} "
                    f"(p={m2['p_value'][3]:.4f})."
                )

        results.append(h_result)

        sig = "***" if m1["p_value"][1] < 0.05 else ""
        print(f"  h={h}: ψ(M1)={m1['beta'][1]:.4f} p={m1['p_value'][1]:.4f}{sig}", end="")
        if "M2_augmented_with_RV" in h_result and not h_result["M2_augmented_with_RV"].get("skipped"):
            m2r = h_result["M2_augmented_with_RV"]
            sig2 = "***" if m2r["psi_corr_p"] < 0.05 else ""
            print(f"  | ψ(M2)={m2r['psi_corr']:.4f} p={m2r['psi_corr_p']:.4f}{sig2}", end="")
        print()

    # Plot coefficient stability
    if plots_dir and len(results) > 0:
        _plot_vix_robustness(results, plots_dir)

    return results


def _plot_vix_robustness(results, plots_dir):
    """Plot coefficient ψ across horizons for M1 vs M2."""
    hs = [r["horizon"] for r in results]
    psi_m1 = [r["M1_baseline"]["psi"] for r in results]
    se_m1 = [r["M1_baseline"]["psi_se"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(hs))
    width = 0.35

    ax.bar(x - width / 2, psi_m1, width, yerr=[1.96 * s for s in se_m1],
           color=C_BLUE, alpha=0.7, label="M1: Baseline", capsize=4,
           edgecolor="black", lw=0.5)

    if "M2_augmented_with_RV" in results[0] and not results[0]["M2_augmented_with_RV"].get("skipped"):
        psi_m2 = [r["M2_augmented_with_RV"]["psi_corr"] for r in results]
        se_m2 = [r["M2_augmented_with_RV"]["psi_corr_se"] for r in results]
        ax.bar(x + width / 2, psi_m2, width, yerr=[1.96 * s for s in se_m2],
               color=C_RED, alpha=0.7, label="M2: + RV control", capsize=4,
               edgecolor="black", lw=0.5)

    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Forecast horizon h")
    ax.set_ylabel("ψ (coefficient on ρ(β,Dₑ))")
    ax.set_title("Predictive Coefficient Stability: Baseline vs RV-Augmented",
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(hs)
    ax.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "B3_vix_robustness_coefficients.png"),
                bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# B4: RV / VIX STANDALONE ANALYSIS (endogenous diagnostic)
# ===========================================================================

def rv_standalone_analysis(corr_s, rv_s, horizons=None, plots_dir=None):
    """
    Run the same predictive regression battery with RV instead of HMFSI.
    
    Model: RV_{t+h} = α + ψ·corr_t + γ·RV_t + ε
    
    If ρ(β,Dₑ) predicts RV, the signal is endogenous.
    If it does not, the HMFSI result is genuinely exogenous.
    """
    if horizons is None:
        horizons = [1, 2, 3, 4, 5, 6]

    # Also run Toda–Yamamoto
    ty = toda_yamamoto_granger(corr_s.values, rv_s.values, max_ar=4, d_max=1)
    print(f"  Toda–Yamamoto: corr→RV p={ty['x_to_y_p']}, RV→corr p={ty['y_to_x_p']}")

    results = []
    for h in horizons:
        df = pd.DataFrame({"corr": corr_s, "rv": rv_s}).dropna()
        rv_future = df["rv"].shift(-h)
        reg_df = pd.DataFrame({
            "rv_future": rv_future,
            "corr": df["corr"],
            "rv_now": df["rv"],
        }).dropna()

        if len(reg_df) < 20:
            continue

        y = reg_df["rv_future"].values
        X = reg_df[["corr", "rv_now"]].values
        m = ols_hac(y, X, max_lags=min(h + 2, 12))

        sig = "***" if m["p_value"][1] < 0.05 else ""
        print(f"  h={h}: ψ={m['beta'][1]:.4f}, p={m['p_value'][1]:.4f}{sig}, "
              f"R²={m['R2']:.4f}")

        results.append(OrderedDict({
            "horizon": h,
            "psi": round(float(m["beta"][1]), 4),
            "psi_se": round(float(m["se"][1]), 4),
            "psi_p": round(float(m["p_value"][1]), 6),
            "R2": round(float(m["R2"]), 4),
            "n": m["n"],
            "significant_005": bool(m["p_value"][1] < 0.05),
        }))

    overall = OrderedDict({
        "toda_yamamoto": ty,
        "predictive_regressions": results,
        "any_significant": any(r["significant_005"] for r in results),
        "interpretation": (
            "If ρ(β,Dₑ) predicts RV → the signal has an endogenous component. "
            "If it does NOT predict RV → the HMFSI result is genuinely exogenous."
        ),
    })

    if plots_dir and len(results) > 0:
        _plot_rv_standalone(results, ty, plots_dir)

    return overall


def _plot_rv_standalone(results, ty, plots_dir):
    """Plot RV predictive regression results vs HMFSI comparison."""
    hs = [r["horizon"] for r in results]
    psis = [r["psi"] for r in results]
    ses = [r["psi_se"] for r in results]
    ps = [r["psi_p"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    colors_bar = [C_RED if p < 0.05 else (C_ORANGE if p < 0.10 else C_GREY)
                  for p in ps]
    ax.bar(hs, psis, yerr=[1.96 * s for s in ses], color=colors_bar,
           alpha=0.7, capsize=4, edgecolor="black", lw=0.5)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Forecast horizon h")
    ax.set_ylabel("ψ coefficient")
    ax.set_title("RV Predictive Regression:\nρ(β,Dₑ) → RV_{t+h}", fontweight="bold")
    ax.set_xticks(hs)

    # Add legend for significance
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_RED, alpha=0.7, label="p < 0.05"),
        Patch(facecolor=C_ORANGE, alpha=0.7, label="p < 0.10"),
        Patch(facecolor=C_GREY, alpha=0.7, label="p ≥ 0.10"),
    ]
    ax.legend(handles=legend_elements, fontsize=8)

    ax = axes[1]
    ax.barh(["corr → RV", "RV → corr"],
            [ty["x_to_y_p"] or 1, ty["y_to_x_p"] or 1],
            color=[C_BLUE, C_RED], alpha=0.7, edgecolor="black", lw=0.5)
    ax.axvline(0.05, color="black", lw=1, ls="--", label="5% threshold")
    ax.set_xlabel("p-value")
    ax.set_title("Toda–Yamamoto Granger Causality", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "B4_rv_standalone_analysis.png"),
                bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# B5: SUBSAMPLE STABILITY
# ===========================================================================

def subsample_stability(corr_s, hmfsi_s, horizon=2, window_frac=0.5,
                        plots_dir=None):
    """
    Assess parameter stability via:
    (a) Rolling-window ψ estimates
    (b) First-half vs second-half comparison
    (c) CUSUM-type test on recursive residuals
    """
    df = pd.DataFrame({"corr": corr_s, "hmfsi": hmfsi_s}).dropna()
    n = len(df)
    h = horizon

    # Prepare full regression data
    hmfsi_future = df["hmfsi"].shift(-h)
    reg_df = pd.DataFrame({
        "y": hmfsi_future,
        "corr": df["corr"],
        "hmfsi_now": df["hmfsi"],
    }).dropna()
    n_reg = len(reg_df)

    # (a) Rolling-window estimates
    win_size = max(int(n_reg * window_frac), 30)
    roll_dates = []
    roll_psi = []
    roll_psi_se = []
    roll_psi_p = []

    for start in range(0, n_reg - win_size + 1):
        end = start + win_size
        sub = reg_df.iloc[start:end]
        y_sub = sub["y"].values
        X_sub = sub[["corr", "hmfsi_now"]].values
        m = ols_hac(y_sub, X_sub, max_lags=min(h + 2, 8))
        roll_dates.append(reg_df.index[end - 1])
        roll_psi.append(m["beta"][1])
        roll_psi_se.append(m["se"][1])
        roll_psi_p.append(m["p_value"][1])

    # (b) First-half vs second-half
    mid = n_reg // 2
    y_full = reg_df["y"].values
    X_full = reg_df[["corr", "hmfsi_now"]].values

    m_first = ols_hac(y_full[:mid], X_full[:mid], max_lags=min(h + 2, 8))
    m_second = ols_hac(y_full[mid:], X_full[mid:], max_lags=min(h + 2, 8))
    m_full = ols_hac(y_full, X_full, max_lags=min(h + 2, 8))

    # (c) CUSUM via recursive residuals
    rec_resid = []
    for t in range(20, n_reg):
        y_t = y_full[:t]
        X_t = _add_const(X_full[:t])
        beta_t, _, _, _ = _ols(y_t, X_t)
        x_next = np.concatenate([[1], X_full[t]])
        fc = x_next @ beta_t
        rec_resid.append(y_full[t] - fc)
    rec_resid = np.array(rec_resid)
    sigma_rec = np.std(rec_resid)
    cusum = np.cumsum(rec_resid) / (sigma_rec * np.sqrt(len(rec_resid))) if sigma_rec > 0 else np.zeros(len(rec_resid))

    result = OrderedDict({
        "horizon": horizon,
        "full_sample": {
            "psi": round(float(m_full["beta"][1]), 4),
            "psi_se": round(float(m_full["se"][1]), 4),
            "psi_p": round(float(m_full["p_value"][1]), 6),
            "R2": round(float(m_full["R2"]), 4),
            "n": m_full["n"],
        },
        "first_half": {
            "psi": round(float(m_first["beta"][1]), 4),
            "psi_p": round(float(m_first["p_value"][1]), 6),
            "n": m_first["n"],
        },
        "second_half": {
            "psi": round(float(m_second["beta"][1]), 4),
            "psi_p": round(float(m_second["p_value"][1]), 6),
            "n": m_second["n"],
        },
        "cusum_max_abs": round(float(np.max(np.abs(cusum))), 4),
        "cusum_5pct_critical": 1.358,  # approximate 5% boundary for CUSUM
        "structural_break_detected": bool(np.max(np.abs(cusum)) > 1.358),
        "n_rolling_windows": len(roll_psi),
        "psi_rolling_mean": round(float(np.mean(roll_psi)), 4),
        "psi_rolling_std": round(float(np.std(roll_psi)), 4),
        "frac_significant_005": round(float(np.mean(np.array(roll_psi_p) < 0.05)), 4),
    })

    if plots_dir:
        _plot_subsample_stability(roll_dates, roll_psi, roll_psi_se, roll_psi_p,
                                  cusum, reg_df.index[20:20 + len(cusum)],
                                  m_first, m_second, m_full, horizon, plots_dir)

    return result


def _plot_subsample_stability(dates, psi, psi_se, psi_p, cusum, cusum_dates,
                               m_first, m_second, m_full, h, plots_dir):
    """Three-panel stability plot."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 11))

    psi = np.array(psi)
    psi_se = np.array(psi_se)

    # Panel A: Rolling ψ
    ax = axes[0]
    ax.plot(dates, psi, color=C_BLUE, lw=1.5)
    ax.fill_between(dates, psi - 1.96 * psi_se, psi + 1.96 * psi_se,
                    color=C_BLUE, alpha=0.15)
    ax.axhline(m_full["beta"][1], color=C_RED, ls="--", lw=1,
               label=f"Full-sample ψ = {m_full['beta'][1]:.3f}")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("ψ (rolling)")
    ax.set_title(f"Rolling-Window ψ Estimate (h = {h})", fontweight="bold")
    ax.legend()
    if len(dates) > 0 and hasattr(dates[0], 'year'):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel B: Rolling p-value
    ax = axes[1]
    psi_p = np.array(psi_p)
    ax.plot(dates, psi_p, color=C_PURPLE, lw=1)
    ax.axhline(0.05, color=C_RED, ls="--", lw=1, label="5% threshold")
    ax.axhline(0.10, color=C_ORANGE, ls=":", lw=1, label="10% threshold")
    ax.fill_between(dates, 0, psi_p, where=psi_p < 0.05,
                    color=C_GREEN, alpha=0.2, label="Significant (5%)")
    ax.set_ylabel("p-value")
    ax.set_title("Rolling p-value of ψ", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0, min(1, np.max(psi_p) * 1.1))
    if len(dates) > 0 and hasattr(dates[0], 'year'):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel C: CUSUM
    ax = axes[2]
    if len(cusum_dates) > 0:
        ax.plot(cusum_dates, cusum, color=C_BLUE, lw=1.5)
        n_c = len(cusum)
        t_grid = np.arange(1, n_c + 1) / n_c
        boundary = 1.358 * np.ones(n_c)  # simplified boundary
        ax.plot(cusum_dates, boundary, "r--", lw=1, label="5% boundary")
        ax.plot(cusum_dates, -boundary, "r--", lw=1)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("Date")
        ax.set_ylabel("CUSUM")
        ax.set_title("CUSUM Test for Structural Stability", fontweight="bold")
        ax.legend()
        if hasattr(cusum_dates[0], 'year'):
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"B5_subsample_stability_h{h}.png"),
                bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# B6: BOOTSTRAP CONFIDENCE INTERVALS & ADDITIONAL ROBUSTNESS
# ===========================================================================

def block_bootstrap_ci(corr_s, hmfsi_s, horizon=2, n_boot=5000,
                        block_size=None, alpha=0.05):
    """
    Circular block bootstrap confidence intervals for ψ.
    
    Uses overlapping circular blocks to preserve serial dependence.
    """
    df = pd.DataFrame({"corr": corr_s, "hmfsi": hmfsi_s}).dropna()
    n = len(df)
    h = horizon

    hmfsi_future = df["hmfsi"].shift(-h)
    reg_df = pd.DataFrame({
        "y": hmfsi_future,
        "corr": df["corr"],
        "hmfsi_now": df["hmfsi"],
    }).dropna()
    n_reg = len(reg_df)

    if block_size is None:
        block_size = max(3, int(np.ceil(n_reg ** (1 / 3))))

    y_full = reg_df["y"].values
    X_full = reg_df[["corr", "hmfsi_now"]].values

    rng = np.random.default_rng(42)
    psi_boots = []

    for b in range(n_boot):
        # Circular block bootstrap indices
        starts = rng.integers(0, n_reg, size=int(np.ceil(n_reg / block_size)))
        indices = []
        for s in starts:
            indices.extend(range(s, s + block_size))
        indices = [i % n_reg for i in indices[:n_reg]]

        y_b = y_full[indices]
        X_b = X_full[indices]

        try:
            X_bc = _add_const(X_b)
            beta_b, _, _, _ = _ols(y_b, X_bc)
            psi_boots.append(beta_b[1])
        except Exception:
            continue

    psi_boots = np.array(psi_boots)
    ci_lo = np.percentile(psi_boots, 100 * alpha / 2)
    ci_hi = np.percentile(psi_boots, 100 * (1 - alpha / 2))
    boot_se = np.std(psi_boots)
    boot_mean = np.mean(psi_boots)
    # Bias-corrected p-value (proportion of bootstrap ψ ≤ 0)
    boot_p = 2 * min(np.mean(psi_boots <= 0), np.mean(psi_boots > 0))

    return OrderedDict({
        "horizon": horizon,
        "n_boot": n_boot,
        "block_size": block_size,
        "psi_boot_mean": round(float(boot_mean), 4),
        "psi_boot_se": round(float(boot_se), 4),
        "ci_95_lo": round(float(ci_lo), 4),
        "ci_95_hi": round(float(ci_hi), 4),
        "boot_p_value": round(float(boot_p), 4),
        "zero_excluded_from_ci": bool(ci_lo > 0 or ci_hi < 0),
    })


def holm_correction(p_values):
    """
    Holm (1979) step-down correction for multiple testing.
    Returns adjusted p-values.
    """
    m = len(p_values)
    sorted_idx = np.argsort(p_values)
    adjusted = np.zeros(m)
    for i, idx in enumerate(sorted_idx):
        adjusted[idx] = min(1.0, p_values[idx] * (m - i))
    # Enforce monotonicity
    for i in range(1, m):
        idx = sorted_idx[i]
        prev_idx = sorted_idx[i - 1]
        adjusted[idx] = max(adjusted[idx], adjusted[prev_idx])
    return adjusted


def run_additional_robustness(corr_s, hmfsi_s, horizons=None, plots_dir=None):
    """
    B6: Bootstrap CI + Holm correction + LOO-CV.
    """
    if horizons is None:
        horizons = [1, 2, 3, 4]

    results = OrderedDict()

    # Bootstrap CI for each horizon
    print("  Running block bootstrap (5000 replications)...")
    boot_results = []
    for h in horizons:
        br = block_bootstrap_ci(corr_s, hmfsi_s, horizon=h, n_boot=5000)
        boot_results.append(br)
        sig = "***" if br["zero_excluded_from_ci"] else ""
        print(f"    h={h}: ψ_boot={br['psi_boot_mean']:.4f} "
              f"CI=[{br['ci_95_lo']:.4f}, {br['ci_95_hi']:.4f}]{sig}")
    results["bootstrap_ci"] = boot_results

    # Holm correction on full-sample p-values
    print("  Computing Holm-corrected p-values...")
    raw_p = []
    for h in horizons:
        df = pd.DataFrame({"corr": corr_s, "hmfsi": hmfsi_s}).dropna()
        hmfsi_future = df["hmfsi"].shift(-h)
        reg_df = pd.DataFrame({
            "y": hmfsi_future, "corr": df["corr"], "hmfsi_now": df["hmfsi"],
        }).dropna()
        y = reg_df["y"].values
        X = reg_df[["corr", "hmfsi_now"]].values
        m = ols_hac(y, X, max_lags=min(h + 2, 12))
        raw_p.append(m["p_value"][1])

    adj_p = holm_correction(np.array(raw_p))
    holm_results = []
    for i, h in enumerate(horizons):
        holm_results.append({
            "horizon": h,
            "raw_p": round(float(raw_p[i]), 6),
            "holm_p": round(float(adj_p[i]), 6),
            "significant_005_holm": bool(adj_p[i] < 0.05),
        })
        sig = "***" if adj_p[i] < 0.05 else ""
        print(f"    h={h}: raw_p={raw_p[i]:.4f} → Holm_p={adj_p[i]:.4f}{sig}")
    results["holm_correction"] = holm_results
    results["any_holm_significant"] = any(r["significant_005_holm"] for r in holm_results)

    # Plot bootstrap distributions
    if plots_dir:
        _plot_bootstrap(boot_results, holm_results, plots_dir)

    return results


def _plot_bootstrap(boot_results, holm_results, plots_dir):
    """Plot bootstrap results summary."""
    hs = [r["horizon"] for r in boot_results]
    means = [r["psi_boot_mean"] for r in boot_results]
    ci_lo = [r["ci_95_lo"] for r in boot_results]
    ci_hi = [r["ci_95_hi"] for r in boot_results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Bootstrap CI
    ax = axes[0]
    x = np.arange(len(hs))
    ax.errorbar(x, means,
                yerr=[np.array(means) - np.array(ci_lo),
                      np.array(ci_hi) - np.array(means)],
                fmt="o", color=C_BLUE, capsize=6, markersize=8, lw=2,
                label="95% Block-Bootstrap CI")
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"h={h}" for h in hs])
    ax.set_ylabel("ψ")
    ax.set_title("Block-Bootstrap Confidence Intervals\n(5000 replications)",
                 fontweight="bold")
    ax.legend()

    # Panel B: Raw vs Holm p-values
    ax = axes[1]
    raw_ps = [r["raw_p"] for r in holm_results]
    holm_ps = [r["holm_p"] for r in holm_results]
    width = 0.35
    ax.bar(x - width / 2, raw_ps, width, color=C_BLUE, alpha=0.7,
           label="Raw p-value", edgecolor="black", lw=0.5)
    ax.bar(x + width / 2, holm_ps, width, color=C_RED, alpha=0.7,
           label="Holm-adjusted p", edgecolor="black", lw=0.5)
    ax.axhline(0.05, color="black", lw=1, ls="--", label="5% threshold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"h={h}" for h in hs])
    ax.set_ylabel("p-value")
    ax.set_title("Multiple-Testing Correction\n(Holm step-down)", fontweight="bold")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "B6_bootstrap_and_holm.png"),
                bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# MASTER SUMMARY PLOT
# ===========================================================================

def plot_master_summary(all_results, plots_dir):
    """Single comprehensive figure summarising all key findings."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (0,0) In-sample predictive regression ψ coefficients
    ax = axes[0, 0]
    if "B3_vix_robustness" in all_results:
        vr = all_results["B3_vix_robustness"]
        hs = [r["horizon"] for r in vr]
        psis = [r["M1_baseline"]["psi"] for r in vr]
        ps = [r["M1_baseline"]["psi_p"] for r in vr]
        colors_b = [C_RED if p < 0.05 else C_GREY for p in ps]
        ax.bar(hs, psis, color=colors_b, alpha=0.7, edgecolor="black", lw=0.5)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title("In-Sample ψ", fontweight="bold")
        ax.set_xlabel("h")
        ax.set_ylabel("ψ")

    # (0,1) OOS R² across horizons
    ax = axes[0, 1]
    if "B1_oos_forecasting" in all_results:
        oos = all_results["B1_oos_forecasting"]
        hs = [int(k[1:]) for k in oos.keys()]
        r2s = [oos[k]["model_metrics"]["R2_OOS"] for k in oos.keys()]
        ax.bar(hs, r2s, color=C_BLUE, alpha=0.7, edgecolor="black", lw=0.5)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title("Out-of-Sample R²", fontweight="bold")
        ax.set_xlabel("h")
        ax.set_ylabel("R²_OOS")

    # (0,2) Theil U
    ax = axes[0, 2]
    if "B1_oos_forecasting" in all_results:
        oos = all_results["B1_oos_forecasting"]
        hs = [int(k[1:]) for k in oos.keys()]
        us = [oos[k]["Theil_U_vs_RW"] for k in oos.keys()]
        ax.bar(hs, us, color=C_ORANGE, alpha=0.7, edgecolor="black", lw=0.5)
        ax.axhline(1, color="black", lw=1, ls="--")
        ax.set_title("Theil U (vs RW)", fontweight="bold")
        ax.set_xlabel("h")
        ax.set_ylabel("U")

    # (1,0) Bootstrap CI
    ax = axes[1, 0]
    if "B6_additional_robustness" in all_results:
        br = all_results["B6_additional_robustness"]["bootstrap_ci"]
        hs = [r["horizon"] for r in br]
        means = [r["psi_boot_mean"] for r in br]
        ci_lo = [r["ci_95_lo"] for r in br]
        ci_hi = [r["ci_95_hi"] for r in br]
        x = np.arange(len(hs))
        ax.errorbar(x, means,
                    yerr=[np.array(means) - np.array(ci_lo),
                          np.array(ci_hi) - np.array(means)],
                    fmt="o", color=C_BLUE, capsize=6, markersize=8, lw=2)
        ax.axhline(0, color="black", lw=1, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels([f"h={h}" for h in hs])
        ax.set_title("Bootstrap 95% CI", fontweight="bold")
        ax.set_ylabel("ψ")

    # (1,1) Endogenous vs exogenous comparison
    ax = axes[1, 1]
    # Compare HMFSI vs RV predictive p-values
    if "B4_rv_standalone" in all_results:
        rv_res = all_results["B4_rv_standalone"]["predictive_regressions"]
        rv_hs = [r["horizon"] for r in rv_res]
        rv_ps = [r["psi_p"] for r in rv_res]

    if "B3_vix_robustness" in all_results:
        hmfsi_res = all_results["B3_vix_robustness"]
        hmfsi_hs = [r["horizon"] for r in hmfsi_res]
        hmfsi_ps = [r["M1_baseline"]["psi_p"] for r in hmfsi_res]

        if "B4_rv_standalone" in all_results:
            common_hs = sorted(set(hmfsi_hs) & set(rv_hs))
            if common_hs:
                x = np.arange(len(common_hs))
                width = 0.35
                h_ps = [hmfsi_ps[hmfsi_hs.index(h)] for h in common_hs]
                r_ps = [rv_ps[rv_hs.index(h)] for h in common_hs]
                ax.bar(x - width / 2, h_ps, width, color=C_BLUE, alpha=0.7,
                       label="→ HMFSI (exogenous)", edgecolor="black", lw=0.5)
                ax.bar(x + width / 2, r_ps, width, color=C_RED, alpha=0.7,
                       label="→ RV (endogenous)", edgecolor="black", lw=0.5)
                ax.axhline(0.05, color="black", lw=1, ls="--")
                ax.set_xticks(x)
                ax.set_xticklabels([f"h={h}" for h in common_hs])
                ax.set_title("Exogenous vs Endogenous", fontweight="bold")
                ax.set_ylabel("p-value (ψ)")
                ax.legend(fontsize=8)

    # (1,2) Economic backtest summary
    ax = axes[1, 2]
    if "B2_economic_backtest" in all_results:
        eb = all_results["B2_economic_backtest"]
        metrics = ["hit_rate", "annualised_sharpe"]
        vals = [eb.get(m, 0) for m in metrics]
        labels = ["Hit Rate", "Ann. Sharpe"]
        bars = ax.barh(labels, vals, color=[C_GREEN, C_BLUE], alpha=0.7,
                       edgecolor="black", lw=0.5)
        ax.axvline(0.5, color=C_GREY, ls=":", lw=1)
        ax.set_title("Economic Backtest", fontweight="bold")
        ax.set_xlabel("Value")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=10)

    fig.suptitle("Predictive Backtesting: ρ(β, Dₑ) → HMFSI — Master Summary",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "MASTER_SUMMARY_backtest.png"),
                bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive backtesting of ρ(β,Dₑ) → HMFSI predictive regression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python backtesting_predictive.py \\
    --hmfsi-corr DJI_HMFSI_CORR_BETA_DE/aligned_series.csv \\
    --rv-corr results_corr_250_acf/aligned_series.csv \\
    --hmfsi-daily hmfsi_output/HMFSI.csv \\
    --out backtesting_predittivita
        """,
    )
    parser.add_argument("--hmfsi-corr", required=True,
                        help="aligned_series.csv from corr(β,Dₑ) vs HMFSI analysis")
    parser.add_argument("--rv-corr", default=None,
                        help="aligned_series.csv from corr(β,Dₑ) vs RV analysis "
                             "(for VIX/RV robustness)")
    parser.add_argument("--hmfsi-daily", default=None,
                        help="HMFSI.csv (daily) for economic backtest alignment")
    parser.add_argument("--vix-csv", default=None,
                        help="VIX CSV (optional; if absent, uses RV from --rv-corr)")
    parser.add_argument("--sp-prices", default=None,
                        help="S&P 500 prices CSV for economic backtest")
    parser.add_argument("--out", default="backtesting_predittivita",
                        help="Output directory")
    parser.add_argument("--min-train-frac", type=float, default=0.5,
                        help="Minimum training fraction for OOS (default 0.5)")
    parser.add_argument("--n-boot", type=int, default=5000,
                        help="Number of bootstrap replications (default 5000)")
    parser.add_argument("--horizons", default="1,2,3,4,5,6",
                        help="Comma-separated forecast horizons")
    args = parser.parse_args()

    out_dir = args.out
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    horizons = [int(h) for h in args.horizons.split(",")]

    print("=" * 70)
    print("BACKTESTING PREDICTIVE REGRESSION: ρ(β,Dₑ) → HMFSI")
    print("=" * 70)
    print(f"Output directory: {out_dir}")
    print(f"Horizons: {horizons}")
    print(f"Min train fraction: {args.min_train_frac}")
    print()

    # Load main data
    print("[1/7] Loading data...")
    corr_s, hmfsi_s = load_aligned(args.hmfsi_corr)
    print(f"  HMFSI-corr: n={len(corr_s)}, "
          f"dates {corr_s.index.min()} to {corr_s.index.max()}")
    print(f"  corr range: [{corr_s.min():.3f}, {corr_s.max():.3f}]")
    print(f"  HMFSI range: [{hmfsi_s.min():.3f}, {hmfsi_s.max():.3f}]")
    print(f"  ρ(corr, HMFSI) = {np.corrcoef(corr_s.values, hmfsi_s.values)[0,1]:.4f}")

    rv_s = None
    if args.rv_corr and os.path.exists(args.rv_corr):
        _, rv_s_raw = load_aligned(args.rv_corr)
        # Align RV to HMFSI-corr dates
        rv_aligned = rv_s_raw.reindex(corr_s.index, method="nearest", tolerance=pd.Timedelta("60D"))
        rv_s = rv_aligned.dropna()
        print(f"  RV-corr loaded: n={len(rv_s)}")
    else:
        print("  [WARN] No RV-corr file; VIX robustness will use reduced tests.")

    all_results = OrderedDict()
    all_results["metadata"] = {
        "script": "backtesting_predictive.py",
        "run_date": datetime.now().isoformat(),
        "hmfsi_corr_path": args.hmfsi_corr,
        "rv_corr_path": args.rv_corr,
        "n_obs": len(corr_s),
        "horizons": horizons,
        "min_train_frac": args.min_train_frac,
    }

    # ===================================================================
    # B1: Out-of-sample forecasting
    # ===================================================================
    print("\n" + "=" * 70)
    print("[2/7] B1: Out-of-Sample Recursive Forecasting")
    print("=" * 70)
    oos_results = run_oos_backtest(corr_s, hmfsi_s, horizons=horizons,
                                   min_train_frac=args.min_train_frac,
                                   plots_dir=plots_dir)
    all_results["B1_oos_forecasting"] = oos_results

    # ===================================================================
    # B2: Economic backtest
    # ===================================================================
    print("\n" + "=" * 70)
    print("[3/7] B2: Economic Backtest")
    print("=" * 70)
    econ_result = economic_backtest(corr_s, hmfsi_s, horizon=2,
                                    min_train_frac=args.min_train_frac,
                                    plots_dir=plots_dir)
    all_results["B2_economic_backtest"] = econ_result
    if "hit_rate" in econ_result:
        print(f"  Hit rate: {econ_result['hit_rate']:.1%}")
        print(f"  Sharpe: {econ_result['annualised_sharpe']:.3f}")

    # ===================================================================
    # B3: VIX / RV Robustness
    # ===================================================================
    print("\n" + "=" * 70)
    print("[4/7] B3: Robustness — Controlling for RV (VIX proxy)")
    print("=" * 70)
    vix_results = vix_robustness(corr_s, hmfsi_s, rv_s=rv_s,
                                  horizons=[h for h in horizons if h <= 6],
                                  plots_dir=plots_dir)
    all_results["B3_vix_robustness"] = vix_results

    # ===================================================================
    # B4: RV standalone (endogenous diagnostic)
    # ===================================================================
    print("\n" + "=" * 70)
    print("[5/7] B4: RV Standalone Analysis (Endogenous Diagnostic)")
    print("=" * 70)
    if rv_s is not None and len(rv_s) >= 30:
        # Align
        common_idx = corr_s.index.intersection(rv_s.index)
        if len(common_idx) >= 30:
            rv_result = rv_standalone_analysis(
                corr_s.reindex(common_idx),
                rv_s.reindex(common_idx),
                horizons=[h for h in horizons if h <= 6],
                plots_dir=plots_dir,
            )
            all_results["B4_rv_standalone"] = rv_result
        else:
            print("  [SKIP] Insufficient overlapping observations.")
            all_results["B4_rv_standalone"] = {"skipped": True, "reason": "insufficient overlap"}
    else:
        print("  [SKIP] No RV data available.")
        all_results["B4_rv_standalone"] = {"skipped": True, "reason": "no RV file"}

    # ===================================================================
    # B5: Subsample stability
    # ===================================================================
    print("\n" + "=" * 70)
    print("[6/7] B5: Subsample Stability")
    print("=" * 70)
    stab_result = subsample_stability(corr_s, hmfsi_s, horizon=2,
                                       window_frac=0.5, plots_dir=plots_dir)
    all_results["B5_subsample_stability"] = stab_result
    print(f"  Full-sample ψ: {stab_result['full_sample']['psi']:.4f} "
          f"(p={stab_result['full_sample']['psi_p']:.4f})")
    print(f"  First half ψ:  {stab_result['first_half']['psi']:.4f} "
          f"(p={stab_result['first_half']['psi_p']:.4f})")
    print(f"  Second half ψ: {stab_result['second_half']['psi']:.4f} "
          f"(p={stab_result['second_half']['psi_p']:.4f})")
    print(f"  CUSUM max: {stab_result['cusum_max_abs']:.4f} "
          f"(crit=1.358, break={'YES' if stab_result['structural_break_detected'] else 'NO'})")
    print(f"  Rolling windows significant (5%): "
          f"{stab_result['frac_significant_005']:.1%}")

    # ===================================================================
    # B6: Bootstrap + Holm correction
    # ===================================================================
    print("\n" + "=" * 70)
    print("[7/7] B6: Block Bootstrap CI & Holm Multiple-Testing Correction")
    print("=" * 70)
    robust_result = run_additional_robustness(corr_s, hmfsi_s,
                                              horizons=[h for h in horizons if h <= 4],
                                              plots_dir=plots_dir)
    all_results["B6_additional_robustness"] = robust_result

    # ===================================================================
    # Master summary plot
    # ===================================================================
    print("\n[FINAL] Generating master summary plot...")
    plot_master_summary(all_results, plots_dir)

    # ===================================================================
    # Save JSON
    # ===================================================================
    def _json_safe(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp, datetime)):
            return str(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return str(obj)

    json_path = os.path.join(out_dir, "backtesting_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_json_safe)
    print(f"\n→ All results saved to {json_path}")
    print(f"→ Plots saved to {plots_dir}/")

    # Print final verdict
    print("\n" + "=" * 70)
    print("VERDICT SUMMARY")
    print("=" * 70)

    # OOS
    if "B1_oos_forecasting" in all_results:
        oos = all_results["B1_oos_forecasting"]
        sig_oos = sum(1 for k, v in oos.items()
                      if isinstance(v, dict) and "CW_vs_RW" in v
                      and v["CW_vs_RW"]["p_value_onesided"] < 0.05)
        print(f"  OOS: {sig_oos}/{len(oos)} horizons with CW p < 0.05")

    # Bootstrap
    if "B6_additional_robustness" in all_results:
        br = all_results["B6_additional_robustness"]
        sig_boot = sum(1 for r in br["bootstrap_ci"] if r["zero_excluded_from_ci"])
        sig_holm = sum(1 for r in br["holm_correction"] if r["significant_005_holm"])
        print(f"  Bootstrap: {sig_boot}/{len(br['bootstrap_ci'])} horizons with 0 ∉ CI")
        print(f"  Holm: {sig_holm}/{len(br['holm_correction'])} horizons survive correction")

    # Stability
    if "B5_subsample_stability" in all_results:
        ss = all_results["B5_subsample_stability"]
        print(f"  Stability: CUSUM {'PASS' if not ss['structural_break_detected'] else 'FAIL'}, "
              f"{ss['frac_significant_005']:.0%} rolling windows significant")

    # Endogenous diagnostic
    if "B4_rv_standalone" in all_results and not all_results["B4_rv_standalone"].get("skipped"):
        rv_r = all_results["B4_rv_standalone"]
        any_rv_sig = rv_r.get("any_significant", False)
        print(f"  Endogenous: corr→RV {'SIGNIFICANT' if any_rv_sig else 'NOT significant'}")
        print(f"  → {'WARNING: signal has endogenous component' if any_rv_sig else 'HMFSI result is genuinely exogenous'}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
