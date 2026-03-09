#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""stress_master_all_tests.py

Five complementary statistical tests of dependence between a financial stress
proxy and target series derived from rolling Baldovin-Stella model parameters.

Implemented tests (ONLY these 5; the legacy project tests are intentionally
ignored as requested):

  (T1) Toda-Yamamoto augmented VAR Wald tests for (Granger-type) causality
       in levels, robust to integration/cointegration pretesting.

  (T2) Quantile "Granger" predictability via quantile regression improvement
       tested by circular-shift permutations (preserves serial dependence
       of the stress series while breaking alignment).

  (T3) Local projections (Jorda) impulse responses of the target to a stress
       shock, with Newey-West HAC standard errors.

  (T4) Distance correlation (nonlinear omnibus dependence) across leads/lags,
       with a max-over-lags test using circular-shift permutations.

  (T5) Wavelet coherence (time-frequency dependence) based on CWT, with
       scale-wise significance thresholds from circular-shift surrogates.

Inputs
------
- Rolling CSV: like rigid_windows_results.csv (must contain a date column,
  typically 'window_date', and parameter columns).

- Prices file: like PRICE_DEF.txt with columns Date, Close.

Stress series options
--------------------
- --stress-source rv
    Realized volatility computed from prices (rolling std of log returns,
    annualized). Always available.

- --stress-source fred:<SERIES_ID>
    Download a stress index from FRED via fredgraph.csv. Works without API key
    in most environments (requires outbound internet).

If downloading fails at runtime, the script falls back to RV automatically.

Target series options
---------------------
- --target col:<COLUMN>
    Use a single column from rolling CSV.

- --target fisher_corr:<COL_A>:<COL_B>
    Compute rolling correlation between COL_A and COL_B on the rolling-CSV
    timeline, then apply Fisher z = atanh(corr). Default corr window = 10.

Outputs
-------
Creates an output directory containing:
- MASTER_SUMMARY.csv      (single-row summary for quick reading)
- results.json            (full structured results)
- plots/*.png             (one or more plots per test)
- aligned_series.csv      (final aligned y and stress series)

Run examples
------------
1) Use FRED STLFSI4 as stress series:

  python stress_master_all_tests.py \
    --rolling rigid_windows_results.csv \
    --prices PRICE_DEF.txt \
    --target fisher_corr:D:beta_hac \
    --stress-source fred:STLFSI4 \
    --out results_5tests

2) Use RV stress fallback (no downloads):

  python stress_master_all_tests.py \
    --rolling rigid_windows_results.csv \
    --prices PRICE_DEF.txt \
    --target fisher_corr:D:beta_hac \
    --stress-source rv \
    --out results_5tests_rv

Notes on statistical robustness
-------------------------------
- Overlapping/rolling constructions induce serial dependence. Where an
  asymptotic test is used (T1, T3), HAC (Newey-West) is used.

- Where a permutation/reference distribution is required (T2, T4, T5),
  we use circular-shift permutations of the stress series. This preserves the
  stress series' autocorrelation and marginal distribution while breaking
  alignment with the target series.

Disclaimer
----------
Wavelet coherence here is implemented as a practical, transparent estimator
with surrogate-based significance. It is suitable for exploratory/robustness
sections, not as the sole inferential pillar.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Core stats
from scipy.stats import chi2

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

try:
    import pywt
except Exception:  # pragma: no cover
    pywt = None

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

# Optional graphics patch for publication-quality figures.
# If PATCH_GRAFICA_7_3 is available in the environment it will be applied.
try:
    from PATCH_GRAFICA_7_3 import apply_patch
    apply_patch()
except ImportError:
    pass
# -----------------------------
# Utilities
# -----------------------------


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _zscore(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    mu = x.mean()
    sd = x.std(ddof=0)
    if sd == 0 or not np.isfinite(sd):
        return x * 0.0
    return (x - mu) / sd


def _safe_atanh(r: pd.Series, eps: float = 1e-6) -> pd.Series:
    r = r.astype(float).clip(-1 + eps, 1 - eps)
    return np.arctanh(r)


def _circular_shift(values: np.ndarray, k: int) -> np.ndarray:
    if len(values) == 0:
        return values
    k = int(k) % len(values)
    if k == 0:
        return values.copy()
    return np.concatenate([values[-k:], values[:-k]])


def _infer_date_col(df: pd.DataFrame) -> str:
    """Infer the rolling timestamp column.

    Prefers end-of-window timestamps (e.g., 'window_end_date') to avoid implicit
    forward-looking when rolling estimates are stamped at the window center.
    Falls back to legacy center/start date columns if end-date is not available.
    """
    preferred = [
        "window_end_date",
        "end_date",
        "date_end",
        "window_end",
        "windowEndDate",
        "win_end_date",
    ]
    for c in preferred:
        if c in df.columns:
            return c

    candidates = [
        "window_date",
        "window_start_date",
        "date",
        "Date",
        "timestamp",
        "time",
        "t",
        "datetime",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    # fallback: first column that parses as datetime at high rate
    for c in df.columns[:5]:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() > 0.8:
                return c
        except Exception:
            continue
    raise ValueError("Could not infer a date column in rolling CSV.")


def _read_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    if "Date" not in df.columns:
        # try lowercase
        for c in df.columns:
            if c.lower() == "date":
                df = df.rename(columns={c: "Date"})
                break
    if "Close" not in df.columns:
        # try common alternatives
        for c in df.columns:
            if c.lower() in {"close", "adj close", "adj_close", "price"}:
                df = df.rename(columns={c: "Close"})
                break
    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError(
            "Prices file must contain columns Date and Close (or recognizable aliases)."
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"]).set_index("Date")
    return df[["Close"]]


def _read_rolling(path: str, prices_index: Optional[pd.Index] = None) -> pd.DataFrame:
    """Read rolling CSV and set a datetime index.

    If 'window_end_date' exists, use it as the index (preferred).
    If 'win_end' exists and prices_index is provided, map the positional index
    in the prices series to a date and use it as end-of-window timestamp.
    Otherwise falls back to _infer_date_col().
    """
    df = pd.read_csv(path)

    if "window_end_date" in df.columns:
        date_col = "window_end_date"
    elif "win_end" in df.columns and prices_index is not None:
        win_end = pd.to_numeric(df["win_end"], errors="coerce")
        if win_end.isna().any():
            raise ValueError(
                "Column 'win_end' contains non-numeric values; cannot map to dates."
            )
        if len(prices_index) == 0:
            raise ValueError("prices_index is empty; cannot map 'win_end' to dates.")
        win_end = win_end.astype(int).clip(lower=0, upper=len(prices_index) - 1)

        # Map positional index to date (trading-day index)
        px_dates = pd.Index(prices_index)
        mapped = px_dates.to_numpy()[win_end.to_numpy()]
        df["window_end_date"] = pd.to_datetime(mapped, errors="coerce")
        date_col = "window_end_date"
    else:
        date_col = _infer_date_col(df)

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
    return df


def _build_target_series(
    rolling: pd.DataFrame, target: str, corr_window: int
) -> Tuple[pd.Series, Dict[str, str]]:
    """Return (y, metadata)."""
    meta: Dict[str, str] = {"target_spec": target}

    if target.startswith("col:"):
        col = target.split(":", 1)[1]
        if col not in rolling.columns:
            raise ValueError(f"Target column '{col}' not found in rolling CSV.")
        y = pd.to_numeric(rolling[col], errors="coerce").dropna()
        meta.update({"target_type": "column", "target_col": col})
        return y, meta

    if target.startswith("fisher_corr:") or target.startswith("corr:"):
        parts = target.split(":")
        if len(parts) != 3:
            raise ValueError(
                "Target fisher_corr requires format fisher_corr:<COL_A>:<COL_B>."
            )
        _, a, b = parts
        if a not in rolling.columns or b not in rolling.columns:
            raise ValueError(f"Columns for correlation not found: {a}, {b}")
        s1 = pd.to_numeric(rolling[a], errors="coerce")
        s2 = pd.to_numeric(rolling[b], errors="coerce")
        df = pd.concat([s1.rename("a"), s2.rename("b")], axis=1).dropna()
        corr = df["a"].rolling(corr_window).corr(df["b"])
        if target.startswith("fisher_corr:"):
            y = _safe_atanh(corr).dropna()
            meta.update(
                {
                    "target_type": "fisher_corr",
                    "col_a": a,
                    "col_b": b,
                    "corr_window": str(corr_window),
                }
            )
        else:
            y = corr.dropna()
            meta.update(
                {
                    "target_type": "corr",
                    "col_a": a,
                    "col_b": b,
                    "corr_window": str(corr_window),
                }
            )
        return y, meta

    raise ValueError(
        "Unsupported --target. Use col:<COL> or fisher_corr:<COLA>:<COLB>."
    )


def _stress_from_rv(prices: pd.DataFrame, rv_window: int) -> pd.Series:
    close = prices["Close"].astype(float)
    rets = np.log(close).diff()
    rv = rets.rolling(rv_window).std(ddof=0) * np.sqrt(252.0)
    rv.name = f"RV_{rv_window}"
    return rv.dropna()


def _stress_from_fred(series_id: str, timeout: int = 30) -> pd.Series:
    if requests is None:
        raise RuntimeError("requests is not available in this environment")
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    from io import StringIO

    df = pd.read_csv(StringIO(r.text))
    if "DATE" not in df.columns or series_id not in df.columns:
        raise ValueError("Unexpected FRED response format")
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    df = df.dropna(subset=["DATE", series_id]).set_index("DATE").sort_index()
    s = df[series_id]
    s.name = series_id
    return s


def _build_stress_series(
    prices: pd.DataFrame, stress_source: str, rv_window: int
) -> Tuple[pd.Series, Dict[str, str]]:
    meta: Dict[str, str] = {"stress_source": stress_source}

    if stress_source.lower() == "rv":
        s = _stress_from_rv(prices, rv_window)
        meta.update({"stress_type": "rv", "rv_window": str(rv_window)})
        return s, meta

    if stress_source.lower().startswith("fred:"):
        sid = stress_source.split(":", 1)[1]
        try:
            s = _stress_from_fred(sid)
            meta.update({"stress_type": "fred", "fred_series": sid})
            return s, meta
        except Exception as e:
            warnings.warn(f"FRED download failed ({e}). Falling back to RV.")
            s = _stress_from_rv(prices, rv_window)
            meta.update(
                {
                    "stress_type": "rv_fallback",
                    "rv_window": str(rv_window),
                    "fred_failed": sid,
                }
            )
            return s, meta

    raise ValueError("Unsupported --stress-source. Use rv or fred:<SERIES_ID>.")


def _align_y_stress(y: pd.Series, s: pd.Series) -> pd.DataFrame:
    """Aligns to y's index; forward-fills stress to target timeline."""
    y = y.sort_index()
    s = s.sort_index()
    # convert to daily-ish index alignment by merging on dates, then forward filling
    df = pd.DataFrame({"y": y})
    df = df.join(s.rename("stress"), how="left")
    df["stress"] = df["stress"].ffill()
    df = df.dropna(subset=["y", "stress"])
    return df


# -----------------------------
# (T1) Toda-Yamamoto causality
# -----------------------------


@dataclass
class TYResult:
    k_ar: int
    d_max: int
    wald_y_from_s: float
    p_y_from_s: float
    wald_s_from_y: float
    p_s_from_y: float


def _integration_order_adf(x: pd.Series, max_d: int = 2, alpha: float = 0.05) -> int:
    """Return d in {0,1,2} based on iterative ADF tests."""
    z = x.astype(float).dropna()
    for d in range(max_d + 1):
        if len(z) < 25:
            return min(d, max_d)
        try:
            p = adfuller(z.values, autolag="AIC")[1]
        except Exception:
            p = 1.0
        if p < alpha:
            return d
        z = z.diff().dropna()
    return max_d


def _select_var_lag(df: pd.DataFrame, maxlags: int) -> int:
    sel = VAR(df).select_order(maxlags)
    # prefer BIC, fall back to AIC
    lag = sel.selected_orders.get("bic", None)
    if lag is None:
        lag = sel.selected_orders.get("aic", None)
    if lag is None or lag <= 0:
        lag = 1
    return int(lag)


def _ty_wald(var_res, caused: str, causing: str, k: int) -> Tuple[float, float]:
    """Wald test on first k lags of `causing` in equation `caused`."""
    params = var_res.params
    covp = var_res.cov_params()

    coef = []
    idx = []
    for i in range(1, k + 1):
        pname = f"L{i}.{causing}"
        if pname not in params.index:
            raise ValueError(f"Missing parameter {pname} in VAR params")
        coef.append(float(params.loc[pname, caused]))
        idx.append((pname, caused))

    b = np.array(coef).reshape(-1, 1)
    S = covp.loc[idx, idx].values
    # numerical safety
    S = (S + S.T) / 2.0
    try:
        Sinv = np.linalg.pinv(S)
    except Exception:
        Sinv = np.linalg.pinv(S + 1e-10 * np.eye(S.shape[0]))

    wald = float((b.T @ Sinv @ b).ravel()[0])
    pval = float(1.0 - chi2.cdf(wald, df=k))
    return wald, pval


def toda_yamamoto_test(y: pd.Series, s: pd.Series, maxlags: int) -> TYResult:
    df = pd.concat([y.rename("y"), s.rename("s")], axis=1).dropna()
    if len(df) < 40:
        raise ValueError("Not enough observations for VAR/Toda-Yamamoto.")

    dmax = max(_integration_order_adf(df["y"]), _integration_order_adf(df["s"]))
    k = _select_var_lag(df, maxlags=maxlags)
    p = k + dmax

    var_res = VAR(df).fit(p)

    w_y_s, p_y_s = _ty_wald(var_res, caused="y", causing="s", k=k)
    w_s_y, p_s_y = _ty_wald(var_res, caused="s", causing="y", k=k)

    return TYResult(
        k_ar=k,
        d_max=dmax,
        wald_y_from_s=w_y_s,
        p_y_from_s=p_y_s,
        wald_s_from_y=w_s_y,
        p_s_from_y=p_s_y,
    )


# -----------------------------
# (T2) Quantile predictability (circular-shift permutation)
# -----------------------------


@dataclass
class QuantileGCResult:
    lags: int
    taus: List[float]
    delta_loss: Dict[str, float]
    p_values: Dict[str, float]
    B: int


def _make_lag_matrix(x: pd.Series, lags: int, prefix: str) -> pd.DataFrame:
    out = {}
    for i in range(1, lags + 1):
        out[f"{prefix}_L{i}"] = x.shift(i)
    return pd.DataFrame(out)


def _check_loss(u: np.ndarray, tau: float) -> float:
    # rho_tau(u) = u*(tau - I(u<0))
    return float(np.sum(u * (tau - (u < 0).astype(float))))


def _fit_quantreg(
    y: pd.Series, X: pd.DataFrame, tau: float
) -> Tuple[np.ndarray, float]:
    mod = sm.QuantReg(y.values, X.values)
    res = mod.fit(q=tau, max_iter=5000)
    beta = res.params
    u = y.values - X.values @ beta
    loss = _check_loss(u, tau)
    return beta, loss


def quantile_granger_test(
    y: pd.Series, s: pd.Series, lags: int, taus: List[float], B: int, seed: int = 12345
) -> QuantileGCResult:
    """Permutation test of incremental predictive content of stress lags.

    H0: stress lags add no improvement to quantile prediction beyond y lags.
    Statistic: Delta loss = Loss_restricted - Loss_unrestricted (>=0 indicates improvement).
    Null distribution built by circularly shifting stress series.
    """
    rng = np.random.default_rng(seed)

    # build design
    y0 = y.astype(float)
    s0 = s.astype(float)
    df = pd.DataFrame({"y": y0, "s": s0})
    Xy = _make_lag_matrix(df["y"], lags, "y")
    Xs = _make_lag_matrix(df["s"], lags, "s")

    # Keep contemporaneous 's' to enable circular-shift permutations.
    base = pd.concat([df[["y", "s"]], Xy, Xs], axis=1).dropna()
    y_t = base["y"]
    Xr = sm.add_constant(
        base[[c for c in base.columns if c.startswith("y_L")]], has_constant="add"
    )
    Xu = sm.add_constant(
        base[[c for c in base.columns if c.startswith("y_L") or c.startswith("s_L")]],
        has_constant="add",
    )

    # observed
    delta_obs: Dict[str, float] = {}
    pvals: Dict[str, float] = {}

    s_vals = base["s"].values

    # precompute y-lag columns and shift mapping for speed
    y_lag_cols = [c for c in base.columns if c.startswith("y_L")]
    s_lag_cols = [c for c in base.columns if c.startswith("s_L")]

    # For shifts, we rebuild only s lag part
    for tau in taus:
        _, loss_r = _fit_quantreg(y_t, Xr, tau)
        _, loss_u = _fit_quantreg(y_t, Xu, tau)
        d0 = max(0.0, loss_r - loss_u)
        delta_obs[f"{tau:.3f}"] = float(d0)

        deltas = []
        for _ in range(B):
            k = int(rng.integers(1, len(s_vals)))
            s_shift = _circular_shift(s_vals, k)
            # rebuild s lags
            tmp = base[["y"] + y_lag_cols].copy()
            tmp["s"] = s_shift
            Xs_shift = _make_lag_matrix(tmp["s"], lags, "s")
            tmp2 = pd.concat([tmp, Xs_shift], axis=1).dropna()
            y_b = tmp2["y"]
            Xr_b = sm.add_constant(tmp2[y_lag_cols], has_constant="add")
            Xu_b = sm.add_constant(tmp2[y_lag_cols + s_lag_cols], has_constant="add")

            try:
                _, lr = _fit_quantreg(y_b, Xr_b, tau)
                _, lu = _fit_quantreg(y_b, Xu_b, tau)
                deltas.append(max(0.0, lr - lu))
            except Exception:
                # if rare convergence issues, skip (conservative)
                continue

        deltas = np.array(deltas, dtype=float)
        # conservative p-value
        p = (
            float((1.0 + np.sum(deltas >= d0)) / (1.0 + len(deltas)))
            if len(deltas)
            else 1.0
        )
        pvals[f"{tau:.3f}"] = p

    return QuantileGCResult(
        lags=lags, taus=taus, delta_loss=delta_obs, p_values=pvals, B=B
    )


# -----------------------------
# (T3) Local Projections IRF
# -----------------------------


@dataclass
class LPResult:
    horizons: List[int]
    beta: List[float]
    se: List[float]
    ci_low: List[float]
    ci_high: List[float]
    shock_method: str
    shock_ar_lags: int


def _select_ar_lag_bic(x: pd.Series, max_ar: int) -> int:
    x = x.dropna().astype(float)
    best_p = 0
    best_bic = np.inf
    for p in range(0, max_ar + 1):
        if p == 0:
            X = np.ones((len(x), 1))
            y = x.values
        else:
            df = pd.DataFrame({"x": x})
            for i in range(1, p + 1):
                df[f"L{i}"] = x.shift(i)
            df = df.dropna()
            y = df["x"].values
            X = sm.add_constant(
                df[[f"L{i}" for i in range(1, p + 1)]].values, has_constant="add"
            )
        if len(y) < 30:
            continue
        res = sm.OLS(y, X).fit()
        bic = res.bic
        if bic < best_bic:
            best_bic = bic
            best_p = p
    return int(best_p)


def local_projections_irf(
    y: pd.Series,
    s: pd.Series,
    horizons: int,
    y_lags: int,
    s_lags: int,
    max_ar_shock: int,
    hac_maxlags_base: int,
) -> LPResult:
    y = y.astype(float)
    s = s.astype(float)

    # Build stress shock as AR residuals (default) for interpretability
    p_ar = _select_ar_lag_bic(s, max_ar=max_ar_shock)
    if p_ar == 0:
        shock = s.diff()
        method = "diff"
    else:
        df_ar = pd.DataFrame({"s": s})
        for i in range(1, p_ar + 1):
            df_ar[f"L{i}"] = s.shift(i)
        df_ar = df_ar.dropna()
        X = sm.add_constant(
            df_ar[[f"L{i}" for i in range(1, p_ar + 1)]], has_constant="add"
        )
        res = sm.OLS(df_ar["s"], X).fit()
        shock = (df_ar["s"] - res.fittedvalues).reindex(s.index)
        method = "ar_resid"

    # Regressions for each horizon
    H = int(horizons)
    betas, ses, lo, hi = [], [], [], []
    hs = list(range(0, H + 1))

    for h in hs:
        dep = y.shift(-h)
        df = pd.DataFrame({"dep": dep, "shock": shock, "y": y, "s": s})
        for i in range(1, y_lags + 1):
            df[f"y_L{i}"] = y.shift(i)
        for i in range(1, s_lags + 1):
            df[f"s_L{i}"] = s.shift(i)
        df = df.dropna()
        if len(df) < 35:
            betas.append(np.nan)
            ses.append(np.nan)
            lo.append(np.nan)
            hi.append(np.nan)
            continue

        Xcols = (
            ["shock"]
            + [f"y_L{i}" for i in range(1, y_lags + 1)]
            + [f"s_L{i}" for i in range(1, s_lags + 1)]
        )
        X = sm.add_constant(df[Xcols], has_constant="add")
        res = sm.OLS(df["dep"], X).fit(
            cov_type="HAC", cov_kwds={"maxlags": max(hac_maxlags_base, h + 1)}
        )
        b = float(res.params["shock"])
        se = float(res.bse["shock"])
        betas.append(b)
        ses.append(se)
        lo.append(b - 1.96 * se)
        hi.append(b + 1.96 * se)

    return LPResult(
        horizons=hs,
        beta=betas,
        se=ses,
        ci_low=lo,
        ci_high=hi,
        shock_method=method,
        shock_ar_lags=p_ar,
    )


# -----------------------------
# (T4) Distance correlation over lags
# -----------------------------


@dataclass
class DistCorrResult:
    max_lag: int
    dcor_by_lag: Dict[int, float]
    p_global: float
    p_by_lag: Dict[int, float]
    B: int


def _distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    n = x.shape[0]
    if n < 5:
        return np.nan

    a = np.abs(x - x.T)
    b = np.abs(y - y.T)
    A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()

    dcov2 = (A * B).mean()
    dvarx = (A * A).mean()
    dvary = (B * B).mean()
    if dvarx <= 0 or dvary <= 0:
        return 0.0
    return float(np.sqrt(max(dcov2, 0.0)) / np.sqrt(np.sqrt(dvarx * dvary)))


def distance_corr_lag_test(
    y: pd.Series, s: pd.Series, max_lag: int, B: int, seed: int = 12345
) -> DistCorrResult:
    rng = np.random.default_rng(seed)

    df0 = pd.DataFrame({"y": y.astype(float), "s": s.astype(float)}).dropna()
    yv = df0["y"].values
    sv = df0["s"].values

    lags = list(range(-max_lag, max_lag + 1))

    def compute_all(s_arr: np.ndarray) -> Dict[int, float]:
        out: Dict[int, float] = {}
        for L in lags:
            if L > 0:
                x = yv[L:]
                z = s_arr[:-L]
            elif L < 0:
                x = yv[:L]
                z = s_arr[-L:]
            else:
                x = yv
                z = s_arr
            out[L] = _distance_correlation(x, z)
        return out

    obs = compute_all(sv)
    obs_max = max(abs(v) for v in obs.values() if np.isfinite(v))

    max_null = []
    null_by_lag: Dict[int, List[float]] = {L: [] for L in lags}

    for _ in range(B):
        k = int(rng.integers(1, len(sv)))
        s_shift = _circular_shift(sv, k)
        vals = compute_all(s_shift)
        m = max(abs(v) for v in vals.values() if np.isfinite(v))
        max_null.append(m)
        for L in lags:
            null_by_lag[L].append(abs(vals[L]))

    max_null = np.array(max_null, dtype=float)
    p_global = float((1.0 + np.sum(max_null >= obs_max)) / (1.0 + len(max_null)))

    p_by_lag: Dict[int, float] = {}
    for L in lags:
        arr = np.array(null_by_lag[L], dtype=float)
        p_by_lag[L] = float((1.0 + np.sum(arr >= abs(obs[L]))) / (1.0 + len(arr)))

    return DistCorrResult(
        max_lag=max_lag,
        dcor_by_lag={int(k): float(v) for k, v in obs.items()},
        p_global=p_global,
        p_by_lag={int(k): float(v) for k, v in p_by_lag.items()},
        B=B,
    )


# -----------------------------
# (T5) Wavelet coherence
# -----------------------------


@dataclass
class WaveletCoherenceResult:
    wavelet: str
    n_scales: int
    scale_min: int
    scale_max: int
    B: int
    global_mean_coherence: float


def _smooth_2d(M: np.ndarray, win_t: int = 5, win_s: int = 3) -> np.ndarray:
    """Simple separable moving-average smoothing."""
    out = M.copy()
    if win_t > 1:
        k = np.ones(win_t) / win_t
        out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, out)
    if win_s > 1:
        k = np.ones(win_s) / win_s
        out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, out)
    return out


def wavelet_coherence(
    y: pd.Series,
    s: pd.Series,
    B: int,
    seed: int,
    wavelet: str = "cmor1.5-1.0",
    max_scales: int = 64,
) -> Tuple[WaveletCoherenceResult, Dict[str, np.ndarray]]:
    if pywt is None:
        raise RuntimeError("pywt (PyWavelets) is required for wavelet coherence")

    rng = np.random.default_rng(seed)

    df = pd.DataFrame({"y": y.astype(float), "s": s.astype(float)}).dropna()
    yv = _zscore(df["y"]).values
    sv = _zscore(df["s"]).values

    n = len(df)
    scale_max = int(min(max_scales, max(8, n // 2)))
    scales = np.arange(1, scale_max + 1)

    Wy, freqs = pywt.cwt(yv, scales, wavelet)
    Ws, _ = pywt.cwt(sv, scales, wavelet)

    Syy = _smooth_2d(np.abs(Wy) ** 2)
    Sss = _smooth_2d(np.abs(Ws) ** 2)
    Sys = _smooth_2d(Wy * np.conj(Ws))

    coh = (np.abs(Sys) ** 2) / (Syy * Sss + 1e-12)
    coh = np.clip(coh.real, 0.0, 1.0)

    # Surrogate significance: circular shifts of stress
    thr = np.zeros(len(scales), dtype=float)
    for i, sc in enumerate(scales):
        vals = []
        for _ in range(B):
            k = int(rng.integers(1, n))
            s_shift = _circular_shift(sv, k)
            Ws_b, _ = pywt.cwt(s_shift, scales, wavelet)
            Sss_b = _smooth_2d(np.abs(Ws_b) ** 2)
            Sys_b = _smooth_2d(Wy * np.conj(Ws_b))
            coh_b = (np.abs(Sys_b) ** 2) / (Syy * Sss_b + 1e-12)
            coh_b = np.clip(coh_b.real, 0.0, 1.0)
            vals.append(coh_b[i, :])
        vals = np.concatenate(vals)
        thr[i] = float(np.quantile(vals, 0.95))

    sig = coh > thr[:, None]

    res = WaveletCoherenceResult(
        wavelet=wavelet,
        n_scales=len(scales),
        scale_min=int(scales.min()),
        scale_max=int(scales.max()),
        B=B,
        global_mean_coherence=float(np.nanmean(coh)),
    )

    arrays = {
        "coh": coh,
        "sig": sig.astype(int),
        "scales": scales,
    }
    return res, arrays


# =============================================================================
# ENHANCED PLOTTING FUNCTIONS (Paper-quality graphics)
# =============================================================================
# These functions produce publication-ready figures that visually communicate
# the key empirical patterns: delayed effects, nonlinearity, state-dependence,
# and tail behavior.
# =============================================================================

# Define a consistent color palette for all plots
COLORS = {
    "stress": "#D62728",  # Red for stress
    "target": "#1F77B4",  # Blue for target
    "ci_fill": "#1F77B4",  # Blue for confidence intervals
    "sig_marker": "#2CA02C",  # Green for significance markers
    "zero_line": "#7F7F7F",  # Gray for reference lines
    "grid": "#E5E5E5",  # Light gray for grid
    "lowess": "#FF7F0E",  # Orange for smoothers
}


def plot_timeseries_enhanced(
    df: pd.DataFrame,
    outpath: str,
    target_label: str = "Target (Fisher-z)",
    stress_label: str = "Financial Stress",
) -> None:
    """PLOT 1: Time series overview (context plot)

    NOTE (matplotlib compatibility):
    Some matplotlib versions cannot handle datetime-like x directly in
    fill_between (it calls np.isfinite on x). We therefore convert the
    DatetimeIndex to matplotlib date numbers explicitly.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Standardize series for visualization
    stress_z = _zscore(df["stress"])
    target_z = _zscore(df["y"])

    # Convert x-axis to matplotlib date numbers for robust plotting
    x_dt = pd.to_datetime(df.index)
    x = mdates.date2num(x_dt.to_pydatetime())

    # Top panel: Stress series
    ax1 = axes[0]
    ax1.plot(x, stress_z, color=COLORS["stress"], linewidth=1.2, alpha=0.9)
    ax1.fill_between(
        x,
        0.0,
        stress_z.to_numpy(dtype=float),
        where=(stress_z.to_numpy(dtype=float) > 0.0),
        color=COLORS["stress"],
        alpha=0.15,
        interpolate=True,
    )
    ax1.axhline(0, color=COLORS["zero_line"], linewidth=0.8, linestyle="-")
    ax1.axhline(
        1.96, color=COLORS["zero_line"], linewidth=0.6, linestyle="--", alpha=0.5
    )
    ax1.axhline(
        -1.96, color=COLORS["zero_line"], linewidth=0.6, linestyle="--", alpha=0.5
    )
    ax1.set_ylabel(f"{stress_label} (standardized)", fontsize=10)
    ax1.set_title(
        "Stress–Target Dynamics: Evidence Against Contemporaneous Co-movement",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax1.grid(True, alpha=0.3, color=COLORS["grid"])
    ax1.set_xlim(float(x.min()), float(x.max()))

    # Add regime shading for high-stress periods (>1 std)
    high_stress = (stress_z > 1.0).to_numpy(dtype=bool)
    ax1.fill_between(
        x,
        ax1.get_ylim()[0],
        ax1.get_ylim()[1],
        where=high_stress,
        color=COLORS["stress"],
        alpha=0.08,
    )

    # Bottom panel: Target series
    ax2 = axes[1]
    ax2.plot(x, target_z, color=COLORS["target"], linewidth=1.2, alpha=0.9)
    ax2.fill_between(
        x,
        0.0,
        target_z.to_numpy(dtype=float),
        where=(target_z.to_numpy(dtype=float) < 0.0),
        color=COLORS["target"],
        alpha=0.15,
        interpolate=True,
    )
    ax2.axhline(0, color=COLORS["zero_line"], linewidth=0.8, linestyle="-")
    ax2.axhline(
        1.96, color=COLORS["zero_line"], linewidth=0.6, linestyle="--", alpha=0.5
    )
    ax2.axhline(
        -1.96, color=COLORS["zero_line"], linewidth=0.6, linestyle="--", alpha=0.5
    )
    ax2.set_ylabel(f"{target_label} (standardized)", fontsize=10)
    ax2.set_xlabel("Date", fontsize=10)
    ax2.grid(True, alpha=0.3, color=COLORS["grid"])

    # Propagate high-stress shading to bottom panel
    ax2.fill_between(
        x,
        ax2.get_ylim()[0],
        ax2.get_ylim()[1],
        where=high_stress,
        color=COLORS["stress"],
        alpha=0.08,
        label="High stress regime",
    )

    # Add correlation annotation
    corr = df["y"].corr(df["stress"])
    ax2.text(
        0.02,
        0.95,
        f"Contemporaneous ρ = {corr:.3f}",
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Format x-axis as dates
    ax2.xaxis_date()
    locator = mdates.AutoDateLocator()
    ax2.xaxis.set_major_locator(locator)
    ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_lp_enhanced(lp: LPResult, outpath: str) -> None:
    """
    PLOT 2: Local Projections impulse response plot

    Shows the dynamic response of the target to a stress shock across horizons:
    - Point estimates with confidence bands
    - Marked significant horizons (where CI excludes zero)
    - Clear visualization of delayed, persistent effects

    Purpose: Demonstrate that stress effects are delayed and persistent,
    not contemporaneous.
    """
    h = np.array(lp.horizons)
    beta = np.array(lp.beta, dtype=float)
    ci_lo = np.array(lp.ci_low, dtype=float)
    ci_hi = np.array(lp.ci_high, dtype=float)

    # Identify significant horizons (CI excludes zero)
    significant = (ci_lo > 0) | (ci_hi < 0)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Confidence band
    ax.fill_between(
        h, ci_lo, ci_hi, color=COLORS["ci_fill"], alpha=0.2, label="95% CI (HAC)"
    )

    # Zero reference line
    ax.axhline(0, color=COLORS["zero_line"], linewidth=1.2, linestyle="-", zorder=1)

    # Point estimates - non-significant
    ax.plot(
        h[~significant],
        beta[~significant],
        "o-",
        color=COLORS["target"],
        linewidth=1.5,
        markersize=6,
        alpha=0.6,
        zorder=2,
    )

    # Point estimates - significant (highlighted)
    if np.any(significant):
        ax.plot(
            h[significant],
            beta[significant],
            "o",
            color=COLORS["sig_marker"],
            markersize=10,
            markeredgecolor="white",
            markeredgewidth=1.5,
            label="Significant (p < 0.05)",
            zorder=3,
        )
        # Connect all points with a line
        ax.plot(
            h, beta, "-", color=COLORS["target"], linewidth=1.5, alpha=0.7, zorder=2
        )
    else:
        ax.plot(
            h, beta, "o-", color=COLORS["target"], linewidth=1.5, markersize=6, zorder=2
        )

    # Annotations
    ax.set_xlabel("Horizon (periods)", fontsize=11)
    ax.set_ylabel("Impulse Response β(h)", fontsize=11)
    ax.set_title(
        f"Local Projections: Target Response to Stress Shock\n"
        f"(Shock: {lp.shock_method}, AR({lp.shock_ar_lags}); HAC standard errors)",
        fontsize=12,
        fontweight="bold",
    )

    # Grid and styling
    ax.grid(True, alpha=0.3, color=COLORS["grid"])
    ax.set_xticks(h)
    ax.legend(loc="best", framealpha=0.9)

    # Add interpretation box
    n_sig = np.sum(significant)
    first_sig = h[significant][0] if n_sig > 0 else None
    if first_sig is not None and first_sig > 0:
        interp_text = (
            f"First significant response at h={first_sig}\n→ Delayed effect confirmed"
        )
    elif first_sig == 0 and n_sig > 0:
        interp_text = (
            f"Contemporaneous effect significant\n({n_sig} horizons significant)"
        )
    else:
        interp_text = "No significant response detected"

    ax.text(
        0.98,
        0.02,
        interp_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_quantile_gc_enhanced(qr: QuantileGCResult, outpath: str) -> None:
    """
    PLOT 3: Quantile causality / tail effects plot

    Shows p-values across quantiles to reveal where predictability is concentrated:
    - Lower quantiles: left tail of target distribution
    - Upper quantiles: right tail of target distribution

    Purpose: Demonstrate that stress effects are concentrated in specific parts
    of the distribution (typically lower tail), not in the conditional mean.
    """
    taus = np.array(qr.taus, dtype=float)
    pvals = np.array([qr.p_values[f"{t:.3f}"] for t in taus], dtype=float)
    delta_loss = np.array([qr.delta_loss[f"{t:.3f}"] for t in taus], dtype=float)

    # Significance threshold
    alpha = 0.05
    sig_mask = pvals < alpha

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[2, 1])

    # ===== TOP PANEL: P-values by quantile =====
    # Plot p-values
    ax1.plot(
        taus,
        pvals,
        "o-",
        color=COLORS["target"],
        linewidth=2,
        markersize=8,
        label="Permutation p-value",
    )

    # Significance threshold line
    ax1.axhline(
        alpha,
        color=COLORS["stress"],
        linewidth=1.5,
        linestyle="--",
        label=f"α = {alpha}",
    )

    # Highlight significant quantiles
    if np.any(sig_mask):
        ax1.scatter(
            taus[sig_mask],
            pvals[sig_mask],
            s=150,
            color=COLORS["sig_marker"],
            marker="*",
            zorder=5,
            label="Significant (p < 0.05)",
        )
        # Add vertical lines to emphasize
        for t, p in zip(taus[sig_mask], pvals[sig_mask]):
            ax1.axvline(t, color=COLORS["sig_marker"], alpha=0.2, linewidth=8)

    # Shading for distribution regions
    ax1.axvspan(0, 0.25, alpha=0.08, color="blue", label="Lower tail")
    ax1.axvspan(0.75, 1.0, alpha=0.08, color="red", label="Upper tail")

    ax1.set_ylabel("P-value", fontsize=11)
    ax1.set_title(
        f"Quantile Granger Causality: Stress → Target\n"
        f"(Lags = {qr.lags}, B = {qr.B} permutations)",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_xlim(0, 1)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3, color=COLORS["grid"])

    # ===== BOTTOM PANEL: Delta loss (predictive improvement) =====
    bars = ax2.bar(
        taus,
        delta_loss,
        width=0.06,
        color=COLORS["target"],
        alpha=0.7,
        edgecolor="white",
        linewidth=1,
    )

    # Color significant bars differently
    for i, (bar, is_sig) in enumerate(zip(bars, sig_mask)):
        if is_sig:
            bar.set_color(COLORS["sig_marker"])
            bar.set_alpha(0.9)

    ax2.set_xlabel("Quantile τ", fontsize=11)
    ax2.set_ylabel("ΔLoss (improvement)", fontsize=11)
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3, color=COLORS["grid"], axis="y")

    # Interpretation annotation
    sig_taus = taus[sig_mask] if np.any(sig_mask) else []
    if len(sig_taus) > 0:
        if np.mean(sig_taus) < 0.5:
            interp = "Predictability concentrated in LOWER tail\n→ Stress predicts negative outcomes"
        elif np.mean(sig_taus) > 0.5:
            interp = "Predictability concentrated in UPPER tail\n→ Stress predicts positive outcomes"
        else:
            interp = "Predictability across distribution"
    else:
        interp = "No significant quantile effects detected"

    ax1.text(
        0.02,
        0.98,
        interp,
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
    )

    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_distcorr_enhanced(dc: DistCorrResult, outpath: str) -> None:
    """
    PLOT 4: Distance correlation by lag plot

    Shows nonlinear dependence structure across leads and lags:
    - Negative lags: target leads stress (reverse causality check)
    - Positive lags: stress leads target (expected causal direction)

    Purpose: Demonstrate delayed and asymmetric nonlinear dependence,
    supporting the hypothesis that stress effects unfold over time.
    """
    lags = sorted(dc.dcor_by_lag.keys())
    vals = np.array([dc.dcor_by_lag[L] for L in lags], dtype=float)
    pvals = np.array([dc.p_by_lag[L] for L in lags], dtype=float)

    sig_mask = pvals < 0.05

    fig, ax = plt.subplots(figsize=(11, 5))

    # Background shading for lead/lag regions
    ax.axvspan(min(lags), -0.5, alpha=0.05, color="gray", label="_nolegend_")
    ax.axvspan(0.5, max(lags), alpha=0.05, color="blue", label="_nolegend_")

    # Main line plot
    ax.plot(
        lags,
        vals,
        "o-",
        color=COLORS["target"],
        linewidth=1.5,
        markersize=6,
        alpha=0.7,
        label="Distance correlation",
    )

    # Zero lag reference
    ax.axvline(0, color=COLORS["zero_line"], linewidth=1.2, linestyle="-", alpha=0.8)

    # Highlight significant lags
    sig_lags = [L for L, s in zip(lags, sig_mask) if s]
    sig_vals = [vals[i] for i, s in enumerate(sig_mask) if s]
    if len(sig_lags) > 0:
        ax.scatter(
            sig_lags,
            sig_vals,
            s=120,
            color=COLORS["sig_marker"],
            marker="*",
            zorder=5,
            label="p < 0.05",
        )

    # Annotations for lag interpretation
    ax.text(
        min(lags) + 0.5,
        ax.get_ylim()[1] * 0.95,
        "Target leads\n(reverse causality)",
        fontsize=9,
        ha="left",
        va="top",
        style="italic",
        alpha=0.7,
    )
    ax.text(
        max(lags) - 0.5,
        ax.get_ylim()[1] * 0.95,
        "Stress leads\n(expected direction)",
        fontsize=9,
        ha="right",
        va="top",
        style="italic",
        alpha=0.7,
    )

    ax.set_xlabel("Lag L (positive = stress leads target)", fontsize=11)
    ax.set_ylabel("Distance Correlation", fontsize=11)
    ax.set_title(
        f"Nonlinear Dependence Structure: Distance Correlation by Lag\n"
        f"(Global max-test p = {dc.p_global:.4f}, B = {dc.B})",
        fontsize=12,
        fontweight="bold",
    )

    ax.grid(True, alpha=0.3, color=COLORS["grid"])
    ax.legend(loc="upper right")
    ax.set_xlim(min(lags) - 0.5, max(lags) + 0.5)

    # Interpretation box
    pos_sig = sum(1 for L in sig_lags if L > 0)
    neg_sig = sum(1 for L in sig_lags if L < 0)
    if pos_sig > neg_sig and pos_sig > 0:
        interp = f"Significant dependence at positive lags\n→ Stress leads target (causal pattern)"
    elif neg_sig > pos_sig and neg_sig > 0:
        interp = (
            f"Significant dependence at negative lags\n→ Target leads stress (feedback)"
        )
    elif dc.p_global < 0.05:
        interp = f"Global nonlinear dependence detected\n(p = {dc.p_global:.4f})"
    else:
        interp = "No significant nonlinear dependence"

    ax.text(
        0.02,
        0.02,
        interp,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
    )

    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_stress_target_scatter(
    df: pd.DataFrame, outpath: str, n_quantiles: int = 3
) -> None:
    """
    PLOT 5: Stress–target state diagram

    Scatter plot of target vs. stress with:
    - Points colored by stress quantile (regime)
    - LOWESS smoother to reveal nonlinearity

    Purpose: Illustrate nonlinearity and regime dependence without
    imposing a parametric model structure.
    """
    stress = df["stress"].values
    target = df["y"].values

    # Compute stress quantiles for coloring
    stress_quantiles = pd.qcut(stress, q=n_quantiles, labels=False, duplicates="drop")
    n_actual = len(np.unique(stress_quantiles))

    # Color mapping
    cmap = plt.cm.RdYlGn_r  # Red = high stress, Green = low stress
    colors = [
        cmap(q / (n_actual - 1)) if n_actual > 1 else cmap(0.5)
        for q in stress_quantiles
    ]

    fig, ax = plt.subplots(figsize=(9, 7))

    # Scatter plot
    scatter = ax.scatter(
        stress,
        target,
        c=stress_quantiles,
        cmap=cmap,
        alpha=0.6,
        s=30,
        edgecolors="white",
        linewidth=0.3,
    )

    # LOWESS smoother
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess

        smoothed = lowess(target, stress, frac=0.3, return_sorted=True)
        ax.plot(
            smoothed[:, 0],
            smoothed[:, 1],
            color=COLORS["lowess"],
            linewidth=3,
            label="LOWESS smoother",
            zorder=5,
        )
    except Exception:
        # Fallback: simple moving average binned approach
        n_bins = 20
        bins = np.linspace(stress.min(), stress.max(), n_bins + 1)
        bin_centers = []
        bin_means = []
        for i in range(n_bins):
            mask = (stress >= bins[i]) & (stress < bins[i + 1])
            if np.sum(mask) > 5:
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
                bin_means.append(np.mean(target[mask]))
        if len(bin_centers) > 3:
            ax.plot(
                bin_centers,
                bin_means,
                color=COLORS["lowess"],
                linewidth=3,
                label="Binned mean",
                zorder=5,
            )

    # Linear fit for comparison
    z = np.polyfit(stress, target, 1)
    p = np.poly1d(z)
    x_line = np.linspace(stress.min(), stress.max(), 100)
    ax.plot(
        x_line,
        p(x_line),
        "--",
        color=COLORS["zero_line"],
        linewidth=1.5,
        alpha=0.7,
        label="Linear fit",
    )

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, label="Stress regime")
    if n_actual == 3:
        cbar.set_ticks([0, 1, 2])
        cbar.set_ticklabels(["Low", "Medium", "High"])
    elif n_actual == 2:
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["Low", "High"])

    ax.set_xlabel("Financial Stress (level)", fontsize=11)
    ax.set_ylabel("Target (level or Fisher-z)", fontsize=11)
    ax.set_title(
        "Stress–Target Relationship: Nonlinearity and State-Dependence",
        fontsize=12,
        fontweight="bold",
    )

    ax.grid(True, alpha=0.3, color=COLORS["grid"])
    ax.legend(loc="best")

    # Add correlation by regime
    corr_text = "Correlation by regime:\n"
    for q in range(n_actual):
        mask = stress_quantiles == q
        if np.sum(mask) > 10:
            r = np.corrcoef(stress[mask], target[mask])[0, 1]
            regime_label = (
                ["Low", "Medium", "High"][q] if n_actual == 3 else ["Low", "High"][q]
            )
            corr_text += f"  {regime_label}: ρ = {r:.3f}\n"

    ax.text(
        0.02,
        0.98,
        corr_text.strip(),
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        family="monospace",
    )

    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_wavelet_enhanced(
    arrays: Dict[str, np.ndarray], index: pd.Index, outpath: str
) -> None:
    """
    PLOT 6 (bonus): Enhanced wavelet coherence plot

    Time-frequency representation of coherence with significance contours.

    Purpose: Reveal time-varying and scale-dependent dependence patterns.
    """
    coh = arrays["coh"]
    sig = arrays["sig"].astype(bool)
    scales = arrays["scales"]

    # Use integer time axis positions
    t = np.arange(len(index))

    fig, ax = plt.subplots(figsize=(12, 5))

    # Coherence heatmap
    im = ax.imshow(
        coh,
        aspect="auto",
        origin="lower",
        extent=[t.min(), t.max(), scales.min(), scales.max()],
        cmap="viridis",
        vmin=0,
        vmax=1,
    )

    # Significance contours
    ax.contour(
        t,
        scales,
        sig.astype(int),
        levels=[0.5],
        colors="white",
        linewidths=1.5,
        linestyles="-",
    )

    # Cone of influence (approximate)
    coi_scale = np.minimum(t, len(t) - t) / 2
    coi_scale = np.clip(coi_scale, scales.min(), scales.max())
    ax.fill_between(t, scales.min(), coi_scale, color="white", alpha=0.3)

    ax.set_ylabel("Scale (proxy for period)", fontsize=11)
    ax.set_xlabel("Time index", fontsize=11)
    ax.set_title(
        "Wavelet Coherence: Time-Frequency Dependence Structure\n"
        "(White contour = 95% significance vs. circular-shift surrogates)",
        fontsize=12,
        fontweight="bold",
    )

    cbar = plt.colorbar(im, ax=ax, label="Coherence", shrink=0.8)

    # Add date labels if possible
    n_labels = 6
    tick_positions = np.linspace(0, len(index) - 1, n_labels).astype(int)
    try:
        tick_labels = [index[i].strftime("%Y-%m") for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    except Exception:
        pass

    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)




# -----------------------------
# Orchestration / main
# -----------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run 5 paper-grade stress/parameter dependence tests."
    )

    p.add_argument(
        "--rolling", required=True, help="Rolling CSV (e.g., rigid_windows_results.csv)"
    )
    p.add_argument("--prices", required=True, help="Prices file (e.g., PRICE_DEF.txt)")
    p.add_argument(
        "--target",
        required=True,
        help="Target: col:<COL> or fisher_corr:<COLA>:<COLB> (or corr:<COLA>:<COLB>)",
    )
    p.add_argument("--out", required=True, help="Output directory")

    p.add_argument("--stress-source", default="rv", help="rv or fred:<SERIES_ID>")
    p.add_argument("--rv-window", type=int, default=22, help="RV window (trading days)")
    p.add_argument(
        "--corr-window",
        type=int,
        default=10,
        help="Rolling corr window for fisher_corr",
    )

    # test parameters
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument(
        "--quick",
        action="store_true",
        help="Reduce resampling sizes for a fast sanity run",
    )

    p.add_argument("--ty-maxlags", type=int, default=10)

    p.add_argument("--qgc-lags", type=int, default=4)
    p.add_argument("--qgc-B", type=int, default=125)
    p.add_argument("--qgc-taus", default="0.05,0.1,0.25,0.5,0.75,0.9,0.95")

    p.add_argument("--lp-horizons", type=int, default=12)
    p.add_argument("--lp-y-lags", type=int, default=4)
    p.add_argument("--lp-s-lags", type=int, default=4)
    p.add_argument("--lp-max-ar-shock", type=int, default=8)
    p.add_argument("--lp-hac-base", type=int, default=4)

    p.add_argument("--dc-maxlag", type=int, default=12)
    p.add_argument("--dc-B", type=int, default=500)

    p.add_argument("--wv-B", type=int, default=50)
    p.add_argument("--wv-max-scales", type=int, default=64)

    return p.parse_args(argv)





def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # quick mode adjustments (sanity run that should finish fast)
    # NOTE: For paper-grade inference increase B and consider more quantiles.
    if args.quick:
        # quantile regressions are the heaviest piece; keep them small in quick mode
        args.qgc_B = min(args.qgc_B, 25)
        if args.qgc_taus == "0.05,0.1,0.25,0.5,0.75,0.9,0.95":
            args.qgc_taus = "0.1,0.5,0.9"

        args.dc_B = min(args.dc_B, 80)
        args.wv_B = min(args.wv_B, 10)
        args.lp_horizons = min(args.lp_horizons, 6)
    out_dir = args.out
    _ensure_dir(out_dir)
    plots_dir = os.path.join(out_dir, "plots")
    _ensure_dir(plots_dir)

    # Load data
    prices = _read_prices(args.prices)
    rolling = _read_rolling(args.rolling, prices_index=prices.index)
    y, meta_y = _build_target_series(rolling, args.target, args.corr_window)
    stress, meta_s = _build_stress_series(prices, args.stress_source, args.rv_window)

    aligned = _align_y_stress(y, stress)

    # Save aligned series
    aligned.to_csv(os.path.join(out_dir, "aligned_series.csv"), index=True)

    # =========================================================================
    # ENHANCED PLOT 1: Time series overview (context plot)
    # =========================================================================
    target_label = meta_y.get("target_type", "Target")
    if target_label == "fisher_corr":
        target_label = (
            f"Fisher-z corr({meta_y.get('col_a', '')}, {meta_y.get('col_b', '')})"
        )
    stress_label = meta_s.get("stress_type", "Stress").upper()

    plot_timeseries_enhanced(
        aligned,
        os.path.join(plots_dir, "timeseries_overview.png"),
        target_label=target_label,
        stress_label=stress_label,
    )

    # Run tests
    results: Dict[str, object] = {
        "meta": {
            **meta_y,
            **meta_s,
            "n_obs": int(len(aligned)),
            "date_start": str(aligned.index.min()),
            "date_end": str(aligned.index.max()),
            "seed": int(args.seed),
        }
    }

    # (T1) Toda-Yamamoto
    try:
        ty = toda_yamamoto_test(
            aligned["y"], aligned["stress"], maxlags=args.ty_maxlags
        )
        results["toda_yamamoto"] = asdict(ty)
    except Exception as e:
        results["toda_yamamoto"] = {"error": str(e)}

    # (T2) Quantile Granger-style predictability (permutation via circular shifts)
    taus = [float(x) for x in args.qgc_taus.split(",") if x.strip()]
    try:
        qgc = quantile_granger_test(
            aligned["y"],
            aligned["stress"],
            lags=args.qgc_lags,
            taus=taus,
            B=args.qgc_B,
            seed=args.seed,
        )
        results["quantile_granger"] = asdict(qgc)
        # =====================================================================
        # ENHANCED PLOT 3: Quantile causality / tail effects
        # =====================================================================
        plot_quantile_gc_enhanced(qgc, os.path.join(plots_dir, "quantile_granger.png"))
    except Exception as e:
        results["quantile_granger"] = {"error": str(e)}

    # (T3) Local projections
    try:
        lp = local_projections_irf(
            aligned["y"],
            aligned["stress"],
            horizons=args.lp_horizons,
            y_lags=args.lp_y_lags,
            s_lags=args.lp_s_lags,
            max_ar_shock=args.lp_max_ar_shock,
            hac_maxlags_base=args.lp_hac_base,
        )
        results["local_projections"] = asdict(lp)
        # =====================================================================
        # ENHANCED PLOT 2: Local Projections IRF
        # =====================================================================
        plot_lp_enhanced(lp, os.path.join(plots_dir, "local_projections_irf.png"))
    except Exception as e:
        results["local_projections"] = {"error": str(e)}

    # (T4) Distance correlation over lags
    try:
        dc = distance_corr_lag_test(
            aligned["y"],
            aligned["stress"],
            max_lag=args.dc_maxlag,
            B=args.dc_B,
            seed=args.seed,
        )
        results["distance_correlation"] = asdict(dc)
        # =====================================================================
        # ENHANCED PLOT 4: Distance correlation by lag
        # =====================================================================
        plot_distcorr_enhanced(
            dc, os.path.join(plots_dir, "distance_correlation_lags.png")
        )
    except Exception as e:
        results["distance_correlation"] = {"error": str(e)}

    # (T5) Wavelet coherence
    if pywt is None:
        results["wavelet_coherence"] = {"error": "pywt not installed"}
    else:
        try:
            wv_res, wv_arrays = wavelet_coherence(
                aligned["y"],
                aligned["stress"],
                B=args.wv_B,
                seed=args.seed,
                max_scales=args.wv_max_scales,
            )
            results["wavelet_coherence"] = asdict(wv_res)
            # Save arrays in npz to avoid huge JSON
            np.savez_compressed(
                os.path.join(out_dir, "wavelet_arrays.npz"), **wv_arrays
            )
            # =================================================================
            # ENHANCED PLOT 6: Wavelet coherence
            # =================================================================
            plot_wavelet_enhanced(
                wv_arrays,
                aligned.index,
                os.path.join(plots_dir, "wavelet_coherence.png"),
            )
        except Exception as e:
            results["wavelet_coherence"] = {"error": str(e)}

    # =========================================================================
    # ENHANCED PLOT 5: Stress-target state diagram (scatter)
    # =========================================================================
    try:
        plot_stress_target_scatter(
            aligned, os.path.join(plots_dir, "stress_target_scatter.png"), n_quantiles=3
        )
    except Exception as e:
        warnings.warn(f"Could not generate stress-target scatter plot: {e}")

    # Save results.json
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # MASTER_SUMMARY.csv
    summary = {}
    summary.update(results["meta"])

    # pull key p-values
    ty_res = results.get("toda_yamamoto", {})
    if isinstance(ty_res, dict) and "p_y_from_s" in ty_res:
        summary["TY_p_y_from_stress"] = ty_res["p_y_from_s"]
        summary["TY_p_stress_from_y"] = ty_res["p_s_from_y"]
        summary["TY_k_ar"] = ty_res["k_ar"]
        summary["TY_d_max"] = ty_res["d_max"]
    else:
        summary["TY_error"] = (
            ty_res.get("error", "") if isinstance(ty_res, dict) else ""
        )

    qgc_res = results.get("quantile_granger", {})
    if isinstance(qgc_res, dict) and "p_values" in qgc_res:
        # store min p across taus (exploratory)
        pvals = list(qgc_res["p_values"].values())
        summary["QGC_min_p"] = float(np.min(pvals)) if len(pvals) else np.nan
    else:
        summary["QGC_error"] = (
            qgc_res.get("error", "") if isinstance(qgc_res, dict) else ""
        )

    dc_res = results.get("distance_correlation", {})
    if isinstance(dc_res, dict) and "p_global" in dc_res:
        summary["DC_global_p"] = dc_res["p_global"]
    else:
        summary["DC_error"] = (
            dc_res.get("error", "") if isinstance(dc_res, dict) else ""
        )

    wv_res = results.get("wavelet_coherence", {})
    if isinstance(wv_res, dict) and "global_mean_coherence" in wv_res:
        summary["WV_global_mean_coh"] = wv_res["global_mean_coherence"]
    else:
        summary["WV_error"] = (
            wv_res.get("error", "") if isinstance(wv_res, dict) else ""
        )

    lp_res = results.get("local_projections", {})
    if isinstance(lp_res, dict) and "beta" in lp_res:
        # store horizon-0 effect and max abs
        beta = np.array(lp_res["beta"], dtype=float)
        summary["LP_beta_h0"] = float(beta[0]) if len(beta) else np.nan
        summary["LP_max_abs_beta"] = (
            float(np.nanmax(np.abs(beta))) if len(beta) else np.nan
        )
    else:
        summary["LP_error"] = (
            lp_res.get("error", "") if isinstance(lp_res, dict) else ""
        )

    pd.DataFrame([summary]).to_csv(
        os.path.join(out_dir, "MASTER_SUMMARY.csv"), index=False
    )

    print(f"Done. Outputs written to: {out_dir}")
    print(f"Enhanced plots saved in: {plots_dir}")

    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    sys.exit(main())
