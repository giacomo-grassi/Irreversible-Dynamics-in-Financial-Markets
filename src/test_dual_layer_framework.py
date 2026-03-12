#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_dual_layer_framework.py
=============================
Statistical tests for the dual-layer (endogenous + exogenous) stress 
decomposition framework described in the stress_decomposition_report.

Tests implemented:
  (F1) Joint regression: β = α₀ + α₁·D_e + α₂·RV + α₃·HMFSI + ε
  (F2) Correlation instability: ρ(β,D_e) = γ₀ + γ₁·RV + γ₂·HMFSI + η  
  (F3) Predictive regression: HMFSI_{t+h} = ψ·ρ(β,D_e)_t + controls
  (F4) Interaction model: β = f(D_e) + δ·D_e·HMFSI + ε

Inputs:
  - aligned_series from RV analysis (with y=beta_acf, stress=RV)
  - aligned_series from HMFSI analysis (with y=beta_acf, stress=HMFSI)
  - aligned_series from HMFSI corr analysis (with y=fisher_corr, stress=HMFSI)

Usage:
  python test_dual_layer_framework.py \
    --rv-beta results_beta_125/aligned_series.csv \
    --rv-de results_de_acf_AR_125/aligned_series.csv \
    --rv-corr results_corr_250_acf/aligned_series.csv \
    --hmfsi-beta results_hmfsi_beta_acf/aligned_series.csv \
    --hmfsi-de results_hmfsi_DE_acf/aligned_series.csv \
    --hmfsi-corr DJI_HMFSI_CORR_BETA_DE/aligned_series.csv \
    --out dual_layer_results
"""

import argparse
import json
import os
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant
    HAS_SM = True
except ImportError:
    HAS_SM = False
    warnings.warn("statsmodels not available; some tests will be skipped.")


def load_aligned(path, y_name="y", stress_name="stress"):
    """Load an aligned_series.csv and return (y, stress) as Series."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df[y_name], df[stress_name]


def newey_west_ols(y, X, max_lags=8):
    """OLS with Newey-West HAC standard errors."""
    X_c = add_constant(X)
    model = OLS(y, X_c, missing="drop").fit(
        cov_type="HAC", cov_kwds={"maxlags": max_lags}
    )
    return model


# =====================================================================
# F1: Joint regression β = α₀ + α₁·D_e + α₂·RV + α₃·HMFSI + ε
# =====================================================================
def test_joint_regression(rv_beta_path, rv_de_path, hmfsi_beta_path):
    """
    Test whether RV and HMFSI have independent predictive power for β,
    controlling for D_e.
    """
    beta_rv, rv_stress = load_aligned(rv_beta_path)
    de_rv, _ = load_aligned(rv_de_path)
    _, hmfsi_stress = load_aligned(hmfsi_beta_path)

    # Align all on common index
    df = pd.DataFrame({
        "beta": beta_rv,
        "De": de_rv,
        "RV": rv_stress,
        "HMFSI": hmfsi_stress
    }).dropna()

    print(f"\n{'='*60}")
    print("F1: Joint regression β = α₀ + α₁·Dₑ + α₂·RV + α₃·HMFSI + ε")
    print(f"{'='*60}")
    print(f"N = {len(df)}")

    if not HAS_SM:
        print("[SKIP] statsmodels not available")
        return None

    model = newey_west_ols(df["beta"], df[["De", "RV", "HMFSI"]])
    print(model.summary2().tables[1].to_string())
    print(f"\nR² = {model.rsquared:.4f}")
    print(f"α₂ (RV):   {model.params['RV']:.4f}, p={model.pvalues['RV']:.4f}")
    print(f"α₃ (HMFSI): {model.params['HMFSI']:.4f}, p={model.pvalues['HMFSI']:.4f}")

    result = OrderedDict({
        "test": "F1_joint_regression",
        "n": len(df),
        "R2": round(model.rsquared, 4),
        "coefficients": {
            "const": {"value": round(model.params["const"], 4), "p": round(model.pvalues["const"], 4)},
            "De": {"value": round(model.params["De"], 4), "p": round(model.pvalues["De"], 4)},
            "RV": {"value": round(model.params["RV"], 4), "p": round(model.pvalues["RV"], 4)},
            "HMFSI": {"value": round(model.params["HMFSI"], 4), "p": round(model.pvalues["HMFSI"], 4)},
        },
        "interpretation": (
            "RV significant → endogenous channel confirmed. "
            "HMFSI not significant → no direct exogenous channel for β."
            if model.pvalues["HMFSI"] > 0.05 else
            "Both RV and HMFSI significant → dual-channel for β."
        )
    })
    return result


# =====================================================================
# F2: Correlation instability ρ(β,Dₑ) = γ₀ + γ₁·RV + γ₂·HMFSI + η
# =====================================================================
def test_correlation_instability(rv_corr_path, hmfsi_corr_path):
    """
    Test whether RV and HMFSI independently predict the β–D_e correlation.
    """
    corr_rv, rv_stress = load_aligned(rv_corr_path)
    _, hmfsi_stress = load_aligned(hmfsi_corr_path)

    df = pd.DataFrame({
        "corr_beta_de": corr_rv,
        "RV": rv_stress,
        "HMFSI": hmfsi_stress
    }).dropna()

    print(f"\n{'='*60}")
    print("F2: Correlation instability ρ(β,Dₑ) = γ₀ + γ₁·RV + γ₂·HMFSI + η")
    print(f"{'='*60}")
    print(f"N = {len(df)}")

    if not HAS_SM:
        print("[SKIP] statsmodels not available")
        return None

    model = newey_west_ols(df["corr_beta_de"], df[["RV", "HMFSI"]])
    print(model.summary2().tables[1].to_string())
    print(f"\nR² = {model.rsquared:.4f}")
    print(f"γ₁ (RV):   {model.params['RV']:.4f}, p={model.pvalues['RV']:.4f}")
    print(f"γ₂ (HMFSI): {model.params['HMFSI']:.4f}, p={model.pvalues['HMFSI']:.4f}")

    result = OrderedDict({
        "test": "F2_correlation_instability",
        "n": len(df),
        "R2": round(model.rsquared, 4),
        "coefficients": {
            "const": {"value": round(model.params["const"], 4), "p": round(model.pvalues["const"], 4)},
            "RV": {"value": round(model.params["RV"], 4), "p": round(model.pvalues["RV"], 4)},
            "HMFSI": {"value": round(model.params["HMFSI"], 4), "p": round(model.pvalues["HMFSI"], 4)},
        }
    })
    return result


# =====================================================================
# F3: Predictive regression HMFSI_{t+h} = ψ·ρ(β,Dₑ)_t + controls
# =====================================================================
def test_predictive_regression(hmfsi_corr_path, horizons=None):
    """
    Test whether ρ(β,Dₑ) predicts future HMFSI at multiple horizons.
    This is the key 'reverse causality' test.
    """
    if horizons is None:
        horizons = [1, 2, 3, 4, 5, 6]

    corr_target, hmfsi_stress = load_aligned(hmfsi_corr_path)

    print(f"\n{'='*60}")
    print("F3: Predictive regression HMFSI_{t+h} = ψ·ρ(β,Dₑ)_t + controls")
    print(f"{'='*60}")

    if not HAS_SM:
        print("[SKIP] statsmodels not available")
        return None

    results = []
    for h in horizons:
        # Forward-shift HMFSI by h steps
        hmfsi_future = hmfsi_stress.shift(-h)
        df = pd.DataFrame({
            "hmfsi_future": hmfsi_future,
            "corr_now": corr_target,
            "hmfsi_now": hmfsi_stress  # control for current level
        }).dropna()

        if len(df) < 20:
            continue

        model = newey_west_ols(
            df["hmfsi_future"],
            df[["corr_now", "hmfsi_now"]],
            max_lags=min(h + 2, 12)
        )

        sig = "***" if model.pvalues["corr_now"] < 0.05 else ""
        print(f"  h={h}: ψ={model.params['corr_now']:.4f}, "
              f"SE={model.bse['corr_now']:.4f}, "
              f"p={model.pvalues['corr_now']:.4f} {sig}, "
              f"n={len(df)}")

        results.append(OrderedDict({
            "horizon": h,
            "psi": round(model.params["corr_now"], 4),
            "se": round(model.bse["corr_now"], 4),
            "p_value": round(model.pvalues["corr_now"], 4),
            "significant_005": model.pvalues["corr_now"] < 0.05,
            "n": len(df),
            "R2": round(model.rsquared, 4)
        }))

    return OrderedDict({
        "test": "F3_predictive_regression",
        "description": "HMFSI_{t+h} regressed on ρ(β,Dₑ)_t, controlling for HMFSI_t",
        "results_by_horizon": results,
        "any_significant": any(r["significant_005"] for r in results)
    })


# =====================================================================
# F4: Interaction model β = f(Dₑ) + δ·Dₑ·HMFSI + ε
# =====================================================================
def test_interaction_model(rv_beta_path, rv_de_path, hmfsi_beta_path):
    """
    Test whether HMFSI modifies the β–Dₑ mapping slope.
    """
    beta_rv, rv_stress = load_aligned(rv_beta_path)
    de_rv, _ = load_aligned(rv_de_path)
    _, hmfsi_stress = load_aligned(hmfsi_beta_path)

    df = pd.DataFrame({
        "beta": beta_rv,
        "De": de_rv,
        "HMFSI": hmfsi_stress
    }).dropna()
    df["De_x_HMFSI"] = df["De"] * df["HMFSI"]

    print(f"\n{'='*60}")
    print("F4: Interaction model β = α₀ + α₁·Dₑ + α₂·HMFSI + δ·Dₑ×HMFSI + ε")
    print(f"{'='*60}")
    print(f"N = {len(df)}")

    if not HAS_SM:
        print("[SKIP] statsmodels not available")
        return None

    model = newey_west_ols(df["beta"], df[["De", "HMFSI", "De_x_HMFSI"]])
    print(model.summary2().tables[1].to_string())
    print(f"\nδ (interaction): {model.params['De_x_HMFSI']:.4f}, "
          f"p={model.pvalues['De_x_HMFSI']:.4f}")

    result = OrderedDict({
        "test": "F4_interaction_model",
        "n": len(df),
        "R2": round(model.rsquared, 4),
        "coefficients": {
            "const": {"value": round(model.params["const"], 4), "p": round(model.pvalues["const"], 4)},
            "De": {"value": round(model.params["De"], 4), "p": round(model.pvalues["De"], 4)},
            "HMFSI": {"value": round(model.params["HMFSI"], 4), "p": round(model.pvalues["HMFSI"], 4)},
            "De_x_HMFSI": {"value": round(model.params["De_x_HMFSI"], 4), "p": round(model.pvalues["De_x_HMFSI"], 4)},
        },
        "interpretation": (
            "Significant interaction → HMFSI modifies the β–Dₑ mapping slope."
            if model.pvalues["De_x_HMFSI"] < 0.05 else
            "No significant interaction → static mapping is robust to exogenous stress."
        )
    })
    return result


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Test dual-layer endogenous/exogenous framework"
    )
    parser.add_argument("--rv-beta", required=True)
    parser.add_argument("--rv-de", required=True)
    parser.add_argument("--rv-corr", required=True)
    parser.add_argument("--hmfsi-beta", required=True)
    parser.add_argument("--hmfsi-de", required=True)
    parser.add_argument("--hmfsi-corr", required=True)
    parser.add_argument("--out", default="dual_layer_results")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    all_results = OrderedDict()

    # F1
    r = test_joint_regression(args.rv_beta, args.rv_de, args.hmfsi_beta)
    if r: all_results["F1"] = r

    # F2
    r = test_correlation_instability(args.rv_corr, args.hmfsi_corr)
    if r: all_results["F2"] = r

    # F3
    r = test_predictive_regression(args.hmfsi_corr)
    if r: all_results["F3"] = r

    # F4
    r = test_interaction_model(args.rv_beta, args.rv_de, args.hmfsi_beta)
    if r: all_results["F4"] = r

    # Save
    out_path = os.path.join(args.out, "dual_layer_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n→ Results saved to {out_path}")


if __name__ == "__main__":
    main()
