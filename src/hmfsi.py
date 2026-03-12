#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hmfsi.py — Historical Macro-Financial Stress Index (HMFSI)
===========================================================

PURPOSE
-------
Construct an *exogenous* financial stress proxy for the entire DJI sample
(1896–2025) that is NOT derived from equity prices, thereby breaking the
endogeneity problem identified in the Baldovin–Stella / stress analysis.

DESIGN PHILOSOPHY
-----------------
Modern financial stress indices (OFR FSI, STLFSI4, KCFSI, NFCI) combine
dozens of variables but start only in the 1970s–1990s.  Our index must
reach back to the 1890s.  We achieve this by:

  1.  Selecting components from non-equity asset classes (bond market,
      monetary conditions, price stability) whose historical data exist
      from at least 1871 (Shiller dataset) or 1919 (Moody's ratings).

  2.  Using an *expanding-window* standardisation so that the z-score at
      each date uses only information available up to that date (no
      look-ahead bias).

  3.  Combining components with equal weights within a *regime-aware*
      splicing framework that adjusts the number of components as data
      become available.

COMPONENTS (all monthly frequency)
-----------------------------------
(C1) Interest-Rate Volatility  (IRV)
     σ₁₂(Δ GS10)  — 12-month rolling std of monthly change in the
     10-year government bond yield.  Captures bond-market / monetary
     uncertainty.  Source: Shiller (1871–present).

(C2) Inflation Surprise  (IS)
     σ₁₂(Δ log CPI)  — 12-month rolling std of monthly log-change in
     CPI.  Captures real-economy / price-stability uncertainty.
     Source: Shiller (1871–present).

(C3) Credit Spread  (CS)
     BAA − AAA  — Moody's seasoned Baa minus Aaa corporate bond yield.
     Captures credit-risk perception and default-risk pricing.
     Source: FRED (1919–present).

(C4) Term Spread Stress  (TSS)
     −(GS10 − TB3MS)  — Negative of the slope of the yield curve.
     When the curve inverts or flattens, stress is elevated.
     Source: FRED (1934–present).

(C5) Real Rate Deviation  (RRD)
     (GS10 − π₁₂) − MA₂₄(GS10 − π₁₂)  — Deviation of the ex-post
     real long rate from its 24-month trailing mean.  Captures
     monetary-tightening shocks.
     Source: Derived from Shiller (1871–present).

REGIME TIMELINE
---------------
  1871 – 1919 :  C1 + C2 + C5        (3 components)
  1919 – 1934 :  C1 + C2 + C3 + C5   (4 components)
  1934 – now  :  C1 + C2 + C3 + C4 + C5  (5 components, full model)

COMBINATION METHOD
------------------
  1.  Each raw component is z-scored with an *expanding* window:
        z_t = (x_t − μ̂_{1:t}) / σ̂_{1:t}
      where μ̂ and σ̂ use all data up to date t.  This avoids look-ahead
      bias and is the standard for real-time index construction.

  2.  At each date, the HMFSI is the simple average of all available
      component z-scores.

  3.  Finally, the resulting index is re-standardised over the full
      sample to have mean 0 and std 1, matching the STLFSI convention
      (positive = above-average stress).

OUTPUT
------
A monthly time series (pandas Series) indexed by date, suitable for:
  • Direct use as --stress-source in stress_master_all_tests.py
  • Alignment with rolling-window parameter estimates via forward-fill

DATA REQUIREMENTS
-----------------
The script can:
  (a) Download everything live from Shiller's website and FRED, or
  (b) Read from local CSV/Excel files if network is unavailable.

ACADEMIC REFERENCES
-------------------
  • Hakkio & Keeton (2009), "Financial stress: What is it, how can it be
    measured, and why does it matter?"  FRBKC Econ Rev 94(2).
  • Kliesen, Owyang & Vermann (2012), "Disentangling diverse measures:
    a survey of financial stress indexes."  FRBSL Review 94(5).
  • Gilchrist & Zakrajšek (2012), "Credit spreads and business cycle
    fluctuations."  AER 102(4).
  • Krishnamurthy & Muir (2020), "How credit cycles across a financial
    crisis."  Working paper.
  • Jurado, Ludvigson & Ng (2015), "Measuring uncertainty."  AER 105(3).

Author : [Author names]
Date   : 2026
License: MIT
"""

from __future__ import annotations

import warnings
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# =====================================================================
# 1.  DATA ACQUISITION
# =====================================================================

SHILLER_URL = (
    "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
)
FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"

# FRED series identifiers
FRED_SERIES = {
    "BAA": "BAA",        # Moody's Baa yield, monthly, 1919-01+
    "AAA": "AAA",        # Moody's Aaa yield, monthly, 1919-01+
    "TB3MS": "TB3MS",    # 3-Month T-Bill rate, monthly, 1934-01+
    "GS10": "GS10",      # 10-Year Treasury CMR, monthly, 1953-04+
}


def _download_fred(series_id: str, timeout: int = 30) -> pd.Series:
    """Download a single FRED series as a monthly pd.Series."""
    import requests
    from io import StringIO

    url = FRED_BASE.format(sid=series_id)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df.columns = [c.strip() for c in df.columns]
    date_col = [c for c in df.columns if "DATE" in c.upper()][0]
    val_col = [c for c in df.columns if c != date_col][0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna().set_index(date_col).sort_index()
    s = df[val_col]
    s.name = series_id
    return s


def _download_shiller(timeout: int = 60) -> pd.DataFrame:
    """
    Download Shiller's ie_data.xls and return a DataFrame with columns:
        Date, SP_Price, GS10, CPI
    indexed by monthly Timestamp.
    """
    import requests
    from io import BytesIO

    r = requests.get(SHILLER_URL, timeout=timeout)
    r.raise_for_status()
    # Shiller's file has header rows that need skipping
    xls = pd.ExcelFile(BytesIO(r.content))
    # The data sheet is typically called "Data"
    sheet_name = [s for s in xls.sheet_names if "data" in s.lower()][0]
    raw = xls.parse(sheet_name, header=None)

    # Find the header row (contains "Date" or similar)
    for i, row in raw.iterrows():
        vals = [str(v).strip().lower() for v in row.values]
        if "date" in vals or "price" in vals:
            header_idx = i
            break
    else:
        header_idx = 6  # fallback

    df = xls.parse(sheet_name, header=header_idx)
    df.columns = [str(c).strip() for c in df.columns]

    # The date column in Shiller is a float like 1871.01
    date_col = [c for c in df.columns if "date" in c.lower()][0]
    df = df[[date_col]].copy()

    # We need: Date, Price (S&P), Rate GS10, CPI
    # Column order in Shiller: Date, P, D, E, CPI, Date_Fraction,
    #   Long_Interest_Rate, ...
    # Re-read with positional logic
    raw2 = xls.parse(sheet_name, header=header_idx)
    raw2.columns = [str(c).strip() for c in raw2.columns]

    # Build output — use positional knowledge of Shiller layout
    out = pd.DataFrame()

    # Parse the fractional date
    date_vals = pd.to_numeric(raw2.iloc[:, 0], errors="coerce")
    years = date_vals.astype(int)
    months = ((date_vals - years) * 100).round().astype(int)
    months = months.clip(1, 12)
    out["Date"] = pd.to_datetime(
        [f"{y}-{m:02d}-01" for y, m in zip(years, months)],
        errors="coerce",
    )
    # S&P Composite Price (column B, index 1)
    out["SP_Price"] = pd.to_numeric(raw2.iloc[:, 1], errors="coerce")
    # CPI (column E, index 4)
    out["CPI"] = pd.to_numeric(raw2.iloc[:, 4], errors="coerce")
    # Long interest rate / GS10 (column G or H, typically index 6)
    # Try to find it by name
    for i, c in enumerate(raw2.columns):
        cl = c.lower()
        if "rate" in cl or "long" in cl or "gs10" in cl or "interest" in cl:
            out["GS10_shiller"] = pd.to_numeric(
                raw2.iloc[:, i], errors="coerce"
            )
            break
    else:
        # Fallback: column index 6
        out["GS10_shiller"] = pd.to_numeric(
            raw2.iloc[:, 6], errors="coerce"
        )

    out = out.dropna(subset=["Date"]).set_index("Date").sort_index()
    out = out[out.index >= "1871-01-01"]
    return out


def load_shiller_from_csv(path: str) -> pd.DataFrame:
    """
    Load a pre-saved Shiller CSV.  Expected columns:
        Date, SP_Price, GS10_shiller, CPI
    """
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df.sort_index()


def load_fred_from_csv(path: str) -> Dict[str, pd.Series]:
    """
    Load a CSV with FRED series (columns: Date, BAA, AAA, TB3MS).
    Returns a dict of series.
    """
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return {c: df[c].dropna() for c in df.columns}


# =====================================================================
# 2.  COMPONENT CONSTRUCTION
# =====================================================================

def _expanding_zscore(s: pd.Series, min_periods: int = 24) -> pd.Series:
    """
    Expanding-window z-score: z_t = (x_t - mean_{1:t}) / std_{1:t}.
    Uses only data available up to date t (no look-ahead).
    """
    mu = s.expanding(min_periods=min_periods).mean()
    sigma = s.expanding(min_periods=min_periods).std()
    z = (s - mu) / sigma
    return z


def compute_irv(gs10: pd.Series, window: int = 12) -> pd.Series:
    """
    C1: Interest-Rate Volatility.
    12-month rolling std of monthly changes in GS10.
    """
    delta_gs10 = gs10.diff()
    irv = delta_gs10.rolling(window=window, min_periods=window).std()
    irv.name = "IRV"
    return irv


def compute_is(cpi: pd.Series, window: int = 12) -> pd.Series:
    """
    C2: Inflation Surprise.
    12-month rolling std of monthly log-changes in CPI.
    """
    log_cpi_change = np.log(cpi).diff()
    infl_vol = log_cpi_change.rolling(
        window=window, min_periods=window
    ).std()
    infl_vol.name = "IS"
    return infl_vol


def compute_cs(baa: pd.Series, aaa: pd.Series) -> pd.Series:
    """
    C3: Credit Spread.
    BAA - AAA (Moody's seasoned corporate bond yields).
    """
    # Align
    aligned = pd.DataFrame({"BAA": baa, "AAA": aaa}).dropna()
    cs = aligned["BAA"] - aligned["AAA"]
    cs.name = "CS"
    return cs


def compute_tss(gs10: pd.Series, tb3ms: pd.Series) -> pd.Series:
    """
    C4: Term Spread Stress.
    Negative of (GS10 - TB3MS).  Positive when curve is flat/inverted.
    """
    aligned = pd.DataFrame({"GS10": gs10, "TB3MS": tb3ms}).dropna()
    tss = -(aligned["GS10"] - aligned["TB3MS"])
    tss.name = "TSS"
    return tss


def compute_rrd(
    gs10: pd.Series, cpi: pd.Series, ma_window: int = 24
) -> pd.Series:
    """
    C5: Real Rate Deviation.
    Deviation of ex-post real long rate from its trailing mean.
    real_rate = GS10 - 12m trailing inflation rate
    RRD = real_rate - MA_24(real_rate)
    """
    # 12-month trailing annualised inflation
    log_cpi = np.log(cpi)
    infl_12m = (log_cpi - log_cpi.shift(12)) * 100  # in percent
    # Align
    aligned = pd.DataFrame({"GS10": gs10, "infl": infl_12m}).dropna()
    real_rate = aligned["GS10"] - aligned["infl"]
    rrd = real_rate - real_rate.rolling(
        window=ma_window, min_periods=ma_window
    ).mean()
    rrd.name = "RRD"
    return rrd


# =====================================================================
# 3.  INDEX ASSEMBLY
# =====================================================================

def build_hmfsi(
    shiller: pd.DataFrame,
    fred_series: Optional[Dict[str, pd.Series]] = None,
    min_z_periods: int = 36,
    start_date: str = "1885-01-01",
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Build the Historical Macro-Financial Stress Index.

    Parameters
    ----------
    shiller : pd.DataFrame
        Must contain columns: GS10_shiller, CPI.
        Indexed by monthly Date.
    fred_series : dict, optional
        Keys: "BAA", "AAA", "TB3MS".  If None, only C1+C2+C5 are used.
    min_z_periods : int
        Minimum observations for expanding z-score.
    start_date : str
        Earliest date to include in the output.

    Returns
    -------
    hmfsi : pd.Series
        The final HMFSI index, monthly.
    components : pd.DataFrame
        All raw and z-scored components for diagnostics.
    """
    gs10 = shiller["GS10_shiller"].dropna()
    cpi = shiller["CPI"].dropna()

    # --- Raw components ---
    c1_raw = compute_irv(gs10)
    c2_raw = compute_is(cpi)
    c5_raw = compute_rrd(gs10, cpi)

    c3_raw, c4_raw = None, None
    if fred_series is not None:
        if "BAA" in fred_series and "AAA" in fred_series:
            c3_raw = compute_cs(fred_series["BAA"], fred_series["AAA"])
        if "TB3MS" in fred_series:
            # For TSS we need GS10 at FRED frequency; use Shiller if needed
            gs10_for_tss = fred_series.get("GS10", gs10)
            c4_raw = compute_tss(gs10_for_tss, fred_series["TB3MS"])

    # --- Assemble into a single DataFrame ---
    all_raw = pd.DataFrame({"C1_IRV": c1_raw, "C2_IS": c2_raw})
    if c3_raw is not None:
        all_raw["C3_CS"] = c3_raw
    if c4_raw is not None:
        all_raw["C4_TSS"] = c4_raw
    all_raw["C5_RRD"] = c5_raw

    # Filter to start_date
    all_raw = all_raw[all_raw.index >= start_date]

    # --- Expanding z-score each component ---
    z_cols = {}
    for col in all_raw.columns:
        z_cols[col + "_z"] = _expanding_zscore(
            all_raw[col], min_periods=min_z_periods
        )
    z_df = pd.DataFrame(z_cols)

    # --- Combine: equal-weight average of available components ---
    hmfsi_raw = z_df.mean(axis=1, skipna=True)

    # Count available components at each date
    n_components = z_df.notna().sum(axis=1)

    # Require at least 2 components
    hmfsi_raw[n_components < 2] = np.nan

    # --- Final standardisation (full-sample) ---
    hmfsi_mean = hmfsi_raw.mean()
    hmfsi_std = hmfsi_raw.std()
    hmfsi = (hmfsi_raw - hmfsi_mean) / hmfsi_std
    hmfsi.name = "HMFSI"

    # --- Diagnostics DataFrame ---
    components = all_raw.copy()
    for col in z_df.columns:
        components[col] = z_df[col]
    components["HMFSI"] = hmfsi
    components["n_components"] = n_components

    return hmfsi, components


# =====================================================================
# 4.  ALIGNMENT WITH ROLLING-WINDOW ESTIMATES
# =====================================================================

def align_hmfsi_to_rolling(
    hmfsi: pd.Series,
    rolling_dates: pd.DatetimeIndex,
    method: str = "ffill",
) -> pd.Series:
    """
    Align the monthly HMFSI to the rolling-window parameter dates.

    The rolling dates typically correspond to the end-of-window dates
    of the Baldovin-Stella parameter estimates.  Since the HMFSI is
    monthly and the rolling dates may be at arbitrary daily frequency,
    we reindex and forward-fill.

    Parameters
    ----------
    hmfsi : pd.Series
        Monthly HMFSI.
    rolling_dates : pd.DatetimeIndex
        Dates from the rolling CSV (window_end_date or window_date).
    method : str
        Interpolation method ('ffill' recommended).

    Returns
    -------
    aligned : pd.Series
        HMFSI values at each rolling date.
    """
    # Create a union index and reindex
    combined_idx = hmfsi.index.union(rolling_dates).sort_values()
    aligned = hmfsi.reindex(combined_idx).interpolate(method="index")
    # Now pick only the rolling dates
    result = aligned.reindex(rolling_dates)
    if method == "ffill":
        result = hmfsi.reindex(rolling_dates, method="ffill")
    result.name = "HMFSI"
    return result


# =====================================================================
# 5.  INTEGRATION WITH stress_master_all_tests.py
# =====================================================================

def hmfsi_as_stress_series(
    hmfsi_csv: str,
    rolling_csv: str,
    date_col: str = "window_end_date",
) -> pd.Series:
    """
    Convenience function: load a pre-built HMFSI CSV and align it
    to a rolling-window CSV, returning a stress series ready for
    the stress_master pipeline.

    Usage in stress_master:
        Instead of --stress-source rv, load the HMFSI and pass it
        directly into the analysis.
    """
    hmfsi_df = pd.read_csv(hmfsi_csv, parse_dates=["Date"], index_col="Date")
    hmfsi = hmfsi_df["HMFSI"]

    rolling = pd.read_csv(rolling_csv)
    rolling[date_col] = pd.to_datetime(rolling[date_col])
    rolling_dates = pd.DatetimeIndex(rolling[date_col])

    return align_hmfsi_to_rolling(hmfsi, rolling_dates)


# =====================================================================
# 6.  VISUALISATION
# =====================================================================

def plot_hmfsi(
    components: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10),
):
    """
    Create a publication-quality diagnostic plot of the HMFSI.

    Panel (a): HMFSI time series with NBER recession shading.
    Panel (b): Individual z-scored components.
    Panel (c): Number of available components over time.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 1]})

    # --- Panel (a): HMFSI ---
    ax = axes[0]
    hmfsi = components["HMFSI"].dropna()
    ax.plot(hmfsi.index, hmfsi.values, color="#1f77b4", linewidth=0.8,
            label="HMFSI")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.fill_between(hmfsi.index, 0, hmfsi.values,
                    where=hmfsi.values > 0, alpha=0.3, color="#d62728",
                    label="Above-average stress")
    ax.fill_between(hmfsi.index, 0, hmfsi.values,
                    where=hmfsi.values <= 0, alpha=0.2, color="#2ca02c",
                    label="Below-average stress")
    ax.set_ylabel("HMFSI (standardised)", fontsize=11)
    ax.set_title(
        "Historical Macro-Financial Stress Index (HMFSI), 1885–2025",
        fontsize=13, fontweight="bold"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel (b): Components ---
    ax = axes[1]
    z_cols = [c for c in components.columns if c.endswith("_z")]
    colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    labels = {
        "C1_IRV_z": "Interest Rate Vol.",
        "C2_IS_z": "Inflation Surprise",
        "C3_CS_z": "Credit Spread",
        "C4_TSS_z": "Term Spread Stress",
        "C5_RRD_z": "Real Rate Deviation",
    }
    for i, col in enumerate(z_cols):
        s = components[col].dropna()
        ax.plot(s.index, s.values, linewidth=0.6, alpha=0.8,
                color=colors[i % len(colors)],
                label=labels.get(col, col))
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Component z-scores", fontsize=11)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- Panel (c): Number of components ---
    ax = axes[2]
    n_comp = components["n_components"].dropna()
    ax.fill_between(n_comp.index, 0, n_comp.values,
                    color="#7f7f7f", alpha=0.4, step="mid")
    ax.set_ylabel("# Components", fontsize=11)
    ax.set_ylim(0, 6)
    ax.set_xlabel("Date", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Format x-axis
    for a in axes:
        a.xaxis.set_major_locator(mdates.YearLocator(20))
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight",
                    facecolor="white")
        logger.info(f"Plot saved to {output_path}")
    plt.close(fig)
    return fig


# =====================================================================
# 7.  MAIN: BUILD & SAVE
# =====================================================================

def main():
    """
    Main entry point.  Downloads data and builds the HMFSI.

    Usage:
        python hmfsi.py [--shiller-csv PATH] [--fred-csv PATH]
                        [--output-dir DIR] [--no-download]
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Build the Historical Macro-Financial Stress Index"
    )
    parser.add_argument(
        "--shiller-csv", type=str, default=None,
        help="Path to a pre-saved Shiller CSV (skip download)."
    )
    parser.add_argument(
        "--fred-csv", type=str, default=None,
        help="Path to a pre-saved FRED CSV with BAA, AAA, TB3MS columns."
    )
    parser.add_argument(
        "--output-dir", type=str, default="hmfsi_output",
        help="Output directory."
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="Do not attempt to download data; use only local files."
    )
    parser.add_argument(
        "--start-date", type=str, default="1885-01-01",
        help="Earliest date for index construction."
    )
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Load Shiller data ----
    if args.shiller_csv:
        print(f"[INFO] Loading Shiller data from {args.shiller_csv}")
        shiller = load_shiller_from_csv(args.shiller_csv)
    elif not args.no_download:
        print("[INFO] Downloading Shiller dataset from Yale...")
        try:
            shiller = _download_shiller()
            # Save for reuse
            shiller.to_csv(outdir / "shiller_data.csv")
            print(f"  → Saved to {outdir / 'shiller_data.csv'}")
        except Exception as e:
            print(f"[ERROR] Shiller download failed: {e}")
            print("  Please download manually from:")
            print(f"    {SHILLER_URL}")
            print("  Then re-run with --shiller-csv <path>")
            return
    else:
        print("[ERROR] No Shiller data source. Use --shiller-csv or allow downloads.")
        return

    print(f"  Shiller data: {shiller.index.min().date()} to {shiller.index.max().date()}")
    print(f"  Columns: {list(shiller.columns)}")

    # ---- Load FRED data ----
    fred_series = None
    if args.fred_csv:
        print(f"[INFO] Loading FRED data from {args.fred_csv}")
        fred_series = load_fred_from_csv(args.fred_csv)
    elif not args.no_download:
        print("[INFO] Downloading FRED series...")
        fred_series = {}
        for sid in ["BAA", "AAA", "TB3MS"]:
            try:
                fred_series[sid] = _download_fred(sid)
                print(f"  → {sid}: {fred_series[sid].index.min().date()} to "
                      f"{fred_series[sid].index.max().date()}")
            except Exception as e:
                print(f"  [WARN] {sid} download failed: {e}")
        if fred_series:
            # Save for reuse
            fred_df = pd.DataFrame(fred_series)
            fred_df.index.name = "Date"
            fred_df.to_csv(outdir / "fred_data.csv")
            print(f"  → Saved to {outdir / 'fred_data.csv'}")
    else:
        print("[INFO] No FRED data — using 3-component model (C1+C2+C5 only).")

    # ---- Build HMFSI ----
    print("\n[INFO] Building HMFSI...")
    hmfsi, components = build_hmfsi(
        shiller, fred_series, start_date=args.start_date
    )

    # ---- Save outputs ----
    hmfsi_out = hmfsi.dropna()
    print(f"  HMFSI range: {hmfsi_out.index.min().date()} to "
          f"{hmfsi_out.index.max().date()}")
    print(f"  N observations: {len(hmfsi_out)}")
    print(f"  Mean: {hmfsi_out.mean():.4f},  Std: {hmfsi_out.std():.4f}")

    # Save
    hmfsi_out.to_frame().to_csv(outdir / "HMFSI.csv")
    components.to_csv(outdir / "HMFSI_components.csv")
    print(f"\n  → HMFSI saved to {outdir / 'HMFSI.csv'}")
    print(f"  → Components saved to {outdir / 'HMFSI_components.csv'}")

    # ---- Plot ----
    try:
        plot_hmfsi(components, output_path=str(outdir / "HMFSI_plot.png"))
        print(f"  → Plot saved to {outdir / 'HMFSI_plot.png'}")
    except Exception as e:
        print(f"  [WARN] Plotting failed: {e}")

    # ---- Summary statistics by era ----
    print("\n" + "=" * 60)
    print("HMFSI SUMMARY BY ERA")
    print("=" * 60)
    eras = [
        ("1885-1919", "1885", "1919"),
        ("1919-1934", "1919", "1934"),
        ("1934-1970", "1934", "1970"),
        ("1970-2000", "1970", "2000"),
        ("2000-2025", "2000", "2026"),
    ]
    for label, s, e in eras:
        mask = (hmfsi_out.index >= s) & (hmfsi_out.index < e)
        sub = hmfsi_out[mask]
        if len(sub) > 0:
            n_comp = components.loc[mask, "n_components"]
            print(
                f"  {label}: n={len(sub):4d}, mean={sub.mean():+.3f}, "
                f"std={sub.std():.3f}, max={sub.max():.2f}, "
                f"components={n_comp.median():.0f}"
            )

    print("\n[DONE] HMFSI construction complete.")
    print(
        "\nTo use with stress_master_all_tests.py, pass the HMFSI CSV\n"
        "as a custom stress source, or modify the stress loading function\n"
        "to read HMFSI.csv instead of computing RV.\n"
    )
    print("Example integration:")
    print("  from hmfsi import hmfsi_as_stress_series")
    print("  stress = hmfsi_as_stress_series('HMFSI.csv', 'rolling.csv')")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
