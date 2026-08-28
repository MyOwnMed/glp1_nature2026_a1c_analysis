#!/usr/bin/env python3
"""
Step 6d (updated): GLP-1 group comparisons with a pooled GEE

- Fit ONE pooled GEE using bs(days, df=best_df) * glp1_group * baseline_a1c_category
  plus non-interacted covariates.
- Within each baseline A1C category, generate predicted trajectories for
  Semaglutide-only vs Tirzepatide-only and their difference.
- Anchor trajectories at Day 0 for BOTH outcomes (weight and A1C).
- Save plots and CSVs per category, plus global model diagnostics.

Usage examples:
  python3 code/step6d_groups_by_glp1.py \
    --outcome weight \
    --input-csv output/step1_prepare_analysis_dataset/analysis_ready_gap150.csv \
    --config-json output/gap_150/step2_select_spline_df/model_config.json \
    --outdir output/gap_150/step6d_glp1_groups_weight \
    --adherence-gap-days 150

  python3 code/step6d_groups_by_glp1.py \
    --outcome a1c \
    --input-csv output/step1_prepare_analysis_dataset_a1c/analysis_ready_a1c_gap150.csv \
    --config-json output/step2_select_spline_df_a1c/model_config_a1c.json \
    --outdir output/gap_150/step6d_glp1_groups_a1c \
    --adherence-gap-days 150 \
    --xlim-min -5 --xlim-max 2.5
"""
import argparse
import os
import json
import logging
from typing import Optional, Sequence, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from patsy import dmatrices, build_design_matrices, bs  # noqa: F401
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.cov_struct import Independence

# Confidence multiplier derived from a single configurable confidence level
# (code-review item 9); replaces a hard-coded 1.96. See analysis_config.py.
import sys as _sys
from pathlib import Path as _Path

for _p in [_Path(__file__).resolve().parent, *_Path(__file__).resolve().parents]:
    if (_p / "analysis_config.py").exists():
        _sys.path.insert(0, str(_p))
        break
from analysis_config import z_critical

Z_CRIT = z_critical()

# Model-specification inputs are never silently defaulted (code-review item 8).
# See model_spec.py.
import sys as _sys_ms
from pathlib import Path as _Path_ms

for _p_ms in [_Path_ms(__file__).resolve().parent, *_Path_ms(__file__).resolve().parents]:
    if (_p_ms / "model_spec.py").exists():
        _sys_ms.path.insert(0, str(_p_ms))
        break
from model_spec import enforce_a1c_order, load_spline_df

# Covariate drops are logged rather than silent (code-review item on
# _select_covariates). See covariates.py.
import sys as _sys_cov
from pathlib import Path as _Path_cov

for _p_cov in [_Path_cov(__file__).resolve().parent, *_Path_cov(__file__).resolve().parents]:
    if (_p_cov / "covariates.py").exists():
        _sys_cov.path.insert(0, str(_p_cov))
        break
from covariates import filter_estimable

A1C_ORDER = [
    "Normal Glycemia",
    "Prediabetes",
    "Type 2 Diabetes",
    "Poorly Controlled Diabetes",
]
GLP1_TWO_GROUPS = ["sema-only", "tirz-only"]
KEY_DAYS = [90, 180, 270, 365, 450, 548, 630, 730]


def configure_logging(level: str = "INFO"):
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(description="Step 6d: GLP-1 group comparisons with pooled GEE")
    p.add_argument("--outcome", choices=["weight", "a1c"], required=True, help="Outcome type")
    p.add_argument("--input-csv", required=True, help="Path to analysis-ready CSV (weight or A1C)")
    p.add_argument("--config-json", required=True, help="Step 2 model config with best df")
    p.add_argument("--outdir", required=False, default=os.path.join("output", "step6d_glp1_groups"), help="Output base directory")
    p.add_argument("--adherence-gap-days", type=int, default=None, help="Gap for gap-specific subfolder (e.g., 150)")
    p.add_argument("--min-nobs", type=int, default=10, help="Minimum observations per A1C stratum to plot")
    p.add_argument("--xlim-min", type=float, default=None, help="Forest x-axis min for differences (unused here)")
    p.add_argument("--xlim-max", type=float, default=None, help="Forest x-axis max for differences (unused here)")
    p.add_argument(
        "--truncate-days", type=int, default=548,
        help=(
            "Truncate follow-up to this maximum day for fitting and predictions "
            "(548 ~ 18 months). The paper's 18-month follow-up limit as a calendar "
            "figure; step8 states the same limit as 18 x 30 = 540 days and step1's "
            "730-day --max-days is a wider outer bound on emitted data for the "
            "730-day persistence sensitivity analysis. See README.md, 'Follow-up "
            "horizon caps'."
        ),
    )
    p.add_argument("--debug-a1c-category", type=str, default=None, help="If set, generate additional diagnostics (raw vs predicted; centered vs uncentered) for this A1C category")
    p.add_argument("--log-level", default="INFO", help="Logging level")
    return p.parse_args(argv)


def _compute_mean_ci(result, X_pred: np.ndarray, z: float = Z_CRIT):
    mean = np.asarray(result.predict(X_pred), dtype=float)
    V = np.asarray(result.cov_params(), dtype=float)
    var = np.einsum("ij,jk,ik->i", X_pred, V, X_pred, optimize=True)
    var = np.clip(var, 0.0, None)
    se = np.sqrt(var)
    low = mean - z * se
    high = mean + z * se
    return mean, low, high


def _compute_diff_mean_ci_from_design(result, X_a: np.ndarray, X_b: np.ndarray, z: float = Z_CRIT):
    """Compute mean and CI for prediction difference b - a using pooled covariance.

    X_a, X_b: shape (n, p). Returns arrays length n.
    """
    params = np.asarray(result.params, dtype=float)
    V = np.asarray(result.cov_params(), dtype=float)
    delta = X_b - X_a  # (n, p)
    mean = np.einsum("ij,j->i", delta, params, optimize=True)
    var = np.einsum("ij,jk,ik->i", delta, V, delta, optimize=True)
    var = np.clip(var, 0.0, None)
    se = np.sqrt(var)
    low = mean - z * se
    high = mean + z * se
    return mean, low, high


def _modal_values(df: pd.DataFrame, cols: List[str]) -> dict:
    ref = {}
    for c in cols:
        if c in df.columns:
            try:
                m = df[c].mode(dropna=True)
                if not m.empty:
                    ref[c] = m.iloc[0]
                elif hasattr(df[c], "cat") and len(df[c].cat.categories) > 0:
                    ref[c] = df[c].cat.categories[0]
            except Exception:
                if hasattr(df[c], "cat") and len(df[c].cat.categories) > 0:
                    ref[c] = df[c].cat.categories[0]
    return ref


def _plot_trajectories(days_grid, mean_sema, mean_tirz, ci_sema, ci_tirz, title, ylabel, out_path, y_limits=None):
    plt.figure(figsize=(10, 6))
    plt.plot(days_grid, mean_sema, label="Semaglutide-only", color="#1f77b4")
    plt.plot(days_grid, mean_tirz, label="Tirzepatide-only", color="#ff7f0e")
    try:
        plt.fill_between(days_grid, ci_sema[0], ci_sema[1], color="#1f77b4", alpha=0.15)
        plt.fill_between(days_grid, ci_tirz[0], ci_tirz[1], color="#ff7f0e", alpha=0.15)
    except Exception as exc:
        logging.debug(
            "could not shade the confidence band on the trajectory plot (%s); the plot is written without shading",
            exc,
        )
    plt.xlabel("Days from Baseline")
    plt.ylabel(ylabel)
    if y_limits is not None:
        plt.ylim(*y_limits)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_difference(days_grid, diff_mean, diff_low, diff_high, title, ylabel, out_path, y_limits=None):
    plt.figure(figsize=(10, 5))
    plt.plot(days_grid, diff_mean, label="Tirzepatide - Semaglutide", color="#2ca02c")
    try:
        plt.fill_between(days_grid, diff_low, diff_high, color="#2ca02c", alpha=0.15)
    except Exception as exc:
        logging.debug(
            "could not shade the confidence band on the difference plot (%s); the plot is written without shading",
            exc,
        )
    plt.axhline(0, color="grey", linestyle="-")
    plt.xlabel("Days from Baseline")
    plt.ylabel(ylabel)
    if y_limits is not None:
        plt.ylim(*y_limits)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _forest_plot(days, est, lo, hi, labels, title, xlabel, out_path, xlim=None):
    fig = plt.figure(figsize=(12, 0.5 * len(labels) + 2))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.5, 1.8])
    ax = fig.add_subplot(gs[0, 0])
    ax_txt = fig.add_subplot(gs[0, 1])
    y = np.arange(len(labels))
    ax.errorbar(est, y, xerr=[est - lo, hi - est], fmt="o", color="black", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(0, color="grey", linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    # Invert y-axis for top-to-bottom ordering
    ax.invert_yaxis()
    # Right-side text of estimates
    ax_txt.axis("off")
    ax_txt.set_title("Estimate (95% CI)")
    if len(y) > 0:
        ax_txt.set_ylim(min(y) - 0.5, max(y) + 0.5)
    for yi, m, l, h in zip(y, est, lo, hi):
        ax_txt.text(0.0, yi, f"{float(m):.2f} ({float(l):.2f}, {float(h):.2f})", va="center", ha="left", fontsize=9)
    ax_txt.set_xlim(0, 1)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _forest_plot_two_groups(est_sema, lo_sema, hi_sema, est_tirz, lo_tirz, hi_tirz, labels, title, xlabel, out_path, xlim=None):
    fig = plt.figure(figsize=(12, 0.5 * len(labels) + 2))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.5, 2.2])
    ax = fig.add_subplot(gs[0, 0])
    ax_txt = fig.add_subplot(gs[0, 1])
    y = np.arange(len(labels))
    # Slight vertical jitter for side-by-side appearance
    ax.errorbar(est_sema, y + 0.1, xerr=[est_sema - lo_sema, hi_sema - est_sema], fmt="o", color="#1f77b4", capsize=3, label="Semaglutide")
    ax.errorbar(est_tirz, y - 0.1, xerr=[est_tirz - lo_tirz, hi_tirz - est_tirz], fmt="o", color="#ff7f0e", capsize=3, label="Tirzepatide")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(0, color="grey", linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.legend(loc="best")
    # Invert y-axis for top-to-bottom ordering
    ax.invert_yaxis()
    # Right-side text: show both groups per line
    ax_txt.axis("off")
    ax_txt.set_title("Estimate (95% CI)")
    if len(y) > 0:
        ax_txt.set_ylim(min(y) - 0.5, max(y) + 0.5)
    for yi, ms, ls, hs, mt, lt, ht in zip(y, est_sema, lo_sema, hi_sema, est_tirz, lo_tirz, hi_tirz):
        txt = f"Sema {float(ms):.2f} ({float(ls):.2f}, {float(hs):.2f}); Tirz {float(mt):.2f} ({float(lt):.2f}, {float(ht):.2f})"
        ax_txt.text(0.0, yi, txt, va="center", ha="left", fontsize=9)
    ax_txt.set_xlim(0, 1)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _fit_pooled(df: pd.DataFrame, df_spline: int, outcome: str):
    """Fit pooled GEE with interactions: bs(time)*glp1_group2*baseline_a1c_category + covariates.
    Returns result and design_info.
    """
    bs_term = f"bs(days_from_baseline, df={df_spline})"
    yvar = "pct_weight_change" if outcome == "weight" else "abs_a1c_change"

    # Add non-interacted covariates if present and varying
    covars = [
        "age_group",
        "gender",
        "baseline_bmi_final_category",
        "race",
        "weight_change_med",
    ]
    covars = filter_estimable(covars, df, context="step6d glp1 group comparison")

    # Interaction core
    core = f"{bs_term} * glp1_group2 * baseline_a1c_category"
    formula = f"{yvar} ~ {core}"
    if covars:
        formula += " + " + " + ".join(covars)

    y, X = dmatrices(formula, df, return_type="dataframe")
    ids = df.loc[y.index, "patient_id"]
    model = GEE(y, X, groups=ids, family=Gaussian(), cov_struct=Independence())
    res = model.fit()
    return res, X.design_info, formula, covars


def _predict_grid_pooled(res, design_info, grid: np.ndarray, ref_row: dict, cat_value: str, group_value: str):
    rows = []
    for d in grid:
        row = {
            **ref_row,
            "baseline_a1c_category": cat_value,
            "glp1_group2": group_value,
            "days_from_baseline": int(d),
        }
        rows.append(row)
    X_pred = build_design_matrices([design_info], pd.DataFrame(rows))[0]
    X_arr = np.asarray(X_pred)
    mean, lo, hi = _compute_mean_ci(res, X_arr)
    return mean, lo, hi, X_arr


def _compute_binned_empirical(df: pd.DataFrame, group_col: str, yvar: str, bin_size: int, max_day: int):
    """Compute empirical means and 95% CI per bin for each group. Returns dict[group]->DataFrame.
    Bins are [k*bin_size, (k+1)*bin_size).
    """
    work = df[["days_from_baseline", group_col, yvar]].dropna().copy()
    work["days_from_baseline"] = pd.to_numeric(work["days_from_baseline"], errors="coerce")
    work = work[(work["days_from_baseline"] >= 0) & (work["days_from_baseline"] <= max_day)]
    work["day_bin"] = (work["days_from_baseline"] // bin_size) * bin_size
    out = {}
    for g in GLP1_TWO_GROUPS:
        gdf = work[work[group_col] == g]
        if gdf.empty:
            continue
        agg = gdf.groupby("day_bin")[yvar].agg(["mean", "std", "count"]).reset_index()
        # standard error and CI
        agg["se"] = agg.apply(lambda r: (r["std"] / np.sqrt(r["count"])) if r["count"] > 1 and pd.notna(r["std"]) else np.nan, axis=1)
        agg["ci_lower"] = agg["mean"] - Z_CRIT * agg["se"]
        agg["ci_upper"] = agg["mean"] + Z_CRIT * agg["se"]
        out[g] = agg.rename(columns={"day_bin": "day"})
    return out


def _plot_raw_vs_predicted(days_grid, pred_sema, pred_tirz, raw_map, title, ylabel, out_path, centered=False, anchor_sema=None, anchor_tirz=None):
    plt.figure(figsize=(11, 6))
    # Predicted
    plt.plot(days_grid, pred_sema, label=("Pred Sema" + (" (centered)" if centered else "")), color="#1f77b4", lw=2)
    plt.plot(days_grid, pred_tirz, label=("Pred Tirz" + (" (centered)" if centered else "")), color="#ff7f0e", lw=2)
    # Raw binned
    for g, color in [("sema-only", "#1f77b4"), ("tirz-only", "#ff7f0e")]:
        rdf = raw_map.get(g)
        if rdf is None or rdf.empty:
            continue
        y = rdf["mean"].values.copy()
        lo = rdf["ci_lower"].values.copy()
        hi = rdf["ci_upper"].values.copy()
        if centered:
            if g == "sema-only" and anchor_sema is not None:
                y = y - anchor_sema
                lo = lo - anchor_sema
                hi = hi - anchor_sema
            if g == "tirz-only" and anchor_tirz is not None:
                y = y - anchor_tirz
                lo = lo - anchor_tirz
                hi = hi - anchor_tirz
        plt.scatter(rdf["day"], y, color=color, s=15, alpha=0.8, label=f"Raw {('Sema' if g=='sema-only' else 'Tirz')} mean{ ' (centered)' if centered else ''}")
        try:
            plt.fill_between(rdf["day"], lo, hi, color=color, alpha=0.12, linewidth=0)
        except Exception as exc:
            logging.debug(
                "could not shade the raw-mean confidence band (%s); the plot is written without shading",
                exc,
            )
    if centered:
        plt.axhline(0, color="grey", linestyle="-", alpha=0.7)
    plt.xlabel("Days from Baseline")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(ncol=2)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    configure_logging(args.log_level)

    # Gap-aware outdir routing
    gap = args.adherence_gap_days
    outdir = args.outdir
    if gap is not None and "/gap_" not in outdir:
        outdir = os.path.join("output", f"gap_{gap}", os.path.basename(outdir))
    os.makedirs(outdir, exist_ok=True)
    # Write run info for debug
    try:
        with open(os.path.join(outdir, "run_info.json"), "w") as f:
            json.dump({
                "outcome": args.outcome,
                "input_csv": args.input_csv,
                "config_json": args.config_json,
                "resolved_outdir": outdir,
                "gap": gap,
            }, f)
    except Exception as exc:
        logging.warning(
            "could not write run_info.json (%s); the analysis outputs are unaffected",
            exc,
        )

    df = pd.read_csv(args.input_csv)
    logging.info("Writing outputs to %s", outdir)

    # If truncation requested, filter days_from_baseline to [0, truncate_days]
    if args.truncate_days is not None:
        if "days_from_baseline" not in df.columns:
            logging.error("days_from_baseline column missing; cannot truncate follow-up")
            return
        before = len(df)
        df["days_from_baseline"] = pd.to_numeric(df["days_from_baseline"], errors="coerce")
        df = df[(df["days_from_baseline"].notna()) & (df["days_from_baseline"] >= 0) & (df["days_from_baseline"] <= args.truncate_days)].copy()
        after = len(df)
        logging.info("Applied truncation to %d days: kept %d/%d rows", args.truncate_days, after, before)
    # Categorical handling
    for cat_col in [
        "baseline_a1c_category",
        "baseline_bmi_final_category",
        "age_group",
        "gender",
        "race",
        "glp1_user_group",
    ]:
        if cat_col in df.columns:
            try:
                df[cat_col] = df[cat_col].astype("category")
            except (TypeError, ValueError) as exc:
                logging.warning(
                    "Could not cast %s to categorical (%s); it will enter the model "
                    "with its original dtype", cat_col, exc,
                )
    # Fails loudly: this fixes the reference level for every contrast
    # (code-review item 8). See model_spec.enforce_a1c_order.
    df = enforce_a1c_order(df, order=A1C_ORDER, context="step6d_groups_by_glp1")

    # Populate GLP-1 group if missing by merging from root_data/step8f.csv
    if "glp1_user_group" not in df.columns:
        try:
            src = os.path.join("root_data", "step8f.csv")
            g = pd.read_csv(src, usecols=["patient_id", "glp1_user_group"]) if os.path.exists(src) else None
            if g is not None and not g.empty:
                g = g.drop_duplicates("patient_id")
                df = df.merge(g, on="patient_id", how="left")
                logging.info("Merged glp1_user_group from %s", src)
        except Exception as e:
            logging.error("Failed to merge glp1_user_group from step8f.csv: %s", e)
    # Normalize GLP-1 group labels to two groups and model with glp1_group2
    if "glp1_user_group" not in df.columns:
        logging.error("glp1_user_group column missing; cannot compare groups")
        return
    lower = df["glp1_user_group"].astype(str).str.lower()
    # More permissive mapping: exclude switchers, map any sema/tirz-only labels
    glp_series = pd.Series(pd.NA, index=df.index, dtype="object")
    glp_series[lower.str.contains("sema") & ~lower.str.contains("switch")] = "sema-only"
    glp_series[lower.str.contains("tirz") & ~lower.str.contains("switch")] = "tirz-only"
    # Accept exact canonical labels too
    glp_series[lower == "sema-only"] = "sema-only"
    glp_series[lower == "tirz-only"] = "tirz-only"
    df["glp1_group2"] = glp_series
    df = df[df["glp1_group2"].isin(GLP1_TWO_GROUPS)].copy()
    df["glp1_group2"] = df["glp1_group2"].astype("category")
    if df.empty:
        logging.error("No rows for sema-only or tirz-only; nothing to do")
        return

    # Best df from config. Fails loudly rather than defaulting the model
    # specification (code-review item 8). See model_spec.load_spline_df.
    df_spline = load_spline_df(args.config_json)
    logging.info("Using spline df=%d", df_spline)

    # Fit pooled model
    res, design_info, formula_used, covars_used = _fit_pooled(df, df_spline, args.outcome)

    # Save global diagnostics
    open(os.path.join(outdir, "pooled_model_summary.txt"), "w").write(str(res.summary()))
    def _coef_df(res_):
        params = res_.params
        bse = res_.bse
        z = Z_CRIT
        ci_low = params - z * bse
        ci_high = params + z * bse
        pvalues = res_.pvalues
        return pd.DataFrame({
            "term": params.index,
            "estimate": np.asarray(params, dtype=float),
            "std_error": np.asarray(bse, dtype=float),
            "ci_lower": np.asarray(ci_low, dtype=float),
            "ci_upper": np.asarray(ci_high, dtype=float),
            "p_value": np.asarray(pvalues, dtype=float),
        })
    _coef_df(res).to_csv(os.path.join(outdir, "pooled_coefficients.csv"), index=False)
    json.dump({"formula": formula_used, "covariates": covars_used, "df_spline": df_spline}, open(os.path.join(outdir, "pooled_model_config_used.json"), "w"))

    # Reference modal values for design building
    ref_covars = [
        "age_group",
        "gender",
        "baseline_bmi_final_category",
        "race",
        "weight_change_med",
    ]

    # Days grid (we'll clamp per-category to overlapping observed ranges of the two groups)
    if args.truncate_days is not None:
        days_grid_base = np.arange(0, int(args.truncate_days) + 1, 14)
        logging.info("Using truncated prediction grid 0..%d step 14 (size=%d)", int(args.truncate_days), int(days_grid_base.size))
    else:
        days_grid_base = np.arange(0, 731, 14)

    # Storage for forest rows by day
    forest_days = [90, 180, 270, 365, 450, 548]
    forest_rows = {d: [] for d in forest_days}

    # Iterate baseline A1C categories in order
    cats = [c for c in (df["baseline_a1c_category"].cat.categories if hasattr(df["baseline_a1c_category"], "cat") else df["baseline_a1c_category"].dropna().unique())]
    cats = [c for c in cats if pd.notna(c)]

    for cat in cats:
        sub = df[df["baseline_a1c_category"] == cat].copy()
        # Per-group counts
        g_sema = sub[sub["glp1_group2"] == "sema-only"].copy()
        g_tirz = sub[sub["glp1_group2"] == "tirz-only"].copy()
        if len(g_sema) < args.min_nobs or len(g_tirz) < args.min_nobs:
            logging.info("Skip %s due to per-group min_nobs (sema=%d, tirz=%d, min_nobs=%d)", cat, len(g_sema), len(g_tirz), args.min_nobs)
            continue

        # Output folders per category
        out_cat_dir = os.path.join(outdir, f"{str(cat).replace(' ', '_').replace('/', '-')}")
        out_sema_dir = os.path.join(out_cat_dir, "sema-only")
        out_tirz_dir = os.path.join(out_cat_dir, "tirz-only")
        os.makedirs(out_sema_dir, exist_ok=True)
        os.makedirs(out_tirz_dir, exist_ok=True)

        # Group overview (counts)
        pd.DataFrame([
            {"baseline_a1c_category": cat, "group": "sema-only", "n_obs": int(len(g_sema)), "n_patients": int(g_sema["patient_id"].nunique())},
        ]).to_csv(os.path.join(out_sema_dir, "overview.csv"), index=False)
        pd.DataFrame([
            {"baseline_a1c_category": cat, "group": "tirz-only", "n_obs": int(len(g_tirz)), "n_patients": int(g_tirz["patient_id"].nunique())},
        ]).to_csv(os.path.join(out_tirz_dir, "overview.csv"), index=False)

        # Reference values (modal) within this category
        ref_vals = _modal_values(sub, ref_covars)

        # Clamp grid to overlap of observed ranges
        def _minmax(df_):
            d = pd.to_numeric(df_.get("days_from_baseline"), errors="coerce")
            d = d[np.isfinite(d)]
            if d.size == 0:
                return 0, 730
            return int(np.nanmin(d)), int(np.nanmax(d))
        min_s, max_s = _minmax(g_sema)
        min_t, max_t = _minmax(g_tirz)
        lo_g = max(min_s, min_t)
        hi_g = min(max_s, max_t)
        grid = days_grid_base[(days_grid_base >= lo_g) & (days_grid_base <= hi_g)]
        if grid.size == 0:
            cap_hi = int(args.truncate_days) if args.truncate_days is not None else 548
            grid = np.arange(max(0, lo_g), min(hi_g, cap_hi) + 1, 14)
        # Log grid info and ensure day 0 inclusion
        includes_day0 = np.any(grid == 0)
        logging.info(
            "Grid info | baseline_a1c_category=%s | lo_g=%s | hi_g=%s | includes_day0=%s | grid_first=%s | grid_last=%s | grid_size=%d",
            cat,
            lo_g,
            hi_g,
            bool(includes_day0),
            int(grid[0]) if grid.size > 0 else None,
            int(grid[-1]) if grid.size > 0 else None,
            int(grid.size),
        )
        if not includes_day0:
            logging.warning("Day 0 missing from grid for %s; removing clamping and forcing grid to start at 0.", cat)
            grid = days_grid_base

        # Predictions from pooled model for each group
        mean_sema, lo_sema, hi_sema, Xs = _predict_grid_pooled(res, design_info, grid, ref_vals, cat, "sema-only")
        mean_tirz, lo_tirz, hi_tirz, Xt = _predict_grid_pooled(res, design_info, grid, ref_vals, cat, "tirz-only")

        # Anchor at Day 0 for BOTH outcomes
        dd0 = 0
        Xs0 = build_design_matrices([design_info], pd.DataFrame([{**ref_vals, "baseline_a1c_category": cat, "glp1_group2": "sema-only", "days_from_baseline": dd0}]))[0]
        Xt0 = build_design_matrices([design_info], pd.DataFrame([{**ref_vals, "baseline_a1c_category": cat, "glp1_group2": "tirz-only", "days_from_baseline": dd0}]))[0]
        mu_s0 = float(np.dot(np.asarray(Xs0)[0], np.asarray(res.params, dtype=float)))
        mu_t0 = float(np.dot(np.asarray(Xt0)[0], np.asarray(res.params, dtype=float)))

        # Center curves
        mean_sema_centered = mean_sema - mu_s0
        mean_tirz_centered = mean_tirz - mu_t0
        lo_sema_centered = lo_sema - mu_s0
        hi_sema_centered = hi_sema - mu_s0
        lo_tirz_centered = lo_tirz - mu_t0
        hi_tirz_centered = hi_tirz - mu_t0

        # Difference and CI from pooled covariance
        diff_mean_raw, diff_lo_raw, diff_hi_raw = _compute_diff_mean_ci_from_design(res, Xs, Xt)
        # Anchor difference at Day 0
        diff0_arr, _, _ = _compute_diff_mean_ci_from_design(res, np.asarray(Xs0), np.asarray(Xt0))
        diff0 = float(np.asarray(diff0_arr).ravel()[0])
        diff_mean_centered = diff_mean_raw - diff0
        diff_lo_centered = diff_lo_raw - diff0
        diff_hi_centered = diff_hi_raw - diff0

        # Plots (centered for both outcomes per request)
        ylabel = "Percent Weight Change" if args.outcome == "weight" else "Absolute A1C Change"
        y_limits = (-15, 5) if args.outcome == "weight" else None
        title = f"{args.outcome.upper()} trajectories by GLP-1 group | Baseline A1C: {cat}"
        _plot_trajectories(
            grid,
            mean_sema_centered,
            mean_tirz_centered,
            (lo_sema_centered, hi_sema_centered),
            (lo_tirz_centered, hi_tirz_centered),
            title,
            ylabel + " (Centered to Day 0)",
            os.path.join(out_cat_dir, "predictive_trajectories.png"),
            y_limits=y_limits,
        )
        _plot_difference(
            grid,
            diff_mean_centered,
            diff_lo_centered,
            diff_hi_centered,
            f"Difference (Tirz - Sema) | Baseline A1C: {cat}",
            ylabel + " (Centered to Day 0)",
            os.path.join(out_cat_dir, "difference_trajectory.png"),
            y_limits=y_limits,
        )
        _plot_difference(
            grid,
            diff_mean_raw,
            diff_lo_raw,
            diff_hi_raw,
            f"Difference (Tirz - Sema) Uncentered | Baseline A1C: {cat}",
            ylabel,
            os.path.join(out_cat_dir, "difference_trajectory_uncentered.png"),
            y_limits=y_limits,
        )

        # Save per-group prediction CSVs (include centered columns)
        pd.DataFrame({
            "baseline_a1c_category": cat,
            "day": grid,
            "mean": mean_sema,
            "ci_lower": lo_sema,
            "ci_upper": hi_sema,
            "mean_centered": mean_sema_centered,
            "ci_lower_centered": lo_sema_centered,
            "ci_upper_centered": hi_sema_centered,
        }).to_csv(os.path.join(out_sema_dir, "predicted_trajectory.csv"), index=False)
        pd.DataFrame({
            "baseline_a1c_category": cat,
            "day": grid,
            "mean": mean_tirz,
            "ci_lower": lo_tirz,
            "ci_upper": hi_tirz,
            "mean_centered": mean_tirz_centered,
            "ci_lower_centered": lo_tirz_centered,
            "ci_upper_centered": hi_tirz_centered,
        }).to_csv(os.path.join(out_tirz_dir, "predicted_trajectory.csv"), index=False)

        # Save difference trajectory CSV (include centered)
        pd.DataFrame({
            "baseline_a1c_category": cat,
            "day": grid,
            "diff_mean": diff_mean_raw,
            "diff_ci_lower": diff_lo_raw,
            "diff_ci_upper": diff_hi_raw,
            "diff_mean_centered": diff_mean_centered,
            "diff_ci_lower_centered": diff_lo_centered,
            "diff_ci_upper_centered": diff_hi_centered,
        }).to_csv(os.path.join(out_cat_dir, "difference_trajectory.csv"), index=False)

        # Collect forest rows for requested days (centered values)
        for d in forest_days:
            if d < grid[0] or d > grid[-1]:
                continue
            # Build 1-row designs for the exact day
            Xs_d = build_design_matrices([design_info], pd.DataFrame([{**ref_vals, "baseline_a1c_category": cat, "glp1_group2": "sema-only", "days_from_baseline": int(d)}]))[0]
            Xt_d = build_design_matrices([design_info], pd.DataFrame([{**ref_vals, "baseline_a1c_category": cat, "glp1_group2": "tirz-only", "days_from_baseline": int(d)}]))[0]
            ms, ls, hs = _compute_mean_ci(res, np.asarray(Xs_d))
            mt, lt, ht = _compute_mean_ci(res, np.asarray(Xt_d))
            # Center using day-0 values
            ms_c = float(ms[0] - mu_s0)
            ls_c = float(ls[0] - mu_s0)
            hs_c = float(hs[0] - mu_s0)
            mt_c = float(mt[0] - mu_t0)
            lt_c = float(lt[0] - mu_t0)
            ht_c = float(ht[0] - mu_t0)
            # Difference (centered by subtracting day-0 difference)
            dm, dl, dh = _compute_diff_mean_ci_from_design(res, np.asarray(Xs_d), np.asarray(Xt_d))
            dm_c = float(dm[0] - diff0)
            dl_c = float(dl[0] - diff0)
            dh_c = float(dh[0] - diff0)
            forest_rows[d].append({
                "baseline_a1c_category": str(cat),
                "day": int(d),
                "sema_mean": ms_c, "sema_ci_lower": ls_c, "sema_ci_upper": hs_c,
                "tirz_mean": mt_c, "tirz_ci_lower": lt_c, "tirz_ci_upper": ht_c,
                "diff_mean": dm_c, "diff_ci_lower": dl_c, "diff_ci_upper": dh_c,
            })

        # Diagnostics: raw vs predicted for a selected A1C category (A1C outcome preferred)
        if args.debug_a1c_category is not None and str(cat) == args.debug_a1c_category:
            yvar = "pct_weight_change" if args.outcome == "weight" else "abs_a1c_change"
            max_day = int(args.truncate_days) if args.truncate_days is not None else 730
            raw_map = _compute_binned_empirical(sub, "glp1_user_group", yvar, bin_size=14, max_day=max_day)
            # Uncentered overlay
            _plot_raw_vs_predicted(
                grid,
                mean_sema, mean_tirz, raw_map,
                f"Raw vs Predicted (uncentered) | {args.outcome.upper()} | Baseline A1C: {cat}",
                ("Percent Weight Change" if args.outcome == "weight" else "Absolute A1C Change"),
                os.path.join(out_cat_dir, "diag_raw_vs_predicted_uncentered.png"),
                centered=False,
            )
            # Center raw using day 0 bin means if available
            anchor_s = None
            anchor_t = None
            if raw_map.get("sema-only") is not None:
                r0 = raw_map["sema-only"][raw_map["sema-only"]["day"] == 0]
                if not r0.empty and pd.notna(r0.iloc[0]["mean"]):
                    anchor_s = float(r0.iloc[0]["mean"])  # raw day-0 mean
                else:
                    anchor_s = float(mu_s0)
            if raw_map.get("tirz-only") is not None:
                r0 = raw_map["tirz-only"][raw_map["tirz-only"]["day"] == 0]
                if not r0.empty and pd.notna(r0.iloc[0]["mean"]):
                    anchor_t = float(r0.iloc[0]["mean"])  # raw day-0 mean
                else:
                    anchor_t = float(mu_t0)
            _plot_raw_vs_predicted(
                grid,
                mean_sema_centered, mean_tirz_centered, raw_map,
                f"Raw vs Predicted (centered at Day 0) | {args.outcome.upper()} | Baseline A1C: {cat}",
                ("Percent Weight Change" if args.outcome == "weight" else "Absolute A1C Change"),
                os.path.join(out_cat_dir, "diag_raw_vs_predicted_centered.png"),
                centered=True,
                anchor_sema=anchor_s,
                anchor_tirz=anchor_t,
            )


    # After category loop, write forest CSVs and plots
    forests_dir = os.path.join(outdir, "forests")
    os.makedirs(forests_dir, exist_ok=True)
    xlim = ( -15, 5 ) if args.outcome == "weight" else (args.xlim_min, args.xlim_max) if (args.xlim_min is not None and args.xlim_max is not None) else None
    for d, rows in forest_rows.items():
        if not rows:
            continue
        fdf = pd.DataFrame(rows)
        # Enforce canonical A1C order for labels
        fdf["order"] = fdf["baseline_a1c_category"].apply(lambda x: A1C_ORDER.index(x) if x in A1C_ORDER else 999)
        fdf = fdf.sort_values("order")
        fdf.to_csv(os.path.join(forests_dir, f"forest_day_{d}.csv"), index=False)
        labels = [f"{c}" for c in fdf["baseline_a1c_category"].tolist()]
        # Difference forest with right-side text
        _forest_plot(
            days=None,
            est=fdf["diff_mean"].to_numpy(),
            lo=fdf["diff_ci_lower"].to_numpy(),
            hi=fdf["diff_ci_upper"].to_numpy(),
            labels=labels,
            title=f"Difference (Tirz - Sema) at Day {d}",
            xlabel=("Percent Weight Change (Centered)" if args.outcome == "weight" else "Absolute A1C Change (Centered)"),
            out_path=os.path.join(forests_dir, f"forest_day_{d}_difference.png"),
            xlim=xlim,
        )
        # Both-medications forest with right-side text
        _forest_plot_two_groups(
            est_sema=fdf["sema_mean"].to_numpy(),
            lo_sema=fdf["sema_ci_lower"].to_numpy(),
            hi_sema=fdf["sema_ci_upper"].to_numpy(),
            est_tirz=fdf["tirz_mean"].to_numpy(),
            lo_tirz=fdf["tirz_ci_lower"].to_numpy(),
            hi_tirz=fdf["tirz_ci_upper"].to_numpy(),
            labels=labels,
            title=f"Predicted by Group at Day {d}",
            xlabel=("Percent Weight Change (Centered)" if args.outcome == "weight" else "Absolute A1C Change (Centered)"),
            out_path=os.path.join(forests_dir, f"forest_day_{d}_both_groups.png"),
            xlim=xlim,
        )

    # Debug: report counts per day before combined plots
    try:
        logging.info("Forest row counts: " + ", ".join([f"{d}={len(forest_rows.get(d, []))}" for d in forest_days]))
    except Exception as exc:
        logging.debug(
            "could not log the forest row counts (%s); diagnostic only",
            exc,
        )

    # Combined forest across all requested days (stacked with spacers)
    combined_csv = []
    labels_all = []
    diff_est_all = []
    diff_lo_all = []
    diff_hi_all = []
    sema_est_all = []
    sema_lo_all = []
    sema_hi_all = []
    tirz_est_all = []
    tirz_lo_all = []
    tirz_hi_all = []
    for idx_d, d in enumerate(forest_days):
        rows = forest_rows.get(d, [])
        if not rows:
            continue
        fdf = pd.DataFrame(rows)
        # Sort by canonical A1C order
        fdf["order"] = fdf["baseline_a1c_category"].apply(lambda x: A1C_ORDER.index(x) if x in A1C_ORDER else 999)
        fdf = fdf.sort_values("order")
        combined_csv.append(fdf)
        for _, r in fdf.iterrows():
            labels_all.append(f"{r['baseline_a1c_category']} (Day {d})")
            diff_est_all.append(float(r["diff_mean"]))
            diff_lo_all.append(float(r["diff_ci_lower"]))
            diff_hi_all.append(float(r["diff_ci_upper"]))
            sema_est_all.append(float(r["sema_mean"]))
            sema_lo_all.append(float(r["sema_ci_lower"]))
            sema_hi_all.append(float(r["sema_ci_upper"]))
            tirz_est_all.append(float(r["tirz_mean"]))
            tirz_lo_all.append(float(r["tirz_ci_lower"]))
            tirz_hi_all.append(float(r["tirz_ci_upper"]))
        # spacer
        labels_all.append(" ")
        diff_est_all.append(np.nan)
        diff_lo_all.append(np.nan)
        diff_hi_all.append(np.nan)
        sema_est_all.append(np.nan)
        sema_lo_all.append(np.nan)
        sema_hi_all.append(np.nan)
        tirz_est_all.append(np.nan)
        tirz_lo_all.append(np.nan)
        tirz_hi_all.append(np.nan)
    if combined_csv:
        combined_df = pd.concat(combined_csv, ignore_index=True)
        combined_df.to_csv(os.path.join(forests_dir, "forest_all_days_combined.csv"), index=False)
        # Build labels and arrays
        labels = labels_all
        # Difference combined (mask NaNs so spacers render as blanks) with right-side text
        est = np.array(diff_est_all, dtype=float)
        lo = np.array(diff_lo_all, dtype=float)
        hi = np.array(diff_hi_all, dtype=float)
        mask = np.isfinite(est) & np.isfinite(lo) & np.isfinite(hi)
        fig = plt.figure(figsize=(12, 0.5 * len(labels) + 2))
        gs = fig.add_gridspec(1, 2, width_ratios=[4.5, 2.2])
        ax = fig.add_subplot(gs[0, 0])
        ax_txt = fig.add_subplot(gs[0, 1])
        y = np.arange(len(labels))
        ax.errorbar(est[mask], y[mask], xerr=[est[mask] - lo[mask], hi[mask] - est[mask]], fmt="o", color="black", capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.axvline(0, color="grey", linestyle="-")
        ax.set_xlabel(("Percent Weight Change (Centered)" if args.outcome == "weight" else "Absolute A1C Change (Centered)"))
        ax.set_title("Difference (Tirz - Sema) at 90 / 180 / 270 / 365 / 450 / 548 Days")
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.invert_yaxis()
        # Right-side text
        ax_txt.axis("off")
        ax_txt.set_title("Estimate (95% CI)")
        if len(y) > 0:
            ax_txt.set_ylim(min(y) - 0.5, max(y) + 0.5)
        for yi, m, l, h in zip(y, est, lo, hi):
            if np.isfinite(m) and np.isfinite(l) and np.isfinite(h):
                ax_txt.text(0.0, yi, f"{float(m):.2f} ({float(l):.2f}, {float(h):.2f})", va="center", ha="left", fontsize=8)
        ax_txt.set_xlim(0, 1)
        fig.tight_layout()
        fig.savefig(os.path.join(forests_dir, "forest_all_days_difference.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Both groups combined with right-side text
        est_s = np.array(sema_est_all, dtype=float)
        lo_s = np.array(sema_lo_all, dtype=float)
        hi_s = np.array(sema_hi_all, dtype=float)
        est_t = np.array(tirz_est_all, dtype=float)
        lo_t = np.array(tirz_lo_all, dtype=float)
        hi_t = np.array(tirz_hi_all, dtype=float)
        mask_s = np.isfinite(est_s) & np.isfinite(lo_s) & np.isfinite(hi_s)
        mask_t = np.isfinite(est_t) & np.isfinite(lo_t) & np.isfinite(hi_t)
        fig = plt.figure(figsize=(12, 0.5 * len(labels) + 2))
        gs = fig.add_gridspec(1, 2, width_ratios=[4.5, 2.2])
        ax = fig.add_subplot(gs[0, 0])
        ax_txt = fig.add_subplot(gs[0, 1])
        ax.errorbar(est_s[mask_s], y[mask_s] + 0.1, xerr=[est_s[mask_s] - lo_s[mask_s], hi_s[mask_s] - est_s[mask_s]], fmt="o", color="#1f77b4", capsize=3, label="Semaglutide")
        ax.errorbar(est_t[mask_t], y[mask_t] - 0.1, xerr=[est_t[mask_t] - lo_t[mask_t], hi_t[mask_t] - est_t[mask_t]], fmt="o", color="#ff7f0e", capsize=3, label="Tirzepatide")
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.axvline(0, color="grey", linestyle="-")
        ax.set_xlabel(("Percent Weight Change (Centered)" if args.outcome == "weight" else "Absolute A1C Change (Centered)"))
        ax.set_title("Predicted by Group at 90 / 180 / 270 / 365 / 450 / 548 Days")
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.legend(loc="best")
        ax.invert_yaxis()
        # Right-side text
        ax_txt.axis("off")
        ax_txt.set_title("Estimate (95% CI)")
        if len(y) > 0:
            ax_txt.set_ylim(min(y) - 0.5, max(y) + 0.5)
        for yi, ms, ls, hs, mt, lt, ht in zip(y, est_s, lo_s, hi_s, est_t, lo_t, hi_t):
            if np.isfinite(ms) and np.isfinite(ls) and np.isfinite(hs) and np.isfinite(mt) and np.isfinite(lt) and np.isfinite(ht):
                txt = f"Sema {float(ms):.2f} ({float(ls):.2f}, {float(hs):.2f}); Tirz {float(mt):.2f} ({float(lt):.2f}, {float(ht):.2f})"
                ax_txt.text(0.0, yi, txt, va="center", ha="left", fontsize=8)
        ax_txt.set_xlim(0, 1)
        fig.tight_layout()
        fig.savefig(os.path.join(forests_dir, "forest_all_days_both_groups.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

# Ensure the script runs when executed directly
if __name__ == "__main__":
    main()
