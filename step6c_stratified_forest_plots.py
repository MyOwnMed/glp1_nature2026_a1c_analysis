#!/usr/bin/env python3
"""
Step 6c: Stratified trajectories and forest plots (A1C outcome)

- For each covariate (sex, age group variants, race, GLP-1+metformin at baseline, BMI):
  - Fit a subgroup-specific GEE model with bs(days)*baseline_a1c_category interactions
    and other covariates except the stratification variable.
  - Generate Step 6-style trajectory plots (uncentered + centered) that overlay
    curves by baseline A1C category within each subgroup (e.g., Female-only A1C curves).
  - Generate Step 5-style forest plots at specific time points per subgroup.
  - Save predictions and contrasts CSVs per subgroup.
- Additionally, create combined plots where helpful.
"""

import os
import argparse
import logging

import numpy as np
import pandas as pd
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


# New: main-effects version (no interaction with baseline A1C)
def _fit_main_effects_stratum(df: pd.DataFrame, df_spline: int, exclude_var: str):
    bs_term = f"bs(days_from_baseline, df={df_spline})"
    formula = f"abs_a1c_change ~ {bs_term} + baseline_a1c_category"
    covariates = [
        "age_group",
        "gender",
        "baseline_bmi_final_category",
        "race",
        # 'metformin_with_glp1_baseline' removed per requirements
        "weight_change_med",
    ]
    covariates = filter_estimable(
        covariates, df, exclude=[exclude_var],
        context=f"step6c a1c, exclude_var={exclude_var}",
    )
    if covariates:
        formula += " + " + " + ".join(covariates)
    y, X = dmatrices(formula, df, return_type="dataframe")
    ids = df.loc[y.index, "patient_id"]
    model = GEE(y, X, groups=ids, family=Gaussian(), cov_struct=Independence())
    result = model.fit()
    return result, X.design_info


def configure_logging(level: str = "INFO"):
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Step 6c: Stratified trajectories and forest plots (A1C outcome)")
    p.add_argument(
        "--input-csv",
        default=os.path.join(
            "output",
            "step1_prepare_analysis_dataset_a1c",
            "analysis_ready_a1c_gap90.csv",
        ),
        help="Path to analysis-ready A1C CSV from Step 1 (A1C pipeline)",
    )
    p.add_argument(
        "--config-json",
        default=os.path.join("output", "step2_select_spline_df_a1c", "model_config_a1c.json"),
        help="Model config JSON from Step 2 (A1C, contains best df)",
    )
    p.add_argument(
        "--outdir",
        default=os.path.join("output", "step6c_stratified_by_covariates_a1c"),
        help="Base directory for stratified outputs (trajectories + forests)",
    )
    p.add_argument(
        "--outdir-main",
        default=os.path.join("output", "step6c_stratified_by_covariates_a1c"),
        help="Base directory for main-effects outputs (will write to subfolder 'main')",
    )
    p.add_argument(
        "--time-days",
        default="90,180,270,365,450,548,630,730",
        help="Comma-separated days from baseline for forest points",
    )
    p.add_argument(
        "--window-days",
        type=int,
        default=28,
        help=(
            "Half-window size in days for counting sample sizes "
            "(unique patients) around each time point"
        ),
    )
    p.add_argument(
        "--min-nobs",
        type=int,
        default=100,
        help="Minimum observations per stratum to fit model",
    )
    p.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    p.add_argument("--adherence-gap-days", type=int, default=None, help="Adherence gap for gap-specific subfolder")
    p.add_argument("--xmax-cap", type=float, default=None, help="Optional max x-axis limit to truncate extreme CIs")
    p.add_argument("--xmin-cap", type=float, default=None, help="Optional min x-axis limit to truncate extreme CIs")
    p.add_argument(
        "--max-days", type=int, default=548,
        help=(
            "Maximum days from baseline to include, capping time points at or below "
            "this (548 ~ 18 months). The paper's 18-month follow-up limit as a "
            "calendar figure; step8 states the same limit as 18 x 30 = 540 days and "
            "step1's 730-day --max-days is a wider outer bound on emitted data for "
            "the 730-day persistence sensitivity analysis. See README.md, "
            "'Follow-up horizon caps'."
        ),
    )
    # Fixed x-axis limits for A1C plots
    p.add_argument("--xlim-min", type=float, default=-4.0, help="Fixed x-axis minimum for A1C forest plots")
    p.add_argument("--xlim-max", type=float, default=1.0, help="Fixed x-axis maximum for A1C forest plots")
    return p.parse_args(argv)


STRAT_VARS = [
    ("gender", "Sex"),
    ("age_group", "Age Group"),
    ("age_group_20_39_vs_40_plus", "Age Group 20-39 vs 40+"),
    ("age_group_20_49_vs_50_plus", "Age Group 20-49 vs 50+"),
    ("baseline_bmi_final_category", "Baseline BMI Category"),
    ("race", "Race"),
    ("metformin_with_glp1_baseline", "GLP-1 + Metformin at Baseline"),
    ("glp1_user_group", "GLP-1 User Group"),
]

A1C_ORDER = [
    "Normal Glycemia",
    "Prediabetes",
    "Type 2 Diabetes",
    "Poorly Controlled Diabetes",
]


def _fit_combined_stratum(df: pd.DataFrame, df_spline: int, exclude_var: str):
    bs_term = f"bs(days_from_baseline, df={df_spline})"
    # Interact spline with baseline A1C category to get category-specific trajectories
    formula = f"abs_a1c_change ~ {bs_term} * baseline_a1c_category"
    # Include other covariates except the stratification variable
    covariates = [
        "age_group",
        "gender",
        "baseline_bmi_final_category",
        "race",
        # 'metformin_with_glp1_baseline' removed per requirements
        "weight_change_med",
    ]
    covariates = filter_estimable(
        covariates, df, exclude=[exclude_var],
        context=f"step6c a1c, exclude_var={exclude_var}",
    )
    if covariates:
        formula += " + " + " + ".join(covariates)

    y, X = dmatrices(formula, df, return_type="dataframe")
    ids = df.loc[y.index, "patient_id"]
    model = GEE(y, X, groups=ids, family=Gaussian(), cov_struct=Independence())
    result = model.fit()
    return result, X.design_info


def _pred_ci(result, xrow: np.ndarray, z: float = Z_CRIT):
    mean = float(np.dot(xrow, result.params))
    V = np.asarray(result.cov_params(), dtype=float)
    var = float(np.dot(xrow, np.dot(V, xrow)))
    var = max(var, 0.0)
    se = float(np.sqrt(var))
    return mean, mean - z * se, mean + z * se, se


def _counts_for_day_cat(df_in: pd.DataFrame, day: int, cat: str, window_days: int):
    dd = df_in[(df_in["baseline_a1c_category"] == cat) & (np.isfinite(df_in["days_from_baseline"]))]
    days = dd["days_from_baseline"].to_numpy()
    ids = dd["patient_id"].to_numpy()
    sel = (days >= day - window_days) & (days <= day + window_days)
    n_obs = int(np.count_nonzero(sel))
    n_unique = int(np.unique(ids[sel]).size) if n_obs > 0 else 0
    return n_obs, n_unique


def _sanitize(x: str) -> str:
    return str(x).replace(" ", "_").replace("/", "-")


def main(argv=None):  # noqa: C901
    args = parse_args(argv)
    configure_logging(args.log_level)
    
    outdir = args.outdir
    outdir_main = args.outdir_main
    # Gap-aware outdir routing (align with other steps)
    gap = getattr(args, "adherence_gap_days", None)
    if gap is None:
        try:
            import re
            m = re.search(r"gap[_]?(\d+)", str(args.input_csv))
            if m:
                gap = int(m.group(1))
        except Exception:
            gap = None
    if gap is not None:
        if "/gap_" not in outdir:
            outdir = os.path.join("output", f"gap_{gap}", os.path.basename(outdir))
        # Always place main-effects into a 'main' subfolder under the gap-aware base
        if "/gap_" not in outdir_main:
            outdir_main = os.path.join("output", f"gap_{gap}", os.path.basename(outdir_main))
        outdir_main = os.path.join(outdir_main, "main")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(outdir_main, exist_ok=True)
    # Write run marker
    with open(os.path.join(outdir, "RUN_OK.txt"), "w") as f:
        f.write("step6c_a1c started\n")

    # Rebind routed paths back to args for the rest of the script
    args.outdir = outdir
    args.outdir_main = outdir_main

    df = pd.read_csv(args.input_csv)
    for cat_col in [
        "gender",
        "baseline_a1c_category",
        "baseline_bmi_final_category",
        "race",
        "age_group",
        "glp1_user_group",
    ]:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype("category")
    # Fails loudly: this fixes the reference level for every contrast
    # (code-review item 8). See model_spec.enforce_a1c_order.
    df = enforce_a1c_order(df, order=A1C_ORDER, context="step6c_stratified_forest_plots")

    # Selected df from Step 2
    df_spline = load_spline_df(args.config_json)
    logging.info("Using spline df=%d", df_spline)

    times = [int(x.strip()) for x in args.time_days.split(",") if x.strip()]
    # Cap to requested horizon
    times = [d for d in times if d <= int(getattr(args, "max_days", 548))]
    a1c_cats = [c for c in A1C_ORDER if c in (list(df["baseline_a1c_category"].cat.categories) if "baseline_a1c_category" in df.columns and hasattr(df["baseline_a1c_category"], "cat") else [])]
    if not a1c_cats:
        a1c_cats = A1C_ORDER
    ref_cat = "Normal Glycemia" if "Normal Glycemia" in a1c_cats else (a1c_cats[0] if a1c_cats else "")

    passes = [
        (args.outdir, _fit_combined_stratum, "two-way"),
        (args.outdir_main, _fit_main_effects_stratum, "main-effects"),
    ]

    for curr_outdir, fit_fun, mode_label in passes:
        os.makedirs(curr_outdir, exist_ok=True)
        # Pre-create per-variable directories to guarantee folder structure
        for var, var_label in STRAT_VARS:
            var_base_outdir = os.path.join(curr_outdir, f"by_{var}")
            forests_dir = os.path.join(var_base_outdir, "forests")
            traj_dir = os.path.join(var_base_outdir, "trajectories")
            traj_unc = os.path.join(traj_dir, "uncentered")
            traj_ctr = os.path.join(traj_dir, "centered")
            traj_grp = os.path.join(traj_dir, "grouped")
            for d in (var_base_outdir, forests_dir, traj_dir, traj_unc, traj_ctr, traj_grp):
                os.makedirs(d, exist_ok=True)
            with open(os.path.join(var_base_outdir, "README.txt"), "w") as rf:
                rf.write(f"Stratified outputs for {var_label} ({mode_label}). Trajectories and forests generated by step6c.\n")
        combined_records = []
        logging.info("Generating stratified outputs (%s) into %s", mode_label, curr_outdir)

        for var, var_label in STRAT_VARS:
            var_base_outdir = os.path.join(curr_outdir, f"by_{var}")
            forests_dir = os.path.join(var_base_outdir, "forests")
            traj_dir = os.path.join(var_base_outdir, "trajectories")
            traj_unc = os.path.join(traj_dir, "uncentered")
            traj_ctr = os.path.join(traj_dir, "centered")
            traj_grp = os.path.join(traj_dir, "grouped")
            # Determine subgroups; skip NaN/Unknown
            if var not in df.columns:
                logging.warning("Column '%s' missing; skipping", var)
                continue
            if hasattr(df[var], "cat"):
                subgroups = [c for c in df[var].cat.categories if pd.notna(c)]
            else:
                subgroups = [v for v in df[var].dropna().unique().tolist()]
                try:
                    subgroups = sorted(subgroups)
                except Exception as exc:
                    logging.debug(
                        "subgroup labels are not sortable (%s); using the order they appear in the data",
                        exc,
                    )
            logging.info("Stratifying forests by %s with %d groups (%s)", var_label, len(subgroups), mode_label)

            # Collect per-variable records for combined plot (days 365/548)
            var_records = [] if mode_label == "two-way" else None

            for subgroup in subgroups:
                sub_df = df[df[var] == subgroup].copy()
                n_obs = len(sub_df)
                n_people = sub_df["patient_id"].nunique()
                if n_obs < args.min_nobs:
                    logging.info(
                        "Skip %s=%s (Obs=%d, N=%d): insufficient data",
                        var_label,
                        subgroup,
                        n_obs,
                        n_people,
                    )
                    continue

                logging.info(
                    "Fitting stratified GEE (forest, %s) for %s=%s (Obs=%d, N=%d)",
                    mode_label,
                    var_label,
                    subgroup,
                    n_obs,
                    n_people,
                )

                try:
                    res, design_info = fit_fun(sub_df, df_spline, exclude_var=var)
                except Exception as e:
                    logging.warning("Fit failed for %s=%s: %s", var_label, subgroup, e)
                    continue

                # Determine supported time points
                finite_days = pd.to_numeric(sub_df["days_from_baseline"], errors="coerce")
                finite_days = finite_days[np.isfinite(finite_days)]
                if finite_days.empty:
                    logging.info("No finite day values for %s=%s; skipping", var_label, subgroup)
                    continue
                d_min = float(np.nanmin(finite_days))
                d_max = float(np.nanmax(finite_days))
                times_in_support = [d for d in times if (d_min <= d <= d_max)]
                if not times_in_support:
                    logging.info(
                        "No requested time points within support for %s=%s (range %.1f-%.1f); skipping",
                        var_label,
                        subgroup,
                        d_min,
                        d_max,
                    )
                    continue

                # Mode values for covariates used in predictions
                mode_vals = {}
                for c in [
                    "age_group",
                    "gender",
                    "baseline_bmi_final_category",
                    "race",
                    # 'metformin_with_glp1_baseline' removed
                    "weight_change_med",
                ]:
                    if c in sub_df.columns:
                        m = sub_df[c].mode()
                        if not m.empty:
                            mode_vals[c] = m.iloc[0]
                        elif hasattr(sub_df[c], "cat"):
                            mode_vals[c] = sub_df[c].cat.categories[0]

                if mode_label == "main-effects":
                    # Compute subgroup-level predictions at supported days (no baseline A1C stratification)
                    pred_rows = []
                    for d in times_in_support:
                        pred_df = pd.DataFrame({"days_from_baseline": [d]})
                        for c, v in mode_vals.items():
                            if c in sub_df.columns:
                                pred_df[c] = [v]
                                if isinstance(sub_df[c].dtype, pd.CategoricalDtype):
                                    pred_df[c] = pd.Categorical(pred_df[c], categories=sub_df[c].cat.categories)
                        # Include baseline A1C category at mode to get average estimate
                        if "baseline_a1c_category" in sub_df.columns:
                            m = sub_df["baseline_a1c_category"].mode()
                            if not m.empty:
                                pred_df["baseline_a1c_category"] = [m.iloc[0]]
                                if isinstance(sub_df["baseline_a1c_category"].dtype, pd.CategoricalDtype):
                                    pred_df["baseline_a1c_category"] = pd.Categorical(
                                        pred_df["baseline_a1c_category"],
                                        categories=sub_df["baseline_a1c_category"].cat.categories,
                                    )
                        Xp = build_design_matrices([design_info], pred_df)[0]
                        Xp = np.asarray(Xp)[0]
                        mean, lo, hi, se = _pred_ci(res, Xp)
                        # Windowed counts within this subgroup for annotation
                        days = pd.to_numeric(sub_df["days_from_baseline"], errors="coerce").to_numpy()
                        ids = sub_df["patient_id"].to_numpy()
                        sel = (days >= d - args.window_days) & (days <= d + args.window_days) & np.isfinite(days)
                        n_unique_w = int(np.unique(ids[sel]).size) if np.count_nonzero(sel) > 0 else 0
                        pred_rows.append({"day": d, "subgroup": subgroup, "pred": mean, "ci_low": lo, "ci_high": hi, "n_unique_window": n_unique_w, "covariate": var_label})
                    # Save subgroup predictions (optional)
                    subpred_path = os.path.join(forests_dir, _sanitize(subgroup), "subgroup_predictions_main.csv")
                    os.makedirs(os.path.dirname(subpred_path), exist_ok=True)
                    pd.DataFrame(pred_rows).to_csv(subpred_path, index=False)
                    # Accumulate only selected days (12m and ~18m) for global plot
                    selected_12 = 365
                    possible_18 = [548, 547, 540]
                    days_avail = sorted([r["day"] for r in pred_rows])
                    sel18 = next((d for d in possible_18 if d in days_avail), None)
                    for d in [selected_12] + ([sel18] if sel18 is not None else []):
                        rr = [r for r in pred_rows if r["day"] == d]
                        if not rr:
                            continue
                        r0 = rr[0]
                        combined_records.append(r0)
                    # Skip per-subgroup A1C-category plots in main-effects
                    continue

                # Two-way interactions branch (retain existing A1C-category stratification)
                pred_rows = []
                contrast_rows = []
                for d in times_in_support:
                    for cat in a1c_cats:
                        pred_df = pd.DataFrame({"days_from_baseline": [d], "baseline_a1c_category": [cat]})
                        for c, v in mode_vals.items():
                            if c in sub_df.columns:
                                pred_df[c] = [v]
                                if isinstance(sub_df[c].dtype, pd.CategoricalDtype):
                                    pred_df[c] = pd.Categorical(pred_df[c], categories=sub_df[c].cat.categories)
                        if isinstance(sub_df["baseline_a1c_category"].dtype, pd.CategoricalDtype):
                            pred_df["baseline_a1c_category"] = pd.Categorical(
                                pred_df["baseline_a1c_category"],
                                categories=sub_df["baseline_a1c_category"].cat.categories,
                            )
                        Xp = build_design_matrices([design_info], pred_df)[0]
                        Xp = np.asarray(Xp)[0]
                        mean, lo, hi, se = _pred_ci(res, Xp)
                        n_obs_w, n_unique_w = _counts_for_day_cat(sub_df, d, cat, args.window_days)
                        pred_rows.append({"day": d, "a1c_group": cat, "pred": mean, "ci_low": lo, "ci_high": hi, "se": se, "n_obs_window": n_obs_w, "n_unique_window": n_unique_w})
                    # Pairwise contrasts vs reference
                    for cat in a1c_cats:
                        if cat == ref_cat:
                            continue
                        df_ref = pd.DataFrame(
                            {"days_from_baseline": [d], "baseline_a1c_category": [ref_cat]}
                        )
                        df_cat = pd.DataFrame(
                            {"days_from_baseline": [d], "baseline_a1c_category": [cat]}
                        )
                        for c, v in mode_vals.items():
                            for P in (df_ref, df_cat):
                                if c in sub_df.columns:
                                    P[c] = [v]
                                    if isinstance(sub_df[c].dtype, pd.CategoricalDtype):
                                        P[c] = pd.Categorical(P[c], categories=sub_df[c].cat.categories)
                        for P in (df_ref, df_cat):
                            if isinstance(sub_df["baseline_a1c_category"].dtype, pd.CategoricalDtype):
                                P["baseline_a1c_category"] = pd.Categorical(
                                    P["baseline_a1c_category"],
                                    categories=sub_df["baseline_a1c_category"].cat.categories,
                                )
                        X_ref = np.asarray(build_design_matrices([design_info], df_ref)[0])[0]
                        X_cat = np.asarray(build_design_matrices([design_info], df_cat)[0])[0]
                        delta = X_cat - X_ref
                        diff, lo, hi, se = _pred_ci(res, delta)
                        contrast_rows.append(
                            {
                                "day": d,
                                "ref": ref_cat,
                                "a1c_group": cat,
                                "diff": diff,
                                "ci_low": lo,
                                "ci_high": hi,
                                "se": se,
                            }
                        )

                preds = pd.DataFrame(pred_rows)
                contrasts = pd.DataFrame(contrast_rows)
                subgroup_dir = os.path.join(forests_dir, _sanitize(subgroup))
                os.makedirs(subgroup_dir, exist_ok=True)
                preds.to_csv(os.path.join(subgroup_dir, "forest_predictions.csv"), index=False)
                contrasts.to_csv(os.path.join(subgroup_dir, "forest_contrasts_vs_ref.csv"), index=False)
                # Per-subgroup overview (sample size)
                pd.DataFrame([
                    {"metric": "unique_people", "value": int(n_people)},
                    {"metric": "observations", "value": int(n_obs)},
                ]).to_csv(os.path.join(subgroup_dir, 'gee_model_overview.csv'), index=False)

                # New: Trajectory plots by baseline A1C within this subgroup (Step 6 style)
                if mode_label == "two-way":
                    try:
                        # Build prediction grid
                        observed_max = int(np.nanmax(sub_df["days_from_baseline"])) if np.isfinite(sub_df["days_from_baseline"]).any() else 366
                        grid_max = int(min(getattr(args, "max_days", 548), observed_max))
                        days_grid = np.arange(0, grid_max + 1, 14)
                        if 0 not in days_grid:
                            days_grid = np.sort(np.append(days_grid, 0))
                        # Mode values for other covariates
                        mode_vals = {}
                        for c in ["age_group", "gender", "baseline_bmi_final_category", "race", "weight_change_med"]:
                            if c in sub_df.columns:
                                m = sub_df[c].mode()
                                if not m.empty:
                                    mode_vals[c] = m.iloc[0]
                                elif hasattr(sub_df[c], "cat"):
                                    mode_vals[c] = sub_df[c].cat.categories[0]
                        overlay = []
                        for cat in a1c_cats:
                            pred_df = pd.DataFrame({"days_from_baseline": days_grid, "baseline_a1c_category": cat})
                            for c, v in mode_vals.items():
                                if c in sub_df.columns:
                                    pred_df[c] = v
                                    if hasattr(sub_df[c], "cat"):
                                        pred_df[c] = pd.Categorical(pred_df[c], categories=sub_df[c].cat.categories)
                            if hasattr(sub_df["baseline_a1c_category"], "cat"):
                                pred_df["baseline_a1c_category"] = pd.Categorical(pred_df["baseline_a1c_category"], categories=sub_df["baseline_a1c_category"].cat.categories)
                            Xp = build_design_matrices([design_info], pred_df)[0]
                            Xp = np.asarray(Xp)
                            mean_u = np.asarray(res.predict(Xp), dtype=float)
                            anchor = mean_u[0]
                            mean_c = mean_u - anchor
                            overlay.append((cat, days_grid, mean_u, mean_c))
                        # Plot uncentered
                        plt.figure(figsize=(10, 6))
                        for cat, xg, mu, _ in overlay:
                            v = np.isfinite(mu)
                            if np.any(v):
                                plt.plot(xg[v], mu[v], label=f"{cat}", linewidth=2)
                        plt.xlabel("Days from Baseline")
                        plt.ylabel("Absolute A1C Change")
                        plt.title(f"{var_label}: {subgroup} — A1C Trajectories (Uncentered)\nstarting sample n = {int(n_people)}")
                        plt.axhline(0, color="black", linestyle="--", linewidth=1)
                        plt.xlim(0, int(getattr(args, "max_days", 548)))
                        plt.legend()
                        plt.grid(True, linestyle="--", alpha=0.7)
                        plt.tight_layout()
                        out_path = os.path.join(traj_unc, f"trajectory_by_a1c_{var}_{str(subgroup).replace(' ', '_')}.png")
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        plt.savefig(out_path, dpi=150)
                        plt.close()
                        # Plot centered
                        plt.figure(figsize=(10, 6))
                        for cat, xg, _, mc in overlay:
                            v = np.isfinite(mc)
                            if np.any(v):
                                mc0 = mc.copy()
                                if 0 in xg:
                                    idx0 = int(np.where(xg == 0)[0][0])
                                    mc0[idx0] = 0.0
                                plt.plot(xg[v], mc0[v], label=f"{cat}", linewidth=2)
                        plt.xlabel("Days from Baseline")
                        plt.ylabel("Absolute A1C Change (Centered to Day 0)")
                        plt.title(f"{var_label}: {subgroup} — A1C Trajectories\nstarting sample n = {int(n_people)}")
                        plt.axhline(0, color="black", linestyle="--", linewidth=1)
                        plt.xlim(0, int(getattr(args, "max_days", 548)))
                        plt.legend()
                        plt.grid(True, linestyle="--", alpha=0.7)
                        plt.tight_layout()
                        out_path = os.path.join(traj_ctr, f"trajectory_by_a1c_{var}_{str(subgroup).replace(' ', '_')}.png")
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        plt.savefig(out_path, dpi=150)
                        plt.close()
                        # Grouped overlay (uncentered)
                        plt.figure(figsize=(12, 7))
                        for cat, xg, mu, _ in overlay:
                            v = np.isfinite(mu)
                            if np.any(v):
                                plt.plot(xg[v], mu[v], label=f"{cat}", linewidth=2)
                        plt.xlabel("Days from Baseline")
                        plt.ylabel("Absolute A1C Change")
                        plt.title(f"{var_label}: {subgroup} — Grouped A1C Trajectories (Uncentered)")
                        plt.axhline(0, color="black", linestyle="--", linewidth=1)
                        plt.legend()
                        plt.grid(True, linestyle="--", alpha=0.7)
                        plt.tight_layout()
                        out_path = os.path.join(traj_grp, f"grouped_trajectory_by_a1c_{var}_{str(subgroup).replace(' ', '_')}.png")
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        plt.savefig(out_path, dpi=150)
                        plt.close()
                    except Exception as e:
                        logging.warning("Trajectory generation failed for %s=%s: %s", var_label, subgroup, e)

                # Accumulate gender-specific records (days 365 and ~548)
                if mode_label == "two-way" and var_records is not None:
                    selected_12 = 365
                    possible_18 = [548, 547, 540]
                    day_values = sorted(preds["day"].unique().tolist())
                    selected_18 = next((dd for dd in possible_18 if dd in day_values), None)
                    for dd in [selected_12] + ([selected_18] if selected_18 is not None else []):
                        for cat in A1C_ORDER:
                            row = preds[(preds["day"] == dd) & (preds["a1c_group"] == cat)]
                            if row.empty:
                                continue
                            r = row.iloc[0]
                            var_records.append({
                                "subgroup": str(subgroup),
                                "a1c_group": cat,
                                "day": int(dd),
                                "pred": float(r["pred"]),
                                "ci_low": float(r["ci_low"]),
                                "ci_high": float(r["ci_high"]),
                                "n_unique_window": int(r["n_unique_window"]),
                            })

                # Per-day forest plots — fixed x-limits
                for d in times_in_support:
                    subp = preds[preds["day"] == d].copy()
                    # Sort strictly by the canonical A1C order
                    subp["order"] = subp["a1c_group"].apply(lambda x: A1C_ORDER.index(x) if x in A1C_ORDER else 999)
                    subp = subp.sort_values("order")
                    finite_mask = np.isfinite(subp["pred"]) & np.isfinite(subp["ci_low"]) & np.isfinite(subp["ci_high"]) 
                    subp = subp[finite_mask]
                    if subp.empty:
                        logging.info("No finite predictions for %s=%s at day %d; skipping plot", var_label, subgroup, d)
                        continue
                    y = np.arange(len(subp))
                    labels = list(subp["a1c_group"])
                    texts = [
                        f"{row['pred']:.2f} ({row['ci_low']:.2f}, {row['ci_high']:.2f})  n={int(row['n_unique_window'])}"
                        for _, row in subp.iterrows()
                    ]

                    fig = plt.figure(figsize=(12, 6))
                    gs = fig.add_gridspec(1, 2, width_ratios=[4.5, 1.8])
                    ax = fig.add_subplot(gs[0, 0])
                    ax_txt = fig.add_subplot(gs[0, 1])

                    ax.errorbar(
                        subp["pred"],
                        y,
                        xerr=[subp["pred"] - subp["ci_low"], subp["ci_high"] - subp["pred"]],
                        fmt="o",
                        capsize=3,
                    )
                    ax.set_yticks(y)
                    ax.set_yticklabels(labels)
                    ax.axvline(0, color="black", linestyle="-", linewidth=1)
                    ax.set_xlabel("Predicted Absolute A1C Change")
                    ax.set_title(f"{var_label}: {subgroup} — A1C (Day {d}) [{mode_label}]")
                    ax.grid(False)

                    x_min = float((subp["ci_low"]).min())
                    x_max = float((subp["ci_high"]).max())
                    x_range = x_max - x_min if x_max > x_min else 1.0
                    left_lim = x_min - 0.1 * x_range
                    right_lim = max(x_max + 0.1 * x_range, 0 + 0.3 * x_range)
                    if getattr(args, "xmax_cap", None) is not None:
                        right_lim = min(right_lim, float(args.xmax_cap))
                    if getattr(args, "xmin_cap", None) is not None:
                        left_lim = max(left_lim, float(args.xmin_cap))
                    # Replace dynamic x-limits with fixed limits for A1C plots
                    ax.set_xlim(args.xlim_min, args.xlim_max)

                    ax_txt.axis("off")
                    ax_txt.set_title("Estimate (95% CI)")
                    # Align text panel y-limits with graphic axis; do not invert here
                    ax_txt.set_ylim(ax.get_ylim())
                    for yi, t in zip(y, texts):
                        ax_txt.text(0.0, yi, t, va="center", ha="left", fontsize=9)
                    ax_txt.set_xlim(0, 1)

                    plt.tight_layout()
                    fig.savefig(
                        os.path.join(subgroup_dir, f"forest_predictions_day_{d}.png"), dpi=150, bbox_inches='tight'
                    )
                    plt.close(fig)

                # Combined forest plot across all time points — fixed x-limits
                combined = preds.copy()
                combined["group_order"] = combined["a1c_group"].apply(lambda x: A1C_ORDER.index(x) if x in A1C_ORDER else 999)
                combined = combined.sort_values(["group_order", "day"])  # group, then time
                day_values = sorted(combined["day"].unique().tolist())
                n_groups = len(a1c_cats)
                n_days = len(day_values)

                records = []
                for g_idx, g in enumerate(a1c_cats):
                    for d_idx, d in enumerate(day_values):
                        row = combined[(combined["a1c_group"] == g) & (combined["day"] == d)]
                        if row.empty:
                            continue
                        r = row.iloc[0].copy()
                        r["y_index"] = g_idx * n_days + d_idx
                        records.append(r)

                stacked = pd.DataFrame(records)
                stacked = stacked[np.isfinite(stacked["pred"]) & np.isfinite(stacked["ci_low"]) & np.isfinite(stacked["ci_high"]) ]
                if not stacked.empty:
                    fig = plt.figure(figsize=(12, 0.4 * n_groups * n_days + 2))
                    gs = fig.add_gridspec(1, 2, width_ratios=[4.5, 1.8])
                    ax = fig.add_subplot(gs[0, 0])
                    ax_txt = fig.add_subplot(gs[0, 1])

                    colors = plt.cm.tab10(np.linspace(0, 1, n_days))
                    for idx, d in enumerate(day_values):
                        subp = stacked[stacked["day"] == d]
                        ax.errorbar(
                            subp["pred"],
                            subp["y_index"],
                            xerr=[subp["pred"] - subp["ci_low"], subp["ci_high"] - subp["pred"]],
                            fmt="o",
                            capsize=3,
                            color=colors[idx],
                            label=f"Day {d}" if idx == 0 or f"Day {d}" not in ax.get_legend_handles_labels()[1] else "",
                        )

                    y_ticks = []
                    y_labels = []
                    texts = []
                    for g_idx, g in enumerate(a1c_cats):
                        for d_idx, d in enumerate(day_values):
                            y_val = g_idx * n_days + d_idx
                            y_ticks.append(y_val)
                            y_labels.append(f"{g} (Day {d})")
                            row = stacked[(stacked["a1c_group"] == g) & (stacked["day"] == d)]
                            if not row.empty:
                                rr = row.iloc[0]
                                texts.append(f"{rr['pred']:.2f} ({rr['ci_low']:.2f}, {rr['ci_high']:.2f})  n={int(rr['n_unique_window'])}")

                    ax.set_yticks(y_ticks)
                    ax.set_yticklabels(y_labels)
                    ax.axvline(0, color="black", linestyle="-", linewidth=1)
                    ax.set_xlabel("Predicted Absolute A1C Change")
                    ax.set_title(f"{var_label}: {subgroup} — A1C by Category and Time [{mode_label}]")
                    ax.grid(False)
                    ax.legend(title="Time from Baseline", bbox_to_anchor=(1.02, 1), loc="upper left")
                    ax.invert_yaxis()

                    x_min = float((stacked["ci_low"]).min())
                    x_max = float((stacked["ci_high"]).max())
                    x_range = x_max - x_min if x_max > x_min else 1.0
                    left_lim = x_min - 0.1 * x_range
                    right_lim = max(x_max + 0.1 * x_range, 0 + 0.3 * x_range)
                    if getattr(args, "xmax_cap", None) is not None:
                        right_lim = min(right_lim, float(args.xmax_cap))
                    if getattr(args, "xmin_cap", None) is not None:
                        left_lim = max(left_lim, float(args.xmin_cap))
                    # Replace dynamic x-limits with fixed limits for A1C plots
                    ax.set_xlim(args.xlim_min, args.xlim_max)

                    ax_txt.axis("off")
                    ax_txt.set_title("Estimate (95% CI)")
                    # Align text panel y-limits with graphic axis; do not invert here
                    ax_txt.set_ylim(ax.get_ylim())
                    for yi, t in zip(y, texts):
                        ax_txt.text(0.0, yi, t, va="center", ha="left", fontsize=9)
                    ax_txt.set_xlim(0, 1)

                    plt.tight_layout()
                    fig.savefig(
                        os.path.join(subgroup_dir, "forest_predictions_grouped_by_day_12m_18m.png"),
                        dpi=150,
                        bbox_inches='tight',
                    )
                    plt.close(fig)

                # Grouped-by-day forest plot for 12 and ~18 months — fixed x-limits
                selected_12 = 365
                possible_18 = [548, 547, 540]
                day_values = sorted(combined["day"].unique().tolist())
                selected_18 = next((d for d in possible_18 if d in day_values), None)

                if selected_12 in day_values and selected_18 is not None:
                    day_clusters = [selected_12, selected_18]
                    gap = 1
                    records_gbd = []
                    y_base = 0
                    for d in day_clusters:
                        for cat in A1C_ORDER:
                            row = combined[(combined["a1c_group"] == cat) & (combined["day"] == d)]
                            if row.empty:
                                continue
                            r = row.iloc[0].copy()
                            r["y_index"] = y_base + (A1C_ORDER.index(cat))
                            records_gbd.append(r)
                        y_base += len(A1C_ORDER) + gap

                    stacked_gbd = pd.DataFrame(records_gbd)
                    stacked_gbd = stacked_gbd[np.isfinite(stacked_gbd["pred"]) & np.isfinite(stacked_gbd["ci_low"]) & np.isfinite(stacked_gbd["ci_high"]) ]
                    if not stacked_gbd.empty:
                        fig = plt.figure(figsize=(12, 0.8 * len(A1C_ORDER) * len(day_clusters) + 2))
                        gs = fig.add_gridspec(1, 2, width_ratios=[4.5, 1.8])
                        ax = fig.add_subplot(gs[0, 0])
                        ax_txt = fig.add_subplot(gs[0, 1])

                        colors = plt.cm.tab10(np.linspace(0, 1, n_days))
                        for idx, d in enumerate(day_values):
                            subp = stacked_gbd[stacked_gbd["day"] == d]
                            ax.errorbar(
                                subp["pred"],
                                subp["y_index"],
                                xerr=[subp["pred"] - subp["ci_low"], subp["ci_high"] - subp["pred"]],
                                fmt="o",
                                capsize=3,
                                color=colors[idx],
                                label=f"Day {d}" if idx == 0 or f"Day {d}" not in ax.get_legend_handles_labels()[1] else "",
                            )

                        y_ticks = []
                        y_labels = []
                        texts = []
                        for g_idx, g in enumerate(a1c_cats):
                            for d_idx, d in enumerate(day_values):
                                y_val = g_idx * n_days + d_idx
                                y_ticks.append(y_val)
                                y_labels.append(f"{g} (Day {d})")
                                row = stacked_gbd[(stacked_gbd["a1c_group"] == g) & (stacked_gbd["day"] == d)]
                                if not row.empty:
                                    rr = row.iloc[0]
                                    texts.append(f"{rr['pred']:.2f} ({rr['ci_low']:.2f}, {rr['ci_high']:.2f})  n={int(rr['n_unique_window'])}")

                        ax.set_yticks(y_ticks)
                        ax.set_yticklabels(y_labels)
                        ax.axvline(0, color="black", linestyle="-", linewidth=1)
                        ax.set_xlabel("Predicted Absolute A1C Change")
                        ax.set_title(f"{var_label}: {subgroup} — A1C at 12 and 18 Months (Grouped) [{mode_label}]")
                        ax.grid(False)
                        ax.invert_yaxis()

                        x_min = float((stacked_gbd["ci_low"]).min())
                        x_max = float((stacked_gbd["ci_high"]).max())
                        x_range = x_max - x_min if x_max > x_min else 1.0
                        left_lim = x_min - 0.1 * x_range
                        right_lim = max(x_max + 0.1 * x_range, 0 + 0.3 * x_range)
                        if getattr(args, "xmax_cap", None) is not None:
                            right_lim = min(right_lim, float(args.xmax_cap))
                        if getattr(args, "xmin_cap", None) is not None:
                            left_lim = max(left_lim, float(args.xmin_cap))
                        # Replace dynamic x-limits with fixed limits for A1C plots
                        ax.set_xlim(args.xlim_min, args.xlim_max)

                        ax_txt.axis("off")
                        ax_txt.set_title("Estimate (95% CI)")
                        if y_ticks:
                            ax_txt.set_ylim(min(y_ticks) - 0.5, max(y_ticks) + 0.5)
                        ax_txt.invert_yaxis()
                        for yv, t in zip(y_ticks, texts):
                            ax_txt.text(0.0, yv, t, va="center", ha="left", fontsize=8)
                        ax_txt.set_xlim(0, 1)

                        plt.tight_layout()
                        fig.savefig(
                            os.path.join(subgroup_dir, "forest_predictions_grouped_by_day_12m_18m.png"),
                            dpi=150,
                            bbox_inches='tight',
                        )
                        plt.close(fig)

                    # Accumulate for global plot
                    for d in day_clusters:
                        for cat in A1C_ORDER:
                            row = preds[(preds["day"] == d) & (preds["a1c_group"] == cat)]
                            if row.empty:
                                continue
                            r = row.iloc[0]
                            combined_records.append(
                                {
                                    "covariate": var_label,
                                    "subgroup": subgroup,
                                    "a1c_group": cat,
                                    "day": int(d),
                                    "pred": float(r["pred"]),
                                    "ci_low": float(r["ci_low"]),
                                    "ci_high": float(r["ci_high"]),
                                    "n_unique_window": int(r["n_unique_window"]),
                                }
                            )

        # Build global combined 12m/18m plot
        if combined_records:
            combined_df = pd.DataFrame(combined_records)
            cov_order = [label for _, label in STRAT_VARS]
            combined_df["cov_order"] = combined_df["covariate"].apply(lambda x: cov_order.index(x) if x in cov_order else 999)
            combined_df = combined_df.sort_values(["cov_order", "subgroup", "day"])  # block, subgroup, time
            selected_days = sorted(combined_df["day"].unique().tolist())
            # Assign y_index with gaps between subgroups and covariates
            records = []
            y_base = 0
            gap_between_subgroups = 1
            gap_between_covariates = 2
            for cov in cov_order:
                cov_block = combined_df[combined_df["covariate"] == cov]
                if cov_block.empty:
                    continue
                for subgroup in cov_block["subgroup"].unique().tolist():
                    sub_block = cov_block[cov_block["subgroup"] == subgroup]
                    for d in selected_days:
                        row = sub_block[sub_block["day"] == d]
                        if row.empty:
                            continue
                        r = row.iloc[0].copy()
                        r["y_index"] = y_base
                        records.append(r)
                        y_base += 1
                    y_base += gap_between_subgroups
                y_base += (gap_between_covariates - gap_between_subgroups)
            stacked_all = pd.DataFrame(records)
            if not stacked_all.empty:
                fig = plt.figure(figsize=(14, max(6, 0.35 * len(stacked_all) + 2)))
                gs = fig.add_gridspec(1, 2, width_ratios=[4.5, 2.2])
                ax = fig.add_subplot(gs[0, 0])
                ax_txt = fig.add_subplot(gs[0, 1])
                colors_map = {selected_days[0]: "tab:blue", (selected_days[1] if len(selected_days) > 1 else None): "tab:orange"}
                for d in selected_days:
                    subp = stacked_all[stacked_all["day"] == d]
                    ax.errorbar(subp["pred"], subp["y_index"], xerr=[subp["pred"] - subp["ci_low"], subp["ci_high"] - subp["pred"]], fmt="o", capsize=3, color=colors_map.get(d, "tab:gray"), label=f"Day {d}")
                y_ticks = []
                y_labels = []
                texts_map = {}
                for _, r in stacked_all.sort_values("y_index").iterrows():
                    yi = int(r["y_index"])
                    label = f"{r['covariate']}: {r['subgroup']} (Day {int(r['day'])})"
                    y_ticks.append(yi)
                    y_labels.append(label)
                    texts_map[yi] = f"{float(r['pred']):.2f} ({float(r['ci_low']):.2f}, {float(r['ci_high']):.2f})  n={int(r['n_unique_window'])}"
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_labels)
                ax.axvline(0, color="black", linestyle="-", linewidth=1)
                ax.set_xlabel("Predicted Absolute A1C Change")
                ax.set_title(f"Stratified Forest — {mode_label} (12 and 18 Months)")
                ax.grid(False)
                ax.invert_yaxis()
                x_min = float((stacked_all["ci_low"]).min())
                x_max = float((stacked_all["ci_high"]).max())
                x_range = x_max - x_min if x_max > x_min else 1.0
                left_lim = x_min - 0.1 * x_range
                right_lim = max(x_max + 0.1 * x_range, 0 + 0.3 * x_range)
                if getattr(args, "xmax_cap", None) is not None:
                    right_lim = min(right_lim, float(args.xmax_cap))
                if getattr(args, "xmin_cap", None) is not None:
                    left_lim = max(left_lim, float(args.xmin_cap))
                # Replace dynamic x-limits with fixed limits for A1C plots (global combined)
                ax.set_xlim(args.xlim_min, args.xlim_max)
                ax_txt.axis("off")
                ax_txt.set_title("Estimate (95% CI)")
                if y_ticks:
                    ax_txt.set_ylim(min(y_ticks) - 0.5, max(y_ticks) + 0.5)
                ax_txt.invert_yaxis()
                for yv, t in zip(y_ticks, texts_map.values()):
                    ax_txt.text(0.0, yv, t, va="center", ha="left", fontsize=8)
                ax_txt.set_xlim(0, 1)
                plt.tight_layout()
                out_path = os.path.join(curr_outdir, "forests", "forest_predictions_grouped_by_day_12m_18m_all_strata.png")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                fig.savefig(out_path, dpi=150, bbox_inches='tight')
                plt.close(fig)

            # Also produce a combined forest grouped by day clusters (12m together, 18m together)
            if combined_records:
                # Reuse combined_df from above scope
                # Build stacked layout with days as outer grouping
                records_by_day = []
                y_base = 0
                gap_between_subgroups = 1
                gap_between_covariates = 2
                gap_between_days = 2
                for d in day_clusters:
                    # Within each day, iterate covariates in cov_order
                    for cov in cov_order:
                        cov_block = combined_df[(combined_df["covariate"] == cov) & (combined_df["day"] == d)]
                        if cov_block.empty:
                            continue
                        # Determine whether we have A1C categories in this pass
                        has_a1c = "a1c_group" in cov_block.columns and not cov_block["a1c_group"].isna().all()
                        for subgroup in cov_block["subgroup"].dropna().unique().tolist():
                            sub_block = cov_block[cov_block["subgroup"] == subgroup]
                            if has_a1c:
                                # Keep category order consistent if present
                                for cat in [c for c in (globals().get("a1c_cats", []))]:
                                    row = sub_block[sub_block.get("a1c_group") == cat]
                                    if row is None or row.empty:
                                        continue
                                    r = row.iloc[0].copy()
                                    r["y_index"] = y_base
                                    records_by_day.append(r)
                                    y_base += 1
                            else:
                                if sub_block.empty:
                                    continue
                                r = sub_block.iloc[0].copy()
                                r["y_index"] = y_base
                                records_by_day.append(r)
                                y_base += 1
                            y_base += gap_between_subgroups
                        y_base += (gap_between_covariates - gap_between_subgroups)
                    y_base += gap_between_days

                stacked_day = pd.DataFrame(records_by_day)
                stacked_day = stacked_day[
                    np.isfinite(stacked_day["pred"]) & np.isfinite(stacked_day["ci_low"]) & np.isfinite(stacked_day["ci_high"]) 
                ] if not pd.DataFrame(records_by_day).empty else stacked_day

                if not stacked_day.empty:
                    fig = plt.figure(figsize=(14, max(6, 0.35 * len(stacked_day) + 2)))
                    gs = fig.add_gridspec(1, 2, width_ratios=[4.5, 2.2])
                    ax = fig.add_subplot(gs[0, 0])
                    ax_txt = fig.add_subplot(gs[0, 1])

                    colors_map = {day_clusters[0]: "tab:blue"}
                    if len(day_clusters) > 1:
                        colors_map[day_clusters[1]] = "tab:orange"
                    for d in day_clusters:
                        subp = stacked_day[stacked_day["day"] == d]
                        ax.errorbar(
                            subp["pred"],
                            subp["y_index"],
                            xerr=[subp["pred"] - subp["ci_low"], subp["ci_high"] - subp["pred"]],
                            fmt="o",
                            capsize=3,
                            color=colors_map.get(d, "tab:gray"),
                            label=f"Day {d}",
                        )

                    y_ticks = []
                    y_labels = []
                    texts_map = {}
                    # Compose labels; include A1C group if present
                    has_a1c = "a1c_group" in stacked_day.columns and not stacked_day["a1c_group"].isna().all()
                    for _, r in stacked_day.sort_values("y_index").iterrows():
                        yi = int(r["y_index"])
                        cov = str(r.get("covariate", ""))
                        subgroup = str(r.get("subgroup", ""))
                        lab_core = f"{cov}: {subgroup}" if cov or subgroup else subgroup
                        if has_a1c and pd.notna(r.get("a1c_group", np.nan)):
                            lab_core = f"{lab_core} — {r['a1c_group']}"
                        label = f"{lab_core} (Day {int(r['day'])})"
                        y_ticks.append(yi)
                        y_labels.append(label)
                        texts_map[yi] = f"{float(r['pred']):.2f} ({float(r['ci_low']):.2f}, {float(r['ci_high']):.2f})  n={int(r['n_unique_window'])}"

                    ax.set_yticks(y_ticks)
                    ax.set_yticklabels(y_labels)
                    ax.axvline(0, color="black", linestyle="-", linewidth=1)
                    ax.set_xlabel("Predicted Absolute A1C Change")
                    ax.set_title(f"Stratified Forest — {mode_label} (12m and 18m grouped by day)")
                    ax.grid(False)
                    ax.invert_yaxis()

                    x_min = float((stacked_day["ci_low"]).min())
                    x_max = float((stacked_day["ci_high"]).max())
                    x_range = x_max - x_min if x_max > x_min else 1.0
                    left_lim = x_min - 0.1 * x_range
                    right_lim = max(x_max + 0.1 * x_range, 0 + 0.3 * x_range)
                    if getattr(args, "xmax_cap", None) is not None:
                        right_lim = min(right_lim, float(args.xmax_cap))
                    if getattr(args, "xmin_cap", None) is not None:
                        left_lim = max(left_lim, float(args.xmin_cap))
                    # Replace dynamic x-limits with fixed limits for A1C plots (grouped by day)
                    ax.set_xlim(args.xlim_min, args.xlim_max)

                    ax_txt.axis("off")
                    ax_txt.set_title("Estimate (95% CI)")
                    if y_ticks:
                        ax_txt.set_ylim(min(y_ticks) - 0.5, max(y_ticks) + 0.5)
                    ax_txt.invert_yaxis()
                    for yv, t in zip(y_ticks, texts_map.items()):
                        ax_txt.text(0.0, yv, t, va="center", ha="left", fontsize=9)
                    ax_txt.set_xlim(0, 1)

                    plt.tight_layout()
                    out_path2 = os.path.join(curr_outdir, "forests", "forest_predictions_by_day_12m_18m_all_strata.png")
                    os.makedirs(os.path.dirname(out_path2), exist_ok=True)
                    fig.savefig(out_path2, dpi=150, bbox_inches='tight')
                    plt.close(fig)

            # After subgroup loop: build combined variable plot (days 365 and ~548) with subgroup colors
            if var_records:
                vr = pd.DataFrame(var_records)
                if not vr.empty:
                    # Resolve day clusters present
                    day_clusters = sorted(vr["day"].unique().tolist())
                    # Consistent subgroup order
                    subgroup_list = [s for s in subgroups if s in vr["subgroup"].unique().tolist()]
                    # Colormap by subgroup
                    cmap = plt.cm.get_cmap('tab20', max(3, len(subgroup_list)))
                    color_by_sub = {sg: cmap(i % cmap.N) for i, sg in enumerate(subgroup_list)}
                    marker_by_day = {day_clusters[0]: 'o'}
                    if len(day_clusters) > 1:
                        marker_by_day[day_clusters[1]] = 's'

                    # Build y-index mapping: iterate by day, then A1C order, then subgroup order
                    records = []
                    y_base = 0
                    gap_between_blocks = 1
                    for dd in day_clusters:
                        for cat in A1C_ORDER:
                            for sg in subgroup_list:
                                row = vr[(vr["day"] == dd) & (vr["a1c_group"] == cat) & (vr["subgroup"] == sg)]
                                if row.empty:
                                    continue
                                r = row.iloc[0].copy()
                                r["y_index"] = y_base
                                records.append(r)
                                y_base += 1
                        y_base += gap_between_blocks

                    stacked = pd.DataFrame(records)
                    if not stacked.empty:
                        fig = plt.figure(figsize=(12, max(6, 0.5 * len(stacked) + 2)))
                        gs = fig.add_gridspec(1, 2, width_ratios=[4.5, 1.8])
                        ax = fig.add_subplot(gs[0, 0])
                        ax_txt = fig.add_subplot(gs[0, 1])
                        # Plot by subgroup color and day marker
                        for sg in subgroup_list:
                            for dd in day_clusters:
                                subp = stacked[(stacked["subgroup"] == sg) & (stacked["day"] == dd)]
                                if subp.empty:
                                    continue
                                ax.errorbar(
                                    subp["pred"],
                                    subp["y_index"],
                                    xerr=[subp["pred"] - subp["ci_low"], subp["ci_high"] - subp["pred"]],
                                    fmt=marker_by_day.get(dd, 'o'),
                                    capsize=3,
                                    color=color_by_sub.get(sg, 'tab:blue'),
                                    label=f"{sg} (Day {dd})" if f"{sg} (Day {dd})" not in ax.get_legend_handles_labels()[1] else None,
                                )
                        # Build labels and right-text aligned to y_index
                        y_ticks = []
                        y_labels = []
                        texts_map = {}
                        for _, r in stacked.sort_values("y_index").iterrows():
                            yi = int(r["y_index"])
                            y_ticks.append(yi)
                            y_labels.append(f"{r['a1c_group']} ({r['subgroup']}, Day {int(r['day'])})")
                            texts_map[yi] = f"{float(r['pred']):.2f} ({float(r['ci_low']):.2f}, {float(r['ci_high']):.2f})  n={int(r['n_unique_window'])}"
                        ax.set_yticks(y_ticks)
                        ax.set_yticklabels(y_labels)
                        ax.axvline(0, color="black", linestyle="-", linewidth=1)
                        ax.set_xlabel("Predicted Absolute A1C Change")
                        ax.set_title(f"{var_label}: Combined Forest at 12m and 18m by Baseline A1C")
                        ax.grid(False)
                        ax.set_xlim(args.xlim_min, args.xlim_max)
                        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Subgroup (Marker = Day)")
                        # Right-side text
                        ax_txt.axis("off")
                        ax_txt.set_title("Estimate (95% CI)")
                        if y_ticks:
                            ax_txt.set_ylim(min(y_ticks) - 0.5, max(y_ticks) + 0.5)
                        ax_txt.invert_yaxis()
                        for yv, t in zip(y_ticks, texts_map.values()):
                            ax_txt.text(0.0, yv, t, va="center", ha="left", fontsize=9)
                        ax_txt.set_xlim(0, 1)
                        plt.tight_layout()
                        out_path = os.path.join(var_base_outdir, "forests", "combined_forest_12m_18m.png")
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        fig.savefig(out_path, dpi=150, bbox_inches='tight')
                        plt.close(fig)


if __name__ == "__main__":
    main()
