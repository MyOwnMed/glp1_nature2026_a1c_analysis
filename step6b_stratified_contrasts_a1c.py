#!/usr/bin/env python3
"""
Step 6b: Stratified contrasts at specified days (A1C outcome)

Purpose:
- For each stratification variable (age, sex, race, baseline BMI, baseline metformin),
  fit separate GEE models per subgroup (excluding the strat variable itself from covariates)
  and compute contrasts (difference in predicted absolute A1C change) at target days.
- Contrasts are absolute differences (in A1C units) vs a chosen reference subgroup.
- 95% CIs use se_diff = sqrt(se_ref^2 + se_sub^2) assuming independent subgroup fits.

Inputs:
- Step 1 A1C analysis-ready CSV
- Step 2 A1C spline df config JSON

Outputs:
- CSV per strat_var listing contrasts vs reference at each target day:
  output/step6b_stratified_contrasts_a1c/contrasts_<strat_var>.csv

Model:
- GEE (Gaussian, identity) with bs(days_from_baseline, df=best_df)
- Covariates: baseline_a1c_category, baseline_bmi_final_category, age_group,
  gender, race, weight_change_med (excluding current strat_var)
"""
import os
import argparse
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
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


STRAT_VARS: List[Tuple[str, str]] = [
    ("age_group", "Age Group"),
    ("age_group_20_39_vs_40_plus", "Age Group 20-39 vs 40+"),
    ("age_group_20_49_vs_50_plus", "Age Group 20-49 vs 50+"),
    ("gender", "Sex"),
    ("race", "Race"),
    ("metformin_with_glp1_baseline", "GLP-1 + Metformin at Baseline"),
    ("baseline_bmi_final_category", "Baseline BMI Category"),
    ("glp1_user_group", "GLP-1 User Group"),
]

A1C_ORDER = [
    "Normal Glycemia",
    "Prediabetes",
    "Type 2 Diabetes",
    "Poorly Controlled Diabetes",
]


def configure_logging(level: str = "INFO"):
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Step 6b: Stratified contrasts at target days (A1C)")
    p.add_argument(
        "--input-csv",
        default=os.path.join("output", "step1_prepare_analysis_dataset_a1c", "analysis_ready_a1c_gap90.csv"),
        help="Path to analysis-ready A1C CSV from Step 1 (A1C pipeline)",
    )
    p.add_argument(
        "--config-json",
        default=os.path.join("output", "step2_select_spline_df_a1c", "model_config_a1c.json"),
        help="Model config JSON from Step 2 (A1C, contains best df)",
    )
    p.add_argument(
        "--outdir",
        default=os.path.join("output", "step6b_stratified_contrasts_a1c"),
        help="Directory to write contrast outputs",
    )
    p.add_argument(
        "--time-days",
        default="365",
        help="Comma-separated days from baseline for contrasts (e.g., 90,180,365)",
    )
    p.add_argument(
        "--min-nobs",
        type=int,
        default=100,
        help="Minimum observations per subgroup to fit model",
    )
    p.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    p.add_argument("--adherence-gap-days", type=int, default=None, help="Adherence gap for gap-specific subfolder")
    p.add_argument(
        "--max-days",
        type=int,
        default=548,
        help=(
            "Maximum day from baseline to allow for contrasts (548 ~ 18 months). "
            "This is the paper's 18-month follow-up limit expressed as a calendar "
            "figure; step8's MAX_FOLLOWUP_DAYS states the same limit as 18 x 30 = "
            "540 days, and step1's 730-day --max-days is a wider outer bound on the "
            "emitted data so the 730-day persistence sensitivity analysis has data "
            "to use. All three are at or beyond 18 months; none is a different "
            "analysis window. See README.md, 'Follow-up horizon caps'."
        ),
    )
    return p.parse_args(argv)


def _compute_mean_ci(result, X_pred: np.ndarray, z: float = Z_CRIT):
    mean = np.asarray(result.predict(X_pred), dtype=float)
    V = np.asarray(result.cov_params(), dtype=float)
    var = np.einsum("ij,jk,ik->i", X_pred, V, X_pred, optimize=True)
    var = np.clip(var, 0.0, None)
    se = np.sqrt(var)
    low = mean - z * se
    high = mean + z * se
    return mean, low, high, se


def _fit_subgroup_model(sub_df: pd.DataFrame, df_spline: int, strat_var: str):
    covariates = [
        "baseline_a1c_category",
        "baseline_bmi_final_category",
        "age_group",
        "gender",
        "race",
        # 'metformin_with_glp1_baseline' intentionally excluded per requirements
        "weight_change_med",
    ]
    covariates = filter_estimable(
        covariates, sub_df, exclude=[strat_var],
        context=f"step6b a1c, strat_var={strat_var}",
    )
    bs_term = f"bs(days_from_baseline, df={df_spline})"
    formula = "abs_a1c_change ~ " + bs_term
    if covariates:
        formula += " + " + " + ".join(covariates)
    y, X = dmatrices(formula, sub_df, return_type="dataframe")
    ids = sub_df.loc[y.index, "patient_id"]
    model = GEE(y, X, groups=ids, family=Gaussian(), cov_struct=Independence())
    res = model.fit()
    return res, X.design_info, covariates


def _build_pred_row(sub_df: pd.DataFrame, design_info, day: int, covariates: List[str]):
    # Clamp prediction day to subgroup's observed range to avoid bs() extrapolation errors
    days = pd.to_numeric(sub_df.get("days_from_baseline"), errors="coerce")
    finite = days[np.isfinite(days)]
    if finite.empty:
        return None
    min_day = int(np.nanmin(finite))
    max_day = int(np.nanmax(finite))
    day_clamped = max(min_day, min(max_day, int(day)))

    pred_df = pd.DataFrame({"days_from_baseline": [day_clamped]})
    for c in covariates:
        if c in sub_df.columns:
            mode_series = sub_df[c].mode()
            mode_val = mode_series.iloc[0] if not mode_series.empty else (
                sub_df[c].cat.categories[0] if hasattr(sub_df[c], "cat") else 0
            )
            pred_df[c] = [mode_val]
            if isinstance(sub_df[c].dtype, pd.CategoricalDtype):  # type: ignore[attr-defined]
                pred_df[c] = pd.Categorical(pred_df[c], categories=sub_df[c].cat.categories)
    X_pred = build_design_matrices([design_info], pred_df)[0]
    return np.asarray(X_pred)


def _youngest_age_bin(subgroups: List[str]) -> str:
    def _lower_bound(s: str) -> int:
        s = str(s)
        if "+" in s:
            try:
                return int(s.replace("+", "").strip())
            except Exception:
                return 9999
        if "-" in s:
            try:
                return int(s.split("-")[0])
            except Exception:
                return 9999
        try:
            return int(s)
        except Exception:
            return 9999
    ordered = sorted(subgroups, key=_lower_bound)
    return ordered[0] if ordered else (subgroups[0] if subgroups else "")


def main(argv=None):
    args = parse_args(argv)
    configure_logging(args.log_level)
    import re
    gap = args.adherence_gap_days
    if gap is None:
        m = re.search(r"gap[_]?(\d+)", str(args.input_csv))
        if m:
            try:
                gap = int(m.group(1))
            except Exception:
                gap = None
    outdir = args.outdir
    if gap is not None and "/gap_" not in outdir:
        outdir = os.path.join("output", f"gap_{gap}", os.path.basename(args.outdir))
    os.makedirs(outdir, exist_ok=True)

    # Read data
    df = pd.read_csv(args.input_csv)

    # Categorical columns for consistent encoding
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
    df = enforce_a1c_order(df, order=A1C_ORDER, context="step6b_stratified_contrasts_a1c")

    # Spline df
    df_spline = load_spline_df(args.config_json)
    logging.info("Using spline df=%d", df_spline)

    # Target days
    target_days = [int(x.strip()) for x in str(args.time_days).split(",") if x.strip()]
    # Clamp to requested horizon
    target_days = [d for d in target_days if d <= int(args.max_days)]

    for strat_var, strat_label in STRAT_VARS:
        if strat_var not in df.columns:
            continue
        # Determine subgroups
        if hasattr(df[strat_var], "cat"):
            subgroups = [c for c in df[strat_var].cat.categories if pd.notna(c)]
        else:
            try:
                subgroups = sorted([v for v in df[strat_var].dropna().unique().tolist()])
            except Exception:
                subgroups = [v for v in df[strat_var].dropna().unique().tolist()]
        # Choose reference subgroup
        counts = (
            df[["patient_id", strat_var]]
            .dropna()
            .drop_duplicates()
            .groupby(strat_var)["patient_id"].nunique()
        )
        ref_subgroup = None
        if strat_var == "age_group":
            ref_subgroup = _youngest_age_bin(subgroups)
        elif strat_var == "age_group_20_39_vs_40_plus":
            ref_subgroup = "20-39" if "20-39" in subgroups else None
        elif strat_var == "age_group_20_49_vs_50_plus":
            ref_subgroup = "20-49" if "20-49" in subgroups else None
        if not ref_subgroup:
            # Fall back to largest-N group
            ref_subgroup = str(counts.idxmax()) if not counts.empty else None
        if not ref_subgroup:
            logging.info("No valid reference for '%s'; skipping", strat_var)
            continue
        logging.info("Strat var '%s': reference subgroup = %s", strat_var, ref_subgroup)

        rows = []
        # Fit per subgroup, collect predicted means and SE at each target day
        subgroup_results = {}
        for subgroup in subgroups:
            sub_df = df[df[strat_var] == subgroup].copy()
            n_obs = len(sub_df)
            n_people = sub_df["patient_id"].nunique() if "patient_id" in sub_df.columns else 0
            if n_obs < args.min_nobs or n_people == 0:
                logging.info(
                    "Skip %s=%s (Obs=%d, N=%d): insufficient data",
                    strat_label,
                    subgroup,
                    n_obs,
                    n_people,
                )
                continue
            try:
                res, design_info, covs = _fit_subgroup_model(sub_df, df_spline, strat_var)
            except Exception as e:
                logging.warning("Fit failed for %s=%s: %s", strat_label, subgroup, e)
                continue
            subgroup_results[subgroup] = (sub_df, res, design_info, covs, n_people, n_obs)

        # Compute contrasts vs reference for available subgroups
        if ref_subgroup not in subgroup_results:
            logging.info("Reference subgroup '%s' not fitted; skipping contrasts for %s", ref_subgroup, strat_var)
            continue
        ref_df, ref_res, ref_design, ref_covs, n_ref, n_obs_ref = subgroup_results[ref_subgroup]
        for day in target_days:
            Xp_ref = _build_pred_row(ref_df, ref_design, day, ref_covs)
            if Xp_ref is None:
                continue
            ref_mean, _, _, ref_se = _compute_mean_ci(ref_res, Xp_ref)
            ref_m = float(ref_mean[0])
            ref_s = float(ref_se[0])
            for subgroup, (sub_df, sub_res, sub_design, sub_covs, n_sub, n_obs_sub) in subgroup_results.items():
                if subgroup == ref_subgroup:
                    continue
                Xp_sub = _build_pred_row(sub_df, sub_design, day, sub_covs)
                if Xp_sub is None:
                    continue
                sub_mean, _, _, sub_se = _compute_mean_ci(sub_res, Xp_sub)
                sub_m = float(sub_mean[0])
                sub_s = float(sub_se[0])
                diff = sub_m - ref_m
                se_diff = float(np.sqrt(max(0.0, ref_s**2 + sub_s**2)))
                z = Z_CRIT
                lo = diff - z * se_diff
                hi = diff + z * se_diff
                rows.append(
                    {
                        "strat_var": strat_var,
                        "ref": str(ref_subgroup),
                        "subgroup": str(subgroup),
                        "day": int(day),
                        "diff": diff,
                        "ci_low": lo,
                        "ci_high": hi,
                        "se": se_diff,
                        "n_people_ref": int(n_ref),
                        "n_obs_ref": int(n_obs_ref),
                        "n_people_sub": int(n_sub),
                        "n_obs_sub": int(n_obs_sub),
                    }
                )
        out = pd.DataFrame(rows)
        out_path = os.path.join(outdir, f"contrasts_{strat_var}.csv")
        out.to_csv(out_path, index=False)
        logging.info("Wrote contrasts for %s to %s (rows=%d)", strat_var, out_path, len(out))


if __name__ == "__main__":
    main()
