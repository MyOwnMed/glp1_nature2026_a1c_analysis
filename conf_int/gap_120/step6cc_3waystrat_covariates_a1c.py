#!/usr/bin/env python3
"""
Step 6cc: 3-way stratification (A1C outcome)
Curves of A1C change over time stratified by Gender × Age (<40 vs 40+).
- Input: Step 1 A1C analysis-ready CSV (gap-specific)
- Output: Single trajectory per stratum (uncentered and centered) and combined overlays across strata
- Gap routing: by default routes outputs into output/gap_120/step6cc_3way_by_age_sex_a1c
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
from model_spec import enforce_a1c_order

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

# Sex-stratified color convention: male = blue, female = red
GENDER_COLORS = {"M": "#1565C0", "F": "#C62828"}
# 3-way (sex × age) color map: Male = blue shades, Female = red shades
GENDER_AGE_COLORS = {
    ("M", "<40"): "#1565C0",
    ("M", "40+"): "#64B5F6",
    ("F", "<40"): "#C62828",
    ("F", "40+"): "#EF9A9A",
}


def configure_logging(level: str = "INFO"):
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Step 6cc: 3-way stratified trajectories (A1C)")
    p.add_argument(
        "--input-csv",
        default=os.path.join(
            "output",
            "step1_prepare_analysis_dataset_a1c",
            "analysis_ready_a1c_gap120.csv",
        ),
        help="Path to analysis-ready A1C CSV from Step 1 (gap=120 by default)",
    )
    p.add_argument(
        "--outdir",
        default=os.path.join("output", "step6cc_3way_by_age_sex_a1c"),
        help="Base directory for 3-way stratified outputs",
    )
    p.add_argument("--max-days", type=int, default=548, help="Max days from baseline to include (default 548)")
    p.add_argument("--spline-df", type=int, default=3, help="Spline df for bs(days) term")
    p.add_argument("--log-level", default="INFO", help="Logging level")
    p.add_argument("--adherence-gap-days", type=int, default=120, help="Gap routing for output subfolder")
    p.add_argument("--ylim-min", type=float, default=None, help="Optional fixed y-axis min for overlays")
    p.add_argument("--ylim-max", type=float, default=None, help="Optional fixed y-axis max for overlays")
    p.add_argument("--window-days", type=int, default=28, help="Half-window for sample size counting around time points")
    return p.parse_args(argv)


def _derive_age_bin(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "age" in df.columns:
        def _bin(a):
            try:
                if pd.isna(a):
                    return "Unknown"
                return "<40" if float(a) < 40 else "40+"
            except Exception:
                return "Unknown"
        df["age_bin_lt40_40plus"] = df["age"].apply(_bin)
    else:
        # Fallback to existing age group variants: treat 20-39 as <40; <20 becomes <40 as well
        if "age_group_20_39_vs_40_plus" in df.columns:
            def _bin2(cat):
                s = str(cat)
                if s in {"20-39", "<20", "Unknown"}:
                    return "<40" if s != "Unknown" else "Unknown"
                return "40+"
            df["age_bin_lt40_40plus"] = df["age_group_20_39_vs_40_plus"].apply(_bin2)
    # Category type
    if "age_bin_lt40_40plus" in df.columns:
        df["age_bin_lt40_40plus"] = pd.Categorical(df["age_bin_lt40_40plus"], categories=["<40", "40+", "Unknown"], ordered=True)
    return df


def _fit_stratum(df: pd.DataFrame, df_spline: int, exclude_vars: list):
    bs_term = f"bs(days_from_baseline, df={df_spline})"
    formula = f"abs_a1c_change ~ {bs_term} + baseline_a1c_category"
    covariates = [
        "age_group",
        "age_group_20_39_vs_40_plus",
        "age_group_20_49_vs_50_plus",
        "gender",
        "baseline_bmi_final_category",
        "race",
        "weight_change_med",
    ]
    covariates = filter_estimable(
        covariates, df, exclude=exclude_vars,
        context=f"step6cc a1c, exclude_vars={list(exclude_vars)}",
    )
    if covariates:
        formula += " + " + " + ".join(covariates)
    y, X = dmatrices(formula, df, return_type="dataframe")
    ids = df.loc[y.index, "patient_id"]
    model = GEE(y, X, groups=ids, family=Gaussian(), cov_struct=Independence())
    result = model.fit()
    return result, X.design_info


def _pred_curve(result, design_info, pred_df: pd.DataFrame) -> np.ndarray:
    Xp = build_design_matrices([design_info], pred_df)[0]
    Xp = np.asarray(Xp)
    mu = np.asarray(result.predict(Xp), dtype=float)
    return mu


def _pred_curve_ci(result, design_info, pred_df: pd.DataFrame, z: float = Z_CRIT):
    """Return mean, ci_low, ci_high from GEE result."""
    try:
        Xp = build_design_matrices([design_info], pred_df)[0]
    except Exception as exc:
        logging.warning("Prediction grid extends beyond training knots: %s", exc)
        n = len(pred_df)
        nan_arr = np.full(n, np.nan)
        return nan_arr, nan_arr.copy(), nan_arr.copy()
    Xp = np.asarray(Xp)
    mu = np.asarray(result.predict(Xp), dtype=float)
    V = np.asarray(result.cov_params(), dtype=float)
    var = np.einsum("ij,jk,ik->i", Xp, V, Xp, optimize=True)
    var = np.clip(var, 0.0, None)
    se = np.sqrt(var)
    return mu, mu - z * se, mu + z * se


def _fill_between_segments(ax, x, y_low, y_high, color, alpha=0.15, min_run_points=3):
    """Fill between CI bounds, skipping NaN gaps."""
    x = np.asarray(x, dtype=float)
    y_low = np.asarray(y_low, dtype=float)
    y_high = np.asarray(y_high, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y_low) & np.isfinite(y_high)
    runs = []
    start = None
    for i, v in enumerate(valid):
        if v:
            if start is None:
                start = i
        else:
            if start is not None:
                runs.append((start, i))
                start = None
    if start is not None:
        runs.append((start, len(valid)))
    for s, e in runs:
        if (e - s) >= min_run_points:
            ax.fill_between(x[s:e], y_low[s:e], y_high[s:e], alpha=alpha, color=color)


def _sanitize(x: str) -> str:
    return str(x).replace(" ", "_").replace("/", "-")


def main(argv=None):
    args = parse_args(argv)
    configure_logging(args.log_level)

    # Gap-aware outdir routing
    outdir = args.outdir
    gap = getattr(args, "adherence_gap_days", None)
    if gap is None:
        import re
        m = re.search(r"gap[_]?(\d+)", str(args.input_csv))
        if m:
            try:
                gap = int(m.group(1))
            except Exception:
                gap = None
    if gap is not None:
        if "/gap_" not in outdir:
            outdir = os.path.join("output", f"gap_{gap}", os.path.basename(outdir))
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "RUN_OK.txt"), "w") as f:
        f.write("step6cc_a1c started\n")

    # Load
    df = pd.read_csv(args.input_csv)
    for cat_col in [
        "gender",
        "baseline_a1c_category",
        "baseline_bmi_final_category",
        "race",
        "age_group",
    ]:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype("category")
    # Enforce A1C baseline category order
    # Fails loudly: this fixes the reference level for every contrast
    # (code-review item 8). See model_spec.enforce_a1c_order.
    df = enforce_a1c_order(df, order=A1C_ORDER, context="step6cc_3waystrat_covariates_a1c")

    # Derive age bin
    df = _derive_age_bin(df)
    # Filter to needed columns
    if "days_from_baseline" not in df.columns:
        raise ValueError("days_from_baseline missing in input")

    # Determine strata combinations
    genders = [g for g in (list(df["gender"].cat.categories) if "gender" in df.columns and hasattr(df["gender"], "cat") else sorted(df["gender"].dropna().unique().tolist())) if g not in (None, "Unknown")]
    age_bins = [b for b in (list(df["age_bin_lt40_40plus"].cat.categories) if "age_bin_lt40_40plus" in df.columns and hasattr(df["age_bin_lt40_40plus"], "cat") else sorted(df["age_bin_lt40_40plus"].dropna().unique().tolist())) if b not in (None, "Unknown")]
    # Default to ['<40','40+'] ordering
    age_bins = [b for b in ["<40", "40+"] if b in set(age_bins)]

    base_traj_dir = os.path.join(outdir, "trajectories")
    traj_unc = os.path.join(base_traj_dir, "uncentered")
    traj_ctr = os.path.join(base_traj_dir, "centered")
    traj_combined = os.path.join(base_traj_dir, "combined_overlays")
    for d in (base_traj_dir, traj_unc, traj_ctr, traj_combined):
        os.makedirs(d, exist_ok=True)
    with open(os.path.join(outdir, "README.txt"), "w") as rf:
        rf.write("3-way stratified A1C trajectories by baseline A1C, stratified by Gender x Age (<40 vs 40+).\n")

    # Prediction grid
    observed_max = int(np.nanmax(pd.to_numeric(df["days_from_baseline"], errors="coerce"))) if np.isfinite(pd.to_numeric(df["days_from_baseline"], errors="coerce")).any() else 366
    grid_max = int(min(int(getattr(args, "max_days", 548)), observed_max))
    days_grid = np.arange(0, grid_max + 1, 14)
    if 0 not in days_grid:
        days_grid = np.sort(np.append(days_grid, 0))

    # Prepare combined overlay across strata (no baseline-category strat)
    combined_strata = []

    # Iterate strata
    for g in genders:
        for ab in age_bins:
            sub = df.copy()
            sub = sub[(sub.get("gender") == g) & (sub.get("age_bin_lt40_40plus") == ab)]
            n_obs = len(sub)
            n_people = sub["patient_id"].nunique() if "patient_id" in sub.columns else n_obs
            if n_obs < 100:
                logging.info("Skip Gender=%s Age=%s (Obs=%d, N=%d)", g, ab, n_obs, n_people)
                continue
            exclude_vars = ["gender", "age_group", "age_group_20_39_vs_40_plus", "age_group_20_49_vs_50_plus", "age_bin_lt40_40plus"]
            try:
                res, design_info = _fit_stratum(sub, int(getattr(args, "spline-df", 3)), exclude_vars)
            except Exception as e:
                logging.warning("Fit failed for Gender=%s Age=%s: %s", g, ab, e)
                continue
            # Mode values for prediction covariates (including baseline_a1c_category)
            pred_df = pd.DataFrame({"days_from_baseline": days_grid})
            for c in ["baseline_a1c_category", "baseline_bmi_final_category", "race", "weight_change_med"]:
                if c in sub.columns:
                    m = sub[c].mode()
                    if not m.empty:
                        pred_df[c] = [m.iloc[0]] * len(days_grid)
                        if hasattr(sub[c], "cat"):
                            pred_df[c] = pd.Categorical(pred_df[c], categories=sub[c].cat.categories)
            # Include stratifier values (not in formula) for completeness
            if "gender" in sub.columns:
                pred_df["gender"] = [g] * len(days_grid)
            if "age_bin_lt40_40plus" in sub.columns:
                pred_df["age_bin_lt40_40plus"] = [ab] * len(days_grid)
            mu, ci_low_u, ci_high_u = _pred_curve_ci(res, design_info, pred_df)
            if np.all(np.isnan(mu)):
                logging.warning("Prediction returned all NaN for Gender=%s Age=%s — skipping", g, ab)
                continue
            anchor = mu[0]
            mc = mu - anchor
            ci_low_c = ci_low_u - anchor
            ci_high_c = ci_high_u - anchor
            # Save per-stratum plots
            _gc = GENDER_COLORS.get(str(g))
            title_core = f"Sex: {g} | Age: {ab}"
            plt.figure(figsize=(10, 6))
            v = np.isfinite(mu)
            if np.any(v):
                (line,) = plt.plot(days_grid[v], mu[v], label=f"{g} {ab}", linewidth=2, **({"color": _gc} if _gc else {}))
                _fill_between_segments(plt.gca(), days_grid, ci_low_u, ci_high_u, color=line.get_color(), alpha=0.15)
            plt.xlabel("Days from Baseline"); plt.ylabel("Absolute A1C Change")
            plt.title(f"{title_core} — A1C Trajectory (Uncentered)\nstarting sample n = {int(n_people)}")
            plt.axhline(0, color="black", linestyle="--", linewidth=1)
            plt.xlim(0, int(getattr(args, "max_days", 548)))
            plt.legend(); plt.grid(True, linestyle="--", alpha=0.7); plt.tight_layout()
            out_path = os.path.join(traj_unc, f"a1c_traj_gender_{_sanitize(str(g))}_age_{_sanitize(str(ab))}.png")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.savefig(out_path, dpi=150); plt.close()
            plt.figure(figsize=(10, 6))
            vc = np.isfinite(mc)
            if np.any(vc):
                mc0 = mc.copy()
                ci_low_c0 = ci_low_c.copy()
                ci_high_c0 = ci_high_c.copy()
                if 0 in days_grid:
                    i0 = int(np.where(days_grid == 0)[0][0]); mc0[i0] = 0.0; ci_low_c0[i0] = 0.0; ci_high_c0[i0] = 0.0
                (line,) = plt.plot(days_grid[vc], mc0[vc], label=f"{g} {ab}", linewidth=2, **({"color": _gc} if _gc else {}))
                _fill_between_segments(plt.gca(), days_grid, ci_low_c0, ci_high_c0, color=line.get_color(), alpha=0.15)
            plt.xlabel("Days from Baseline"); plt.ylabel("Absolute A1C Change (Centered)")
            plt.title(f"{title_core} — A1C Trajectory")
            plt.axhline(0, color="black", linestyle="--", linewidth=1)
            plt.xlim(0, int(getattr(args, "max_days", 548)))
            plt.legend(); plt.grid(True, linestyle="--", alpha=0.7); plt.tight_layout()
            out_path = os.path.join(traj_ctr, f"a1c_traj_gender_{_sanitize(str(g))}_age_{_sanitize(str(ab))}.png")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.savefig(out_path, dpi=150); plt.close()
            # Accumulate for combined overlays across strata
            combined_strata.append({
                "label": f"{g} | {ab}",
                "gender": str(g),
                "age_bin": str(ab),
                "x": days_grid,
                "y_unc": mu,
                "y_ctr": mc,
                "ci_low_u": ci_low_u,
                "ci_high_u": ci_high_u,
                "ci_low_c": ci_low_c,
                "ci_high_c": ci_high_c,
            })

            # Track combined overlays: per baseline A1C category, overlay all strata
            combined_by_cat = {cat: [] for cat in A1C_ORDER}
            # Also keep stratum-level series for overall overlays
            combined_strata = []

            # Iterate strata
            for g in genders:
                for ab in age_bins:
                    sub = df.copy()
                    sub = sub[(sub.get("gender") == g) & (sub.get("age_bin_lt40_40plus") == ab)]
                    n_obs = len(sub)
                    n_people = sub["patient_id"].nunique() if "patient_id" in sub.columns else n_obs
                    if n_obs < 100:
                        logging.info("Skip Gender=%s Age=%s (Obs=%d, N=%d)", g, ab, n_obs, n_people)
                        continue
                    exclude_vars = ["gender", "age_group", "age_group_20_39_vs_40_plus", "age_group_20_49_vs_50_plus", "age_bin_lt40_40plus"]
                    try:
                        res, design_info = _fit_stratum(sub, int(getattr(args, "spline-df", 3)), exclude_vars)
                    except Exception as e:
                        logging.warning("Fit failed for Gender=%s Age=%s: %s", g, ab, e)
                        continue
                    # Mode values for prediction covariates (including baseline_a1c_category)
                    pred_df = pd.DataFrame({"days_from_baseline": days_grid})
                    for c in ["baseline_a1c_category", "baseline_bmi_final_category", "race", "weight_change_med"]:
                        if c in sub.columns:
                            m = sub[c].mode()
                            if not m.empty:
                                pred_df[c] = [m.iloc[0]] * len(days_grid)
                                if hasattr(sub[c], "cat"):
                                    pred_df[c] = pd.Categorical(pred_df[c], categories=sub[c].cat.categories)
                    # Include stratifier values (not in formula) for completeness
                    if "gender" in sub.columns:
                        pred_df["gender"] = [g] * len(days_grid)
                    if "age_bin_lt40_40plus" in sub.columns:
                        pred_df["age_bin_lt40_40plus"] = [ab] * len(days_grid)
                    mu, ci_low_u, ci_high_u = _pred_curve_ci(res, design_info, pred_df)
                    if np.all(np.isnan(mu)):
                        logging.warning("Prediction returned all NaN for Gender=%s Age=%s (loop2) — skipping", g, ab)
                        continue
                    anchor = mu[0]
                    mc = mu - anchor
                    ci_low_c = ci_low_u - anchor
                    ci_high_c = ci_high_u - anchor
                    # Save per-stratum plots
                    _gc = GENDER_COLORS.get(str(g))
                    title_core = f"Sex: {g} | Age: {ab}"
                    plt.figure(figsize=(10, 6))
                    v = np.isfinite(mu)
                    if np.any(v):
                        (line,) = plt.plot(days_grid[v], mu[v], label=f"{g} {ab}", linewidth=2, **({"color": _gc} if _gc else {}))
                        _fill_between_segments(plt.gca(), days_grid, ci_low_u, ci_high_u, color=line.get_color(), alpha=0.15)
                    plt.xlabel("Days from Baseline"); plt.ylabel("Absolute A1C Change")
                    plt.title(f"{title_core} — A1C Trajectory (Uncentered)\nstarting sample n = {int(n_people)}")
                    plt.axhline(0, color="black", linestyle="--", linewidth=1)
                    plt.xlim(0, int(getattr(args, "max_days", 548)))
                    plt.legend(); plt.grid(True, linestyle="--", alpha=0.7); plt.tight_layout()
                    out_path = os.path.join(traj_unc, f"a1c_traj_gender_{_sanitize(str(g))}_age_{_sanitize(str(ab))}.png")
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    plt.savefig(out_path, dpi=150); plt.close()
                    plt.figure(figsize=(10, 6))
                    vc = np.isfinite(mc)
                    if np.any(vc):
                        mc0 = mc.copy()
                        ci_low_c0 = ci_low_c.copy()
                        ci_high_c0 = ci_high_c.copy()
                        if 0 in days_grid:
                            i0 = int(np.where(days_grid == 0)[0][0]); mc0[i0] = 0.0; ci_low_c0[i0] = 0.0; ci_high_c0[i0] = 0.0
                        (line,) = plt.plot(days_grid[vc], mc0[vc], label=f"{g} {ab}", linewidth=2, **({"color": _gc} if _gc else {}))
                        _fill_between_segments(plt.gca(), days_grid, ci_low_c0, ci_high_c0, color=line.get_color(), alpha=0.15)
                    plt.xlabel("Days from Baseline"); plt.ylabel("Absolute A1C Change (Centered)")
                    plt.title(f"{title_core} — A1C Trajectory")
                    plt.axhline(0, color="black", linestyle="--", linewidth=1)
                    plt.xlim(0, int(getattr(args, "max_days", 548)))
                    plt.legend(); plt.grid(True, linestyle="--", alpha=0.7); plt.tight_layout()
                    out_path = os.path.join(traj_ctr, f"a1c_traj_gender_{_sanitize(str(g))}_age_{_sanitize(str(ab))}.png")
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    plt.savefig(out_path, dpi=150); plt.close()
                    # Accumulate for combined overlays across strata
                    combined_strata.append({
                        "label": f"{g} | {ab}",
                        "gender": str(g),
                        "age_bin": str(ab),
                        "x": days_grid,
                        "y_unc": mu,
                        "y_ctr": mc,
                        "ci_low_u": ci_low_u,
                        "ci_high_u": ci_high_u,
                        "ci_low_c": ci_low_c,
                        "ci_high_c": ci_high_c,
                    })

                    # Build overlay across baseline A1C categories
                    a1c_cats = [c for c in A1C_ORDER if c in (list(sub["baseline_a1c_category"].cat.categories) if "baseline_a1c_category" in sub.columns and hasattr(sub["baseline_a1c_category"], "cat") else A1C_ORDER)]
                    for cat in a1c_cats:
                        # Mode values for prediction covariates (including baseline_a1c_category)
                        pred_df = pd.DataFrame({"days_from_baseline": days_grid})
                        for c in ["baseline_bmi_final_category", "race", "weight_change_med"]:
                            if c in sub.columns:
                                m = sub[c].mode()
                                if not m.empty:
                                    pred_df[c] = [m.iloc[0]] * len(days_grid)
                                    if hasattr(sub[c], "cat"):
                                        pred_df[c] = pd.Categorical(pred_df[c], categories=sub[c].cat.categories)
                        # Include stratifier values (not in formula) for completeness
                        if "gender" in sub.columns:
                            pred_df["gender"] = [g] * len(days_grid)
                        if "age_bin_lt40_40plus" in sub.columns:
                            pred_df["age_bin_lt40_40plus"] = [ab] * len(days_grid)
                        # Specific to this baseline A1C category
                        pred_df["baseline_a1c_category"] = cat
                        mu_cat, ci_low_cat, ci_high_cat = _pred_curve_ci(res, design_info, pred_df)
                        if np.all(np.isnan(mu_cat)):
                            logging.warning("Prediction NaN for Gender=%s Age=%s cat=%s — skipping", g, ab, cat)
                            continue
                        anchor_cat = mu_cat[0]
                        mc_cat = mu_cat - anchor_cat
                        ci_low_c_cat = ci_low_cat - anchor_cat
                        ci_high_c_cat = ci_high_cat - anchor_cat
                        # Store for combined overlays
                        combined_by_cat.setdefault(cat, []).append({
                            "gender": g,
                            "age_bin": ab,
                            "x": days_grid,
                            "y_unc": mu_cat,
                            "y_ctr": mc_cat,
                            "ci_low_u": ci_low_cat,
                            "ci_high_u": ci_high_cat,
                            "ci_low_c": ci_low_c_cat,
                            "ci_high_c": ci_high_c_cat,
                        })
                    # Per-category overlays (no stratification)
                    for cat, series in combined_by_cat.items():
                        if not series:
                            continue
                        # Uncentered
                        plt.figure(figsize=(12, 7))
                        for s in series:
                            xg = s["x"]; mu = s["y_unc"]; lab = s.get("age_bin", "")
                            _sgc = GENDER_AGE_COLORS.get((str(s.get("gender", "")), str(s.get("age_bin", ""))))
                            v = np.isfinite(mu)
                            if np.any(v):
                                line, = plt.plot(xg[v], mu[v], label=lab, linewidth=2, **({"color": _sgc} if _sgc else {}))
                                _fill_between_segments(plt.gca(), xg, s["ci_low_u"], s["ci_high_u"], color=line.get_color(), alpha=0.12)
                        plt.xlabel("Days from Baseline"); plt.ylabel("Absolute A1C Change")
                        plt.title(f"A1C ({cat}) — Overlay by Sex × Age (<40 vs 40+)")
                        plt.axhline(0, color="black", linestyle="--", linewidth=1)
                        plt.legend(); plt.grid(True, linestyle="--", alpha=0.7); plt.tight_layout()
                        out_path = os.path.join(traj_combined, f"overlay_uncentered_{_sanitize(cat)}.png")
                        plt.savefig(out_path, dpi=150); plt.close()
                        # Centered
                        plt.figure(figsize=(12, 7))
                        for s in series:
                            xg = s["x"]; mc = s["y_ctr"]; lab = s.get("age_bin", "")
                            _sgc = GENDER_AGE_COLORS.get((str(s.get("gender", "")), str(s.get("age_bin", ""))))
                            ci_low_c = s["ci_low_c"]; ci_high_c = s["ci_high_c"]
                            v = np.isfinite(mc)
                            if np.any(v):
                                mc0 = mc.copy()
                                ci_low_c0 = ci_low_c.copy()
                                ci_high_c0 = ci_high_c.copy()
                                if 0 in xg:
                                    i0 = int(np.where(xg == 0)[0][0]); mc0[i0] = 0.0; ci_low_c0[i0] = 0.0; ci_high_c0[i0] = 0.0
                                line, = plt.plot(xg[v], mc0[v], label=lab, linewidth=2, **({"color": _sgc} if _sgc else {}))
                                _fill_between_segments(plt.gca(), xg, ci_low_c0, ci_high_c0, color=line.get_color(), alpha=0.12)
                        plt.xlabel("Days from Baseline"); plt.ylabel("Absolute A1C Change (Centered)")
                        plt.title(f"A1C ({cat}) — Overlay by Sex × Age (<40 vs 40+)")
                        plt.axhline(0, color="black", linestyle="--", linewidth=1)
                        plt.legend(); plt.grid(True, linestyle="--", alpha=0.7); plt.tight_layout()
                        out_path = os.path.join(traj_combined, f"overlay_centered_{_sanitize(cat)}.png")
                        plt.savefig(out_path, dpi=150); plt.close()

    # Combined overlays across strata (no baseline-category separation)
    if combined_strata:
        # Compute common y-limits for uncentered and centered overlays unless provided
        def _common_limits(series_list, key: str):
            vals = []
            for s in series_list:
                y = s.get(key)
                if y is not None:
                    y = np.asarray(y, dtype=float)
                    vals.append(y[np.isfinite(y)])
            if not vals:
                return None, None
            v = np.concatenate(vals) if len(vals) > 1 else vals[0]
            if v.size == 0:
                return None, None
            lo, hi = float(np.min(v)), float(np.max(v))
            pad = 0.05 * (hi - lo if hi > lo else 1.0)
            return lo - pad, hi + pad

        unc_lo, unc_hi = _common_limits(combined_strata, "y_unc")
        ctr_lo, ctr_hi = _common_limits(combined_strata, "y_ctr")
        # Apply CLI overrides if set
        if args.ylim_min is not None and args.ylim_max is not None:
            unc_lo, unc_hi = float(args.ylim_min), float(args.ylim_max)
            ctr_lo, ctr_hi = float(args.ylim_min), float(args.ylim_max)

        def _counts_label(s: dict) -> str:
            c = s.get("counts", {})
            return f"n0={c.get('0',0)}, n6={c.get('180',0)}, n12={c.get('365',0)}, n18={c.get('548',0)}"
        def _scatter_markers(xg, yg, color):
            if yg is None or xg is None:
                return
            tpoints = [0, 180, 365, 548]
            for t in tpoints:
                if t > (xg.max() if hasattr(xg, 'max') else max(xg)):
                    continue
                idx = int(np.argmin(np.abs(xg - t)))
                if idx < len(yg) and np.isfinite(yg[idx]):
                    plt.scatter([xg[idx]], [yg[idx]], color=color, s=20, zorder=3)

        # Uncentered overlay
        plt.figure(figsize=(12, 7))
        for s in combined_strata:
            xg = s["x"]; mu = s["y_unc"]; lab = f"{s['label']} ({_counts_label(s)})"
            _sgc = GENDER_AGE_COLORS.get((str(s.get("gender", "")), str(s.get("age_bin", ""))))
            v = np.isfinite(mu)
            if np.any(v):
                line, = plt.plot(xg[v], mu[v], label=lab, linewidth=2, **({"color": _sgc} if _sgc else {}))
                _fill_between_segments(plt.gca(), xg, s["ci_low_u"], s["ci_high_u"], color=line.get_color(), alpha=0.12)
                _scatter_markers(xg, mu, line.get_color())
        plt.xlabel("Days from Baseline"); plt.ylabel("Absolute A1C Change")
        plt.title("A1C — Overlay by Sex × Age (<40 vs 40+)")
        plt.axhline(0, color="black", linestyle="--", linewidth=1)
        plt.legend(); plt.grid(True, linestyle="--", alpha=0.7); plt.tight_layout()
        out_path = os.path.join(traj_combined, "overlay_uncentered_all_strata.png")
        if unc_lo is not None and unc_hi is not None:
            plt.ylim(unc_lo, unc_hi)
        plt.savefig(out_path, dpi=150); plt.close()

        # Centered overlay
        plt.figure(figsize=(12, 7))
        for s in combined_strata:
            xg = s["x"]; mc = s["y_ctr"]; lab = f"{s['label']} ({_counts_label(s)})"
            _sgc = GENDER_AGE_COLORS.get((str(s.get("gender", "")), str(s.get("age_bin", ""))))
            ci_low_c = s["ci_low_c"]; ci_high_c = s["ci_high_c"]
            v = np.isfinite(mc)
            if np.any(v):
                mc0 = mc.copy()
                ci_low_c0 = ci_low_c.copy()
                ci_high_c0 = ci_high_c.copy()
                if 0 in xg:
                    i0 = int(np.where(xg == 0)[0][0]); mc0[i0] = 0.0; ci_low_c0[i0] = 0.0; ci_high_c0[i0] = 0.0
                line, = plt.plot(xg[v], mc0[v], label=lab, linewidth=2, **({"color": _sgc} if _sgc else {}))
                _fill_between_segments(plt.gca(), xg, ci_low_c0, ci_high_c0, color=line.get_color(), alpha=0.12)
                _scatter_markers(xg, mc0, line.get_color())
        plt.xlabel("Days from Baseline"); plt.ylabel("Absolute A1C Change (Centered)")
        plt.title("A1C — Overlay by Sex × Age (<40 vs 40+)")
        plt.axhline(0, color="black", linestyle="--", linewidth=1)
        plt.legend(); plt.grid(True, linestyle="--", alpha=0.7); plt.tight_layout()
        out_path = os.path.join(traj_combined, "overlay_centered_all_strata.png")
        if ctr_lo is not None and ctr_hi is not None:
            plt.ylim(ctr_lo, ctr_hi)
        plt.savefig(out_path, dpi=150); plt.close()

        # Male-only overlay by age bins
        male_series = [s for s in combined_strata if str(s.get("gender", "")).lower().startswith("m")]
        if male_series:
            # Uncentered
            plt.figure(figsize=(12, 7))
            for s in male_series:
                xg = s["x"]; mu = s["y_unc"]; lab = f"{s.get('age_bin','')} ({_counts_label(s)})"
                _mgc = GENDER_AGE_COLORS.get(("M", str(s.get("age_bin", ""))))
                v = np.isfinite(mu)
                if np.any(v):
                    line, = plt.plot(xg[v], mu[v], label=lab, linewidth=2, **({"color": _mgc} if _mgc else {}))
                    _fill_between_segments(plt.gca(), xg, s["ci_low_u"], s["ci_high_u"], color=line.get_color(), alpha=0.12)
                    _scatter_markers(xg, mu, line.get_color())
            plt.xlabel("Days from Baseline"); plt.ylabel("Absolute A1C Change")
            plt.title("A1C — Male Overlay by Age (<40 vs 40+)")
            plt.axhline(0, color="black", linestyle="--", linewidth=1)
            plt.legend(title="Age Bin"); plt.grid(True, linestyle="--", alpha=0.7); plt.tight_layout()
            out_path = os.path.join(traj_combined, "overlay_uncentered_male_by_age.png")
            if unc_lo is not None and unc_hi is not None:
                plt.ylim(unc_lo, unc_hi)
            plt.savefig(out_path, dpi=150); plt.close()
            # Centered
            plt.figure(figsize=(12, 7))
            for s in male_series:
                xg = s["x"]; mc = s["y_ctr"]; lab = f"{s.get('age_bin','')} ({_counts_label(s)})"
                _mgc = GENDER_AGE_COLORS.get(("M", str(s.get("age_bin", ""))))
                ci_low_c = s["ci_low_c"]; ci_high_c = s["ci_high_c"]
                v = np.isfinite(mc)
                if np.any(v):
                    mc0 = mc.copy()
                    ci_low_c0 = ci_low_c.copy()
                    ci_high_c0 = ci_high_c.copy()
                    if 0 in xg:
                        i0 = int(np.where(xg == 0)[0][0]); mc0[i0] = 0.0; ci_low_c0[i0] = 0.0; ci_high_c0[i0] = 0.0
                    line, = plt.plot(xg[v], mc0[v], label=lab, linewidth=2, **({"color": _mgc} if _mgc else {}))
                    _fill_between_segments(plt.gca(), xg, ci_low_c0, ci_high_c0, color=line.get_color(), alpha=0.12)
                    _scatter_markers(xg, mc0, line.get_color())
            plt.xlabel("Days from Baseline"); plt.ylabel("Absolute A1C Change (Centered)")
            plt.title("A1C — Male Overlay by Age (<40 vs 40+)")
            plt.axhline(0, color="black", linestyle="--", linewidth=1)
            plt.legend(title="Age Bin"); plt.grid(True, linestyle="--", alpha=0.7); plt.tight_layout()
            out_path = os.path.join(traj_combined, "overlay_centered_male_by_age.png")
            if ctr_lo is not None and ctr_hi is not None:
                plt.ylim(ctr_lo, ctr_hi)
            plt.savefig(out_path, dpi=150); plt.close()

        # Female-only overlay by age bins
        female_series = [s for s in combined_strata if str(s.get("gender", "")).lower().startswith("f")]
        if female_series:
            # Uncentered
            plt.figure(figsize=(12, 7))
            for s in female_series:
                xg = s["x"]; mu = s["y_unc"]; lab = f"{s.get('age_bin','')} ({_counts_label(s)})"
                _fgc = GENDER_AGE_COLORS.get(("F", str(s.get("age_bin", ""))))
                v = np.isfinite(mu)
                if np.any(v):
                    line, = plt.plot(xg[v], mu[v], label=lab, linewidth=2, **({"color": _fgc} if _fgc else {}))
                    _fill_between_segments(plt.gca(), xg, s["ci_low_u"], s["ci_high_u"], color=line.get_color(), alpha=0.12)
                    _scatter_markers(xg, mu, line.get_color())
            plt.xlabel("Days from Baseline"); plt.ylabel("Absolute A1C Change")
            plt.title("A1C — Female Overlay by Age (<40 vs 40+)")
            plt.axhline(0, color="black", linestyle="--", linewidth=1)
            plt.legend(title="Age Bin"); plt.grid(True, linestyle="--", alpha=0.7); plt.tight_layout()
            out_path = os.path.join(traj_combined, "overlay_uncentered_female_by_age.png")
            if unc_lo is not None and unc_hi is not None:
                plt.ylim(unc_lo, unc_hi)
            plt.savefig(out_path, dpi=150); plt.close()
            # Centered
            plt.figure(figsize=(12, 7))
            for s in female_series:
                xg = s["x"]; mc = s["y_ctr"]; lab = f"{s.get('age_bin','')} ({_counts_label(s)})"
                _fgc = GENDER_AGE_COLORS.get(("F", str(s.get("age_bin", ""))))
                ci_low_c = s["ci_low_c"]; ci_high_c = s["ci_high_c"]
                v = np.isfinite(mc)
                if np.any(v):
                    mc0 = mc.copy()
                    ci_low_c0 = ci_low_c.copy()
                    ci_high_c0 = ci_high_c.copy()
                    if 0 in xg:
                        i0 = int(np.where(xg == 0)[0][0]); mc0[i0] = 0.0; ci_low_c0[i0] = 0.0; ci_high_c0[i0] = 0.0
                    line, = plt.plot(xg[v], mc0[v], label=lab, linewidth=2, **({"color": _fgc} if _fgc else {}))
                    _fill_between_segments(plt.gca(), xg, ci_low_c0, ci_high_c0, color=line.get_color(), alpha=0.12)
                    _scatter_markers(xg, mc0, line.get_color())
            plt.xlabel("Days from Baseline"); plt.ylabel("Absolute A1C Change (Centered)")
            plt.title("A1C — Female Overlay by Age (<40 vs 40+)")
            plt.axhline(0, color="black", linestyle="--", linewidth=1)
            plt.legend(title="Age Bin"); plt.grid(True, linestyle="--", alpha=0.7); plt.tight_layout()
            out_path = os.path.join(traj_combined, "overlay_centered_female_by_age.png")
            if ctr_lo is not None and ctr_hi is not None:
                plt.ylim(ctr_lo, ctr_hi)
            plt.savefig(out_path, dpi=150); plt.close()

    logging.info("Done: outputs written to %s", outdir)


if __name__ == "__main__":
    main()
