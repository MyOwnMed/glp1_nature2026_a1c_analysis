#!/usr/bin/env python3
"""
Step 6: Stratified GEE analyses by key covariates, controlling for baseline A1C (A1C outcome)

Audience-friendly summary:
- We create separate analyses for subgroups (age group, sex, race, GLP-1+metformin at baseline, BMI).
- Within each subgroup, we estimate absolute A1C change over time.
- We control for baseline A1C category (confounder) and other covariates, but NOT the stratification variable.
- This helps see if results are consistent across different populations.
"""

import os
import argparse
import logging

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


CI_WINDOW_DAYS = int(os.getenv("PLOT_CI_WINDOW_DAYS", "28"))


def configure_logging(level: str = "INFO"):
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Step 6: Stratified GEE by covariates (A1C outcome)"
    )
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
        default=os.path.join(
            "output", "step2_select_spline_df_a1c", "model_config_a1c.json"
        ),
        help="Model config JSON from Step 2 (A1C, contains best df)",
    )
    p.add_argument(
        "--outdir",
        default=os.path.join("output", "step6_stratified_by_covariates_a1c"),
        help="Directory to write outputs",
    )
    p.add_argument(
        "--min-nobs",
        type=int,
        default=100,
        help="Minimum observations per stratum to fit model",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    p.add_argument(
        "--adherence-gap-days",
        type=int,
        default=None,
        help="Adherence gap in days to display in plot titles (e.g., 90, 120, 180)",
    )
    p.add_argument(
        "--max-days",
        type=int,
        default=548,
        help="Maximum days from baseline to include in predictions and plots (cap)",
    )
    p.add_argument(
        "--suppress-grouped-titles",
        action="store_true",
        help="If set, omit titles from grouped overlay plots.",
    )
    p.add_argument(
        "--suppress-grouped-legend",
        action="store_true",
        help="If set, omit legends from grouped overlay plots.",
    )
    p.add_argument(
        "--write-grouped-legend-only",
        action="store_true",
        help=(
            "If set, in addition to grouped overlay plots, also write a "
            "legend-only PNG per stratification variable."
        ),
    )
    return p.parse_args(argv)


STRAT_VARS = [
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

AGE_GROUP_ORDER = [
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "70-79",
    "80+",
]


BMI_CATEGORY_ORDER = [
    "Underweight",
    "Normal",
    "Overweight",
    "Obese I",
    "Obese II",
    "Obese III",
]

# Sex-stratified color convention: male = blue, female = red
GENDER_COLORS = {"M": "#1565C0", "F": "#C62828"}


def _support_mask_fraction(
    days: pd.Series,
    ids: pd.Series,
    grid_days: np.ndarray,
    window_days: int = CI_WINDOW_DAYS,
    min_fraction: float = 0.10,
) -> np.ndarray:
    d = np.asarray(days, dtype=float)
    i = np.asarray(ids)
    finite = np.isfinite(d)
    d = d[finite]
    i = i[finite]
    grid = np.asarray(grid_days, dtype=float)
    if d.size == 0:
        return np.zeros_like(grid, dtype=bool)
    baseline_n = np.unique(i).size
    if baseline_n == 0:
        return np.zeros_like(grid, dtype=bool)
    out = np.zeros_like(grid, dtype=bool)
    for k, g in enumerate(grid):
        sel = (d >= g - window_days) & (d <= g + window_days)
        if not np.any(sel):
            continue
        frac = np.unique(i[sel]).size / float(baseline_n)
        out[k] = frac >= min_fraction
    return out


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


def _compute_mean_ci(result, X_pred: np.ndarray, z: float = Z_CRIT):
    mean = np.asarray(result.predict(X_pred), dtype=float)
    V = np.asarray(result.cov_params(), dtype=float)
    var = np.einsum("ij,jk,ik->i", X_pred, V, X_pred, optimize=True)
    var = np.clip(var, 0.0, None)
    se = np.sqrt(var)
    low = mean - z * se
    high = mean + z * se
    return mean, low, high


def main(argv=None):
    args = parse_args(argv)
    configure_logging(args.log_level)
    import re, os
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

    df = pd.read_csv(args.input_csv)
    gap_suffix = (
        f" (adherence gap = {args.adherence_gap_days} days)"
        if getattr(args, "adherence_gap_days", None)
        else ""
    )

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
    df = enforce_a1c_order(df, order=A1C_ORDER, context="step6_stratified_by_covariates_a1c")

    # Selected df from Step 2
    df_spline = load_spline_df(args.config_json)
    logging.info("Using spline df=%d", df_spline)

    consolidated_rows = []

    for var, var_label in STRAT_VARS:
        var_outdir = os.path.join(outdir, f"by_{var}")
        plots_unc = os.path.join(var_outdir, "plots", "uncentered")
        plots_ctr = os.path.join(var_outdir, "plots", "centered")
        plots_grp = os.path.join(var_outdir, "plots", "grouped")
        models_dir = os.path.join(var_outdir, "models")
        for d in (
            var_outdir,
            os.path.join(var_outdir, "plots"),
            plots_unc,
            plots_ctr,
            plots_grp,
            models_dir,
        ):
            os.makedirs(d, exist_ok=True)
        if var not in df.columns:
            logging.warning("Column '%s' missing; skipping", var)
            continue

        # Determine subgroups; skip NaN and optionally drop/reorder BMI levels
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

        # For baseline BMI, remove the explicit "Unknown" category and
        # enforce ordering: Normal, Overweight, Obese I, Obese II, Obese III,
        # with any other residual categories (e.g., Underweight) appended.
        if var == "baseline_bmi_final_category" and subgroups:
            subgroups = [s for s in subgroups if str(s) != "Unknown"]
            ordered = [s for s in BMI_CATEGORY_ORDER if s in subgroups]
            extras = [s for s in subgroups if s not in ordered]
            subgroups = ordered + extras
        logging.info("Stratifying by %s with %d groups", var_label, len(subgroups))
        overlay = []
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
                "Fitting GEE for %s=%s (Obs=%d, N=%d)",
                var_label,
                subgroup,
                n_obs,
                n_people,
            )
            try:
                covariates = [
                    "baseline_a1c_category",
                    "baseline_bmi_final_category",
                    "age_group",
                    "gender",
                    "race",
                    # 'metformin_with_glp1_baseline' removed per requirements
                    "weight_change_med",
                ]
                covariates = filter_estimable(
                    covariates, sub_df, exclude=[var],
                    context=f"step6 stratified a1c, var={var}",
                )
                bs_term = f"bs(days_from_baseline, df={df_spline})"
                formula = "abs_a1c_change ~ " + bs_term
                if covariates:
                    formula += " + " + " + ".join(covariates)
                y, X = dmatrices(formula, sub_df, return_type="dataframe")
                ids = sub_df.loc[y.index, "patient_id"]
                model = GEE(y, X, groups=ids, family=Gaussian(), cov_struct=Independence())
                res = model.fit()

                # Write per-subgroup CSV outputs (coefficients and covariance)
                sub_dir = os.path.join(models_dir, str(subgroup).replace(" ", "_"))
                os.makedirs(sub_dir, exist_ok=True)
                coef_df = pd.DataFrame({
                    'term': res.params.index if hasattr(res.params, 'index') else np.arange(len(res.params)),
                    'estimate': np.asarray(res.params, dtype=float),
                    'std_error': np.asarray(res.bse, dtype=float),
                    'p_value': np.asarray(res.pvalues, dtype=float)
                })
                try:
                    ci = res.conf_int()
                    coef_df['ci_lower'] = np.asarray(ci[0], dtype=float)
                    coef_df['ci_upper'] = np.asarray(ci[1], dtype=float)
                except Exception as exc:
                    logging.warning(
                        "could not attach confidence bounds to the coefficient table (%s); gee_coefficients.csv is written without ci_lower/ci_upper",
                        exc,
                    )
                coef_df.to_csv(os.path.join(sub_dir, 'gee_coefficients.csv'), index=False)
                try:
                    res.cov_params().to_csv(os.path.join(sub_dir, 'gee_covariance_matrix.csv'))
                except Exception as exc:
                    logging.warning(
                        "could not write gee_covariance_matrix.csv (%s); the model summary and coefficients are unaffected",
                        exc,
                    )
                # Per-subgroup overview
                pd.DataFrame([
                    {"metric": "unique_people", "value": int(n_people)},
                    {"metric": "observations", "value": int(n_obs)},
                ]).to_csv(os.path.join(sub_dir, 'gee_model_overview.csv'), index=False)

                # Prediction grid
                max_day = (
                    int(np.nanmax(sub_df["days_from_baseline"]))
                    if np.isfinite(sub_df["days_from_baseline"]).any()
                    else 365
                )
                # Cap to requested horizon (e.g., 548 days ~ 1.5 years)
                max_day = min(max_day, args.max_days)
                days_grid = np.arange(0, max_day + 1, 14)
                pred_df = pd.DataFrame({"days_from_baseline": days_grid})
                for c in [
                    "baseline_a1c_category",
                    "baseline_bmi_final_category",
                    "age_group",
                    "gender",
                    "race",
                    # 'metformin_with_glp1_baseline' removed
                    "weight_change_med",
                ]:
                    if c in sub_df.columns:
                        mode_val = (
                            sub_df[c].mode().iloc[0]
                            if not sub_df[c].mode().empty
                            else (
                                sub_df[c].cat.categories[0]
                                if hasattr(sub_df[c], "cat")
                                else 0
                            )
                        )
                        pred_df[c] = mode_val
                        if isinstance(sub_df[c].dtype, pd.CategoricalDtype):
                            pred_df[c] = pd.Categorical(
                                pred_df[c], categories=sub_df[c].cat.categories
                            )
                X_pred = build_design_matrices([X.design_info], pred_df)[0]
                X_pred = np.asarray(X_pred)
                mean_u, ci_low_u, ci_high_u = _compute_mean_ci(res, X_pred)
                anchor = mean_u[0]
                mean_c = mean_u - anchor
                ci_low_c = ci_low_u - anchor
                ci_high_c = ci_high_u - anchor

                support_mask = _support_mask_fraction(
                    sub_df["days_from_baseline"],
                    sub_df["patient_id"],
                    days_grid,
                    window_days=CI_WINDOW_DAYS,
                    min_fraction=0.10,
                )
                if not np.any(support_mask):
                    head_k = min(5, len(support_mask))
                    support_mask[:head_k] = True
            except Exception as e:  # pragma: no cover - defensive
                logging.warning(
                    "Fit/predict failed for %s=%s: %s", var_label, subgroup, e
                )
                continue

            summ_path = os.path.join(
                var_outdir,
                f"gee_summary_{var}_{str(subgroup).replace(' ', '_')}.txt",
            )
            with open(summ_path, "w") as f:
                f.write(
                    f"Stratified by {var_label} = {subgroup}\nUnique people: {n_people}\nObservations: {n_obs}\n\n"
                )
                f.write(str(res.summary()))

            import matplotlib.pyplot as plt

            # Uncentered
            _gc = GENDER_COLORS.get(str(subgroup)) if var == "gender" else None
            plt.figure(figsize=(10, 6))
            v = np.isfinite(mean_u)
            if np.any(v):
                (line,) = plt.plot(
                    days_grid[v],
                    mean_u[v],
                    label=f"{var_label}: {subgroup}",
                    linewidth=2,
                    **({"color": _gc} if _gc else {}),
                )
                _fill_between_segments(
                    plt.gca(), days_grid, ci_low_u, ci_high_u,
                    color=line.get_color(), alpha=0.15,
                )
            plt.xlabel("Days from Baseline")
            plt.ylabel("Absolute A1C Change")
            plt.ylim(-3, 1)
            plt.title(
                f"{var_label}: {subgroup} - A1C Change Over Time (Uncentered){gap_suffix}\nstarting sample n = {n_people}"
            )
            plt.axhline(0, color="black", linestyle="--", linewidth=1)
            plt.xlim(0, args.max_days)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()
            out_path = os.path.join(
                plots_unc, f"trajectory_{var}_{str(subgroup).replace(' ', '_')}.png"
            )
            plt.savefig(out_path, dpi=300)
            plt.close()

            # Centered
            plt.figure(figsize=(10, 6))
            v = np.isfinite(mean_c)
            if np.any(v):
                (line,) = plt.plot(
                    days_grid[v],
                    mean_c[v],
                    label=f"{var_label}: {subgroup}",
                    linewidth=2,
                    **({"color": _gc} if _gc else {}),
                )
                _fill_between_segments(
                    plt.gca(), days_grid, ci_low_c, ci_high_c,
                    color=line.get_color(), alpha=0.15,
                )
            plt.xlabel("Days from Baseline")
            plt.ylabel("Absolute A1C Change (Centered to Day 0)")
            plt.ylim(-3, 1)
            plt.title(
                f"{var_label}: {subgroup} - A1C Change Over Time{gap_suffix}\nstarting sample n = {n_people}"
            )
            plt.axhline(0, color="black", linestyle="--", linewidth=1)
            plt.xlim(0, args.max_days)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()
            out_path = os.path.join(
                plots_ctr, f"trajectory_{var}_{str(subgroup).replace(' ', '_')}.png"
            )
            plt.savefig(out_path, dpi=300)
            plt.close()

            overlay.append((subgroup, days_grid, mean_c, ci_low_c, ci_high_c, support_mask, int(n_people)))
            consolidated_rows.append(
                {
                    "strat_var": var,
                    "subgroup": subgroup,
                    "n_people": int(n_people),
                    "n_obs": int(n_obs),
                    "scale": float(res.scale),
                }
            )

        # Grouped overlay across subgroups
        if overlay:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(12, 7))
            total_n = 0
            handles = []
            labels = []
            for subgroup, days_grid, mean_c, ci_low_c, ci_high_c, support_mask, n_people in overlay:
                days_grid = np.asarray(days_grid, dtype=float)
                mean_c = np.asarray(mean_c, dtype=float)
                ci_low_c = np.asarray(ci_low_c, dtype=float)
                ci_high_c = np.asarray(ci_high_c, dtype=float)
                v = np.isfinite(mean_c)
                if np.any(v):
                    _gc2 = GENDER_COLORS.get(str(subgroup)) if var == "gender" else None
                    (line,) = ax.plot(
                        days_grid[v],
                        mean_c[v],
                        label=f"{var_label}: {subgroup}",
                        linewidth=2,
                        **({"color": _gc2} if _gc2 else {}),
                    )
                    _fill_between_segments(
                        ax, days_grid, ci_low_c, ci_high_c,
                        color=line.get_color(), alpha=0.12,
                    )
                    handles.append(line)
                    labels.append(f"{var_label}: {subgroup}")
                total_n += int(n_people)
            ax.set_xlabel("Days from Baseline")
            ax.set_ylabel("Absolute A1C Change (Centered to Day 0)")
            ax.set_ylim(-3, 1)
            if not args.suppress_grouped_titles:
                ax.set_title(
                    f"HbA1c Change by {var_label}{gap_suffix}\n(n\u2009=\u2009{total_n})"
                )
            ax.axhline(0, color="black", linestyle="--", linewidth=1)
            ax.set_xlim(0, args.max_days)
            if not args.suppress_grouped_legend and handles:
                ax.legend()
            ax.grid(True, linestyle="--", alpha=0.7)
            fig.tight_layout()
            fig.text(0.01, -0.02,
                "GEE spline model; trajectories centered to GLP-1 initiation (day\u20090). "
                "Shaded bands\u2009=\u200995% CI. Adjusted for age, sex, race, baseline HbA1c, and BMI.",
                fontsize=8, color='#444444', ha='left', va='top', wrap=True)
            out_path = os.path.join(plots_grp, f"grouped_trajectory_{var}.png")
            fig.savefig(out_path, dpi=300, bbox_inches='tight')

            # Optional legend-only PNG for use in panel figures
            if args.write_grouped_legend_only and handles:
                fig_leg, ax_leg = plt.subplots(figsize=(6, 4))
                ax_leg.axis("off")
                ax_leg.legend(
                    handles,
                    labels,
                    loc="center",
                    frameon=False,
                    fontsize=10,
                )
                out_leg = os.path.join(
                    plots_grp,
                    f"grouped_trajectory_{var}_legend.png",
                )
                fig_leg.savefig(out_leg, dpi=300, bbox_inches="tight")
                plt.close(fig_leg)

            plt.close(fig)

    if consolidated_rows:
        pd.DataFrame(consolidated_rows).to_csv(
            os.path.join(outdir, "stratified_summary_counts.csv"),
            index=False,
        )
        logging.info("Wrote stratified summary counts CSV")


if __name__ == "__main__":
    main()
