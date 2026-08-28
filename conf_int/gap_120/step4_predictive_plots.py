#!/usr/bin/env python3
"""Step 4 (weight outcome): Predictive plots of pct_weight_change over time by baseline A1C.

This adapts the A1C predictive script you provided to the weight
outcome, using pct_weight_change as the response.
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
from model_spec import load_spline_df

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
CI_MIN_COUNT = int(os.getenv("PLOT_CI_MIN_COUNT", "40"))
CI_QLOW = float(os.getenv("PLOT_CI_QLOW", "0.05"))
CI_QHIGH = float(os.getenv("PLOT_CI_QHIGH", "0.95"))
CI_MAX_ABS_WIDTH = float(os.getenv("PLOT_CI_MAX_ABS_WIDTH", "5.0"))
CI_WIDTH_RATIO = float(os.getenv("PLOT_CI_WIDTH_RATIO", "3.0"))
CI_MIN_UNIQUE = int(os.getenv("PLOT_CI_MIN_UNIQUE", "30"))
CI_ERODE_POINTS = int(os.getenv("PLOT_CI_ERODE_POINTS", "1"))
SHOW_CI = os.getenv("PLOT_SHOW_CI", "1") == "1"
TRAJECTORY_CI_STYLE = os.getenv("PLOT_TRAJECTORY_CI_STYLE", "fill").strip().lower()
SUPPORT_WINDOW_DAYS = int(os.getenv("PLOT_SUPPORT_WINDOW_DAYS", str(CI_WINDOW_DAYS)))
MIN_SUPPORT_FRACTION = float(os.getenv("PLOT_MIN_SUPPORT_FRACTION", "0.10"))


def configure_logging(level: str = "INFO"):
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Step 4 (weight): Predictive plots by baseline A1C category")
    p.add_argument("--input-csv", default=os.path.join("output", "step1_prepare_analysis_dataset", "analysis_ready_gap90.csv"), help="Path to weight analysis-ready CSV from Step 1 (weight)")
    p.add_argument("--config-json", default=os.path.join("output", "step2_select_spline_df", "model_config.json"), help="Model config JSON from Step 2 (weight)")
    p.add_argument("--outdir", default=os.path.join("output", "step4_predictive_plots"), help="Directory to write plots")
    p.add_argument("--adherence-gap-days", type=int, default=None, help="Adherence gap in days to display in plot titles (e.g., 90, 120, 180)")
    p.add_argument("--min-nobs", type=int, default=100, help="Minimum observations per A1C stratum to plot")
    p.add_argument("--max-days", type=int, default=548, help="Maximum days from baseline to include in prediction grid (default: 548)")
    p.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    # Cosmetic options for grouped plots; default False so existing pipelines are unchanged
    p.add_argument("--suppress-grouped-titles", action="store_true", help="If set, suppress titles on grouped trajectory plots (centered and uncentered)")
    p.add_argument("--suppress-grouped-legend", action="store_true", help="If set, suppress legends on grouped trajectory plots (centered and uncentered)")
    p.add_argument("--write-grouped-legend-only", action="store_true", help="If set, write a separate PNG with just the grouped trajectory legend (for panel figures)")
    return p.parse_args(argv)


def _compute_mean_ci(result, X_pred: np.ndarray, z: float = Z_CRIT):
    mean = np.asarray(result.predict(X_pred), dtype=float)
    V = np.asarray(result.cov_params(), dtype=float)
    var = np.einsum('ij,jk,ik->i', X_pred, V, X_pred, optimize=True)
    var = np.clip(var, 0.0, None)
    se = np.sqrt(var)
    low = mean - z * se
    high = mean + z * se
    return mean, low, high


def _supported_mask(day_values, grid_days, window_days: int = CI_WINDOW_DAYS, min_count: int = CI_MIN_COUNT):
    day_vals = np.asarray(day_values, dtype=float)
    day_vals = day_vals[np.isfinite(day_vals)]
    grid = np.asarray(grid_days, dtype=float)
    if day_vals.size == 0:
        return np.zeros_like(grid, dtype=bool)
    counts = np.array([
        np.count_nonzero((day_vals >= d - window_days) & (day_vals <= d + window_days))
        for d in grid
    ])
    return counts >= min_count


def _supported_mask_patients(day_values, id_values, grid_days, window_days: int = CI_WINDOW_DAYS, min_unique: int = CI_MIN_UNIQUE):
    days = np.asarray(day_values, dtype=float)
    ids = np.asarray(id_values)
    finite = np.isfinite(days)
    days = days[finite]
    ids = ids[finite]
    grid = np.asarray(grid_days, dtype=float)
    if days.size == 0:
        return np.zeros_like(grid, dtype=bool)
    out = np.zeros_like(grid, dtype=bool)
    for i, d in enumerate(grid):
        sel = np.abs(days - d) <= window_days
        if not np.any(sel):
            out[i] = False
            continue
        out[i] = np.unique(ids[sel]).size >= min_unique
    return out


def _mask_by_ci_width(ci_low, ci_high, max_abs: float = CI_MAX_ABS_WIDTH, ratio: float = CI_WIDTH_RATIO):
    lo = np.asarray(ci_low, dtype=float)
    hi = np.asarray(ci_high, dtype=float)
    width = np.abs(hi - lo)
    finite = np.isfinite(width)
    if not np.any(finite):
        return np.zeros_like(width, dtype=bool)
    median_w = np.nanmedian(width[finite])
    ok_abs = width <= max_abs
    ok_rel = width <= (ratio * median_w if np.isfinite(median_w) and median_w > 0 else width)
    return (ok_abs & ok_rel & finite)


def _erode_mask(mask: np.ndarray, erode_points: int = CI_ERODE_POINTS) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    if erode_points <= 0 or m.size == 0:
        return m
    idx = np.where(m)[0]
    if idx.size == 0:
        return m
    splits = np.where(np.diff(idx) > 1)[0] + 1
    runs = np.split(idx, splits)
    out = m.copy()
    out[:] = False
    for r in runs:
        if r.size <= 2 * erode_points:
            continue
        trimmed = r[erode_points:-erode_points]
        out[trimmed] = True
    return out


def _support_mask_fraction(day_values, id_values, grid_days, window_days: int, min_fraction: float) -> np.ndarray:
    days = np.asarray(day_values, dtype=float)
    ids = np.asarray(id_values)
    finite = np.isfinite(days)
    days = days[finite]
    ids = ids[finite]
    grid = np.asarray(grid_days, dtype=float)
    if days.size == 0:
        return np.zeros_like(grid, dtype=bool)
    baseline_n = np.unique(ids).size
    if baseline_n == 0:
        return np.zeros_like(grid, dtype=bool)
    out = np.zeros_like(grid, dtype=bool)
    for i, d in enumerate(grid):
        sel = (days >= d - window_days) & (days <= d + window_days)
        if not np.any(sel):
            continue
        frac = np.unique(ids[sel]).size / float(baseline_n)
        out[i] = frac >= min_fraction
    return out


def _fill_between_segments(ax, x, y_low, y_high, color=None, alpha=0.15, zorder=1, min_run_points: int = 3):
    x = np.asarray(x, dtype=float)
    y_low = np.asarray(y_low, dtype=float)
    y_high = np.asarray(y_high, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y_low) & np.isfinite(y_high)
    if not np.any(valid):
        return
    lo = np.minimum(y_low, y_high)
    hi = np.maximum(y_low, y_high)
    idx = np.where(valid)[0]
    splits = np.where(np.diff(idx) > 1)[0] + 1
    blocks = np.split(idx, splits)
    for b in blocks:
        if b.size < max(2, int(min_run_points)):
            continue
        ax.fill_between(x[b], lo[b], hi[b], alpha=alpha, color=color, zorder=zorder)


def _plot_line_segments(ax, x, y, color=None, alpha=0.8, zorder=1, linestyle='--', linewidth=1.0, min_run_points: int = 3):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return
    idx = np.where(valid)[0]
    splits = np.where(np.diff(idx) > 1)[0] + 1
    blocks = np.split(idx, splits)
    for b in blocks:
        if b.size < max(2, int(min_run_points)):
            continue
        ax.plot(x[b], y[b], color=color, alpha=alpha, zorder=zorder, linestyle=linestyle, linewidth=linewidth)


def _fit_gee(sub_df: pd.DataFrame, df_spline: int):
    # Remove metformin_with_glp1_baseline from covariates per requirements
    covariates = [
        "age_group",
        "gender",
        "baseline_bmi_final_category",
        "race",
        "baseline_a1c_category",
    ]
    if "weight_change_med" in sub_df.columns:
        covariates.append("weight_change_med")
    covariates = filter_estimable(covariates, sub_df, context="step4 predictive weight")
    bs_term = f"bs(days_from_baseline, df={df_spline})"
    formula = f"pct_weight_change ~ {bs_term}" + (" + " + " + ".join(covariates) if covariates else "")
    y, X = dmatrices(formula, sub_df, return_type='dataframe')
    ids = sub_df.loc[y.index, 'patient_id']
    model = GEE(y, X, groups=ids, family=Gaussian(), cov_struct=Independence())
    result = model.fit()
    return result, X.design_info


def _legend_label_for_a1c_category(subgroup: str) -> str:
    """Return legend labels with A1C ranges for each baseline category.

    Removes the "A1C:" prefix and attaches clinically meaningful ranges.
    """

    mapping = {
        "Normal Glycemia": "Normal glycemia (<5.7%)",
        "Prediabetes": "Prediabetes (5.7% to <6.5%)",
        "Type 2 Diabetes": "Type 2 diabetes (6.5% to <9.0%)",
        "Poorly Controlled Diabetes": "Poorly controlled diabetes (>=9.0%)",
    }
    return mapping.get(str(subgroup), str(subgroup))


def main(argv=None):
    args = parse_args(argv)
    configure_logging(args.log_level)
    # Gap-aware outdir routing
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
    uncentered_out = os.path.join(outdir, "uncentered")
    centered_out = os.path.join(outdir, "centered")
    grouped_out = os.path.join(outdir, "grouped")
    tables_out = os.path.join(outdir, "tables")
    summaries_out = os.path.join(outdir, "summaries")
    models_out = os.path.join(outdir, "models")
    for d in (uncentered_out, centered_out, grouped_out, tables_out, summaries_out, models_out):
        os.makedirs(d, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    for cat_col in ["gender", "baseline_a1c_category", "baseline_bmi_final_category", "race", "age_group"]:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype("category")

    df_spline = load_spline_df(args.config_json)
    logging.info("Using spline df=%d", df_spline)
    gap_suffix = f" (adherence gap = {args.adherence_gap_days} days)" if args.adherence_gap_days else ""

    a1c_col = "baseline_a1c_category"
    A1C_ORDER = [
        "Normal Glycemia",
        "Prediabetes",
        "Type 2 Diabetes",
        "Poorly Controlled Diabetes",
        "Unknown",
    ]
    strata = [cat for cat in (list(df[a1c_col].cat.categories) if hasattr(df[a1c_col], 'cat') else A1C_ORDER) if pd.notna(cat) and cat in A1C_ORDER]
    a1c_rank = {cat: i for i, cat in enumerate(A1C_ORDER)}
    overlay_results = {}
    tables = []

    for subgroup in strata:
        sub_df = df[df[a1c_col] == subgroup].copy()
        n_obs = len(sub_df)
        n_people = sub_df['patient_id'].nunique()
        if n_obs < args.min_nobs:
            logging.info("Skip A1C=%s (Obs=%d, N=%d): insufficient data", subgroup, n_obs, n_people)
            continue
        res, design_info = _fit_gee(sub_df, df_spline)
        summ_path = os.path.join(summaries_out, f"gee_summary_weight_outcome_pct_weight_change_{str(subgroup).replace(' ', '_')}.txt")
        with open(summ_path, 'w') as f:
            f.write("Outcome: pct_weight_change (%)\n")
            f.write(f"A1C Category: {subgroup}\nUnique people: {n_people}\nObservations: {n_obs}\n\n")
            f.write(str(res.summary()))
        # Write CSV artifacts for each model run (coefficients and covariance)
        sub_dir = os.path.join(models_out, str(subgroup).replace(' ', '_'))
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
        # Per-model overview (sample size) like step 8
        pd.DataFrame([
            {"metric": "unique_people", "value": int(n_people)},
            {"metric": "observations", "value": int(n_obs)},
        ]).to_csv(os.path.join(sub_dir, 'gee_model_overview.csv'), index=False)

        # Build prediction grid capped at max-days
        observed_max = int(np.nanmax(sub_df['days_from_baseline'])) if np.isfinite(sub_df['days_from_baseline']).any() else 366
        grid_max = int(min(args.max_days, observed_max))
        days_grid = np.arange(0, grid_max + 1, 14)
        if 0 not in days_grid:
            days_grid = np.sort(np.append(days_grid, 0))
        pred_df = pd.DataFrame({'days_from_baseline': days_grid})
        for c in ["age_group", "gender", "baseline_bmi_final_category", "race", "weight_change_med"]:
            if c in sub_df.columns:
                mode_val = sub_df[c].mode().iloc[0] if not sub_df[c].mode().empty else (sub_df[c].cat.categories[0] if hasattr(sub_df[c], 'cat') else 0)
                pred_df[c] = mode_val
                if isinstance(sub_df[c].dtype, pd.CategoricalDtype):
                    pred_df[c] = pd.Categorical(pred_df[c], categories=sub_df[c].cat.categories)
        pred_df[a1c_col] = subgroup
        if isinstance(sub_df[a1c_col].dtype, pd.CategoricalDtype):
            pred_df[a1c_col] = pd.Categorical(pred_df[a1c_col], categories=sub_df[a1c_col].cat.categories)
        X_pred = build_design_matrices([design_info], pred_df)[0]
        X_pred = np.asarray(X_pred)

        pred_mean_u, ci_low_u, ci_high_u = _compute_mean_ci(res, X_pred)
        x_vals = np.asarray(days_grid, dtype=float)
        supported_counts = _supported_mask(sub_df['days_from_baseline'], days_grid)
        supported_patients = _supported_mask_patients(sub_df['days_from_baseline'], sub_df['patient_id'], days_grid)
        supported_fraction = _support_mask_fraction(
            sub_df['days_from_baseline'],
            sub_df['patient_id'],
            days_grid,
            window_days=SUPPORT_WINDOW_DAYS,
            min_fraction=MIN_SUPPORT_FRACTION,
        )
        obs_days = sub_df['days_from_baseline'].to_numpy()
        obs_days = obs_days[np.isfinite(obs_days)]
        if obs_days.size > 0:
            qlo = np.quantile(obs_days, CI_QLOW)
            qhi = np.quantile(obs_days, CI_QHIGH)
            within_q = (x_vals >= qlo - CI_WINDOW_DAYS) & (x_vals <= qhi + CI_WINDOW_DAYS)
        else:
            within_q = np.zeros_like(x_vals, dtype=bool)
        width_ok = _mask_by_ci_width(ci_low_u, ci_high_u)
        raw_supported = supported_counts & supported_patients & supported_fraction & within_q & width_ok
        supported_mask = _erode_mask(raw_supported)
        if 0 in days_grid:
            zero_idx = int(np.where(days_grid == 0)[0][0])
            supported_mask[zero_idx] = True

        try:
            import matplotlib as mpl
            cmap = mpl.colormaps.get_cmap('tab10')
            subgroup_idx = strata.index(subgroup) if subgroup in strata else 0
            color = cmap(subgroup_idx % 10)
        except Exception:
            color = None

        # Uncentered plot: solid colored line for full window; CI over full range
        plt.figure(figsize=(10, 6))
        if SHOW_CI:
            if TRAJECTORY_CI_STYLE in ("line", "lines", "dash", "dashed"):
                _plot_line_segments(plt.gca(), x_vals, ci_low_u, color=color)
                _plot_line_segments(plt.gca(), x_vals, ci_high_u, color=color)
            else:
                _fill_between_segments(plt.gca(), x_vals, ci_low_u, ci_high_u, color=color)
        plt.plot(x_vals, pred_mean_u, label=f"A1C: {subgroup}", color=color)
        plt.xlabel('Days from Baseline')
        plt.ylabel('Percent Weight Change (%)')
        plt.ylim(-10, 4)
        plt.title(f'A1C: {subgroup} - Percent Weight Change Over Time (Uncentered){gap_suffix}\nstarting sample n = {n_people}')
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        out_path = os.path.join(uncentered_out, f"{a1c_rank.get(subgroup,99)+1:02d}_trajectory_weight_outcome_pct_weight_change_{str(subgroup).replace(' ', '_')}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()

        # Centered plot: solid colored line for full window; CI over full range
        anchor = pred_mean_u[0]
        pred_mean_c = pred_mean_u - anchor
        ci_low_c = ci_low_u - anchor
        ci_high_c = ci_high_u - anchor
        plt.figure(figsize=(10, 6))
        if SHOW_CI:
            if TRAJECTORY_CI_STYLE in ("line", "lines", "dash", "dashed"):
                _plot_line_segments(plt.gca(), x_vals, ci_low_c, color=color)
                _plot_line_segments(plt.gca(), x_vals, ci_high_c, color=color)
            else:
                _fill_between_segments(plt.gca(), x_vals, ci_low_c, ci_high_c, color=color)
        # Force Day 0 to 0 visually
        pmc = pred_mean_c.copy()
        zero_idx = np.where(x_vals == 0)[0]
        if zero_idx.size > 0:
            pmc[zero_idx[0]] = 0.0
        plt.plot(x_vals, pmc, label=_legend_label_for_a1c_category(subgroup), color=color)
        plt.xlabel('Days from Baseline')
        plt.ylabel('Percent Weight Change (Centered to Day 0)')
        plt.ylim(-10, 4)
        plt.title(f'A1C: {subgroup} - Percent Weight Change Over Time{gap_suffix}\nstarting sample n = {n_people}')
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        out_path = os.path.join(centered_out, f"{a1c_rank.get(subgroup,99)+1:02d}_trajectory_weight_outcome_pct_weight_change_{str(subgroup).replace(' ', '_')}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()

        overlay_results[subgroup] = {
            'pred_mean_unc': pred_mean_u,
            'pred_mean_ctr': pred_mean_c,
            'ci_low_unc': ci_low_u,
            'ci_high_unc': ci_high_u,
            'ci_low_ctr': ci_low_c,
            'ci_high_ctr': ci_high_c,
            'supported_mask': supported_mask,
            'days_grid': days_grid,
            'label': _legend_label_for_a1c_category(subgroup),
            'n_people': int(n_people),
        }

        tbl = pd.DataFrame({
            'a1c_group': subgroup,
            'day': x_vals,
            'pred_mean_uncentered': pred_mean_u,
            'ci_low_uncentered': ci_low_u,
            'ci_high_uncentered': ci_high_u,
            'pred_mean_centered': pred_mean_c,
            'ci_low_centered': ci_low_c,
            'ci_high_centered': ci_high_c,
            'supported': overlay_results[subgroup]['supported_mask'],
        })
        tables.append(tbl)

    plt.figure(figsize=(12, 7))
    # Enforce legend order: Normal, Prediabetes, Type 2, Poorly Controlled, then others
    preferred = [
        "Normal Glycemia",
        "Prediabetes",
        "Type 2 Diabetes",
        "Poorly Controlled Diabetes",
    ]
    ordered = [c for c in preferred if c in overlay_results]
    remaining = [g for g in overlay_results.keys() if g not in ordered]
    legend_order = ordered + remaining
    for subgroup in legend_order:
        res = overlay_results[subgroup]
        try:
            import matplotlib as mpl
            cmap = mpl.colormaps.get_cmap('tab10')
            color = cmap(legend_order.index(subgroup) % 10)
        except Exception:
            color = None
        xg = np.asarray(res['days_grid'], dtype=float)
        mg = np.asarray(res['pred_mean_unc'], dtype=float)
        plt.plot(xg, mg, label=res['label'], color=color)
        # 95% CI bands on grouped overlay
        if SHOW_CI:
            ci_lo = np.asarray(res['ci_low_unc'], dtype=float)
            ci_hi = np.asarray(res['ci_high_unc'], dtype=float)
            _fill_between_segments(plt.gca(), xg, ci_lo, ci_hi, color=color, alpha=0.12)
    plt.xlabel('Days from Baseline')
    plt.ylabel('Percent Weight Change (%)')
    plt.ylim(-10, 4)
    total_n = int(sum(r['n_people'] for r in overlay_results.values())) if overlay_results else 0
    if not args.suppress_grouped_titles:
        plt.title(f'Baseline A1C Category - Grouped Trajectories (Uncentered, weight outcome){gap_suffix}\nstarting sample n = {total_n}')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    if not args.suppress_grouped_legend:
        plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(grouped_out, "grouped_trajectory_uncentered_outcome_weight.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(12, 7))
    for subgroup in legend_order:
        res = overlay_results[subgroup]
        try:
            import matplotlib as mpl
            cmap = mpl.colormaps.get_cmap('tab10')
            color = cmap(legend_order.index(subgroup) % 10)
        except Exception:
            color = None
        xg = np.asarray(res['days_grid'], dtype=float)
        mg = np.asarray(res['pred_mean_ctr'], dtype=float)
        zero_idx = np.where(xg == 0)[0]
        if zero_idx.size > 0:
            mg = mg.copy()
            mg[zero_idx[0]] = 0.0
        if SHOW_CI:
            ci_lo = np.asarray(res['ci_low_ctr'], dtype=float)
            ci_hi = np.asarray(res['ci_high_ctr'], dtype=float)
            _fill_between_segments(plt.gca(), xg, ci_lo, ci_hi, color=color, alpha=0.12)
        plt.plot(xg, mg, label=res['label'], color=color)
    plt.xlabel('Days from Baseline')
    plt.ylabel('Percent Weight Change (Centered to Day 0)')
    plt.ylim(-10, 4)
    if not args.suppress_grouped_titles:
        plt.title(f'Baseline A1C Category - Grouped Trajectories (weight outcome){gap_suffix}\nstarting sample n = {total_n}')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    if not args.suppress_grouped_legend:
        plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    centered_path = os.path.join(grouped_out, "grouped_trajectory_centered_outcome_weight.png")
    plt.savefig(centered_path, dpi=300)

    # Optionally emit a legend-only PNG for use in panel figures
    if args.write_grouped_legend_only:
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            import matplotlib.pyplot as _plt
            fig_leg = _plt.figure(figsize=(4, 2))
            # Match legend font size to the panel legend title size
            fig_leg.legend(handles, labels, loc='center', ncol=1, frameon=False, fontsize=12)
            legend_path = os.path.join(grouped_out, "grouped_trajectory_legend_weight.png")
            fig_leg.savefig(legend_path, dpi=300, bbox_inches='tight')
            _plt.close(fig_leg)

    plt.close()

    if tables:
        pd.concat(tables, ignore_index=True).to_csv(os.path.join(tables_out, "predicted_means_by_day_weight_outcome_pct_weight_change.csv"), index=False)


if __name__ == "__main__":
    main()
