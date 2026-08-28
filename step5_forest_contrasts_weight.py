#!/usr/bin/env python3
"""
Step 5: Forest-style contrasts across baseline A1C categories (Weight)

Audience-friendly summary:
- We fit one combined GEE model that includes time-by-A1C interactions.
- We estimate average percent weight change at 3, 6, 9, 12, 15, 18, 21, and 24 months
  (approx. 90, 180, 270, 365, 450, 548, 630, 730 days).
- We produce forest-style plots that compare categories and show uncertainty.
- We also compute pairwise contrasts vs a reference category to quantify differences.
"""

import os
import argparse
import logging
from pathlib import Path

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


CI_WINDOW_DAYS = int(os.getenv("FOREST_WINDOW_DAYS", "28"))

def configure_logging(level: str = "INFO"):
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Step 5: Forest-style contrasts across A1C categories (Weight)")
    p.add_argument(
        "--input-csv",
        default=os.path.join("output", "step1_prepare_analysis_dataset", "analysis_ready_gap90.csv"),
        help="Path to analysis-ready weight CSV from Step 1",
    )
    p.add_argument(
        "--config-json",
        default=os.path.join("output", "step2_select_spline_df", "model_config.json"),
        help="Model config JSON from Step 2 (contains best df)",
    )
    p.add_argument(
        "--outdir",
        default=os.path.join("output", "step5_forest_contrasts_weight"),
        help="Directory to write outputs",
    )
    p.add_argument(
        "--adherence-gap-days",
        type=int,
        default=None,
        help="Adherence gap in days for gap-specific output folder (e.g., 90)",
    )
    # Months: 3, 6, 9, 12, 15, 18, 21, 24 -> days: 90, 180, 270, 365, 450, 548, 630, 730
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
    p.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    return p.parse_args(argv)


def _fit_combined(df: pd.DataFrame, df_spline: int):
    bs_term = f"bs(days_from_baseline, df={df_spline})"
    formula = (
        f"pct_weight_change ~ {bs_term} * baseline_a1c_category + "
        f"age_group + gender + baseline_bmi_final_category + race"
    )
    # Log the inclusion decision rather than taking it silently, so the covariate
    # set behind every fit is visible in the run log (code-review item on
    # silently dropped covariates).
    if "weight_change_med" not in df.columns:
        logging.warning(
            "weight_change_med is absent from the input; the model will NOT adjust "
            "for concomitant weight-affecting medications"
        )
    elif df["weight_change_med"].nunique(dropna=True) <= 1:
        logging.warning(
            "weight_change_med has a single value (%r); dropped as not estimable, "
            "so this model does not adjust for concomitant weight-affecting "
            "medications",
            df["weight_change_med"].dropna().unique()[:1],
        )
    else:
        formula += " + weight_change_med"
        logging.info("Model adjusts for weight_change_med")
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


def main(argv=None):
    args = parse_args(argv)
    configure_logging(args.log_level)
    # Force non-interactive backend for headless runs
    try:
        import matplotlib
        matplotlib.use("Agg")
    except Exception as exc:
        logging.debug(
            "could not select the Agg matplotlib backend (%s); the configured backend is used instead",
            exc,
        )

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
    # Enforce A1C order
    a1c_order = [
        "Normal Glycemia",
        "Prediabetes",
        "Type 2 Diabetes",
        "Poorly Controlled Diabetes",
    ]
    # Fails loudly: this fixes the reference level for every contrast
    # (code-review item 8). See model_spec.enforce_a1c_order.
    df = enforce_a1c_order(df, order=a1c_order, context="step5_forest_contrasts_weight")

    # Helper: windowed counts by day and category
    def _counts_for_day_cat(df_in: pd.DataFrame, day: int, cat: str, window_days: int):
        dd = df_in[
            (df_in["baseline_a1c_category"] == cat)
            & (np.isfinite(df_in["days_from_baseline"]))
        ]
        days = dd["days_from_baseline"].to_numpy()
        ids = dd["patient_id"].to_numpy()
        sel = (days >= day - window_days) & (days <= day + window_days)
        n_obs = int(np.count_nonzero(sel))
        n_unique = int(np.unique(ids[sel]).size) if n_obs > 0 else 0
        return n_obs, n_unique

    df_spline = load_spline_df(args.config_json)

    res, design_info = _fit_combined(df, df_spline)
    # Save combined model summary and coefficients
    with open(os.path.join(outdir, "gee_summary_combined.txt"), "w") as f:
        f.write(str(res.summary()))
    # Robust CI extraction across statsmodels versions
    ci = res.conf_int()
    try:
        ci_lower = ci.iloc[:, 0]
        ci_upper = ci.iloc[:, 1]
    except Exception:
        # Fallback if conf_int returns ndarray
        ci = np.asarray(ci)
        ci_lower = ci[:, 0]
        ci_upper = ci[:, 1]
    # Code-review item 4: this table was written with index=False, dropping the
    # Series index that held the term names, so gee_combined_coefficients.csv had
    # Coef / CI Lower / CI Upper / p-value columns against unlabelled rows. Every
    # sibling script wrote the names; the weight version was the only one that lost
    # them. Nothing in the pipeline reads this file back, so no result was affected
    # — it is a record-keeping defect. Mirrors the A1c twin, which does it correctly.
    coef_df = pd.DataFrame({
        "term": (
            res.params.index
            if hasattr(res.params, "index")
            else np.arange(len(res.params))
        ),
        "Coef": np.asarray(res.params, dtype=float),
        "CI Lower": np.asarray(ci_lower, dtype=float),
        "CI Upper": np.asarray(ci_upper, dtype=float),
        "p-value": np.asarray(res.pvalues, dtype=float),
    })
    coef_df.to_csv(
        os.path.join(outdir, "gee_combined_coefficients.csv"), index=False
    )
    logging.info(
        "Wrote %d labelled coefficients to gee_combined_coefficients.csv",
        len(coef_df),
    )

    # Try to load Step 7 adherence milestone counts for overall n annotations
    counts_overall = {}
    try:
        scenario_dir = Path(args.outdir).parent  # output/<scenario>
        step7_path = scenario_dir / "step7_adherence_counts" / "adherence_counts.csv"
        if step7_path.exists():
            acc = pd.read_csv(step7_path)
            acc_overall = acc[(acc["scope"] == "overall")]
            # Map day -> n_unique_window
            counts_overall = {
                int(r["day"]): int(r["n_unique_window"])
                for _, r in acc_overall.iterrows()
            }
    except Exception:
        counts_overall = {}

    a1c_cats = list(df["baseline_a1c_category"].cat.categories)
    # Choose reference: Normal Glycemia if present, else first category
    ref_cat = "Normal Glycemia" if "Normal Glycemia" in a1c_cats else a1c_cats[0]

    # Build prediction rows at specified times
    times = [int(x.strip()) for x in args.time_days.split(",") if x.strip()]
    mode_vals = {}
    for c in [
        "age_group",
        "gender",
        "baseline_bmi_final_category",
        "race",
        "weight_change_med",
    ]:
        if c in df.columns:
            m = df[c].mode()
            if not m.empty:
                mode_vals[c] = m.iloc[0]
            elif hasattr(df[c], "cat"):
                mode_vals[c] = df[c].cat.categories[0]

    pred_rows = []
    contrast_rows = []

    for d in times:
        for cat in a1c_cats:
            pred_df = pd.DataFrame(
                {"days_from_baseline": [d], "baseline_a1c_category": [cat]}
            )
            for c, v in mode_vals.items():
                if c in df.columns:
                    pred_df[c] = [v]
                    if isinstance(df[c].dtype, pd.CategoricalDtype):
                        pred_df[c] = pd.Categorical(
                            pred_df[c], categories=df[c].cat.categories
                        )
            if isinstance(df["baseline_a1c_category"].dtype, pd.CategoricalDtype):
                pred_df["baseline_a1c_category"] = pd.Categorical(
                    pred_df["baseline_a1c_category"],
                    categories=df["baseline_a1c_category"].cat.categories,
                )
            Xp = build_design_matrices([design_info], pred_df)[0]
            Xp = np.asarray(Xp)[0]
            mean, lo, hi, se = _pred_ci(res, Xp)
            n_obs, n_unique = _counts_for_day_cat(df, d, cat, args.window_days)
            pred_rows.append(
                {
                    "day": d,
                    "a1c_group": cat,
                    "pred": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "se": se,
                    "n_obs_window": n_obs,
                    "n_unique_window": n_unique,
                }
            )
        # Pairwise contrasts vs reference
        for cat in a1c_cats:
            if cat == ref_cat:
                continue
            # Build rows for ref and cat
            df_ref = pd.DataFrame(
                {"days_from_baseline": [d], "baseline_a1c_category": [ref_cat]}
            )
            df_cat = pd.DataFrame(
                {"days_from_baseline": [d], "baseline_a1c_category": [cat]}
            )
            for c, v in mode_vals.items():
                for P in (df_ref, df_cat):
                    if c in df.columns:
                        P[c] = [v]
                        if isinstance(df[c].dtype, pd.CategoricalDtype):
                            P[c] = pd.Categorical(P[c], categories=df[c].cat.categories)
            for P in (df_ref, df_cat):
                if isinstance(df["baseline_a1c_category"].dtype, pd.CategoricalDtype):
                    P["baseline_a1c_category"] = pd.Categorical(
                        P["baseline_a1c_category"],
                        categories=df["baseline_a1c_category"].cat.categories,
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
    preds.to_csv(os.path.join(outdir, "forest_predictions.csv"), index=False)
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(
        os.path.join(outdir, "forest_contrasts_vs_ref.csv"), index=False
    )

    # Build map for n by (day, group) for reuse in annotations
    counts_map = {
        (int(r.day), str(r.a1c_group)): int(r.n_unique_window)
        for _, r in preds.iterrows()
    }

    # Forest plots for each time point (predictions) — right-side text column with header
    for d in times:
        sub = preds[preds["day"] == d].copy()
        sub["order"] = sub["a1c_group"].apply(lambda x: a1c_cats.index(x))
        sub = sub.sort_values("order")
        y = np.arange(len(sub))
        labels = list(sub["a1c_group"])
        texts = [
            f"{row['pred']:.2f} ({row['ci_low']:.2f}, {row['ci_high']:.2f})  n={int(row['n_unique_window'])}"
            for _, row in sub.iterrows()
        ]

        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[4.5, 1.8])
        ax = fig.add_subplot(gs[0, 0])
        ax_txt = fig.add_subplot(gs[0, 1])

        ax.errorbar(
            sub["pred"],
            y,
            xerr=[sub["pred"] - sub["ci_low"], sub["ci_high"] - sub["pred"]],
            fmt="o",
            capsize=3,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Predicted Percent Weight Change")
        n_all = counts_overall.get(d)
        title_suffix = f" (n={n_all})" if n_all is not None else ""
        ax.set_title(
            f"Predicted Weight Change by Baseline A1C Category (Day {d})" f"{title_suffix}"
        )
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)

        x_min = float((sub["ci_low"]).min())
        x_max = float((sub["ci_high"]).max())
        x_range = x_max - x_min if x_max > x_min else 1.0
        left_lim = x_min - 0.1 * x_range
        right_lim = max(x_max + 0.1 * x_range, 0 + 0.3 * x_range)
        ax.set_xlim(left_lim, right_lim)

        # Right-side text column
        ax_txt.axis("off")
        ax_txt.set_title("Estimate (95% CI)")
        ax_txt.set_ylim(y.min() - 0.5, y.max() + 0.5)
        ax_txt.invert_yaxis()  # ensure text rows align with left axis order
        for yi, t in zip(y, texts):
            ax_txt.text(0.0, yi, t, va="center", ha="left", fontsize=9, transform=ax_txt.transData)
        ax_txt.set_xlim(0, 1)

        plt.tight_layout()
        fig.savefig(
            os.path.join(outdir, f"forest_predictions_day_{d}.png"), dpi=150, bbox_inches='tight'
        )
        plt.close(fig)

    # Combined forest plot across all time points (predictions) — right-side text column with header
    combined = preds.copy()
    combined["group_order"] = combined["a1c_group"].apply(lambda x: a1c_cats.index(x))
    combined = combined.sort_values(["group_order", "day"])  # group, then time

    day_values = sorted(times)
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

    fig = plt.figure(figsize=(12, 0.4 * n_groups * n_days + 2))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.5, 1.8])
    ax = fig.add_subplot(gs[0, 0])
    ax_txt = fig.add_subplot(gs[0, 1])

    colors = plt.cm.tab10(np.linspace(0, 1, n_days))
    for idx, d in enumerate(day_values):
        sub = stacked[stacked["day"] == d]
        ax.errorbar(
            sub["pred"],
            sub["y_index"],
            xerr=[sub["pred"] - sub["ci_low"], sub["ci_high"] - sub["pred"]],
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
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted Percent Weight Change")
    ax.set_title("Predicted Weight Change by Baseline A1C Category and Time")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    ax.legend(title="Time from Baseline", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.invert_yaxis()

    if not stacked.empty:
        x_min = float((stacked["ci_low"]).min())
        x_max = float((stacked["ci_high"]).max())
        x_range = x_max - x_min if x_max > x_min else 1.0
        left_lim = x_min - 0.1 * x_range
        right_lim = max(x_max + 0.1 * x_range, 0 + 0.3 * x_range)
        ax.set_xlim(left_lim, right_lim)

    # Right text column
    ax_txt.axis("off")
    ax_txt.set_title("Estimate (95% CI)")
    if y_ticks:
        ax_txt.set_ylim(min(y_ticks) - 0.5, max(y_ticks) + 0.5)
    ax_txt.invert_yaxis()  # align text rows with inverted left axis
    for yv, t in zip(y_ticks, texts):
        ax_txt.text(0.0, yv, t, va="center", ha="left", fontsize=8)
    ax_txt.set_xlim(0, 1)

    plt.tight_layout()
    fig.savefig(
        os.path.join(outdir, "forest_predictions_all_days.png"), dpi=150, bbox_inches='tight'
    )
    plt.close(fig)

    # New: Grouped multi-day forest with spacers (all groups’ predictions, not differences)
    # Uses the time points provided via --time-days (e.g., 90/180/365/548) and inserts a blank spacer between clusters
    day_clusters = day_values
    gap = 1
    records_gbd = []
    y_base = 0
    for d in day_clusters:
        for cat_idx, cat in enumerate(a1c_cats):
            row = combined[(combined["a1c_group"] == cat) & (combined["day"] == d)]
            if row.empty:
                continue
            r = row.iloc[0].copy()
            r["y_index"] = y_base + cat_idx
            records_gbd.append(r)
        y_base += len(a1c_cats) + gap
    stacked_gbd = pd.DataFrame(records_gbd)

    if not stacked_gbd.empty:
        fig = plt.figure(figsize=(12, 0.8 * len(a1c_cats) * len(day_clusters) + 2))
        gs = fig.add_gridspec(1, 2, width_ratios=[4.5, 1.8])
        ax = fig.add_subplot(gs[0, 0])
        ax_txt = fig.add_subplot(gs[0, 1])

        # Color by day for visual separation
        colors_map = {d: plt.cm.tab10(idx % 10) for idx, d in enumerate(day_clusters)}
        for d in day_clusters:
            sub = stacked_gbd[stacked_gbd["day"] == d].sort_values("y_index")
            ax.errorbar(
                sub["pred"],
                sub["y_index"],
                xerr=[sub["pred"] - sub["ci_low"], sub["ci_high"] - sub["pred"]],
                fmt="o",
                capsize=3,
                color=colors_map.get(d, "tab:gray"),
                label=f"Day {d}",
            )

        # Build ticks, labels, and text values mapped by y-index to guarantee alignment
        y_ticks = []
        y_labels = []
        texts_map = {}
        y_base = 0
        for d in day_clusters:
            for cat_idx, cat in enumerate(a1c_cats):
                y_val = y_base + cat_idx
                y_ticks.append(y_val)
                y_labels.append(f"{cat} (Day {d})")
                row = stacked_gbd[(stacked_gbd["a1c_group"] == cat) & (stacked_gbd["day"] == d)]
                if not row.empty:
                    rr = row.iloc[0]
                    texts_map[y_val] = f"{rr['pred']:.2f} ({rr['ci_low']:.2f}, {rr['ci_high']:.2f})  n={int(rr['n_unique_window'])}"
            y_base += len(a1c_cats) + gap
            if gap > 0:
                # Add a blank separating row
                y_ticks.append(y_base - 1)
                y_labels.append("")
                texts_map[y_base - 1] = ""

        # Remove trailing gap if present
        if y_labels and y_labels[-1] == "":
            y_labels = y_labels[:-1]
            y_ticks = y_ticks[:-1]
            texts_map.pop(max(texts_map.keys()), None)

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Predicted Percent Weight Change")
        # Title lists the specific day values
        title_days = " / ".join(str(d) for d in day_clusters)
        ax.set_title(f"Predicted Weight Change at {title_days} Days")
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)
        ax.legend(title="Time from Baseline", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.invert_yaxis()

        x_min = float((stacked_gbd["ci_low"]).min())
        x_max = float((stacked_gbd["ci_high"]).max())
        x_range = x_max - x_min if x_max > x_min else 1.0
        left_lim = x_min - 0.1 * x_range
        right_lim = max(x_max + 0.1 * x_range, 0 + 0.3 * x_range)
        ax.set_xlim(left_lim, right_lim)

        # Right text column
        ax_txt.axis("off")
        ax_txt.set_title("Estimate (95% CI)")
        if y_ticks:
            ax_txt.set_ylim(min(y_ticks) - 0.5, max(y_ticks) + 0.5)
        ax_txt.invert_yaxis()
        for yv in y_ticks:
            t = texts_map.get(yv, "")
            ax_txt.text(0.0, yv, t, va="center", ha="left", fontsize=9)
        ax_txt.set_xlim(0, 1)

        plt.tight_layout()
        fig.savefig(
            os.path.join(
                outdir,
                "forest_predictions_grouped_by_day_4_days.png",
            ),
            dpi=150,
            bbox_inches='tight',
        )
        plt.close(fig)

    # Grouped-by-day forest for 12 and ~18 months (Weight) — right-side text column
    selected_12 = 365
    possible_18 = [548, 547, 540]
    selected_18 = next((d for d in possible_18 if d in day_values), None)

    if selected_12 in day_values and selected_18 is not None:
        day_clusters = [selected_12, selected_18]
        gap = 1
        records_gbd = []
        y_base = 0
        for d in day_clusters:
            for cat_idx, cat in enumerate(a1c_cats):
                row = combined[(combined["a1c_group"] == cat) & (combined["day"] == d)]
                if row.empty:
                    continue
                r = row.iloc[0].copy()
                r["y_index"] = y_base + cat_idx
                records_gbd.append(r)
            y_base += len(a1c_cats) + gap

        stacked_gbd = pd.DataFrame(records_gbd)
        fig = plt.figure(figsize=(12, 0.8 * len(a1c_cats) * len(day_clusters) + 2))
        gs = fig.add_gridspec(1, 2, width_ratios=[4.5, 1.8])
        ax = fig.add_subplot(gs[0, 0])
        ax_txt = fig.add_subplot(gs[0, 1])

        colors_map = {selected_12: "tab:blue", selected_18: "tab:orange"}
        for d in day_clusters:
            sub = stacked_gbd[stacked_gbd["day"] == d].sort_values("y_index")
            ax.errorbar(
                sub["pred"],
                sub["y_index"],
                xerr=[sub["pred"] - sub["ci_low"], sub["ci_high"] - sub["pred"]],
                fmt="o",
                capsize=3,
                color=colors_map.get(d, "tab:gray"),
                label=f"Day {d}",
            )

        # Build ticks, labels, and text values mapped by y-index to guarantee alignment
        y_ticks = []
        y_labels = []
        texts_map = {}
        y_base = 0
        for d in day_clusters:
            for cat_idx, cat in enumerate(a1c_cats):
                y_val = y_base + cat_idx
                y_ticks.append(y_val)
                y_labels.append(f"{cat} (Day {d})")
                row = stacked_gbd[(stacked_gbd["a1c_group"] == cat) & (stacked_gbd["day"] == d)]
                if not row.empty:
                    rr = row.iloc[0]
                    texts_map[y_val] = f"{rr['pred']:.2f} ({rr['ci_low']:.2f}, {rr['ci_high']:.2f})  n={int(rr['n_unique_window'])}"
            y_base += len(a1c_cats) + gap
            if gap > 0:
                # Add a blank separating row
                y_ticks.append(y_base - 1)
                y_labels.append("")
                texts_map[y_base - 1] = ""

        # Remove trailing gap if present
        if y_labels and y_labels[-1] == "":
            y_labels = y_labels[:-1]
            y_ticks = y_ticks[:-1]
            texts_map.pop(max(texts_map.keys()), None)

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Predicted Percent Weight Change")
        ax.set_title("Predicted Weight Change at 12 and 18 Months (Grouped by Day)")
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)
        # Remove legend per request (colors self-explanatory)
        # ax.legend(title="Time from Baseline", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.invert_yaxis()

        if not stacked_gbd.empty:
            x_min = float((stacked_gbd["ci_low"]).min())
            x_max = float((stacked_gbd["ci_high"]).max())
            x_range = x_max - x_min if x_max > x_min else 1.0
            left_lim = x_min - 0.1 * x_range
            right_lim = max(x_max + 0.1 * x_range, 0 + 0.3 * x_range)
            ax.set_xlim(left_lim, right_lim)

        # Right text column
        ax_txt.axis("off")
        ax_txt.set_title("Estimate (95% CI)")
        if y_ticks:
            ax_txt.set_ylim(min(y_ticks) - 0.5, max(y_ticks) + 0.5)
        ax_txt.invert_yaxis()  # align text rows with inverted left axis
        for yv in y_ticks:
            t = texts_map.get(yv, "")
            ax_txt.text(0.0, yv, t, va="center", ha="left", fontsize=9)
        ax_txt.set_xlim(0, 1)

        plt.tight_layout()
        fig.savefig(
            os.path.join(
                outdir,
                "forest_predictions_grouped_by_day_12m_18m.png",
            ),
            dpi=150,
            bbox_inches='tight',
        )
        plt.close(fig)

    # Forest plots for contrasts vs reference — right-side text column with header
    for d in times:
        sub = contrasts[contrasts["day"] == d].copy()
        sub["order"] = sub["a1c_group"].apply(lambda x: a1c_cats.index(x))
        sub = sub.sort_values("order")
        y = np.arange(len(sub))
        labels = list(sub["a1c_group"])
        texts = []
        for _, row in sub.iterrows():
            nval = counts_map.get((int(row["day"]), str(row["a1c_group"])) , None)
            nstr = f"  n={int(nval)}" if nval is not None else ""
            texts.append(f"{row['diff']:.2f} ({row['ci_low']:.2f}, {row['ci_high']:.2f}){nstr}")

        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[4.5, 1.8])
        ax = fig.add_subplot(gs[0, 0])
        ax_txt = fig.add_subplot(gs[0, 1])

        ax.errorbar(
            sub["diff"],
            y,
            xerr=[sub["diff"] - sub["ci_low"], sub["ci_high"] - sub["diff"]],
            fmt="o",
            capsize=3,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        n_all = counts_overall.get(d)
        title_suffix = f" (n={n_all})" if n_all is not None else ""
        ax.set_xlabel(f"Difference vs {ref_cat} (Percent Weight Change)")
        ax.set_title(
            f"Contrasts vs {ref_cat} by Baseline A1C (Day {d})" f"{title_suffix}"
        )
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)

        x_min = float((sub["ci_low"]).min())
        x_max = float((sub["ci_high"]).max())
        x_range = x_max - x_min if x_max > x_min else 1.0
        left_lim = x_min - 0.1 * x_range
        right_lim = max(x_max + 0.1 * x_range, 0 + 0.3 * x_range)
        ax.set_xlim(left_lim, right_lim)

        # Right text column
        ax_txt.axis("off")
        ax_txt.set_title("Estimate (95% CI)")
        ax_txt.set_ylim(y.min() - 0.5, y.max() + 0.5)
        ax_txt.invert_yaxis()  # align text rows with inverted left axis
        for yi, t in zip(y, texts):
            ax_txt.text(0.0, yi, t, va="center", ha="left", fontsize=9)
        ax_txt.set_xlim(0, 1)

        plt.tight_layout()
        fig.savefig(
            os.path.join(
                outdir,
                f"forest_contrasts_vs_{ref_cat.replace(' ', '_')}_day_{d}.png",
            ),
            dpi=150,
            bbox_inches='tight',
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
