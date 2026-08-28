#!/usr/bin/env python3
"""Step 8: Publication-oriented survival plots and Cox models.

This script takes the event-level CSVs produced by:
  - step8_survival_time_to_weight_loss.py
  - step8_survival_time_to_a1c_drop.py

and produces:
  - Kaplan–Meier curves (1 - S) by baseline A1C category and threshold
  - Cox proportional hazards models with baseline A1C category and
    weight_change_med flag as covariates (no metformin term)
  - Summary tables of median times and hazard ratios

Outputs are written under output/step8_survival_* and figures under
presentation/step8_survival/.
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
import matplotlib.pyplot as plt

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


# Define A1C order for legends
A1C_ORDER = [
    "Normal Glycemia",
    "Prediabetes",
    "Type 2 Diabetes",
    "Poorly Controlled Diabetes",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Step 8 survival plots and Cox models")
    p.add_argument(
        "--weight-events-csv",
        default="output/step8_survival_weight/step8_weight_time_to_threshold_events.csv",
        help="Path to weight time-to-threshold events CSV",
    )
    p.add_argument(
        "--a1c-events-csv",
        default="output/step8_survival_a1c/step8_a1c_time_to_threshold_events.csv",
        help="Path to A1C time-to-threshold events CSV",
    )
    p.add_argument(
        "--analysis-a1c-csv",
        default="output/step1_prepare_analysis_dataset_a1c/analysis_ready_a1c_gap90.csv",
        help="A1C analysis dataset (for covariates including weight_change_med)",
    )
    p.add_argument(
        "--analysis-weight-csv",
        default="output/step1_prepare_analysis_dataset/analysis_ready_gap90.csv",
        help="Weight analysis dataset (for covariates including weight_change_med)",
    )
    p.add_argument(
        "--outdir-base",
        default="output/step8_survival_plots_and_cox",
        help="Base directory for tabular outputs",
    )
    p.add_argument(
        "--figdir-base",
        default="output/step8_survival_plots_and_cox/plots",
        help="Base directory for figures",
    )
    p.add_argument(
        "--adherence-gap-days",
        type=int,
        default=None,
        help="Adherence gap in days for gap-specific output folder (e.g., 90)",
    )
    # Cosmetic options for KM plots; default False so existing pipelines are unchanged
    p.add_argument(
        "--suppress-km-titles",
        action="store_true",
        help="If set, suppress titles on KM plots (used for panel figures)",
    )
    p.add_argument(
        "--suppress-km-legend",
        action="store_true",
        help="If set, suppress legends on KM plots (used for panel figures)",
    )
    p.add_argument(
        "--write-km-legend-only",
        action="store_true",
        help="If set, also write legend-only PNGs for KM plots (for panel figures)",
    )
    return p.parse_args(argv)


def _legend_label_for_a1c_category(subgroup: str) -> str:
    """Return legend labels with A1C ranges for each baseline category.

    Mirrors the step 4 legend text (no 'A1C:' prefix, includes ranges).
    """

    mapping = {
        "Normal Glycemia": "Normal glycemia (<5.7%)",
        "Prediabetes": "Prediabetes (5.7% to <6.5%)",
        "Type 2 Diabetes": "Type 2 diabetes (6.5% to <9.0%)",
        "Poorly Controlled Diabetes": "Poorly controlled diabetes (>=9.0%)",
    }
    return mapping.get(str(subgroup), str(subgroup))


def _get_weight_meds_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Create a per-patient weight_change_med flag.

    Prefer an explicit ``weight_change_med`` column (as in ``step8f.csv``
    and the Step 1 analysis datasets). If not present, default to a
    0/1 indicator column of zeros.
    """

    if "weight_change_med" in df.columns:
        col = "weight_change_med"
        tmp = df.drop_duplicates("patient_id")[ ["patient_id", col] ].copy()
        tmp.rename(columns={col: "weight_change_med"}, inplace=True)
        # Coerce to 0/1
        tmp["weight_change_med"] = (
            pd.to_numeric(tmp["weight_change_med"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        return tmp
    else:
        df_out = df.drop_duplicates("patient_id")[ ["patient_id"] ].copy()
        df_out["weight_change_med"] = 0
        return df_out


def _km_plot_one_endpoint(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    group_col: str,
    threshold_col: str,
    threshold_value,
    title: str,
    figpath: str,
    suppress_titles: bool = False,
    suppress_legend: bool = False,
    write_legend_only: bool = False,
):
    os.makedirs(os.path.dirname(figpath), exist_ok=True)

    plt.figure(figsize=(7, 4.5))
    kmf = KaplanMeierFitter()

    sub = df[df[threshold_col] == threshold_value].copy()
    if sub.empty:
        plt.close()
        return

    groups = list(sub[group_col].dropna().unique())
    # Order legend/groups explicitly when grouping by baseline A1C
    if group_col == "baseline_a1c_category":
        groups = [g for g in A1C_ORDER if g in groups]
    else:
        groups.sort()

    for g in groups:
        gdf = sub[sub[group_col] == g]
        if gdf[time_col].notna().sum() == 0:
            continue
        label = (
            _legend_label_for_a1c_category(g)
            if group_col == "baseline_a1c_category"
            else str(g)
        )
        kmf.fit(
            durations=gdf[time_col],
            event_observed=gdf[event_col],
            label=label,
        )
        # Plot standard KM survival S(t) (proportion not yet at threshold)
        surv_fn = kmf.survival_function_.copy()
        line = plt.step(
            surv_fn.index,
            surv_fn[kmf._label],
            where="post",
            label=label,
        )
        # 95% CI bands around KM curve
        line_color = line[0].get_color()
        ci = kmf.confidence_interval_survival_function_
        ci_lo = ci.iloc[:, 0].values
        ci_hi = ci.iloc[:, 1].values
        t_days = ci.index.values
        plt.fill_between(t_days, ci_lo, ci_hi, alpha=0.15, color=line_color, step="post")

    plt.xlabel("Days from baseline")
    plt.ylabel("Survival probability")
    if not suppress_titles:
        plt.title(title)
    # Always keep y-axis on the 0–1 scale for comparability
    plt.ylim(0, 1.0)
    # Cap x-axis at 548 days (~18 months)
    plt.xlim(0, 548)

    # Add a horizontal reference line at survival = 0.5; the point where
    # each KM curve crosses this line corresponds to the median time
    # if a median exists.
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)

    # Keep the legend anchored in the lower-left corner for consistency,
    # and use a reader-friendly title for baseline HbA1c.
    legend_title = "Baseline HbA1c Category" if group_col == "baseline_a1c_category" else group_col
    if not suppress_legend:
        plt.legend(title=legend_title, fontsize=8, loc="lower left")

    plt.tight_layout()
    plt.savefig(figpath, dpi=600)

    # Optionally emit a legend-only PNG for use in panel figures
    if write_legend_only and group_col == "baseline_a1c_category":
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            import matplotlib.pyplot as _plt

            fig_leg = _plt.figure(figsize=(4, 2))
            fig_leg.legend(
                handles,
                labels,
                loc="center",
                ncol=1,
                frameon=False,
                fontsize=12,
            )
            legend_path = figpath.replace(".png", "_legend.png")
            fig_leg.savefig(legend_path, dpi=300, bbox_inches="tight")
            _plt.close(fig_leg)

    plt.close()


def _format_covariate_name_to_phreg(row_covariate: str) -> tuple[str, str]:
    """Split dummy-coded covariate names into (Parameter, Level) for SAS-like output.

    Examples:
      - 'baseline_a1c_category_Prediabetes' -> ("baseline_a1c_category", "Prediabetes")
      - 'gender_M' -> ("gender", "M")
      - 'weight_change_med' -> ("weight_change_med", "1 vs 0")
    """

    # Special-case baseline A1C so the Parameter column is meaningful and
    # matches what downstream code (e.g., Step 8b) expects.
    if row_covariate.startswith("baseline_a1c_category_"):
        level = row_covariate.replace("baseline_a1c_category_", "")
        return "baseline_a1c_category", level

    if "_" in row_covariate and not row_covariate.startswith("age_group_20_"):
        # Most dummies will be var_level
        parts = row_covariate.split("_", 1)
        param = parts[0]
        level = parts[1]
        return param, level
    # binary numeric
    return row_covariate, "1 vs 0"


def _cox_for_endpoint(
    events_df: pd.DataFrame,
    covariates_df: pd.DataFrame,
    time_col: str,
    event_col: str,
    threshold_col: str,
    threshold_value,
    out_csv: str,
):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    sub = events_df[events_df[threshold_col] == threshold_value].copy()
    if sub.empty:
        pd.DataFrame().to_csv(out_csv, index=False)
        return

    wanted_covs = [
        "patient_id",
        "baseline_a1c_category",
        "age_group",
        "gender",
        "baseline_bmi_final_category",
        "race",
        # "metformin_with_glp1_baseline",  # removed per request
        "weight_change_med",
    ]
    cov_cols = [c for c in wanted_covs if c in covariates_df.columns]
    cov = covariates_df.drop_duplicates("patient_id")[cov_cols].copy()

    A1C_ORDER = [
        "Normal Glycemia",
        "Prediabetes",
        "Type 2 Diabetes",
        "Poorly Controlled Diabetes",
    ]
    # For baseline A1C, we rely on the events_df (sub) rather than duplicating
    # it in the covariate frame to avoid creating "_x"/"_y" columns on merge.

    dat = sub.merge(cov, on="patient_id", how="left")
    dat = dat.dropna(subset=[time_col, event_col])
    if dat.empty:
        pd.DataFrame().to_csv(out_csv, index=False)
        return

    # Ensure categorical types for consistent dummy encoding (drop_first uses first category)
    for cat_col in ["baseline_a1c_category", "age_group", "gender", "baseline_bmi_final_category", "race"]:
        if cat_col in dat.columns and not isinstance(dat[cat_col].dtype, pd.CategoricalDtype):  # type: ignore[attr-defined]
            if cat_col == "baseline_a1c_category":
                dat[cat_col] = pd.Categorical(dat[cat_col], categories=A1C_ORDER, ordered=True)
            else:
                dat[cat_col] = dat[cat_col].astype("category")

    cat_vars = [
        v for v in ["baseline_a1c_category", "age_group", "gender", "baseline_bmi_final_category", "race"] if v in dat.columns
    ]
    num_vars = [v for v in ["weight_change_med"] if v in dat.columns]

    X_parts = []
    base = dat[[time_col, event_col]].reset_index(drop=True).copy()
    X_parts.append(base)
    if cat_vars:
        dummies = pd.get_dummies(dat[cat_vars], drop_first=True)
        dummies = dummies.reset_index(drop=True)
        X_parts.append(dummies)
    if num_vars:
        nums = dat[num_vars].apply(pd.to_numeric, errors="coerce").fillna(0)
        nums = nums.reset_index(drop=True)
        X_parts.append(nums)

    X = pd.concat(X_parts, axis=1)

    cph = CoxPHFitter()
    try:
        cph.fit(X, duration_col=time_col, event_col=event_col)
    except Exception as e:
        logging.warning("Cox PH fit failed for threshold=%s: %s — skipping Cox output", threshold_value, e)
        pd.DataFrame().to_csv(out_csv, index=False)
        return

    try:
        ph_test = proportional_hazard_test(cph, X, time_transform="rank")
    except Exception:
        ph_test = None

    summary = cph.summary.reset_index().rename(columns={"index": "covariate"})
    summary["HR"] = np.exp(summary["coef"])
    summary["HR_lower_95"] = np.exp(summary["coef"] - Z_CRIT * summary["se(coef)"])
    summary["HR_upper_95"] = np.exp(summary["coef"] + Z_CRIT * summary["se(coef)"])

    block_p = np.nan
    if ph_test is not None:
        try:
            a1c_rows = [i for i in ph_test.summary.index if str(i).startswith("baseline_a1c_category")]
            if a1c_rows:
                block_p = ph_test.summary.loc[a1c_rows, "p"].max()
        except Exception:
            block_p = np.nan
    summary["ph_global_p_for_a1c_block"] = block_p

    # Write HR summary
    summary.to_csv(out_csv, index=False)

    # Create additional report-style artifacts in subfolders
    prefix = "cox_a1c" if threshold_col == "threshold_abs" else "cox_weight"
    thr_tag = str(threshold_value).replace(".", "p") if prefix == "cox_a1c" else f"{int(abs(threshold_value))}"
    model_dir = os.path.join(os.path.dirname(out_csv), "models")
    os.makedirs(model_dir, exist_ok=True)

    # Overview: sample sizes and events overall and by A1C strata
    overview_rows = []
    n_unique = int(dat["patient_id"].nunique()) if "patient_id" in dat.columns else len(dat)
    n_events = int(dat[event_col].sum())
    overview_rows.append({"metric": "total_patients", "value": n_unique})
    overview_rows.append({"metric": "total_events", "value": n_events})
    if "baseline_a1c_category" in dat.columns:
        grp = dat.groupby("baseline_a1c_category")
        for cat, g in grp:
            overview_rows.append({"metric": f"patients_{cat}", "value": int(g["patient_id"].nunique())})
            overview_rows.append({"metric": f"events_{cat}", "value": int(g[event_col].sum())})
    overview_df = pd.DataFrame(overview_rows)

    # Covariates table: HRs with CIs and p-values (PHREG-like)
    cov_tbl = summary[["covariate", "HR", "HR_lower_95", "HR_upper_95", "z", "p"]].copy()
    cov_tbl.rename(columns={
        "HR": "HazardRatio",
        "HR_lower_95": "HRLowerCL",
        "HR_upper_95": "HRUpperCL",
        "z": "WaldZ",
        "p": "PrChiSq",
    }, inplace=True)
    cov_tbl["WaldChiSq"] = cov_tbl["WaldZ"] ** 2
    # Split covariate into Parameter and Level for dummy-coded terms
    param_level = cov_tbl["covariate"].apply(_format_covariate_name_to_phreg)
    cov_tbl["Parameter"] = [pl[0] for pl in param_level]
    cov_tbl["Level"] = [pl[1] for pl in param_level]
    cov_tbl = cov_tbl[["Parameter", "Level", "HazardRatio", "HRLowerCL", "HRUpperCL", "WaldChiSq", "PrChiSq", "covariate"]]

    # Combine into single CSV with a section indicator
    overview_df["section"] = "overview"
    overview_df["Parameter"] = ""
    overview_df["Level"] = ""
    overview_df["HazardRatio"] = ""
    overview_df["HRLowerCL"] = ""
    overview_df["HRUpperCL"] = ""
    overview_df["WaldChiSq"] = ""
    overview_df["PrChiSq"] = ""
    overview_df["covariate"] = ""
    overview_df = overview_df[[
        "section", "metric", "value", "Parameter", "Level", "HazardRatio", "HRLowerCL", "HRUpperCL", "WaldChiSq", "PrChiSq", "covariate"
    ]]

    cov_tbl["section"] = "covariates"
    cov_tbl["metric"] = ""
    cov_tbl["value"] = ""
    cov_tbl = cov_tbl[[
        "section", "metric", "value", "Parameter", "Level", "HazardRatio", "HRLowerCL", "HRUpperCL", "WaldChiSq", "PrChiSq", "covariate"
    ]]

    combined = pd.concat([overview_df, cov_tbl], axis=0, ignore_index=True)
    out_combined = os.path.join(model_dir, f"{prefix}_time_to_{thr_tag}_{'reduction' if prefix=='cox_a1c' else 'pct_loss'}_phreg.csv")
    combined.to_csv(out_combined, index=False)

    # Remove baseline hazard/survival outputs per request (do not write long lists)
    # ...existing code...


def main(argv=None):
    args = parse_args(argv)

    # Load event datasets
    weight_events = pd.read_csv(args.weight_events_csv)
    a1c_events = pd.read_csv(args.a1c_events_csv)

    # Load analysis datasets to obtain covariates and weight_change_med flag
    a1c_analysis = pd.read_csv(args.analysis_a1c_csv)
    weight_analysis = pd.read_csv(args.analysis_weight_csv)

    # Determine gap-specific base outdir
    import re, os
    gap = args.adherence_gap_days
    if gap is None:
        for s in [args.analysis_weight_csv, args.analysis_a1c_csv]:
            m = re.search(r"gap[_]?(\d+)", str(s))
            if m:
                try:
                    gap = int(m.group(1))
                    break
                except Exception as exc:
                    logging.debug(
                        "could not parse a gap number from the input path (%s); falling back to the --adherence-gap-days value",
                        exc,
                    )
    outdir_base = args.outdir_base
    if gap is not None and "/gap_" not in outdir_base:
        outdir_base = os.path.join("output", f"gap_{gap}", os.path.basename(args.outdir_base))

    figdir_base = args.figdir_base
    if gap is not None and "/gap_" not in figdir_base:
        figdir_base = os.path.join("output", f"gap_{gap}", os.path.basename(args.figdir_base))

    # Construct per-patient weight_change_med flag
    a1c_meds = _get_weight_meds_flag(a1c_analysis)
    wt_meds = _get_weight_meds_flag(weight_analysis)
    meds = (
        wt_meds.set_index("patient_id")
        .combine_first(a1c_meds.set_index("patient_id"))
        .reset_index()
    )

    # Build covariate frame primarily from A1C analysis (more complete covariates), merged with meds
    # Covariates aligned with Step 6: baseline A1C category (primary exposure),
    # plus age group, sex, baseline BMI category, race, and weight_change_med.
    cov_keep = [
        "patient_id",
        "age_group",
        "gender",
        "baseline_bmi_final_category",
        "race",
        "weight_change_med",
    ]
    cov_base = a1c_analysis.drop_duplicates("patient_id")[[c for c in cov_keep if c in a1c_analysis.columns]].copy()
    # Ensure weight_change_med is present via the meds frame
    base_cov = cov_base.merge(meds, on="patient_id", how="left")

    # Ensure output dirs exist under gap
    weight_outdir = os.path.join(outdir_base, "step8_survival_weight")
    a1c_outdir = os.path.join(outdir_base, "step8_survival_a1c")
    os.makedirs(weight_outdir, exist_ok=True)
    os.makedirs(a1c_outdir, exist_ok=True)
    # Plots subdirs per user request
    weight_figdir = os.path.join(weight_outdir, "plots")
    a1c_figdir = os.path.join(a1c_outdir, "plots")
    os.makedirs(weight_figdir, exist_ok=True)
    os.makedirs(a1c_figdir, exist_ok=True)

    # --- KM plots ---
    # A1C: thresholds (including 0.5, 1.0, 1.5, 2.0 where available)
    for thr in sorted(a1c_events["threshold_abs"].dropna().unique()):
        title = f"Time to ≥{thr:g}-point A1C reduction"
        figpath = os.path.join(
            a1c_figdir,
            f"km_a1c_time_to_{str(thr).replace('.', 'p')}_reduction_by_baseline_a1c.png",
        )
        _km_plot_one_endpoint(
            df=a1c_events,
            time_col="time_days",
            event_col="event",
            group_col="baseline_a1c_category",
            threshold_col="threshold_abs",
            threshold_value=thr,
            title=title,
            figpath=figpath,
            suppress_titles=args.suppress_km_titles,
            suppress_legend=args.suppress_km_legend,
            write_legend_only=args.write_km_legend_only,
        )

    # Weight: thresholds in percent
    for thr in sorted(weight_events["threshold_pct"].dropna().unique()):
        title = f"Time to ≥{abs(thr):g}% weight loss"
        figpath = os.path.join(
            weight_figdir,
            f"km_weight_time_to_{int(abs(thr))}pct_loss_by_baseline_a1c.png",
        )
        _km_plot_one_endpoint(
            df=weight_events,
            time_col="time_days",
            event_col="event",
            group_col="baseline_a1c_category",
            threshold_col="threshold_pct",
            threshold_value=thr,
            title=title,
            figpath=figpath,
            suppress_titles=args.suppress_km_titles,
            suppress_legend=args.suppress_km_legend,
            write_legend_only=args.write_km_legend_only,
        )

    # --- Cox models ---
    try:
        print("[Cox] A1C thresholds:", sorted(a1c_events["threshold_abs"].dropna().unique().tolist()))
        print("[Cox] Weight thresholds:", sorted(weight_events["threshold_pct"].dropna().unique().tolist()))
    except Exception as exc:
        logging.debug(
            "could not print the threshold diagnostic (%s); no effect on the fitted models",
            exc,
        )
    for thr in sorted(a1c_events["threshold_abs"].dropna().unique()):
        out_csv = os.path.join(
            a1c_outdir,
            f"cox_a1c_time_to_{str(thr).replace('.', 'p')}_reduction_by_baseline_a1c.csv",
        )
        _cox_for_endpoint(
            events_df=a1c_events,
            covariates_df=base_cov,
            time_col="time_days",
            event_col="event",
            threshold_col="threshold_abs",
            threshold_value=thr,
            out_csv=out_csv,
        )

    for thr in sorted(weight_events["threshold_pct"].dropna().unique()):
        out_csv = os.path.join(
            weight_outdir,
            f"cox_weight_time_to_{int(abs(thr))}pct_loss_by_baseline_a1c.csv",
        )
        _cox_for_endpoint(
            events_df=weight_events,
            covariates_df=base_cov,
            time_col="time_days",
            event_col="event",
            threshold_col="threshold_pct",
            threshold_value=thr,
            out_csv=out_csv,
        )
    try:
        print("[Cox] Completed all thresholds.")
    except Exception as exc:
        logging.debug(
            "could not print the completion diagnostic (%s); no effect on the fitted models",
            exc,
        )


if __name__ == "__main__":
    main()
