#!/usr/bin/env python3
"""Step 0: Baseline analysis population table by baseline A1C.

This script creates a publication-style baseline characteristics table
for the main gap analysis population, including categorical summaries,
achievement flags, and numeric summaries (mean/median/mode/SD/range)
for baseline age, weight, height, BMI, and A1C.
"""

import argparse
import logging
import os
from typing import Dict, List

import pandas as pd


A1C_ORDER = [
    "Normal Glycemia",
    "Prediabetes",
    "Type 2 Diabetes",
    "Poorly Controlled Diabetes",
]


def configure_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Step 0: Baseline analysis population table by baseline A1C "
            "for a specified adherence gap (e.g., 90 days)"
        )
    )
    p.add_argument(
        "--input-csv",
        default=os.path.join(
            "output", "step1_prepare_analysis_dataset", "analysis_ready_gap90.csv"
        ),
        help=(
            "Path to weight analysis-ready CSV from Step 1 "
            "(default is gap90; override when using other gaps)"
        ),
    )
    p.add_argument(
        "--outdir",
        default=os.path.join("output", "step0_analysis_population_table"),
        help="Directory to write the baseline table CSV",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    p.add_argument(
        "--adherence-gap-days",
        type=int,
        default=90,
        help="Adherence gap in days for gap-specific output folder (e.g., 90)",
    )
    return p.parse_args(argv)


def _unique_patient_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per patient_id with baseline covariates."""
    if "patient_id" not in df.columns:
        raise ValueError("Expected 'patient_id' column in analysis dataset")
    sort_cols: List[str] = []
    if "baseline_glp1_date" in df.columns:
        sort_cols.append("baseline_glp1_date")
    if "event_date" in df.columns:
        sort_cols.append("event_date")
    if sort_cols:
        df = df.sort_values(sort_cols)
    return df.drop_duplicates(subset=["patient_id"]).copy()


def _format_cell(n: int, denom: int) -> str:
    if denom <= 0:
        return f"{n} (0.0%)"
    pct = 100.0 * n / float(denom)
    return f"{n} ({pct:.1f}%)"


def _summarize_categorical(
    df: pd.DataFrame,
    var: str,
    col_totals: Dict[str, int],
    a1c_col: str = "baseline_a1c_category",
) -> List[Dict[str, object]]:
    if var not in df.columns:
        logging.warning("Column '%s' missing; skipping", var)
        return []
    rows: List[Dict[str, object]] = []
    total_n = col_totals["Total"]
    if hasattr(df[var], "cat"):
        levels = [c for c in df[var].cat.categories if pd.notna(c)]
    else:
        levels = sorted(df[var].dropna().unique().tolist())
    for lvl in levels:
        mask_lvl = df[var] == lvl
        n_total = df.loc[mask_lvl, "patient_id"].nunique()
        row = {
            "variable": var,
            "level": str(lvl),
            "Total": _format_cell(n_total, total_n),
        }
        for a1c_cat in A1C_ORDER:
            mask_col = df[a1c_col] == a1c_cat
            denom = col_totals.get(a1c_cat, 0)
            n = df.loc[mask_lvl & mask_col, "patient_id"].nunique() if denom > 0 else 0
            row[a1c_cat] = _format_cell(n, denom)
        rows.append(row)
    return rows


def _summarize_binary_from_flag(
    df: pd.DataFrame,
    var: str,
    label_yes: str,
    col_totals: Dict[str, int],
    a1c_col: str = "baseline_a1c_category",
) -> List[Dict[str, object]]:
    if var not in df.columns:
        logging.warning("Column '%s' missing; skipping", var)
        return []
    flag = df[var].fillna(0).astype(bool)
    df_flag = df.copy()
    df_flag[var] = flag
    rows: List[Dict[str, object]] = []
    total_n = col_totals["Total"]
    for value, level_label in [(True, label_yes), (False, "No")]:
        mask_lvl = df_flag[var] == value
        n_total = df_flag.loc[mask_lvl, "patient_id"].nunique()
        row = {
            "variable": var,
            "level": level_label,
            "Total": _format_cell(n_total, total_n),
        }
        for a1c_cat in A1C_ORDER:
            mask_col = df_flag[a1c_col] == a1c_cat
            denom = col_totals.get(a1c_cat, 0)
            n = df_flag.loc[mask_lvl & mask_col, "patient_id"].nunique() if denom > 0 else 0
            row[a1c_cat] = _format_cell(n, denom)
        rows.append(row)
    return rows


def _compute_achievement_flags(df_weight: pd.DataFrame, gap_days: int) -> pd.DataFrame:
    """Compute per-patient achievement flags for weight loss thresholds using df_weight.
    If abs_a1c_change is available in the weight df, compute A1C flags too; otherwise
    attempt to load the gap-specific A1C analysis dataset and compute from there.
    Returns a DataFrame with patient_id and boolean flags.
    """
    pid_col = "patient_id"
    out = pd.DataFrame({pid_col: df_weight[pid_col].dropna().unique()})

    # Weight achievements from df_weight (pct_weight_change negative indicates loss)
    if "pct_weight_change" in df_weight.columns:
        agg_w = df_weight[[pid_col, "pct_weight_change"]].copy()
        agg_w["pct_weight_change"] = pd.to_numeric(agg_w["pct_weight_change"], errors="coerce")
        wmin = agg_w.groupby(pid_col)["pct_weight_change"].min()
        out = out.merge(wmin.rename("min_pct_weight_change"), left_on=pid_col, right_index=True, how="left")
        out["achieved_weight_loss_5pct"] = out["min_pct_weight_change"] <= -5.0
        out["achieved_weight_loss_10pct"] = out["min_pct_weight_change"] <= -10.0
        out["achieved_weight_loss_15pct"] = out["min_pct_weight_change"] <= -15.0
    else:
        out["achieved_weight_loss_5pct"] = False
        out["achieved_weight_loss_10pct"] = False
        out["achieved_weight_loss_15pct"] = False

    # A1C achievements
    a1c_source = None
    if "abs_a1c_change" in df_weight.columns:
        a1c_source = df_weight[[pid_col, "abs_a1c_change"]].copy()
    else:
        # Try to load gap-specific A1C dataset
        try:
            a1c_path = os.path.join("output", "step1_prepare_analysis_dataset_a1c", f"analysis_ready_a1c_gap{gap_days}.csv")
            if os.path.exists(a1c_path):
                a1c_df = pd.read_csv(a1c_path, usecols=[pid_col, "abs_a1c_change"])
                a1c_source = a1c_df
        except Exception as e:
            logging.warning("Could not load A1C dataset for achievements: %s", e)
            a1c_source = None
    if a1c_source is not None and not a1c_source.empty:
        a1c_source["abs_a1c_change"] = pd.to_numeric(a1c_source["abs_a1c_change"], errors="coerce")
        amin = a1c_source.groupby(pid_col)["abs_a1c_change"].min()
        out = out.merge(amin.rename("min_abs_a1c_change"), left_on=pid_col, right_index=True, how="left")
        # Decrease thresholds: change <= -threshold
        def _flag(thr: float):
            return (out["min_abs_a1c_change"] <= -thr).fillna(False)
        out["achieved_a1c_reduction_0p5"] = _flag(0.5)
        out["achieved_a1c_reduction_1p0"] = _flag(1.0)
        out["achieved_a1c_reduction_1p5"] = _flag(1.5)
        out["achieved_a1c_reduction_2p0"] = _flag(2.0)
        out["achieved_a1c_reduction_2p5"] = _flag(2.5)
    else:
        for nm in [
            "achieved_a1c_reduction_0p5",
            "achieved_a1c_reduction_1p0",
            "achieved_a1c_reduction_1p5",
            "achieved_a1c_reduction_2p0",
            "achieved_a1c_reduction_2p5",
        ]:
            out[nm] = False

    return out


def _standardize_baseline_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Detect alternate baseline brand/ingredient columns and standardize names."""
    df = df.copy()
    brand_candidates = [
        "baseline_glp1_brand_final",
        "baseline_glp1_brand",
        "first_glp1_brand",
    ]
    ing_candidates = [
        "baseline_glp1_ingredient_final",
        "baseline_glp1_ingredient",
        "first_glp1_ingredient",
        "baseline_glp1",  # legacy ingredient-like
    ]
    for col in brand_candidates:
        if col in df.columns:
            df["baseline_glp1_brand_final"] = df[col]
            break
    for col in ing_candidates:
        if col in df.columns:
            df["baseline_glp1_ingredient_final"] = df[col]
            break
    return df


def _summarize_numeric_stats(
    df: pd.DataFrame,
    var: str,
    label: str,
    a1c_col: str = "baseline_a1c_category",
) -> List[Dict[str, object]]:
    """Create rows for Mean, Median, Mode, SD, and Range for a numeric column,
    overall and by baseline A1C category.
    """
    if var not in df.columns:
        logging.warning("Numeric column '%s' missing; skipping", var)
        return []

    def stats_for(series: pd.Series) -> Dict[str, str]:
        series = pd.to_numeric(series, errors="coerce").dropna()
        if series.empty:
            return {"Mean": "NA", "Median": "NA", "Mode": "NA", "SD": "NA", "Range": "NA"}
        mean = f"{series.mean():.1f}"
        median = f"{series.median():.1f}"
        try:
            mvals = series.mode(dropna=True)
            mode_val = mvals.iloc[0] if not mvals.empty else float("nan")
            mode = f"{mode_val:.1f}" if pd.notna(mode_val) else "NA"
        except Exception:
            mode = "NA"
        sd = f"{series.std(ddof=1):.1f}" if series.size > 1 else "NA"
        rng = f"{series.min():.1f}-{series.max():.1f}"
        return {"Mean": mean, "Median": median, "Mode": mode, "SD": sd, "Range": rng}

    rows: List[Dict[str, object]] = []
    total_stats = stats_for(df[var])
    for stat_name in ["Mean", "Median", "Mode", "SD", "Range"]:
        row: Dict[str, object] = {"variable": label, "level": stat_name, "Total": total_stats[stat_name]}
        for a1c_cat in A1C_ORDER:
            mask = (df[a1c_col] == a1c_cat) if a1c_col in df.columns else pd.Series(False, index=df.index)
            stat_val = stats_for(df.loc[mask, var])[stat_name]
            row[a1c_cat] = stat_val
        rows.append(row)
    return rows


def _ensure_baseline_a1c(df_unique: pd.DataFrame, gap_days: int) -> pd.DataFrame:
    """Ensure baseline A1C numeric column is available on the unique-patient frame.

    If the weight analysis dataset already has baseline_a1c_final or baseline_a1c,
    nothing is changed. Otherwise, try to pull baseline A1C from the gap-specific
    A1C analysis dataset produced in Step 1 and merge it by patient_id.
    """
    df_unique = df_unique.copy()
    if "baseline_a1c_final" in df_unique.columns or "baseline_a1c" in df_unique.columns:
        return df_unique

    try:
        a1c_path = os.path.join(
            "output",
            "step1_prepare_analysis_dataset_a1c",
            f"analysis_ready_a1c_gap{gap_days}.csv",
        )
        if os.path.exists(a1c_path):
            a1c_df = pd.read_csv(a1c_path, usecols=["patient_id", "baseline_a1c_final"])
            a1c_df = a1c_df.dropna(subset=["baseline_a1c_final"]).drop_duplicates("patient_id")
            df_unique = df_unique.merge(a1c_df, on="patient_id", how="left")
        else:
            logging.warning("A1C analysis dataset not found at %s; baseline A1C stats will be skipped", a1c_path)
    except Exception as e:
        logging.warning("Could not merge baseline A1C from A1C dataset: %s", e)

    return df_unique


def build_baseline_table(df: pd.DataFrame, gap_days: int) -> pd.DataFrame:
    """Build baseline characteristics table for gap analysis population."""
    df_unique = _unique_patient_frame(df)
    df_unique = _standardize_baseline_cols(df_unique)

    # If standardized baseline missing, merge from step8f and standardize
    need_baseline = ["baseline_glp1_brand_final", "baseline_glp1_ingredient_final"]
    if any(col not in df_unique.columns for col in need_baseline):
        try:
            src = os.path.join("root_data", "step8f.csv")
            if os.path.exists(src):
                base = pd.read_csv(src, low_memory=False)
                base = _standardize_baseline_cols(base)
                sel = [c for c in ["patient_id"] + need_baseline if c in base.columns]
                if len(sel) >= 2:
                    base = base[sel].drop_duplicates("patient_id")
                    df_unique = df_unique.merge(base, on="patient_id", how="left")
                else:
                    logging.warning("Baseline brand/ingredient columns not found in step8f.csv headers")
        except Exception as e:
            logging.warning("Could not merge baseline brand/ingredient from step8f.csv: %s", e)

    if "baseline_a1c_category" not in df_unique.columns:
        raise ValueError("Expected 'baseline_a1c_category' in analysis dataset")

    # Bring in baseline A1C values from the A1C analysis dataset if needed
    df_unique = _ensure_baseline_a1c(df_unique, gap_days=gap_days)

    # Column totals (unique patients)
    col_totals: Dict[str, int] = {"Total": df_unique["patient_id"].nunique()}
    for a1c_cat in A1C_ORDER:
        col_totals[a1c_cat] = df_unique.loc[df_unique["baseline_a1c_category"] == a1c_cat, "patient_id"].nunique()

    rows: List[Dict[str, object]] = []

    # Numeric summaries (baseline values)
    if "age" in df_unique.columns:
        rows.extend(_summarize_numeric_stats(df_unique, var="age", label="Baseline Age (years)"))
    # Weight: prefer baseline_weight_final else weight_in_pounds_final
    if "baseline_weight_final" in df_unique.columns:
        rows.extend(_summarize_numeric_stats(df_unique, var="baseline_weight_final", label="Baseline Weight (lbs)"))
    elif "weight_in_pounds_final" in df_unique.columns:
        rows.extend(_summarize_numeric_stats(df_unique, var="weight_in_pounds_final", label="Baseline Weight (lbs)"))
    # Height: prefer height_in_inches_final else baseline_height_final
    if "height_in_inches_final" in df_unique.columns:
        rows.extend(_summarize_numeric_stats(df_unique, var="height_in_inches_final", label="Baseline Height (in)"))
    elif "baseline_height_final" in df_unique.columns:
        rows.extend(_summarize_numeric_stats(df_unique, var="baseline_height_final", label="Baseline Height (in)"))
    # BMI: prefer baseline_bmi_final else BMI_final
    if "baseline_bmi_final" in df_unique.columns:
        rows.extend(_summarize_numeric_stats(df_unique, var="baseline_bmi_final", label="Baseline BMI"))
    elif "BMI_final" in df_unique.columns:
        rows.extend(_summarize_numeric_stats(df_unique, var="BMI_final", label="Baseline BMI"))
    # A1C: prefer baseline_a1c_final else baseline_a1c (may have been merged above)
    if "baseline_a1c_final" in df_unique.columns:
        rows.extend(_summarize_numeric_stats(df_unique, var="baseline_a1c_final", label="Baseline A1C"))
    elif "baseline_a1c" in df_unique.columns:
        rows.extend(_summarize_numeric_stats(df_unique, var="baseline_a1c", label="Baseline A1C"))

    # Categorical summaries
    rows.extend(_summarize_categorical(df_unique, var="baseline_a1c_category", col_totals=col_totals, a1c_col="baseline_a1c_category"))
    rows.extend(_summarize_categorical(df_unique, var="baseline_bmi_final_category", col_totals=col_totals))
    rows.extend(_summarize_categorical(df_unique, var="age_group", col_totals=col_totals))
    rows.extend(_summarize_categorical(df_unique, var="gender", col_totals=col_totals))
    rows.extend(_summarize_categorical(df_unique, var="race", col_totals=col_totals))

    # Baseline GLP-1 ingredient and brand
    if "baseline_glp1_ingredient_final" in df_unique.columns:
        rows.extend(_summarize_categorical(df_unique, var="baseline_glp1_ingredient_final", col_totals=col_totals))
    elif "baseline_glp1" in df_unique.columns:
        rows.extend(_summarize_categorical(df_unique, var="baseline_glp1", col_totals=col_totals))
    if "baseline_glp1_brand_final" in df_unique.columns:
        rows.extend(_summarize_categorical(df_unique, var="baseline_glp1_brand_final", col_totals=col_totals))

    # Achievement flags
    flags = _compute_achievement_flags(df, gap_days)
    df_flags = df_unique[["patient_id", "baseline_a1c_category"]].merge(flags, on="patient_id", how="left")
    ach_vars = [
        ("achieved_weight_loss_5pct", ">=5% loss"),
        ("achieved_weight_loss_10pct", ">=10% loss"),
        ("achieved_weight_loss_15pct", ">=15% loss"),
        ("achieved_a1c_reduction_0p5", ">=0.5pt A1C drop"),
        ("achieved_a1c_reduction_1p0", ">=1.0pt A1C drop"),
        ("achieved_a1c_reduction_1p5", ">=1.5pt A1C drop"),
        ("achieved_a1c_reduction_2p0", ">=2.0pt A1C drop"),
        ("achieved_a1c_reduction_2p5", ">=2.5pt A1C drop"),
    ]
    for col, lbl in ach_vars:
        if col in df_flags.columns:
            rows.extend(_summarize_binary_from_flag(df_flags, var=col, label_yes=lbl, col_totals=col_totals))

    table_df = pd.DataFrame(rows)
    ordered_cols = ["variable", "level", "Total"] + A1C_ORDER
    table_df = table_df[ordered_cols]
    table_df.sort_values(["variable", "level"], inplace=True)
    return table_df


def main(argv=None) -> None:
    args = parse_args(argv)
    configure_logging(args.log_level)

    # Gap-aware outdir routing
    outdir = args.outdir
    gap = args.adherence_gap_days
    if gap is not None and "/gap_" not in outdir:
        outdir = os.path.join("output", f"gap_{gap}", os.path.basename(args.outdir))
    os.makedirs(outdir, exist_ok=True)

    logging.info("Reading analysis dataset from %s", args.input_csv)
    df = pd.read_csv(args.input_csv)

    logging.info("Building baseline analysis population table (gap%d)", gap)
    table_df = build_baseline_table(df, gap_days=gap or 90)

    out_filename = f"step0_baseline_table_gap{gap}.csv" if gap is not None else "step0_baseline_table.csv"
    out_path = os.path.join(outdir, out_filename)
    table_df.to_csv(out_path, index=False)
    logging.info("Wrote baseline table to %s", out_path)


if __name__ == "__main__":
    main()
