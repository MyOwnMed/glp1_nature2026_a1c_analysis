#!/usr/bin/env python3

"""Step 8b: Combine Cox results for threshold outcomes into publication-style tables.

This script reads the Step 8 KM summaries and Cox model outputs and produces
markdown tables summarizing hazard ratios for weight-loss and A1c-reduction
thresholds (5%, 10%, 15% weight loss; 0.5, 1.0, 1.5, 2.0 percentage-point
reductions in A1c), stratified by baseline glycemic status. The script is
gap-aware and, by default, builds tables for gap_120.
"""

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


A1C_ORDER = [
    "Normal Glycemia",
    "Prediabetes",
    "Type 2 Diabetes",
    "Poorly Controlled Diabetes",
]

# Mapping from internal baseline_a1c_category values to display labels
BASELINE_DISPLAY = {
    "Prediabetes": "Prediabetes",
    "Type 2 Diabetes": "Type 2 diabetes",
    "Poorly Controlled Diabetes": "Poorly controlled diabetes",
}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Step 8b: Combine gap-specific Cox results for weight and A1c "
            "thresholds into publication-style tables."
        )
    )
    p.add_argument(
        "--adherence-gap-days",
        "--gap",
        dest="gap",
        type=int,
        default=120,
        help="Adherence gap in days (e.g., 120). Used to locate gap-specific outputs.",
    )
    p.add_argument(
        "--out-markdown",
        default=None,
        help=(
            "Optional path for the markdown table output. If not provided, "
            "a default under output/gap_<gap>/ will be used."
        ),
    )
    return p.parse_args(argv)


def _format_hr_ci(hr: float, lci: float, uci: float) -> str:
    """Format HR and confidence interval as 'HR (LCI–UCI)' with two decimals."""

    return f"{hr:.2f} ({lci:.2f}-{uci:.2f})"


def _format_p(p: float) -> str:
    """Format p-value with three decimals, using '<0.001' when very small."""

    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def _load_km_summary_weight(gap: int) -> pd.DataFrame:
    path = os.path.join(
        "output",
        f"gap_{gap}",
        "step8_survival_time_to_weight_loss",
        "step8_weight_time_to_threshold_summary.csv",
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Weight KM summary not found at: {path}")

    df = pd.read_csv(path)
    # Keep only the thresholds of interest
    keep = [-5.0, -10.0, -15.0]
    df = df[df["threshold_pct"].isin(keep)].copy()
    # Map thresholds to display labels
    thr_map = {-5.0: "≥5%", -10.0: "≥10%", -15.0: "≥15%"}
    df["threshold_label"] = df["threshold_pct"].map(thr_map)
    return df


def _load_km_summary_a1c(gap: int) -> pd.DataFrame:
    path = os.path.join(
        "output",
        f"gap_{gap}",
        "step8_survival_time_to_a1c_drop",
        "step8_a1c_time_to_threshold_summary.csv",
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"A1c KM summary not found at: {path}")

    df = pd.read_csv(path)
    keep = [0.5, 1.0, 1.5, 2.0]
    df = df[df["threshold_abs"].isin(keep)].copy()
    thr_map = {0.5: "≥0.5%", 1.0: "≥1.0%", 1.5: "≥1.5%", 2.0: "≥2.0%"}
    df["threshold_label"] = df["threshold_abs"].map(thr_map)
    return df


def _load_weight_hrs_by_a1c_category(gap: int, pct_loss: int) -> Dict[str, Tuple[str, str]]:
    """Return mapping {baseline_a1c_category: (HR_CI_str, p_str)} for a % loss.

    Uses the PHREG-style outputs where A1c category dummies are encoded as
    covariates named "baseline_a1c_category_<Category>" with Normal Glycemia
    as the reference. The returned HRs are interpreted as the relative hazard
    of achieving the threshold vs Normal Glycemia, within the same model.
    """

    phreg_path = os.path.join(
        "output",
        f"gap_{gap}",
        "step8_survival_plots_and_cox",
        "step8_survival_weight",
        "models",
        f"cox_weight_time_to_{pct_loss}_pct_loss_phreg.csv",
    )
    if not os.path.exists(phreg_path):
        raise FileNotFoundError(f"Weight Cox PHREG output not found at: {phreg_path}")

    df = pd.read_csv(phreg_path)
    out: Dict[str, Tuple[str, str]] = {}

    # In the PHREG-style file, baseline A1C effects appear as rows with
    # Parameter == 'baseline_a1c_category' and Level equal to the category.
    mask = df["Parameter"].astype(str) == "baseline_a1c_category"
    sub = df[mask].copy()
    if sub.empty:
        # If the model happened not to include baseline A1C (shouldn't happen),
        # just return an empty mapping.
        return out

    for _, row in sub.iterrows():
        cat = str(row["Level"])
        hr = float(row["HazardRatio"])
        lci = float(row["HRLowerCL"])
        uci = float(row["HRUpperCL"])
        p = float(row["PrChiSq"])
        out[cat] = (_format_hr_ci(hr, lci, uci), _format_p(p))

    return out


def _load_a1c_hrs_by_a1c_category(gap: int, thr_abs: float) -> Dict[str, Tuple[str, str]]:
    """Return mapping {baseline_a1c_category: (HR_CI_str, p_str)} for an A1c drop.

    thr_abs should be one of 0.5, 1.0, 1.5, 2.0. HRs are for each baseline A1c
    category vs Normal Glycemia within the same Cox model.
    """

    tag_map: Dict[float, str] = {0.5: "0p5", 1.0: "1p0", 1.5: "1p5", 2.0: "2p0"}
    if thr_abs not in tag_map:
        raise ValueError(f"Unexpected A1c threshold: {thr_abs}")

    tag = tag_map[thr_abs]
    phreg_path = os.path.join(
        "output",
        f"gap_{gap}",
        "step8_survival_plots_and_cox",
        "step8_survival_a1c",
        "models",
        f"cox_a1c_time_to_{tag}_reduction_phreg.csv",
    )
    if not os.path.exists(phreg_path):
        raise FileNotFoundError(f"A1c Cox PHREG output not found at: {phreg_path}")

    df = pd.read_csv(phreg_path)
    out: Dict[str, Tuple[str, str]] = {}

    mask = df["Parameter"].astype(str) == "baseline_a1c_category"
    sub = df[mask].copy()
    if sub.empty:
        return out

    for _, row in sub.iterrows():
        cat = str(row["Level"])
        hr = float(row["HazardRatio"])
        lci = float(row["HRLowerCL"])
        uci = float(row["HRUpperCL"])
        p = float(row["PrChiSq"])
        out[cat] = (_format_hr_ci(hr, lci, uci), _format_p(p))

    return out


def _build_weight_section(gap: int) -> pd.DataFrame:
    km = _load_km_summary_weight(gap)

    # Index KM summary for quick lookup by (threshold_label, baseline_a1c_category)
    km_idx: Dict[Tuple[str, str], Tuple[int, int]] = {}
    for _, row in km.iterrows():
        key = (str(row["threshold_label"]), str(row["baseline_a1c_category"]))
        n = int(row["n"])
        e = int(row["n_events"])
        km_idx[key] = (e, n)

    # Order of thresholds and baseline groups
    thresholds: List[Tuple[float, str, int]] = [
        (-5.0, "≥5%", 5),
        (-10.0, "≥10%", 10),
        (-15.0, "≥15%", 15),
    ]
    baseline_groups = ["Prediabetes", "Type 2 Diabetes", "Poorly Controlled Diabetes"]

    rows: List[Dict[str, str]] = []

    for thr_val, thr_label, thr_int in thresholds:
        hr_by_cat = _load_weight_hrs_by_a1c_category(gap, thr_int)
        for j, cat in enumerate(baseline_groups):
            key = (thr_label, cat)
            e, n = km_idx.get(key, (np.nan, np.nan))
            en_str = "" if not np.isfinite(e) or not np.isfinite(n) else f"{int(e)} / {int(n)}"

            hr_ci, p_str = hr_by_cat.get(cat, ("", ""))

            rows.append(
                {
                    "Outcome": "Weight loss" if j == 0 else "",
                    "Threshold": thr_label if j == 0 else "",
                    "Baseline glycemic status": BASELINE_DISPLAY.get(cat, cat),
                    "e / n": en_str,
                    "HR (LCI-UCI)": hr_ci,
                    "p": p_str,
                }
            )

    return pd.DataFrame(rows)


def _build_a1c_section(gap: int) -> pd.DataFrame:
    km = _load_km_summary_a1c(gap)

    km_idx: Dict[Tuple[str, str], Tuple[int, int]] = {}
    for _, row in km.iterrows():
        key = (str(row["threshold_label"]), str(row["baseline_a1c_category"]))
        n = int(row["n"])
        e = int(row["n_events"])
        km_idx[key] = (e, n)

    thresholds: List[Tuple[float, str]] = [
        (0.5, "≥0.5%"),
        (1.0, "≥1.0%"),
        (1.5, "≥1.5%"),
        (2.0, "≥2.0%"),
    ]
    baseline_groups = ["Prediabetes", "Type 2 Diabetes", "Poorly Controlled Diabetes"]

    rows: List[Dict[str, str]] = []

    for thr_abs, thr_label in thresholds:
        hr_by_cat = _load_a1c_hrs_by_a1c_category(gap, thr_abs)
        for j, cat in enumerate(baseline_groups):
            key = (thr_label, cat)
            e, n = km_idx.get(key, (np.nan, np.nan))
            en_str = "" if not np.isfinite(e) or not np.isfinite(n) else f"{int(e)} / {int(n)}"

            hr_ci, p_str = hr_by_cat.get(cat, ("", ""))

            rows.append(
                {
                    "Outcome": "A1c reduction" if j == 0 else "",
                    "Threshold": thr_label if j == 0 else "",
                    "Baseline glycemic status": BASELINE_DISPLAY.get(cat, cat),
                    "e / n": en_str,
                    "HR (LCI-UCI)": hr_ci,
                    "p": p_str,
                }
            )

    return pd.DataFrame(rows)


def main(argv=None) -> None:
    args = parse_args(argv)
    gap = args.gap

    weight_df = _build_weight_section(gap)
    a1c_df = _build_a1c_section(gap)

    # Determine output directory (defaults to output/gap_<gap>/step8b)
    if args.out_markdown is not None:
        # If a path was provided, use its directory as the output folder
        outdir = os.path.dirname(args.out_markdown) or "."
    else:
        outdir = os.path.join("output", f"gap_{gap}", "step8b")

    os.makedirs(outdir, exist_ok=True)

    weight_csv = os.path.join(outdir, "step8b_weight_threshold_cox_summary.csv")
    a1c_csv = os.path.join(outdir, "step8b_a1c_threshold_cox_summary.csv")

    weight_df.to_csv(weight_csv, index=False)
    a1c_df.to_csv(a1c_csv, index=False)


if __name__ == "__main__":
    main()
