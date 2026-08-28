#!/usr/bin/env python3
"""
Structured-Only Sensitivity Analysis — Step 0: Filter gap-120 datasets.

Reads the existing analysis-ready gap-120 CSVs (weight and A1C) and
produces structured-only versions by keeping only observation rows where
the outcome measurement has a structured-data source flag.

Filtering rules:
  - A1C dataset:   keep rows where a1c_has_structured == 1
  - Weight dataset: keep rows where weight_has_structured == 1
  - Baseline (day 0) carried rows: keep if a1c_has_structured or
    weight_has_structured is set, OR if neither flag is set (carried
    baseline row — retained so patients keep their baseline anchor).
  - After filtering rows, require each patient has at least 1
    post-baseline observation (days_from_baseline > 0).
  - Recompute abs_a1c_change and pct_weight_change from the retained
    baseline values (values already consistent in-row).

Outputs:
  output/structured_only/gap_120/data/analysis_ready_a1c_gap120.csv
  output/structured_only/gap_120/data/analysis_ready_gap120.csv
  output/structured_only/gap_120/data/filtering_summary.md
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]


def _bool_col(df: pd.DataFrame, col: str) -> pd.Series:
    return df.get(col, pd.Series(0, index=df.index)).fillna(0).astype(int)


def _inject_flags_from_source(df: pd.DataFrame, flag_col: str, flag_src_csv: Path) -> pd.Series:
    """Load per-patient flag from flag_src_csv and merge onto df by patient_id."""
    log.info("'%s' missing — loading from %s", flag_col, flag_src_csv)
    flag_df = pd.read_csv(flag_src_csv, usecols=["patient_id", flag_col])
    # Take the max per patient (1 if any row has it set)
    flag_map = flag_df.groupby("patient_id")[flag_col].max().fillna(0).astype(int)
    return df["patient_id"].map(flag_map).fillna(0).astype(int)


def filter_a1c(src_csv: Path, out_csv: Path, flag_src_csv: Path | None = None) -> dict:
    """Filter A1C dataset to structured observations only."""
    df = pd.read_csv(src_csv)
    n_rows_orig = len(df)
    n_pat_orig = df["patient_id"].nunique()
    log.info("A1C source: %d rows, %d patients", n_rows_orig, n_pat_orig)

    # Flag columns — inject from flag_src_csv if missing
    if "a1c_has_structured" not in df.columns and flag_src_csv and flag_src_csv.exists():
        df["a1c_has_structured"] = _inject_flags_from_source(df, "a1c_has_structured", flag_src_csv)
    s = _bool_col(df, "a1c_has_structured")

    # Identify baseline rows (day 0) — these may be carried rows with no flags
    is_baseline = df["days_from_baseline"] == 0

    # Keep: structured rows OR baseline rows (preserve baseline anchor)
    keep_mask = (s == 1) | is_baseline
    df = df[keep_mask].copy()
    log.info("After structured filter (keeping baselines): %d rows", len(df))

    # Require at least 1 post-baseline structured observation per patient
    s_filtered = _bool_col(df, "a1c_has_structured")
    has_post = (df["days_from_baseline"] > 0) & (s_filtered == 1)
    keep_pids = set(df.loc[has_post, "patient_id"].unique())
    df = df[df["patient_id"].isin(keep_pids)].copy()
    n_rows_final = len(df)
    n_pat_final = df["patient_id"].nunique()
    log.info("After requiring post-baseline structured obs: %d rows, %d patients",
             n_rows_final, n_pat_final)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log.info("Wrote %s", out_csv)

    return {
        "domain": "A1C",
        "rows_orig": n_rows_orig,
        "patients_orig": n_pat_orig,
        "rows_final": n_rows_final,
        "patients_final": n_pat_final,
        "rows_dropped": n_rows_orig - n_rows_final,
        "patients_dropped": n_pat_orig - n_pat_final,
        "pct_rows_retained": n_rows_final / n_rows_orig * 100 if n_rows_orig else 0,
        "pct_patients_retained": n_pat_final / n_pat_orig * 100 if n_pat_orig else 0,
    }


def filter_weight(src_csv: Path, out_csv: Path, flag_src_csv: Path | None = None) -> dict:
    """Filter weight dataset to structured observations only."""
    df = pd.read_csv(src_csv)
    n_rows_orig = len(df)
    n_pat_orig = df["patient_id"].nunique()
    log.info("Weight source: %d rows, %d patients", n_rows_orig, n_pat_orig)

    # Inject weight_has_structured from flag_src_csv if missing
    if "weight_has_structured" not in df.columns and flag_src_csv and flag_src_csv.exists():
        df["weight_has_structured"] = _inject_flags_from_source(df, "weight_has_structured", flag_src_csv)
    s = _bool_col(df, "weight_has_structured")
    is_baseline = df["days_from_baseline"] == 0

    keep_mask = (s == 1) | is_baseline
    df = df[keep_mask].copy()
    log.info("After structured filter (keeping baselines): %d rows", len(df))

    s_filtered = _bool_col(df, "weight_has_structured")
    has_post = (df["days_from_baseline"] > 0) & (s_filtered == 1)
    keep_pids = set(df.loc[has_post, "patient_id"].unique())
    df = df[df["patient_id"].isin(keep_pids)].copy()
    n_rows_final = len(df)
    n_pat_final = df["patient_id"].nunique()
    log.info("After requiring post-baseline structured obs: %d rows, %d patients",
             n_rows_final, n_pat_final)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log.info("Wrote %s", out_csv)

    return {
        "domain": "Weight",
        "rows_orig": n_rows_orig,
        "patients_orig": n_pat_orig,
        "rows_final": n_rows_final,
        "patients_final": n_pat_final,
        "rows_dropped": n_rows_orig - n_rows_final,
        "patients_dropped": n_pat_orig - n_pat_final,
        "pct_rows_retained": n_rows_final / n_rows_orig * 100 if n_rows_orig else 0,
        "pct_patients_retained": n_pat_final / n_pat_orig * 100 if n_pat_orig else 0,
    }


def write_summary(results: list, out_md: Path):
    lines = [
        "# Structured-Only Filtering Summary (Gap 120)\n",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## Filtering Rule\n",
        "- Keep only observation rows where the outcome measurement has `*_has_structured == 1`",
        "- Baseline (day 0) rows are always retained to preserve the baseline anchor",
        "- Patients must have at least 1 post-baseline structured observation\n",
        "## Results\n",
        "| Domain | Orig Rows | Orig Patients | Final Rows | Final Patients | Rows Retained % | Patients Retained % |",
        "|--------|----------:|--------------:|-----------:|--------------:|----------------:|--------------------:|",
    ]
    for r in results:
        lines.append(
            f"| {r['domain']} | {r['rows_orig']:,} | {r['patients_orig']:,} | "
            f"{r['rows_final']:,} | {r['patients_final']:,} | "
            f"{r['pct_rows_retained']:.1f}% | {r['pct_patients_retained']:.1f}% |"
        )
    lines.append("")
    for r in results:
        lines.append(f"### {r['domain']}")
        lines.append(f"- Rows dropped: {r['rows_dropped']:,}")
        lines.append(f"- Patients dropped: {r['patients_dropped']:,}")
        lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))
    log.info("Wrote %s", out_md)


def main():
    parser = argparse.ArgumentParser(
        description="Filter gap-N analysis datasets to structured-only observations",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=120,
        help="Adherence gap in days (used to derive default CSV paths and output filenames)",
    )
    parser.add_argument(
        "--a1c-csv",
        default=None,
        help="Source A1C gap CSV (default: auto-derived from --gap)",
    )
    parser.add_argument(
        "--weight-csv",
        default=None,
        help="Source weight gap CSV (default: auto-derived from --gap)",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory for filtered datasets (default: auto-derived from --gap)",
    )
    parser.add_argument(
        "--flag-a1c-csv",
        default=None,
        help="Fallback CSV with a1c_has_structured column (used when source CSV lacks the flag)",
    )
    parser.add_argument(
        "--flag-weight-csv",
        default=None,
        help="Fallback CSV with weight_has_structured column (used when source CSV lacks the flag)",
    )
    args = parser.parse_args()

    gap = args.gap
    a1c_csv = Path(args.a1c_csv) if args.a1c_csv else \
        ROOT / "output" / "step1_prepare_analysis_dataset_a1c" / f"analysis_ready_a1c_gap{gap}.csv"
    weight_csv = Path(args.weight_csv) if args.weight_csv else \
        ROOT / "output" / "step1_prepare_analysis_dataset" / f"analysis_ready_gap{gap}.csv"
    outdir = Path(args.outdir) if args.outdir else \
        ROOT / "output" / "structured_only" / f"gap_{gap}" / "data"

    # Default flag sources: gap_120 all_data CSVs (they have the structured flag columns)
    gap120_a1c = ROOT / "output" / "step1_prepare_analysis_dataset_a1c" / "analysis_ready_a1c_gap120.csv"
    gap120_weight = ROOT / "output" / "step1_prepare_analysis_dataset" / "analysis_ready_gap120.csv"
    flag_a1c_csv = Path(args.flag_a1c_csv) if args.flag_a1c_csv else gap120_a1c
    flag_weight_csv = Path(args.flag_weight_csv) if args.flag_weight_csv else gap120_weight

    results = []
    results.append(filter_a1c(
        a1c_csv,
        outdir / f"analysis_ready_a1c_gap{gap}.csv",
        flag_src_csv=flag_a1c_csv,
    ))
    results.append(filter_weight(
        weight_csv,
        outdir / f"analysis_ready_gap{gap}.csv",
        flag_src_csv=flag_weight_csv,
    ))
    write_summary(results, outdir / "filtering_summary.md")

    print("\n=== Summary ===")
    for r in results:
        print(f"  {r['domain']}: {r['patients_orig']} → {r['patients_final']} patients "
              f"({r['pct_patients_retained']:.1f}%), "
              f"{r['rows_orig']:,} → {r['rows_final']:,} rows "
              f"({r['pct_rows_retained']:.1f}%)")


if __name__ == "__main__":
    main()
