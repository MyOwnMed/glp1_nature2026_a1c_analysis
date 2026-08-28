#!/usr/bin/env python3
"""
Structured-Only Sensitivity Analysis — Step 0: Pre-filter raw data.

Reads the raw step8g CSV and produces a structured-only version by:
  1. Zeroing out glp1_event_for_adherance where glp1_has_structured == 0
     (so the adherence gap algorithm only sees structured GLP-1 evidence)
  2. Nulling out weight_in_pounds_final (and pct_weight_change) where
     weight_has_structured == 0
  3. Nulling out a1c_value (and abs_a1c_change) where
     a1c_has_structured == 0

This MUST run BEFORE step1, so that adherence windows are computed
from structured data only.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]


def _flag(df: pd.DataFrame, col: str) -> pd.Series:
    """Return integer flag series, 0 where missing or absent."""
    if col not in df.columns:
        return pd.Series(0, index=df.index, dtype=int)
    return df[col].fillna(0).astype(int)


def prefilter(src_csv: Path, out_csv: Path) -> dict:
    """Pre-filter raw data to structured-only signals."""
    log.info("Reading %s", src_csv)
    df = pd.read_csv(src_csv)
    n_rows = len(df)
    n_pat = df["patient_id"].nunique()
    log.info("Raw: %d rows, %d patients", n_rows, n_pat)

    stats = {"raw_rows": n_rows, "raw_patients": n_pat}

    # ── 1. GLP-1 adherence: zero out unstructured-only events ──────
    glp1_struct = _flag(df, "glp1_has_structured")
    if "glp1_event_for_adherance" in df.columns:
        adh = df["glp1_event_for_adherance"].fillna(0).astype(float)
        unstructured_glp1 = (glp1_struct == 0) & (adh.isin([1, 2]))
        n_zeroed = int(unstructured_glp1.sum())
        df.loc[unstructured_glp1, "glp1_event_for_adherance"] = 0
        stats["glp1_events_zeroed"] = n_zeroed
        log.info("GLP-1 adherence events zeroed (unstructured): %d", n_zeroed)
    else:
        stats["glp1_events_zeroed"] = 0
        log.warning("glp1_event_for_adherance column not found")

    # ── 2. Weight: null out unstructured-only measurements ─────────
    wt_struct = _flag(df, "weight_has_structured")
    wt_unstruct_mask = wt_struct == 0
    for col in ["weight_in_pounds_final", "pct_weight_change"]:
        if col in df.columns:
            n_before = df[col].notna().sum()
            df.loc[wt_unstruct_mask, col] = np.nan
            n_after = df[col].notna().sum()
            stats[f"weight_{col}_nulled"] = int(n_before - n_after)
            log.info("Weight %s nulled: %d values", col, n_before - n_after)

    # ── 3. A1C: null out unstructured-only measurements ────────────
    a1c_struct = _flag(df, "a1c_has_structured")
    a1c_unstruct_mask = a1c_struct == 0
    for col in ["a1c_value", "abs_a1c_change", "A1C_SOURCE"]:
        if col in df.columns:
            n_before = df[col].notna().sum()
            df.loc[a1c_unstruct_mask, col] = np.nan
            n_after = df[col].notna().sum()
            stats[f"a1c_{col}_nulled"] = int(n_before - n_after)
            log.info("A1C %s nulled: %d values", col, n_before - n_after)

    # Write pre-filtered CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log.info("Wrote %s (%d rows)", out_csv, len(df))
    stats["output_rows"] = len(df)
    stats["output_patients"] = df["patient_id"].nunique()
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Pre-filter raw CSV to structured-only signals (before step1)",
    )
    parser.add_argument(
        "--input-csv",
        default=str(ROOT / "root_data" / "merged"
                     / "step8g_with_unstructured_flags_with_assessments_weightcleaned.csv"),
        help="Raw input CSV",
    )
    parser.add_argument(
        "--output-csv",
        default=str(ROOT / "output" / "structured_only" / "prefiltered_structured_only.csv"),
        help="Output pre-filtered CSV",
    )
    args = parser.parse_args()

    stats = prefilter(Path(args.input_csv), Path(args.output_csv))
    log.info("Pre-filter stats: %s", stats)


if __name__ == "__main__":
    main()
