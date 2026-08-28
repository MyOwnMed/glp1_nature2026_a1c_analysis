#!/usr/bin/env python3
"""
Run all add_unstructured analyses for a single adherence-gap cohort.

Usage:
    python3 code/add_unstructured/run_for_gap.py --gap 120

What it does:
  1. Loads each domain's prepared data from output/add_unstructured/data/
  2. Applies per-patient adherence censoring derived from the step1 gap file
     (post-GLP1 rows beyond each patient's max observed day are dropped;
      pre-GLP1 rows are never censored)
  3. Writes gap-censored data to output/submitted_analysis/gap_<N>/add_unstructured/data/
  4. Runs run_its_analysis, run_trajectory_plots, run_baseline_anchor_analysis,
     and run_elevated_analysis as subprocesses with AU_DATADIR / AU_OUTROOT
     environment variables pointing to the gap-specific paths.

Output root for each gap:
    output/submitted_analysis/gap_<N>/add_unstructured/
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT   = Path(__file__).resolve().parents[2]
_CI_DIR   = os.environ.get('CONF_INT_DIR')
CONF_INT  = Path(_CI_DIR) if _CI_DIR else ROOT / "output" / "submitted_analysis"
BASE_DATA = CONF_INT / "1_no_adherence" / "data"
STEP1     = ROOT / "output" / "step1_prepare_analysis_dataset"

DOMAINS = [
    "phq9",
    "pain_score",
    "waist_circumference",
    "alcohol",
    "muscle_strength",
]

SCRIPTS = [
    ROOT / "code" / "add_unstructured" / "run_its_analysis.py",
    ROOT / "code" / "add_unstructured" / "run_elevated_analysis.py",
    ROOT / "code" / "add_unstructured" / "run_trajectory_plots.py",
    ROOT / "code" / "add_unstructured" / "run_baseline_anchor_analysis.py",
    ROOT / "code" / "add_unstructured" / "forest_point_estimates.py",
    ROOT / "code" / "add_unstructured" / "run_time_varying_covar.py",
]


def parse_args():
    p = argparse.ArgumentParser(description="Run add_unstructured analyses for one gap cohort")
    p.add_argument("--gap", type=int, required=True,
                   help="Adherence gap threshold in days (e.g. 30, 60, 90, 120, 150, 180, 365, 548, 730)")
    p.add_argument("--name", default="add_unstructured",
                   help="Output subdirectory name under conf_int/gap_N/ (default: add_unstructured)")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if output directory already has files")
    return p.parse_args()


def load_gap_cutoffs(gap: int) -> pd.Series:
    """Load per-patient max days_from_baseline from the step1 gap file."""
    fpath = STEP1 / f"analysis_ready_gap{gap}.csv"
    if not fpath.exists():
        raise FileNotFoundError(f"Step1 gap file not found: {fpath}")
    tmp = pd.read_csv(fpath, usecols=["patient_id", "days_from_baseline"])
    cutoffs = tmp.groupby("patient_id")["days_from_baseline"].max()
    log.info("Gap %d: loaded cutoffs for %d patients (max days range %.0f–%.0f)",
             gap, len(cutoffs), cutoffs.min(), cutoffs.max())
    return cutoffs


def apply_gap_censoring(domain_df: pd.DataFrame, cutoffs: pd.Series) -> pd.DataFrame:
    """
    Restrict to the gap cohort and censor post-GLP1 rows beyond each
    patient's adherence cutoff.  Pre-GLP1 rows (days < 0) are kept as-is.
    """
    cohort_ids = set(cutoffs.index)
    df = domain_df[domain_df["patient_id"].isin(cohort_ids)].copy()
    cutoff_map = cutoffs.to_dict()
    df["_cutoff"] = df["patient_id"].map(cutoff_map)
    df = df[
        (df["days_from_baseline"] < 0) |
        (df["days_from_baseline"] <= df["_cutoff"])
    ].drop(columns=["_cutoff"])

    # Recompute has_both_periods flag after censoring
    pre_ids  = set(df.loc[df["post"] == 0, "patient_id"])
    post_ids = set(df.loc[df["post"] == 1, "patient_id"])
    both     = pre_ids & post_ids
    df["has_both_periods"] = df["patient_id"].isin(both).astype(int)

    return df


def prepare_gap_data(gap: int, gap_data_dir: Path, cutoffs: pd.Series):
    """Censor each domain CSV and write to gap_data_dir."""
    gap_data_dir.mkdir(parents=True, exist_ok=True)

    for domain in DOMAINS:
        src = BASE_DATA / f"{domain}_prepared.csv"
        if not src.exists():
            log.warning("  Base data missing for %s — skipping", domain)
            continue
        base_df = pd.read_csv(src)
        base_df["days_from_baseline"] = pd.to_numeric(
            base_df["days_from_baseline"], errors="coerce")

        censored = apply_gap_censoring(base_df, cutoffs)

        n_pts  = censored["patient_id"].nunique()
        n_both = censored.loc[censored["has_both_periods"] == 1, "patient_id"].nunique()
        log.info("  %s: %d patients (%d with both periods)", domain, n_pts, n_both)

        out_path = gap_data_dir / f"{domain}_prepared.csv"
        censored.to_csv(out_path, index=False)

    log.info("Gap %d data written to %s", gap, gap_data_dir)


def run_script(script: Path, env: dict, gap: int):
    """Run a Python analysis script as a subprocess with custom env."""
    name = script.stem
    log.info("Running %s ...", name)
    result = subprocess.run(
        [sys.executable, str(script)],
        env=env,
        capture_output=False,
    )
    if result.returncode != 0:
        log.error("  %s FAILED (exit %d)", name, result.returncode)
    else:
        log.info("  %s done", name)
    return result.returncode


def main():
    args = parse_args()
    gap  = args.gap

    out_root    = CONF_INT / f"gap_{gap}" / args.name
    gap_data_dir = out_root / "data"

    log.info("=" * 60)
    log.info("add_unstructured analyses for gap=%d days", gap)
    log.info("Output root: %s", out_root)
    log.info("=" * 60)

    # Skip if already complete and not forced
    if not args.force and (out_root / "ITS" / "figures").exists():
        existing = list((out_root / "ITS" / "figures").glob("*.png"))
        if existing:
            log.info("Output already exists (%d figures). Use --force to re-run.",
                     len(existing))
            return

    # -- 1. Load cutoffs and prepare data --
    cutoffs = load_gap_cutoffs(gap)
    prepare_gap_data(gap, gap_data_dir, cutoffs)

    # -- 2. Build environment for subprocesses --
    env = os.environ.copy()
    env["AU_DATADIR"] = str(gap_data_dir)
    env["AU_OUTROOT"] = str(out_root)
    env["AU_STEP1"]   = str(STEP1 / f"analysis_ready_gap{gap}.csv")
    env["MPLBACKEND"] = "Agg"

    # -- 3. Run each analysis script --
    failures = []
    for script in SCRIPTS:
        if not script.exists():
            log.error("Script not found: %s", script)
            failures.append(script.name)
            continue
        rc = run_script(script, env, gap)
        if rc != 0:
            failures.append(script.name)

    # -- 4. Summary --
    log.info("=" * 60)
    if failures:
        log.error("Completed with failures: %s", ", ".join(failures))
        sys.exit(1)
    else:
        log.info("All analyses complete for gap=%d", gap)
        log.info("Output: %s", out_root)


if __name__ == "__main__":
    main()
