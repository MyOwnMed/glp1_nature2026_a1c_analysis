#!/usr/bin/env python3
"""Step 0a: patients remaining in follow-up, month by month, per persistence definition.

Produces the supplementary attrition table ("Cohort attrition under 120-day
persistence definition"): for each persistence definition and each 30-day month
from baseline through 18 months, how many patients are still within persistence.

────────────────────────────────────────────────────────────────────────────────
WHAT CHANGED IN PART II v2 (code-review item 1, two defects)
────────────────────────────────────────────────────────────────────────────────
1. Patient-identifier dtype. The submitted version built its patient set as
   ``set(df["patient_id"].astype(str))`` but tested membership against censor-map
   keys that had kept whatever dtype ``read_csv`` inferred. With numeric-looking
   identifiers, ``pid in censor_map`` would have been False for every patient:
   month 0 uses ``len(patients)`` and so would have looked right, while every
   later month collapsed to zero. This project's identifiers are UUID strings, so
   the comparison happened to hold — the escape route the reviewer anticipated.
   Both sides are now normalised through ``persistence.normalize_patient_id``,
   the one place identifier form is decided, so the failure mode is gone
   regardless of what a future CSV export looks like.

2. Where the censor map is computed from. The submitted version recomputed it
   from ``analysis_ready_gap{g}.csv``, which step1 had already gap-censored:
   "last active day + gap" measured on truncated data can only give a censor day
   at or before the true one, so the curve was censored twice. This version
   computes it from the untrimmed source by default (``--censor-from source``).

   The published table is nonetheless correct, in all 171 cells (9 persistence
   definitions x 19 monthly time points) — verified against the untrimmed source;
   see E1 (``table_s1_true_vs_published.csv``) and ``verification/``. GLP-1
   mentions ride on the same rows as the measurements, and the trimming rule and
   the recomputation use the same time gap, so any mention sitting on a removed
   row is necessarily more than *g* days from the last kept mention and cannot
   extend a patient's counted follow-up. Computing from the untrimmed source is
   the safer construction, not a correction to the published numbers.

   ``--censor-from gapfiles`` reproduces the submitted construction exactly and is
   retained so the equivalence stays checkable. It is no longer the default.

The follow-up rule itself now lives in ``persistence.py`` — one implementation,
shared with step1 (code-review item 2).

Run from the study root:
  python3 code/step0a_samplesize_analysis.py \\
      --analysis-dir output/submitted_analysis/step1_weight \\
      --source-csv root_data/merged/step8g_with_unstructured_flags_with_assessments_weightcleaned.csv \\
      --outdir output/step0a_samplesize_analysis
"""

import argparse
import logging
import re
import sys as _sys
from pathlib import Path
from pathlib import Path as _Path
from typing import Dict, List, Optional, Sequence, Set

import pandas as pd

for _p in [_Path(__file__).resolve().parent, *_Path(__file__).resolve().parents]:
    if (_p / "persistence.py").exists():
        _sys.path.insert(0, str(_p))
        break
from persistence import censor_days, mention_timeline, normalize_patient_id

# Row semantics used when rebuilding the mention timeline from the untrimmed
# source. These mirror step1.load_and_prepare: the observation date is `date`,
# the day index is recomputed against baseline_glp1_date, and a row is kept when
# it carries a non-missing outcome and falls in [0, MAX_DAYS].
SOURCE_MAX_DAYS = 730
SOURCE_CHUNK_ROWS = 1_500_000


def discover_gap_files(analysis_dir: Path) -> Dict[int, Path]:
    """Find analysis_ready_gap*.csv files and return mapping gap_days -> path."""
    gap_files: Dict[int, Path] = {}
    for p in sorted(analysis_dir.glob("analysis_ready_gap*.csv")):
        m = re.search(r"gap(\d+)", p.name)
        if not m:
            continue
        # Skip the prebaseline special file if matched by the pattern.
        if "step7_prebaseline" in p.name:
            continue
        gap_files[int(m.group(1))] = p
    return gap_files


def month_days(n_months: int, month_length_days: int = 30) -> List[int]:
    """Day thresholds from 0 months through n_months, e.g. [0, 30, ..., 540]."""
    return [m * month_length_days for m in range(n_months + 1)]


def cohort_from_gap_file(path: Path) -> Set[str]:
    """Normalised patient identifiers present in one analysis-ready file."""
    ids = pd.read_csv(path, usecols=["patient_id"])["patient_id"].dropna()
    return set(normalize_patient_id(ids))


def timeline_from_gap_file(path: Path) -> Dict[str, list]:
    """Mention timeline read from an already-gap-censored analysis file.

    This is the submitted construction, retained for ``--censor-from gapfiles``.
    """
    needed = ["patient_id", "glp1_event_for_adherance", "glp1_days_from_baseline"]
    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in needed if c not in header]
    if missing:
        raise KeyError(f"{path.name} lacks required columns {missing}")
    return mention_timeline(pd.read_csv(path, usecols=needed))


def timeline_from_source(
    source_csv: Path,
    cohort: Set[str],
    max_days: int = SOURCE_MAX_DAYS,
    chunksize: int = SOURCE_CHUNK_ROWS,
    outcome_col: str = "pct_weight_change",
) -> Dict[str, list]:
    """Mention timeline rebuilt from the UNTRIMMED source, before any censoring.

    Replicates step1.load_and_prepare's row semantics and step1's day-0 anchoring
    (a patient with no observed day-0 row gets the created day-0 mention that
    records treatment initiation), then returns the full per-patient timeline.
    Read in chunks: the source is ~1.24M rows x 169 columns.
    """
    usecols = [
        "patient_id",
        "date",
        "baseline_glp1_date",
        outcome_col,
        "glp1_event_for_adherance",
        "glp1_days_from_baseline",
    ]
    header = pd.read_csv(source_csv, nrows=0).columns
    missing = [c for c in usecols if c not in header]
    if missing:
        raise KeyError(
            f"{source_csv.name} lacks columns required to rebuild the mention "
            f"timeline: {missing}"
        )

    pairs: Dict[str, set] = {}
    has_observed_day0: Set[str] = set()
    seen: Set[str] = set()
    total = kept = 0

    reader = pd.read_csv(source_csv, usecols=usecols, chunksize=chunksize, low_memory=False)
    for i, chunk in enumerate(reader, 1):
        total += len(chunk)
        chunk["patient_id"] = normalize_patient_id(chunk["patient_id"])
        chunk = chunk[chunk["patient_id"].isin(cohort)]
        if not chunk.empty:
            day = (
                pd.to_datetime(chunk["date"], errors="coerce", cache=True)
                - pd.to_datetime(chunk["baseline_glp1_date"], errors="coerce", cache=True)
            ).dt.days
            keep = (
                chunk[outcome_col].notna()
                & day.notna()
                & (day >= 0)
                & (day <= max_days)
            )
            sub = chunk.loc[keep].copy()
            sub["_day"] = day[keep]
            kept += len(sub)

            seen.update(sub["patient_id"].unique())
            has_observed_day0.update(sub.loc[sub["_day"] == 0, "patient_id"].unique())

            values = pd.to_numeric(sub["glp1_event_for_adherance"], errors="coerce")
            event_days = pd.to_numeric(sub["glp1_days_from_baseline"], errors="coerce")
            m = values.notna() & event_days.notna() & (event_days >= 0)
            for pid, value, event_day in zip(
                sub.loc[m, "patient_id"],
                values[m].to_numpy(dtype=float),
                event_days[m].to_numpy(dtype=float),
            ):
                pairs.setdefault(pid, set()).add((float(value), float(event_day)))
        if i % 5 == 0:
            logging.info("  chunk %d: %s rows scanned, %s cohort rows kept", i,
                         f"{total:,}", f"{kept:,}")

    # step1 anchors every trajectory at day 0; that created row carries
    # glp1_event_for_adherance = 1, so the timeline must include it here too or
    # the from-source map would not describe the dataset step1 actually emits.
    anchored = 0
    for pid in seen - has_observed_day0:
        pairs.setdefault(pid, set()).add((1.0, 0.0))
        anchored += 1

    missing_from_scan = cohort - seen
    if missing_from_scan:
        raise RuntimeError(
            f"{len(missing_from_scan)} of {len(cohort)} cohort patients were not "
            f"found in {source_csv.name}. The source does not cover the cohort in "
            "the analysis directory; check that both refer to the same run."
        )

    logging.info(
        "Source scan complete: %s rows scanned, %s cohort rows kept, %d patients, "
        "%d day-0 mentions added for patients without an observed day-0 row",
        f"{total:,}", f"{kept:,}", len(seen), anchored,
    )
    return {pid: sorted(p, key=lambda t: t[1]) for pid, p in pairs.items()}


def aggregate_counts(
    gap_to_file: Dict[int, Path],
    n_months: int = 18,
    month_length_days: int = 30,
    censor_from: str = "source",
    source_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """Monthly within-persistence counts for each gap.

    Columns: month_number, days_from_baseline, then gap_<days> per gap.
    """
    days = month_days(n_months, month_length_days)
    out = pd.DataFrame({"month_number": list(range(n_months + 1)), "days_from_baseline": days})

    source_timeline: Optional[Dict[str, list]] = None

    for gap, path in sorted(gap_to_file.items()):
        logging.info("Processing gap %s (cohort from %s)", gap, path.name)
        cohort = cohort_from_gap_file(path)

        if censor_from == "source":
            if source_csv is None:
                raise ValueError("--source-csv is required when --censor-from source")
            if source_timeline is None:
                # The cohort is the same at every gap (step1 emits one cohort and
                # censors follow-up within it), so the source is scanned once.
                source_timeline = timeline_from_source(source_csv, cohort)
            timeline = source_timeline
        else:
            timeline = timeline_from_gap_file(path)

        cmap = censor_days(timeline, gap)
        counts = [
            len(cohort) if d == 0 else sum(1 for pid in cohort if cmap.get(pid, -1) >= d)
            for d in days
        ]
        out[f"gap_{gap}"] = counts
        logging.info("  gap %s: %d patients, %d with a persistence day", gap,
                     len(cohort), len(cmap))

    return out


def save_counts_csv(df: pd.DataFrame, outdir: Path, filename: str = "samplesize_by_month.csv") -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / filename
    df.to_csv(out_path, index=False)
    logging.info("Wrote %s", out_path)
    return out_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description="Step0a: patients in follow-up over time, per persistence definition",
    )
    p.add_argument(
        "--analysis-dir",
        default=str(Path("output") / "step1_prepare_analysis_dataset"),
        help="Directory containing analysis_ready_gap*.csv files (defines the cohort)",
    )
    p.add_argument(
        "--outdir",
        default=str(Path("output") / "step0a_samplesize_analysis"),
        help="Output directory for the single CSV",
    )
    p.add_argument(
        "--source-csv",
        default=str(
            Path("root_data") / "merged"
            / "step8g_with_unstructured_flags_with_assessments_weightcleaned.csv"
        ),
        help=(
            "Untrimmed merged source CSV. The mention timeline is rebuilt from "
            "this, before any gap censoring (default construction)."
        ),
    )
    p.add_argument(
        "--censor-from",
        choices=("source", "gapfiles"),
        default="source",
        help=(
            "Where to compute the persistence timeline from. 'source' (default) "
            "rebuilds it from the untrimmed source CSV. 'gapfiles' reproduces the "
            "submitted construction, which read the already-gap-censored "
            "analysis_ready_gap*.csv files; retained for verification only."
        ),
    )
    p.add_argument("--months", type=int, default=18,
                   help="Number of months from baseline to include (default 18 = 1.5 years)")
    p.add_argument("--month-length-days", type=int, default=30,
                   help="Days per month bin (default 30)")
    p.add_argument("--log-level", default="INFO",
                   help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
    )

    analysis_dir = Path(args.analysis_dir)
    gap_files = discover_gap_files(analysis_dir)
    if not gap_files:
        # Fail loudly: the submitted version logged an error and returned 0, so a
        # scripted run could not tell "no data" from "table written".
        raise FileNotFoundError(
            f"No analysis_ready_gap*.csv files found in {analysis_dir}. Run step1 "
            "first, or point --analysis-dir at the directory holding them."
        )

    source_csv = Path(args.source_csv) if args.censor_from == "source" else None
    if source_csv is not None and not source_csv.is_file():
        raise FileNotFoundError(
            f"--source-csv not found: {source_csv}. The persistence timeline is "
            "computed from the untrimmed source; pass --censor-from gapfiles to "
            "reproduce the submitted construction from the censored files instead."
        )

    counts_df = aggregate_counts(
        gap_files,
        n_months=args.months,
        month_length_days=args.month_length_days,
        censor_from=args.censor_from,
        source_csv=source_csv,
    )
    save_counts_csv(counts_df, Path(args.outdir))


if __name__ == "__main__":
    main()
