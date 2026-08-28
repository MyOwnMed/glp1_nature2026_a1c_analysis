#!/usr/bin/env python
"""Step 1 (weight): build the gap-censored, analysis-ready weight dataset.

Baseline window and day-0 anchoring — see README.md for the full statement in
Methods wording. In brief: baseline GLP-1 exposure is the first qualifying GLP-1
RA medication event; baseline measurements come from a -60/+14-day window around
that date, taking the value closest in time with ties resolved on or before
baseline. Every patient contributes an observation at day 0. Where a patient's
baseline weight was measured inside the window but not on day 0 itself,
``_ensure_baseline_rows`` writes the day-0 row carrying that recorded baseline
value, with change-from-baseline zero by definition. Those rows are marked
``baseline_carried_to_day0 = 1`` and the marker is carried through to the output CSVs.
"""

import argparse
import json
import logging
import sys as _sys
from pathlib import Path
from pathlib import Path as _Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

for _p in [_Path(__file__).resolve().parent, *_Path(__file__).resolve().parents]:
    if (_p / "persistence.py").exists():
        _sys.path.insert(0, str(_p))
        break
from persistence import adherence_flags

# ── Cohort definition: the injectable GLP-1 RA agents, one source of truth ────
# Addresses code-review item "GLP1_INJECTABLE_NAMES is dead". The submitted code
# carried a 6-name set that nothing read, alongside inline regexes in the live
# cohort filter that happened to list the same agents. Two lists agreeing by
# coincidence is one edit away from disagreeing, so the dead set is gone and the
# filter patterns below are built from these names — change an agent here and
# the cohort filter changes with it.
GLP1_BRAND_NAMES = ("ozempic", "wegovy", "mounjaro", "zepbound")
GLP1_INGREDIENT_NAMES = ("semaglutide", "tirzepatide")


def _word_boundary_pattern(names: Sequence[str]) -> str:
    """Regex matching any of *names* as a whole word, e.g. ``\\bozempic\\b|...``."""
    return "|".join(rf"\b{name}\b" for name in names)


GLP1_BRAND_PATTERN = _word_boundary_pattern(GLP1_BRAND_NAMES)
GLP1_INGREDIENT_PATTERN = _word_boundary_pattern(GLP1_INGREDIENT_NAMES)
GLP1_ANY_PATTERN = _word_boundary_pattern(GLP1_BRAND_NAMES + GLP1_INGREDIENT_NAMES)

GAPS_DEFAULT = [30, 60, 90, 120, 150, 180, 365, 730]

# Marker column distinguishing rows created to anchor a trajectory at day 0 from
# rows carrying an observed measurement (code-review item 3). 1 = created here,
# 0 = observed. Written to every analysis-ready CSV; see _ensure_baseline_rows.
BASELINE_CARRIED_COL = "baseline_carried_to_day0"

# Follow-up horizon cap for step1, in days (the ``--max-days`` default).
#
# 730 days is the widest horizon any script uses; it is an outer bound on the
# data step1 emits, NOT the analysis window. The paper's follow-up limit is 18
# months, and the downstream caps sit at or just beyond it:
#
#   step1 (here)                       730  outer bound on emitted follow-up
#   step6b/step6c --max-days           548  ~18 months, contrast/plot horizon
#   step6d --truncate-days             548  ~18 months, fitting horizon
#   step8_survival_* MAX_FOLLOWUP_DAYS 540  18 x 30-day months, event-time cap
#
# 540 and 548 are both "18 months" under different month conventions (18x30 vs
# calendar); 730 exists so that sensitivity analyses at the 730-day persistence
# definition have data to use. No estimate is reported beyond 18 months.
MAX_DAYS_DEFAULT = 730


def _safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _fill_exclusion_flag(df: pd.DataFrame, col: str, default: int = 0) -> None:
    """Ensure exclusion flag *col* exists on *df* and carries no missing values.

    Addresses the code-review point on step1:330-331. The submitted code wrote
    ``df[col] = df.get(col, 0).fillna(0)``, which raises
    ``AttributeError: 'int' object has no attribute 'fillna'`` whenever the
    column is absent, because ``DataFrame.get`` returns the scalar default rather
    than a Series. The column is present in the study source, so the pipeline
    never hit it; on any input lacking the column, step1 died at the first line
    of ``load_and_prepare`` instead of applying the default.
    """
    if col in df.columns:
        df[col] = df[col].fillna(default)
    else:
        logging.info(
            "Column %s absent from input; defaulting to %s for the exclusion filter",
            col,
            default,
        )
        df[col] = default


def _pct_weight_change(df: pd.DataFrame) -> pd.Series:
    """Percent change from baseline weight, vectorised.

    Replaces a row-wise ``df.apply(_compute_pct_weight_change, axis=1)`` over the
    full long-format dataset (code-review performance point). Semantics are
    unchanged: NaN when either weight is missing or the baseline is zero.
    """
    baseline = pd.to_numeric(df.get("baseline_weight_final"), errors="coerce")
    current = pd.to_numeric(df.get("weight_in_pounds_final"), errors="coerce")
    if baseline is None or current is None:
        return pd.Series(np.nan, index=df.index)
    out = 100.0 * (current - baseline) / baseline
    return out.where(baseline.notna() & current.notna() & (baseline != 0), np.nan)


def _rows_before_first_lapse(df: pd.DataFrame) -> pd.Series:
    """Boolean mask keeping rows strictly before a patient's first lapse day.

    Replaces a row-wise ``df.apply(_ok, axis=1)`` over the full long-format
    dataset, repeated once per persistence definition (code-review performance
    point). Semantics are unchanged: a patient with no lapse (``first_zero_day``
    missing) keeps every row; a row with no ``days_from_baseline`` is dropped once
    the patient has a lapse day.
    """
    first_zero = pd.to_numeric(df["first_zero_day"], errors="coerce")
    day = pd.to_numeric(df["days_from_baseline"], errors="coerce")
    return first_zero.isna() | (day.notna() & (day < first_zero))


def _col_true(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].fillna(0).astype(bool)


def _glp1_evidence_therapy_flag(df: pd.DataFrame) -> pd.Series:
    is_med = _col_true(df, "is_glp1_event_med")
    is_clin = _col_true(df, "is_glp1_event_clinical")
    event_type = df.get("event_type")
    if event_type is not None:
        event_type = event_type.fillna("").astype(str).str.lower()
        is_event_type_med = event_type.eq("medication")
    else:
        is_event_type_med = pd.Series(False, index=df.index)
    return is_med | is_clin | is_event_type_med


def _collect_glp1_event_timeline(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["glp1_evidence_therapy"].fillna(False).astype(bool)
    cols = [c for c in ["patient_id", "event_date"] if c in df.columns]
    return df.loc[mask, cols].copy()


def _has_metformin_near_baseline(
    medication_history: object,
    baseline_glp1_date: object,
    window_days: int = 180,
) -> bool:
    """True when medication_history records metformin within *window_days* of baseline.

    Addresses the code-review point that this helper was biased toward True. The
    submitted version failed *open* in two places: an unparseable entry date
    returned True, and any JSON parse failure (or a missing baseline date) fell
    back to a bare substring search over the whole history with no date window at
    all — so a metformin exposure years away from baseline counted as "near
    baseline". Both paths now fail *closed*: evidence that cannot be placed in
    time does not count as evidence inside the window. Unparseable input is
    logged at DEBUG rather than swallowed, so the frequency is inspectable.

    Context for the change list: the flag this helper produces
    (``metformin_with_glp1_baseline``) is a candidate covariate that no reported
    model includes — concomitant metformin is adjusted for inside the
    ``weight_change_med`` covariate, which every reported model carries. The
    fail-open behaviour therefore could not propagate into a published estimate.
    """
    if pd.isna(medication_history):
        return False
    raw = str(medication_history)
    # Fast path: if 'metformin' is absent entirely, skip the expensive parsing.
    if "metformin" not in raw.lower():
        return False

    baseline = pd.to_datetime(baseline_glp1_date, errors="coerce")
    if pd.isna(baseline):
        # Without a baseline date there is no window to test against. Fail
        # closed: an undatable exposure is not evidence of exposure near baseline.
        logging.debug(
            "metformin present in medication_history but baseline_glp1_date is "
            "unparseable (%r); not counted as near-baseline",
            baseline_glp1_date,
        )
        return False

    try:
        entries = json.loads(raw)
    except (ValueError, TypeError) as exc:
        # Fail closed rather than degrading to an undated substring match.
        logging.debug(
            "medication_history is not parseable JSON (%s); metformin mention "
            "not counted as near-baseline",
            exc,
        )
        return False

    if not isinstance(entries, list):
        logging.debug(
            "medication_history parsed to %s, expected a list of entries; "
            "metformin mention not counted as near-baseline",
            type(entries).__name__,
        )
        return False

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("medication_name", "") or "")
        if "metformin" not in name.lower():
            continue
        entry_date = pd.to_datetime(entry.get("medication_date"), errors="coerce")
        if pd.isna(entry_date):
            # Fail closed: an entry we cannot date cannot be shown to sit in
            # the window. The submitted code returned True here.
            logging.debug(
                "metformin entry has an unparseable medication_date (%r); "
                "not counted as near-baseline",
                entry.get("medication_date"),
            )
            continue
        if abs((entry_date - baseline).days) <= window_days:
            return True
    return False


# ── Removed in Part II v2 (code-review item 2) ────────────────────────────────
# Three follow-up-rule variants that lived here are gone; persistence.py is now
# the only implementation (see its docstring for the rule and for why replacing
# them changed no published number):
#
#   _first_nonadherence_gap          helper reachable only from the dead variant
#   _compute_censor_map_from_step8f  live only via step0a; step0a now imports
#                                    persistence.censor_days
#   _apply_adherence_censoring       defined and never called from anywhere
#
def _fill_covariates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "baseline_a1c_category" in df.columns:
        df["baseline_a1c_category"] = df["baseline_a1c_category"].fillna("Unknown")
    if "baseline_bmi_final_category" in df.columns:
        df["baseline_bmi_final_category"] = df["baseline_bmi_final_category"].fillna("Unknown")
    if "age" in df.columns:
        df["age"] = df["age"].fillna(-1)
    if "gender" in df.columns:
        df["gender"] = df["gender"].fillna("Unknown")
    if "race" in df.columns:
        df["race"] = df["race"].fillna("Unknown")
    return df


def _derive_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "age" not in df.columns:
        return df

    def _age_group(age: float) -> str:
        if pd.isna(age) or age < 0:
            return "Unknown"
        if age < 20:
            return "<20"
        if age >= 80:
            return "80+"
        lo = int(age // 10) * 10
        hi = lo + 9
        return f"{lo}-{hi}"

    df["age_group"] = df["age"].apply(_age_group)

    def _grp_20_39_vs_40_plus(age: float) -> str:
        if pd.isna(age) or age < 20:
            return "Unknown"
        if 20 <= age <= 39:
            return "20-39"
        return "40+"

    def _grp_20_49_vs_50_plus(age: float) -> str:
        if pd.isna(age) or age < 20:
            return "Unknown"
        if 20 <= age <= 49:
            return "20-49"
        return "50+"

    df["age_group_20_39_vs_40_plus"] = df["age"].apply(_grp_20_39_vs_40_plus)
    df["age_group_20_49_vs_50_plus"] = df["age"].apply(_grp_20_49_vs_50_plus)
    return df


def _ensure_glp1_evidence_and_metformin(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["glp1_evidence_therapy"] = _glp1_evidence_therapy_flag(df)

    # Always recompute — the pre-computed column in the raw 8g dataset is unreliable
    # because it was generated before medication_history was fully populated and uses
    # a plain string search with no date window.
    #
    # Performance: medication_history and baseline_glp1_date are the same on every
    # row for a given patient, so we compute once per patient then merge back.
    pat_cols = ["patient_id", "medication_history", "baseline_glp1_date"]
    available = [c for c in pat_cols if c in df.columns]
    if "patient_id" in available and len(available) > 1:
        per_patient = (
            df[available]
            .drop_duplicates(subset=["patient_id"])
            .copy()
        )
        per_patient["metformin_with_glp1_baseline"] = per_patient.apply(
            lambda r: _has_metformin_near_baseline(
                r.get("medication_history"), r.get("baseline_glp1_date")
            ),
            axis=1,
        )
        met_map = per_patient.set_index("patient_id")["metformin_with_glp1_baseline"]
        df["metformin_with_glp1_baseline"] = df["patient_id"].map(met_map).fillna(False).astype(bool)
    else:
        df["metformin_with_glp1_baseline"] = df.apply(
            lambda r: _has_metformin_near_baseline(
                r.get("medication_history"), r.get("baseline_glp1_date")
            ),
            axis=1,
        )

    rx_timeline = _collect_glp1_event_timeline(df)
    return df, rx_timeline


def load_and_prepare(
    input_csv: Path,
    max_days: int = -1,
    adherence_gap_days: int = 90,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("Reading input CSV %s", input_csv)
    df = pd.read_csv(input_csv)

    _fill_exclusion_flag(df, "pregnant_during_glp1")
    _fill_exclusion_flag(df, "bariatric_surgery_flag")

    logging.info("Initial rows: %d", len(df))
    df = df[df["pregnant_during_glp1"] == 0]
    logging.info("After pregnancy exclusion: %d", len(df))

    df = df[df["bariatric_surgery_flag"] == 0]
    logging.info("After bariatric exclusion: %d", len(df))

    # Baseline GLP-1 cohort filter. Patterns come from GLP1_BRAND_NAMES /
    # GLP1_INGREDIENT_NAMES at the top of this module — the single source of
    # truth for which agents define the cohort.
    keep_mask = None
    if "baseline_glp1_brand_final" in df.columns:
        brands = df["baseline_glp1_brand_final"].astype(str).str.lower().str.strip()
        bm = brands.str.contains(GLP1_BRAND_PATTERN, regex=True, na=False)
        keep_mask = bm if keep_mask is None else (keep_mask | bm)
    if "baseline_glp1_ingredient_final" in df.columns:
        ings = df["baseline_glp1_ingredient_final"].astype(str).str.lower().str.strip()
        im = ings.str.contains(GLP1_INGREDIENT_PATTERN, regex=True, na=False)
        keep_mask = im if keep_mask is None else (keep_mask | im)
    if keep_mask is None and "baseline_glp1" in df.columns:
        base = df["baseline_glp1"].astype(str).str.lower().str.strip()
        keep_mask = base.str.contains(GLP1_ANY_PATTERN, regex=True, na=False)
    if keep_mask is None:
        df = df.iloc[0:0]
    else:
        df = df[keep_mask]
    logging.info("After injectable GLP-1 baseline filter: %d", len(df))

    if "baseline_weight_final" in df.columns:
        df = df[df["baseline_weight_final"].notna()]
        logging.info("After baseline_weight_final not null: %d", len(df))

    if "baseline_a1c_final" in df.columns:
        df = df[df["baseline_a1c_final"].notna()]
        logging.info("After baseline_a1c_final not null: %d", len(df))

    for col in ["baseline_glp1_date", "date", "medication_date"]:
        if col in df.columns:
            df[col] = _safe_to_datetime(df[col])

    if "date" in df.columns and df["date"].notna().any():
        df["event_date"] = df["date"]
    else:
        df["event_date"] = df.get("medication_date")

    if "pct_weight_change" not in df.columns:
        df["pct_weight_change"] = _pct_weight_change(df)
    df = df[df["pct_weight_change"].notna()]
    logging.info("After non-missing pct_weight_change: %d", len(df))

    df = _fill_covariates(df)
    df = _derive_age_groups(df)

    if "baseline_a1c_category" in df.columns:
        cat_type = pd.CategoricalDtype(
            [
                "Normal Glycemia",
                "Prediabetes",
                "Type 2 Diabetes",
                "Poorly Controlled Diabetes",
                "Unknown",
            ],
            ordered=True,
        )
        df["baseline_a1c_category"] = df["baseline_a1c_category"].astype(cat_type)
        df = df[df["baseline_a1c_category"] != "Unknown"]
        logging.info("After baseline_a1c_category ordering & Unknown drop: %d", len(df))

    df, rx_timeline = _ensure_glp1_evidence_and_metformin(df)

    if "baseline_glp1_date" in df.columns and "event_date" in df.columns:
        df["days_from_baseline"] = (df["event_date"] - df["baseline_glp1_date"]).dt.days
        # For the main gap90 analysis we keep only post-baseline
        # observations here; Step 7 has its own loader that retains
        # 6 months of pre-baseline data.
        if max_days is not None and max_days > 0:
            df = df[(df["days_from_baseline"] >= 0) & (df["days_from_baseline"] <= max_days)]
        else:
            df = df[df["days_from_baseline"] >= 0]
        logging.info("After days_from_baseline windowing: %d", len(df))

    return df, rx_timeline


def _ensure_baseline_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Anchor every patient's trajectory at day 0, marking rows created to do so.

    The Methods require that "all individuals contributed a baseline observation
    at day 0 corresponding to treatment initiation". A patient whose baseline
    weight was measured inside the -60/+14-day baseline window but not on day 0
    itself has a valid baseline value and no day-0 row. This function writes that
    row, carrying the patient's real recorded ``baseline_weight_final``; the
    outcome on it, change from baseline, is 0.0 by definition, being the baseline
    compared with itself.

    Addresses code-review item 3. Every row created here now carries
    ``baseline_carried_to_day0 = 1``; observed rows carry 0. The marker is listed in
    ``keep_cols`` and reaches the written CSVs, so downstream work can separate
    anchored from measured rows and a leave-them-out sensitivity analysis is
    possible without re-deriving which rows these were. At the primary 120-day
    persistence definition the flag marks 2,967 of 16,061 patients (18.5%).

    The created row also sets ``glp1_event_for_adherance = 1``, recording
    treatment initiation as the first tick of the persistence clock — a
    documented event that every patient in the cohort has by construction.
    """
    if "patient_id" not in df.columns:
        return df

    # Observed rows are marked before any row is created, so the flag is defined
    # for every row in the frame rather than only for the created ones.
    if BASELINE_CARRIED_COL not in df.columns:
        df = df.copy()
        df[BASELINE_CARRIED_COL] = 0

    have_zero = df["days_from_baseline"].eq(0) if "days_from_baseline" in df.columns else pd.Series(False, index=df.index)
    missing_ids = set(df.loc[~have_zero, "patient_id"].unique()) - set(df.loc[have_zero, "patient_id"].unique())
    if not missing_ids:
        logging.info(
            "Day-0 anchoring: every patient already has an observed day-0 row; "
            "0 rows created, %s summing to 0",
            BASELINE_CARRIED_COL,
        )
        return df

    cov_cols = [
        "patient_id",
        "baseline_a1c_category",
        "baseline_bmi_final_category",
        "age_group",
        "age_group_20_39_vs_40_plus",
        "age_group_20_49_vs_50_plus",
        "gender",
        "race",
        "weight_change_med",
        # include legacy and new baseline GLP-1 identifiers
        "baseline_glp1",
        "baseline_glp1_brand_final",
        "baseline_glp1_ingredient_final",
        "baseline_glp1_date",
        "baseline_weight_final",
        "metformin_with_glp1_baseline",
        "glp1_evidence_therapy",
        # Ensure GLP-1 user group is carried onto the day-0 carried baseline row
        "glp1_user_group",
    ]
    cov_cols = [c for c in cov_cols if c in df.columns]
    base = df.drop_duplicates("patient_id")[cov_cols].copy()
    base = base[base["patient_id"].isin(missing_ids)].copy()
    if base.empty:
        return df

    carried = base.copy()
    carried["days_from_baseline"] = 0
    if "baseline_glp1_date" in carried.columns:
        carried["event_date"] = carried["baseline_glp1_date"]
    if "baseline_weight_final" in carried.columns:
        carried["weight_in_pounds_final"] = carried["baseline_weight_final"]
    carried["pct_weight_change"] = 0.0
    carried["glp1_days_from_baseline"] = 0
    # Treatment initiation is the first tick of the persistence clock.
    carried["glp1_event_for_adherance"] = 1
    # Ensure evidence flag true at baseline
    if "glp1_evidence_therapy" in carried.columns:
        carried["glp1_evidence_therapy"] = True
    # Mark every created row (code-review item 3).
    carried[BASELINE_CARRIED_COL] = 1

    common_cols = list(set(df.columns) | set(carried.columns))
    carried = carried.reindex(columns=common_cols)
    df = df.reindex(columns=common_cols)
    out = pd.concat([df, carried], axis=0, ignore_index=True)
    out = out.sort_values(["patient_id", "days_from_baseline"]).reset_index(drop=True)
    # Reindexing can introduce NaN in the marker for pre-existing rows.
    out[BASELINE_CARRIED_COL] = out[BASELINE_CARRIED_COL].fillna(0).astype(int)
    logging.info(
        "Day-0 anchoring: created %d day-0 rows for %d patients lacking an "
        "observed day-0 measurement (%s == 1)",
        len(carried),
        len(missing_ids),
        BASELINE_CARRIED_COL,
    )
    return out


def _compute_days_since_last_glp1(df: pd.DataFrame) -> pd.DataFrame:
    """For each row, compute days since the previous GLP-1 mention/event.
    Previous events are rows with glp1_event_for_adherance in {1,2} and use glp1_days_from_baseline.
    If none exist before the row day, result is NaN. Carried baseline rows will yield 0.
    Requires columns: patient_id, days_from_baseline, glp1_event_for_adherance, glp1_days_from_baseline.
    """
    need = {"patient_id", "days_from_baseline", "glp1_event_for_adherance", "glp1_days_from_baseline"}
    if not need.issubset(set(df.columns)):
        df["days_since_last_glp1"] = np.nan
        return df

    df_local = df[["patient_id", "days_from_baseline", "glp1_event_for_adherance", "glp1_days_from_baseline"]].copy()
    df_local["days_from_baseline"] = pd.to_numeric(df_local["days_from_baseline"], errors="coerce")
    df_local["glp1_days_from_baseline"] = pd.to_numeric(df_local["glp1_days_from_baseline"], errors="coerce")

    out = np.full(len(df_local), np.nan)

    for pid, grp in df_local.groupby("patient_id"):
        row_labels = grp.index.to_numpy()
        row_pos = df_local.index.get_indexer(row_labels)
        row_days = grp["days_from_baseline"].to_numpy()
        # Mention days: 1 or 2
        mvals = grp["glp1_event_for_adherance"].astype(float).to_numpy()
        mdays = grp["glp1_days_from_baseline"].to_numpy()
        mention_days = np.sort(mdays[(mvals == 1) | (mvals == 2)])
        if mention_days.size == 0:
            continue
        # For each row day, find index of previous mention day
        idx = np.searchsorted(mention_days, row_days, side="right") - 1
        prev = np.full(row_days.shape, np.nan)
        valid = idx >= 0
        prev[valid] = mention_days[idx[valid]]
        diff = row_days - prev
        out[row_pos] = diff

    df["days_since_last_glp1"] = out
    return df


def _compute_days_since_last_glp1_evidence(df: pd.DataFrame) -> pd.DataFrame:
    """Compute days since previous GLP-1 evidence therapy event for each row.
    Uses rows flagged glp1_evidence_therapy==True and their days_from_baseline as the evidence timeline.
    If no prior evidence exists for a row, result is NaN; carried baseline rows yield 0 when baseline is in evidence.
    Requires columns: patient_id, days_from_baseline, glp1_evidence_therapy.
    """
    need = {"patient_id", "days_from_baseline", "glp1_evidence_therapy"}
    if not need.issubset(set(df.columns)):
        df["days_since_last_glp1_evidence"] = np.nan
        return df

    df_local = df[["patient_id", "days_from_baseline", "glp1_evidence_therapy"]].copy()
    df_local["days_from_baseline"] = pd.to_numeric(df_local["days_from_baseline"], errors="coerce")

    out = np.full(len(df_local), np.nan)

    # Build evidence day arrays per patient once
    evid_map = {}
    for pid, grp in df_local.groupby("patient_id"):
        evid_days = np.sort(grp.loc[grp["glp1_evidence_therapy"].astype(bool), "days_from_baseline"].dropna().to_numpy())
        evid_map[pid] = evid_days

    for pid, grp in df_local.groupby("patient_id"):
        row_labels = grp.index.to_numpy()
        row_pos = df_local.index.get_indexer(row_labels)
        row_days = grp["days_from_baseline"].to_numpy()
        evid_days = evid_map.get(pid, np.array([], dtype=float))
        if evid_days.size == 0:
            continue
        idx = np.searchsorted(evid_days, row_days, side="right") - 1
        prev = np.full(row_days.shape, np.nan)
        valid = idx >= 0
        prev[valid] = evid_days[idx[valid]]
        diff = row_days - prev
        out[row_pos] = diff

    df["days_since_last_glp1_evidence"] = out
    return df


def _compute_glp1_distance_columns(df: pd.DataFrame) -> pd.DataFrame:
    need_mention = {"patient_id", "days_from_baseline", "glp1_event_for_adherance", "glp1_days_from_baseline"}
    need_evid = {"patient_id", "days_from_baseline", "glp1_evidence_therapy"}
    # Initialize columns
    df["days_to_prev_glp1"] = np.nan
    df["days_to_next_glp1"] = np.nan
    df["days_to_nearest_glp1"] = np.nan
    df["days_to_prev_glp1_evidence"] = np.nan
    df["days_to_next_glp1_evidence"] = np.nan
    df["days_to_nearest_glp1_evidence"] = np.nan

    # Mentions
    if need_mention.issubset(set(df.columns)):
        df_local = df[["patient_id", "days_from_baseline", "glp1_event_for_adherance", "glp1_days_from_baseline"]].copy()
        df_local["days_from_baseline"] = pd.to_numeric(df_local["days_from_baseline"], errors="coerce")
        df_local["glp1_days_from_baseline"] = pd.to_numeric(df_local["glp1_days_from_baseline"], errors="coerce")
        prev_out = np.full(len(df_local), np.nan)
        next_out = np.full(len(df_local), np.nan)
        near_out = np.full(len(df_local), np.nan)
        for pid, grp in df_local.groupby("patient_id"):
            row_labels = grp.index.to_numpy()
            row_pos = df_local.index.get_indexer(row_labels)
            row_days = grp["days_from_baseline"].to_numpy()
            mvals = grp["glp1_event_for_adherance"].astype(float).to_numpy()
            mdays = grp["glp1_days_from_baseline"].to_numpy()
            mention_days = np.sort(mdays[(mvals == 1) | (mvals == 2)])
            if mention_days.size == 0:
                continue
            # prev
            idx_prev = np.searchsorted(mention_days, row_days, side="right") - 1
            prev = np.full(row_days.shape, np.nan)
            valid_prev = idx_prev >= 0
            prev[valid_prev] = row_days[valid_prev] - mention_days[idx_prev[valid_prev]]
            # next
            idx_next = np.searchsorted(mention_days, row_days, side="left")
            next_ = np.full(row_days.shape, np.nan)
            valid_next = idx_next < mention_days.size
            next_[valid_next] = mention_days[idx_next[valid_next]] - row_days[valid_next]
            # nearest
            nearest = np.nanmin(np.vstack([
                np.where(np.isnan(prev), np.inf, prev),
                np.where(np.isnan(next_), np.inf, next_)
            ]), axis=0)
            nearest[np.isinf(nearest)] = np.nan
            prev_out[row_pos] = prev
            next_out[row_pos] = next_
            near_out[row_pos] = nearest
        df["days_to_prev_glp1"] = prev_out
        df["days_to_next_glp1"] = next_out
        df["days_to_nearest_glp1"] = near_out

    # Evidence
    if need_evid.issubset(set(df.columns)):
        df_local = df[["patient_id", "days_from_baseline", "glp1_evidence_therapy"]].copy()
        df_local["days_from_baseline"] = pd.to_numeric(df_local["days_from_baseline"], errors="coerce")
        prev_out = np.full(len(df_local), np.nan)
        next_out = np.full(len(df_local), np.nan)
        near_out = np.full(len(df_local), np.nan)
        evid_map = {}
        for pid, grp in df_local.groupby("patient_id"):
            evid_days = np.sort(grp.loc[grp["glp1_evidence_therapy"].astype(bool), "days_from_baseline"].dropna().to_numpy())
            evid_map[pid] = evid_days
        for pid, grp in df_local.groupby("patient_id"):
            row_labels = grp.index.to_numpy()
            row_pos = df_local.index.get_indexer(row_labels)
            row_days = grp["days_from_baseline"].to_numpy()
            evid_days = evid_map.get(pid, np.array([], dtype=float))
            if evid_days.size == 0:
                continue
            idx_prev = np.searchsorted(evid_days, row_days, side="right") - 1
            prev = np.full(row_days.shape, np.nan)
            valid_prev = idx_prev >= 0
            prev[valid_prev] = row_days[valid_prev] - evid_days[idx_prev[valid_prev]]
            idx_next = np.searchsorted(evid_days, row_days, side="left")
            next_ = np.full(row_days.shape, np.nan)
            valid_next = idx_next < evid_days.size
            next_[valid_next] = evid_days[idx_next[valid_next]] - row_days[valid_next]
            nearest = np.nanmin(np.vstack([
                np.where(np.isnan(prev), np.inf, prev),
                np.where(np.isnan(next_), np.inf, next_)
            ]), axis=0)
            nearest[np.isinf(nearest)] = np.nan
            prev_out[row_pos] = prev
            next_out[row_pos] = next_
            near_out[row_pos] = nearest
        df["days_to_prev_glp1_evidence"] = prev_out
        df["days_to_next_glp1_evidence"] = next_out
        df["days_to_nearest_glp1_evidence"] = near_out

    return df


def run_for_gaps(
    input_csv: Path,
    outdir: Path,
    max_days: int,
    adherence_gaps: Iterable[int],
) -> None:
    base_df, rx_timeline = load_and_prepare(input_csv, max_days=max_days)

    outdir.mkdir(parents=True, exist_ok=True)

    # Anchor every trajectory at day 0 before computing flags. This is a
    # structural step the Methods require, not an optional enrichment: swallowing
    # a failure here would silently change which rows the models see, so it is
    # allowed to raise (code-review item 8, silent-handler sweep).
    base_df = _ensure_baseline_rows(base_df)

    # Compute helper metrics
    try:
        base_df = _compute_days_since_last_glp1(base_df)
    except Exception as e:
        logging.warning("Could not compute days_since_last_glp1: %s", e)
        base_df["days_since_last_glp1"] = np.nan
    try:
        base_df = _compute_days_since_last_glp1_evidence(base_df)
    except Exception as e:
        logging.warning("Could not compute days_since_last_glp1_evidence: %s", e)
        base_df["days_since_last_glp1_evidence"] = np.nan
    try:
        base_df = _compute_glp1_distance_columns(base_df)
    except Exception as e:
        logging.warning("Could not compute GLP-1 distance columns: %s", e)

    # Compute multi-gap adherence flags
    gaps = list(adherence_gaps) if isinstance(adherence_gaps, (list, tuple)) else list(adherence_gaps)
    if not gaps:
        gaps = GAPS_DEFAULT
    base_df = adherence_flags(base_df, gaps)

    # For each gap, censor at first row where adherence_{gap}==0 (drop subsequent rows), keep rows with days>=0
    for gap in gaps:
        logging.info("Processing adherence gap via flags: %s", gap)
        df_gap = base_df.copy()
        if "days_from_baseline" in df_gap.columns:
            df_gap = df_gap[df_gap["days_from_baseline"] >= 0]
        first_zero = (
            df_gap.loc[df_gap[f"adherence_{gap}"] == 0, ["patient_id", "days_from_baseline"]]
            .groupby("patient_id", as_index=False)["days_from_baseline"].min()
            .rename(columns={"days_from_baseline": "first_zero_day"})
        )
        df_gap = df_gap.merge(first_zero, on="patient_id", how="left")
        keep = _rows_before_first_lapse(df_gap)
        df_gap = df_gap.loc[keep].copy()
        df_gap.drop(columns=["first_zero_day"], inplace=True)

        if "pct_weight_change" not in df_gap.columns:
            df_gap["pct_weight_change"] = _pct_weight_change(df_gap)

        keep_cols = [
            "patient_id",
            "days_from_baseline",
            # Marker for day-0 rows created by _ensure_baseline_rows
            # (code-review item 3). Carried through to the written CSV so that
            # anchored and measured rows can be told apart downstream.
            BASELINE_CARRIED_COL,
            "glp1_event_for_adherance",
            "glp1_days_from_baseline",
            "days_since_last_glp1",
            "days_since_last_glp1_evidence",
            "days_to_prev_glp1",
            "days_to_next_glp1",
            "days_to_nearest_glp1",
            "days_to_prev_glp1_evidence",
            "days_to_next_glp1_evidence",
            "days_to_nearest_glp1_evidence",
            f"adherence_{gap}",
            "pct_weight_change",
            "baseline_a1c_category",
            "baseline_bmi_final_category",
            "age_group",
            "age_group_20_39_vs_40_plus",
            "age_group_20_49_vs_50_plus",
            "gender",
            "race",
            "weight_change_med",
            # include legacy and new baseline identifiers
            "baseline_glp1",
            "baseline_glp1_brand_final",
            "baseline_glp1_ingredient_final",
            "baseline_glp1_date",
            "baseline_weight_final",
            "weight_in_pounds_final",
            "metformin_with_glp1_baseline",
            "glp1_evidence_therapy",
            # Newly kept numeric baseline variables for descriptive stats
            "age",
            "baseline_height_final",
            "height_in_inches_final",
            "baseline_bmi_final",
            "BMI_final",
            # NEW: carry unstructured vs structured flags from step8g
            "a1c_has_unstructured",
            "a1c_has_structured",
            "weight_has_unstructured",
            "weight_has_structured",
            "glp1_has_unstructured",
            "glp1_has_structured",
        ]
        keep_cols = [c for c in keep_cols if c in df_gap.columns]
        df_out = df_gap[keep_cols].copy()

        out_path = outdir / f"analysis_ready_gap{gap}.csv"
        if BASELINE_CARRIED_COL in df_out.columns:
            n_carried_rows = int(df_out[BASELINE_CARRIED_COL].sum())
            n_carried_pat = int(
                df_out.loc[df_out[BASELINE_CARRIED_COL] == 1, "patient_id"].nunique()
            )
            logging.info(
                "gap %s: %s marks %d rows across %d unique patients",
                gap,
                BASELINE_CARRIED_COL,
                n_carried_rows,
                n_carried_pat,
            )
        else:
            # The marker must reach every output file; its absence means
            # _ensure_baseline_rows or keep_cols was changed inconsistently.
            raise RuntimeError(
                f"{BASELINE_CARRIED_COL} is missing from the gap {gap} output. The "
                "day-0 anchor marker is required in every analysis-ready file."
            )
        logging.info("Writing %s with %d rows", out_path, len(df_out))
        df_out.to_csv(out_path, index=False)


def parse_adherence_gaps(values: List[str]) -> List[int]:
    gaps: List[int] = []
    for v in values:
        try:
            gaps.append(int(v))
        except ValueError:
            logging.warning("Could not parse adherence gap '%s'", v)
    return gaps


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Step 1: Prepare analysis-ready GLP-1 dataset",
    )
    parser.add_argument(
        "--input-csv",
        default="root_data/step8g_with_unstructured_flags.csv",
        help="Input CSV path (default: root_data/step8g_with_unstructured_flags.csv)",
    )
    parser.add_argument(
        "--outdir",
        default="output/step1_prepare_analysis_dataset",
        help="Output directory for analysis-ready datasets",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=MAX_DAYS_DEFAULT,
        help=(
            f"Maximum follow-up days; <=0 means no upper bound (default "
            f"{MAX_DAYS_DEFAULT}). This is an outer bound on emitted data, not "
            "the analysis window; see MAX_DAYS_DEFAULT for how it relates to the "
            "540/548-day caps downstream."
        ),
    )
    parser.add_argument(
        "--adherence-gaps",
        nargs="+",
        default=["90"],
        help="One or more adherence gap lengths in days (e.g., 90 120)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
    )

    input_csv = Path(args.input_csv)
    outdir = Path(args.outdir)

    adherence_gaps = parse_adherence_gaps(args.adherence_gaps)

    run_for_gaps(
        input_csv=input_csv,
        outdir=outdir,
        max_days=args.max_days,
        adherence_gaps=adherence_gaps,
    )


if __name__ == "__main__":
    main()
