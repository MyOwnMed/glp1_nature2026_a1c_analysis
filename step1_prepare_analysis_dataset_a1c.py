#!/usr/bin/env python
"""Step 1 (HbA1c): build the gap-censored, analysis-ready HbA1c dataset.

Baseline window and day-0 anchoring — see README.md for the full statement in
Methods wording. Note the contrast with the weight analysis: an HbA1c value
appears in this dataset only on the date it was measured, so
``_ensure_baseline_rows_a1c`` creates no rows on this data and no HbA1c value is
anchored to day 0. Baseline HbA1c is the measurement closest to initiation within
the -60/+14-day window; for the 38.0% of the cohort measured on the index date
that row is an observed day-0 row, and the remaining 62.0% contribute no day-0
observation rather than an anchored one. The ``baseline_carried_to_day0`` marker is therefore
present in the HbA1c outputs and sums to zero — the column exists so that the
absence is auditable rather than assumed.
"""

import argparse
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

from step1_prepare_analysis_dataset import (
    GLP1_ANY_PATTERN,
    GLP1_BRAND_PATTERN,
    GLP1_INGREDIENT_PATTERN,
    BASELINE_CARRIED_COL,
    MAX_DAYS_DEFAULT,
    _derive_age_groups,
    _ensure_glp1_evidence_and_metformin,
    _fill_covariates,
    _fill_exclusion_flag,
    _rows_before_first_lapse,
    _safe_to_datetime,
)

GAPS_DEFAULT = [30, 60, 90, 120, 150, 180, 365, 730]


def _ensure_a1c_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "baseline_a1c_final" not in df.columns:
        raise ValueError("baseline_a1c_final is required for A1c analyses")

    has_a1c_value = "a1c_value" in df.columns
    has_abs_change = "abs_a1c_change" in df.columns

    if not has_abs_change and not has_a1c_value:
        raise ValueError(
            "Neither abs_a1c_change nor a1c_value present; cannot construct A1c outcomes",
        )

    if not has_abs_change and has_a1c_value:
        df["abs_a1c_change"] = df["a1c_value"] - df["baseline_a1c_final"]

    if has_abs_change and not has_a1c_value:
        df["a1c_value"] = df["baseline_a1c_final"] + df["abs_a1c_change"]

    return df


def load_and_prepare_a1c(
    input_csv: Path,
    max_days: int = -1,
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

    # Baseline GLP-1 cohort filter. Patterns are imported from
    # step1_prepare_analysis_dataset, the single source of truth for which agents
    # define the cohort, so the weight and HbA1c cohorts cannot drift apart.
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

    df = _ensure_a1c_variables(df)

    if "baseline_glp1_date" in df.columns and "event_date" in df.columns:
        df["days_from_baseline"] = (df["event_date"] - df["baseline_glp1_date"]).dt.days
        if max_days is not None and max_days > 0:
            df = df[(df["days_from_baseline"] >= 0) & (df["days_from_baseline"] <= max_days)]
        else:
            df = df[df["days_from_baseline"] >= 0]
        logging.info("After days_from_baseline windowing: %d", len(df))

    # Drop patients who never have a post-baseline A1c measurement
    # (i.e., no row with days_from_baseline > 0 and non-missing a1c_value).
    if {"patient_id", "a1c_value", "days_from_baseline"}.issubset(df.columns):
        prev_n = df["patient_id"].nunique()
        has_post_a1c = (df["days_from_baseline"] > 0) & df["a1c_value"].notna()
        keep_ids = set(df.loc[has_post_a1c, "patient_id"].unique())
        df = df[df["patient_id"].isin(keep_ids)].copy()
        curr_n = df["patient_id"].nunique()
        logging.info(
            "After requiring at least one post-baseline A1c value: %d patients (dropped %d)",
            curr_n,
            prev_n - curr_n,
        )

    return df, rx_timeline


# ── Removed in Part II v2 (code-review item 2) ────────────────────────────────
# Two more copies of the follow-up rule lived here; persistence.py is now the only
# implementation:
#
#   _compute_adherence_flag   singular, per-gap variant using a nearest-mention
#                             (|d - m| <= g, forward as well as backward) test —
#                             defined and never called from anywhere. Not among
#                             the three variants the reviewer identified; found
#                             while consolidating. Because it was never called it
#                             contributed to nothing.
#   _compute_adherence_flags  a line-for-line duplicate of the weight version,
#                             which is now persistence.adherence_flags.
#


def _ensure_baseline_rows_a1c(df: pd.DataFrame) -> pd.DataFrame:
    """Anchor every patient's trajectory at day 0, marking rows created to do so.

    Structurally the twin of ``_ensure_baseline_rows`` in the weight pipeline, but
    on this data it creates nothing. An HbA1c value appears in this dataset only
    on the date it was measured — verified across the full source — so a patient
    either has a genuine day-0 measurement (38.0% of the cohort, measured on the
    index date) or contributes no day-0 observation at all. There is consequently
    no anchored HbA1c value and no zero-variance point at t = 0 in the HbA1c
    models.

    Addresses code-review item 3. ``baseline_carried_to_day0`` is written to the HbA1c outputs
    and sums to zero. The column is emitted even though it is everywhere 0 so that
    a reader can confirm the absence from the data rather than take it on trust.
    """
    if "patient_id" not in df.columns:
        return df

    if BASELINE_CARRIED_COL not in df.columns:
        df = df.copy()
        df[BASELINE_CARRIED_COL] = 0

    have_zero = df["days_from_baseline"].eq(0) if "days_from_baseline" in df.columns else pd.Series(False, index=df.index)
    missing_ids = set(df.loc[~have_zero, "patient_id"].unique()) - set(df.loc[have_zero, "patient_id"].unique())
    if not missing_ids:
        logging.info(
            "Day-0 anchoring (A1c): no patient needs a created day-0 row; "
            "%s sums to 0",
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
        # include legacy and new baseline GLP-1 identifiers
        "baseline_glp1",
        "baseline_glp1_brand_final",
        "baseline_glp1_ingredient_final",
        "baseline_glp1_date",
        "baseline_weight_final",
        "baseline_a1c_final",
        "metformin_with_glp1_baseline",
        "glp1_evidence_therapy",
        # Ensure GLP-1 user group is carried to baseline row
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
    if "baseline_a1c_final" in carried.columns:
        carried["a1c_value"] = carried["baseline_a1c_final"]
    carried["abs_a1c_change"] = 0.0
    carried["glp1_days_from_baseline"] = 0
    # Treatment initiation is the first tick of the persistence clock.
    carried["glp1_event_for_adherance"] = 1
    if "glp1_evidence_therapy" in carried.columns:
        carried["glp1_evidence_therapy"] = True
    # Mark every created row (code-review item 3).
    carried[BASELINE_CARRIED_COL] = 1

    common_cols = list(set(df.columns) | set(carried.columns))
    carried = carried.reindex(columns=common_cols)
    df = df.reindex(columns=common_cols)
    out = pd.concat([df, carried], axis=0, ignore_index=True)
    out = out.sort_values(["patient_id", "days_from_baseline"]).reset_index(drop=True)
    out[BASELINE_CARRIED_COL] = out[BASELINE_CARRIED_COL].fillna(0).astype(int)
    logging.info(
        "Day-0 anchoring (A1c): created %d day-0 rows for %d patients lacking an "
        "observed day-0 measurement (%s == 1)",
        len(carried),
        len(missing_ids),
        BASELINE_CARRIED_COL,
    )
    return out


def _compute_days_since_last_glp1(df: pd.DataFrame) -> pd.DataFrame:
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
        mvals = grp["glp1_event_for_adherance"].astype(float).to_numpy()
        mdays = grp["glp1_days_from_baseline"].to_numpy()
        mention_days = np.sort(mdays[(mvals == 1) | (mvals == 2)])
        if mention_days.size == 0:
            continue
        idx = np.searchsorted(mention_days, row_days, side="right") - 1
        prev = np.full(row_days.shape, np.nan)
        valid = idx >= 0
        prev[valid] = mention_days[idx[valid]]
        diff = row_days - prev
        out[row_pos] = diff
    df["days_since_last_glp1"] = out
    return df


def _compute_days_since_last_glp1_evidence(df: pd.DataFrame) -> pd.DataFrame:
    need = {"patient_id", "days_from_baseline", "glp1_evidence_therapy"}
    if not need.issubset(set(df.columns)):
        df["days_since_last_glp1_evidence"] = np.nan
        return df
    df_local = df[["patient_id", "days_from_baseline", "glp1_evidence_therapy"]].copy()
    df_local["days_from_baseline"] = pd.to_numeric(df_local["days_from_baseline"], errors="coerce")
    out = np.full(len(df_local), np.nan)
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
    df["days_to_prev_glp1"] = np.nan
    df["days_to_next_glp1"] = np.nan
    df["days_to_nearest_glp1"] = np.nan
    df["days_to_prev_glp1_evidence"] = np.nan
    df["days_to_next_glp1_evidence"] = np.nan
    df["days_to_nearest_glp1_evidence"] = np.nan
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
            idx_prev = np.searchsorted(mention_days, row_days, side="right") - 1
            prev = np.full(row_days.shape, np.nan)
            valid_prev = idx_prev >= 0
            prev[valid_prev] = row_days[valid_prev] - mention_days[idx_prev[valid_prev]]
            idx_next = np.searchsorted(mention_days, row_days, side="left")
            next_ = np.full(row_days.shape, np.nan)
            valid_next = idx_next < mention_days.size
            next_[valid_next] = mention_days[idx_next[valid_next]] - row_days[valid_next]
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


def run_for_gaps_a1c(
    input_csv: Path,
    outdir: Path,
    max_days: int,
    adherence_gaps: Iterable[int],
) -> None:
    base_df, rx_timeline = load_and_prepare_a1c(input_csv, max_days=max_days)
    outdir.mkdir(parents=True, exist_ok=True)

    # Anchor every trajectory at day 0 before computing flags. Allowed to raise:
    # swallowing a failure here would silently change which rows the models see
    # (code-review item 8, silent-handler sweep).
    base_df = _ensure_baseline_rows_a1c(base_df)

    # Compute helper
    try:
        base_df = _compute_days_since_last_glp1(base_df)
    except Exception as e:
        logging.warning("Could not compute days_since_last_glp1 (A1C): %s", e)
        base_df["days_since_last_glp1"] = np.nan
    try:
        base_df = _compute_days_since_last_glp1_evidence(base_df)
    except Exception as e:
        logging.warning("Could not compute days_since_last_glp1_evidence (A1C): %s", e)
        base_df["days_since_last_glp1_evidence"] = np.nan
    try:
        base_df = _compute_glp1_distance_columns(base_df)
    except Exception as e:
        logging.warning("Could not compute GLP-1 distance columns (A1C): %s", e)

    # Compute flags
    gaps = list(adherence_gaps) if isinstance(adherence_gaps, (list, tuple)) else list(adherence_gaps)
    if not gaps:
        gaps = GAPS_DEFAULT
    base_df = adherence_flags(base_df, gaps)

    for gap in gaps:
        logging.info("Processing A1c adherence gap via flags: %s", gap)
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

        keep_cols = [
            "patient_id",
            "days_from_baseline",
            # Marker for day-0 rows created by _ensure_baseline_rows_a1c
            # (code-review item 3). Everywhere 0 on this data; emitted so the
            # absence of anchored HbA1c values is auditable.
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
            "baseline_a1c_category",
            "baseline_bmi_final_category",
            "age_group",
            "age_group_20_39_vs_40_plus",
            "age_group_20_49_vs_50_plus",
            "gender",
            "race",
            # include legacy and new baseline identifiers
            "baseline_glp1",
            "baseline_glp1_brand_final",
            "baseline_glp1_ingredient_final",
            "baseline_glp1_date",
            "baseline_weight_final",
            "weight_in_pounds_final",
            "baseline_a1c_final",
            "a1c_value",
            "abs_a1c_change",
            "metformin_with_glp1_baseline",
            "glp1_evidence_therapy",
            "weight_change_med",
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

        out_path = outdir / f"analysis_ready_a1c_gap{gap}.csv"
        if BASELINE_CARRIED_COL not in df_out.columns:
            raise RuntimeError(
                f"{BASELINE_CARRIED_COL} is missing from the A1c gap {gap} output. The "
                "day-0 anchor marker is required in every analysis-ready file."
            )
        logging.info(
            "A1c gap %s: %s marks %d rows across %d unique patients",
            gap,
            BASELINE_CARRIED_COL,
            int(df_out[BASELINE_CARRIED_COL].sum()),
            int(df_out.loc[df_out[BASELINE_CARRIED_COL] == 1, "patient_id"].nunique()),
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
        description="Step 1 (A1c): Prepare analysis-ready GLP-1 A1c dataset",
    )
    parser.add_argument(
        "--input-csv",
        default="root_data/step8g_with_unstructured_flags.csv",
        help="Input CSV path (default: root_data/step8g_with_unstructured_flags.csv)",
    )
    parser.add_argument(
        "--outdir",
        default="output/step1_prepare_analysis_dataset_a1c",
        help="Output directory for A1c analysis-ready datasets",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=MAX_DAYS_DEFAULT,
        help=(
            f"Maximum follow-up days; <=0 means no upper bound (default "
            f"{MAX_DAYS_DEFAULT}). Outer bound on emitted data, not the analysis "
            "window; see MAX_DAYS_DEFAULT in step1_prepare_analysis_dataset.py."
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

    run_for_gaps_a1c(
        input_csv=input_csv,
        outdir=outdir,
        max_days=args.max_days,
        adherence_gaps=adherence_gaps,
    )


if __name__ == "__main__":
    main()
