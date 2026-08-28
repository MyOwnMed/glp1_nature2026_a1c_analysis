#!/usr/bin/env python3
"""The persistence-of-therapy (follow-up) rule — one implementation.

Part II v2 (2026-08-17). Addresses code-review item 2.

In the submitted package the follow-up rule existed in five places:

  1. ``_compute_adherence_flags``          step1_prepare_analysis_dataset.py      LIVE
  2. ``_compute_adherence_flags``          step1_prepare_analysis_dataset_a1c.py  LIVE (duplicate of 1)
  3. ``_compute_censor_map_from_step8f``   step1_prepare_analysis_dataset.py      LIVE (used by step0a)
  4. ``_apply_adherence_censoring``        step1_prepare_analysis_dataset.py      DEAD, never called
  5. ``_compute_adherence_flag``           step1_prepare_analysis_dataset_a1c.py  DEAD, never called

The reviewer identified 1, 3 and 4. Variant 5 was found while consolidating.
All five are gone. This module is the only implementation, and every caller —
step1 (weight), step1 (A1c) and step0a — uses it.

────────────────────────────────────────────────────────────────────────────────
THE RULE
────────────────────────────────────────────────────────────────────────────────
The surviving rule is the one that produced the manuscript: the *flags* path
(variant 1/2). Stated once, in full:

  Let ``mentions`` be the set of days (``glp1_days_from_baseline`` >= 0) on which
  ``glp1_event_for_adherance`` is 1 (evidence of ongoing therapy) or 2 (an
  explicit stop). Let ``stop_day`` be the earliest day with value 2, if any.

  An observation on day ``d`` is *within persistence* for gap ``g`` when

      (a) some mention day ``m`` satisfies  m <= d  and  d - m <= g,   and
      (b) d < stop_day   (the stop day itself is excluded, strictly).

  A patient with no mentions at all is never within persistence.

``adherence_flags`` evaluates (a) and (b) per observation row. ``censor_days``
evaluates the same rule as a per-patient last-within-persistence day, which is
what an attrition count needs: walking the sorted mention days, coverage runs
from the first mention until either an interval longer than ``g`` opens (coverage
ends ``g`` days after the mention preceding it) or the mentions run out (coverage
ends ``g`` days after the last one); ``stop_day - 1`` caps it.

────────────────────────────────────────────────────────────────────────────────
ONE RULE, TWO READINGS — and the one place they part company
────────────────────────────────────────────────────────────────────────────────
The two entry points answer two different questions about the same rule, and the
difference is worth stating rather than glossing:

  ``adherence_flags``  is this observation within persistence?     (per row)
  ``censor_days``      how far does the persistence window reach?  (per patient)

step1 uses the first to censor each patient at their first non-persistent
observation. step0a uses the second, because the supplementary attrition table
reports how far each patient's persistence window extends over calendar time, not
how many observations happen to survive censoring. Those are genuinely different
quantities: the second is bounded by the rule, the first also by whether the
patient has a visit on a given day. Counting observations retained in the dataset
instead would change 162 of the 171 published cells; it is a different (and much
smaller) measure, reported separately as ``table_s1_retention_in_dataset.csv``.

Within the projection there is one edge case where a per-patient reading has to
make a choice the per-row reading makes implicitly. Coverage is the union of
``[m, m + g]`` over mention days. If a mention lands exactly ``g + 1`` days after
the previous one, the two intervals abut with no uncovered day between them, so a
day-by-day walk would continue; a projection that breaks on
``day - prev > g`` stops at ``prev + g``.

This implementation breaks on ``> g``, matching the construction that produced
the published table. Verified 2026-08-17: this reproduces the published attrition
table in all 171 cells of the supplementary layout and all 225 cells of the full
table, from the untrimmed source. Breaking on ``> g + 1`` instead would change 79
of the 171 cells. Since no published number may change, and the reviewer's item
was about *which rule is used*, not about this boundary, the published
construction is retained and the boundary is documented here.

Both readings share the two properties that define the surviving rule: mentions
include explicit stops (value 2), and the stop day is excluded strictly. Those
are the two ways the retired ``_compute_censor_map_from_step8f`` differed, and
neither is exercised by this dataset (see below).

────────────────────────────────────────────────────────────────────────────────
WHY REPLACING VARIANT 3 CHANGED NO PUBLISHED NUMBER
────────────────────────────────────────────────────────────────────────────────
Variant 3 (which produced the published supplementary attrition table) differed
from the surviving rule in two ways: it used only value-1 days to build the gap
sequence, ignoring value 2; and it kept the stop day rather than excluding it.
Both differences are reachable only through the value 2.

``glp1_event_for_adherance`` never takes the value 2 in this dataset. Verified
2026-08-17 over the full untrimmed source
(``step8g_with_unstructured_flags_with_assessments_weightcleaned.csv``,
1,239,561 rows): the only values present are 0 (931,665 rows) and 1 (307,896).
There are no explicit-stop events, so the two live rules are not merely close
here, they are identical patient by patient — confirmed for all 16,061 weight-
cohort patients at all nine persistence definitions, giving the identical 171-cell
attrition table (see verification/ and E1).

The value-2 branch is retained in this implementation because the rule is
defined over it and a future data refresh may contain stops; it is simply not
exercised by the data behind the manuscript.
"""

from __future__ import annotations

import math
from typing import Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "MENTION_VALUES",
    "STOP_VALUE",
    "REQUIRED_COLUMNS",
    "normalize_patient_id",
    "mention_timeline",
    "censor_day_for_patient",
    "censor_days",
    "adherence_flags",
]

#: ``glp1_event_for_adherance`` values that count as evidence of ongoing therapy.
MENTION_VALUES: Tuple[float, ...] = (1.0, 2.0)

#: ``glp1_event_for_adherance`` value marking an explicit discontinuation.
STOP_VALUE: float = 2.0

#: Columns the rule needs on an input frame.
REQUIRED_COLUMNS = frozenset(
    {"patient_id", "glp1_event_for_adherance", "glp1_days_from_baseline"}
)


def normalize_patient_id(values):
    """Put patient identifiers in one canonical form: stripped strings.

    Addresses code-review item 1. Identifier normalisation happens *here and
    only here*, so a censor map produced by this module and a patient set built
    by a caller cannot end up on opposite sides of a dtype mismatch. In the
    submitted code ``step0a`` compared ``set(df["patient_id"].astype(str))``
    against dict keys that had kept whatever dtype ``read_csv`` inferred; with
    numeric-looking identifiers every membership test would have returned False
    and every post-baseline count would have collapsed to zero. This project's
    identifiers are UUID strings, so the comparison happened to hold, but the
    defect did not depend on that and neither does the fix.

    Accepts a scalar, a Series or any iterable; returns the same shape.
    """
    if isinstance(values, pd.Series):
        return values.astype(str).str.strip()
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        return str(values).strip()
    return [str(v).strip() for v in values]


def _coerce_day(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def mention_timeline(df: pd.DataFrame) -> Dict[str, List[Tuple[float, float]]]:
    """Per-patient ``[(value, day), ...]`` mention pairs, ids normalised.

    Keeps rows with a parseable, non-negative ``glp1_days_from_baseline``. Pairs
    are de-duplicated and returned sorted by day.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise KeyError(
            "mention_timeline requires columns "
            f"{sorted(REQUIRED_COLUMNS)}; missing {sorted(missing)}"
        )
    ev = df[list(REQUIRED_COLUMNS)].copy()
    ev["glp1_days_from_baseline"] = _coerce_day(ev["glp1_days_from_baseline"])
    ev["glp1_event_for_adherance"] = pd.to_numeric(
        ev["glp1_event_for_adherance"], errors="coerce"
    )
    ev = ev.dropna(subset=["patient_id", "glp1_days_from_baseline"])
    ev = ev[ev["glp1_days_from_baseline"] >= 0]

    out: Dict[str, set] = {}
    for pid, value, day in zip(
        normalize_patient_id(ev["patient_id"]),
        ev["glp1_event_for_adherance"].to_numpy(dtype=float),
        ev["glp1_days_from_baseline"].to_numpy(dtype=float),
    ):
        out.setdefault(pid, set()).add((float(value), float(day)))
    return {pid: sorted(pairs, key=lambda p: p[1]) for pid, pairs in out.items()}


def censor_day_for_patient(
    pairs: Sequence[Tuple[float, float]], gap_days: int
) -> Optional[int]:
    """Last day this patient is still within persistence, or None if never.

    *pairs* is ``[(glp1_event_for_adherance, day), ...]`` for one patient, days
    non-negative. See the module docstring for the rule.

    Assumes the day-0 mention that step1 guarantees. Every patient in the cohort
    has a mention on day 0 — either an observed one or the row
    ``_ensure_baseline_rows`` creates to anchor the trajectory at treatment
    initiation, which carries ``glp1_event_for_adherance = 1``. Under that
    invariant this function and :func:`adherence_flags` are two readings of one
    rule, and the regression suite asserts they agree.

    The invariant matters. Coverage is the union of ``[m, m + g]`` over mention
    days, and this function returns the end of the *first* such block. If a
    patient's first mention were at, say, day 250, the row-level rule would place
    days 0-249 outside persistence, so step1 — which censors at the first
    non-persistent observation — would drop that patient entirely rather than
    credit the later block. Anchoring at day 0 is what makes the first block the
    only one that can matter, so the two readings coincide.
    """
    if gap_days is None or gap_days <= 0:
        raise ValueError(f"gap_days must be a positive number of days, got {gap_days!r}")

    stop_candidates = [d for v, d in pairs if v == STOP_VALUE]
    stop_day = min(stop_candidates) if stop_candidates else None

    mention_days = sorted(d for v, d in pairs if v in MENTION_VALUES)
    if not mention_days:
        return None

    # Walk the mention days forward; coverage ends gap days after the mention that
    # precedes the first inter-mention interval longer than gap, or gap days after
    # the last mention if no such interval occurs.
    #
    # See "ONE RULE, TWO READINGS" in the module docstring for why the break test
    # is `> gap_days` and not `> gap_days + 1`.
    coverage_end = None
    prev = mention_days[0]
    for day in mention_days[1:]:
        if (day - prev) > gap_days:
            coverage_end = prev + gap_days
            break
        prev = day
    if coverage_end is None:
        coverage_end = mention_days[-1] + gap_days

    if stop_day is not None:
        # Condition (b) excludes the stop day strictly, so the last day still
        # within persistence is the day before it.
        coverage_end = min(coverage_end, stop_day - 1)

    if not np.isfinite(coverage_end) or coverage_end < 0:
        return None
    return int(math.floor(coverage_end))


def censor_days(
    source: pd.DataFrame | Mapping[Hashable, Sequence[Tuple[float, float]]],
    gap_days: int,
) -> Dict[str, int]:
    """``{patient_id: last day within persistence}`` for one gap definition.

    *source* is either a frame carrying the three required columns or an
    already-built mention timeline (as returned by :func:`mention_timeline`).
    Patients with no mentions are absent from the result. Keys are normalised
    by :func:`normalize_patient_id`.
    """
    timeline = (
        mention_timeline(source)
        if isinstance(source, pd.DataFrame)
        else {normalize_patient_id(k): v for k, v in source.items()}
    )
    out: Dict[str, int] = {}
    for pid, pairs in timeline.items():
        day = censor_day_for_patient(pairs, gap_days)
        if day is not None:
            out[pid] = day
    return out


def adherence_flags(df: pd.DataFrame, gap_days_list: Iterable[int]) -> pd.DataFrame:
    """Add an ``adherence_{g}`` 0/1 column per gap, evaluating the rule per row.

    Backward-looking: a row is flagged 1 when the most recent mention day at or
    before it is within *g* days and the row precedes any explicit stop day.
    Mutates and returns *df*.
    """
    gaps = list(gap_days_list)
    for gap in gaps:
        df[f"adherence_{gap}"] = 0
    if not REQUIRED_COLUMNS.issubset(df.columns) or "days_from_baseline" not in df.columns:
        return df

    local = df[
        [
            "patient_id",
            "days_from_baseline",
            "glp1_event_for_adherance",
            "glp1_days_from_baseline",
        ]
    ].copy()
    local["days_from_baseline"] = _coerce_day(local["days_from_baseline"])
    local["glp1_days_from_baseline"] = _coerce_day(local["glp1_days_from_baseline"])

    out = {gap: np.zeros(len(local), dtype=int) for gap in gaps}

    for _pid, grp in local.groupby("patient_id", sort=False):
        row_pos = local.index.get_indexer(grp.index.to_numpy())
        row_days = grp["days_from_baseline"].to_numpy(dtype=float)
        values = grp["glp1_event_for_adherance"].astype(float).to_numpy()
        event_days = grp["glp1_days_from_baseline"].to_numpy(dtype=float)

        mention_days = np.sort(event_days[np.isin(values, MENTION_VALUES)])
        stop_candidates = event_days[values == STOP_VALUE]
        stop_day = np.min(stop_candidates) if stop_candidates.size else None

        if mention_days.size == 0:
            continue

        idx_prev = np.searchsorted(mention_days, row_days, side="right") - 1
        has_prev = idx_prev >= 0
        prev_day = np.full(row_days.shape, np.nan)
        prev_day[has_prev] = mention_days[idx_prev[has_prev]]
        delta = row_days - prev_day

        for gap in gaps:
            mask = np.zeros(row_days.shape, dtype=bool)
            valid = has_prev & np.isfinite(delta)
            mask[valid] = delta[valid] <= gap
            if stop_day is not None and np.isfinite(stop_day):
                mask &= row_days < stop_day
            out[gap][row_pos] = mask.astype(int)

    for gap in gaps:
        df[f"adherence_{gap}"] = out[gap]
    return df
