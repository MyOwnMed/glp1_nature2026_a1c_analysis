#!/usr/bin/env python3
"""Regression tests for the Part II v2 corrections.

Run from the study root (no pytest required):

    python3 output/part_ii_v2_2026-08-17/code/tests/test_persistence_and_step0a.py

Covers:
  * code-review item 1 — patient-identifier dtype. A cohort whose identifiers
    parse as integers must still produce non-zero counts at every month. This is
    the test the submitted code would have failed.
  * code-review item 2 — one follow-up rule. The consolidated rule must agree
    with the rule that produced the published attrition table, and the removed
    variants must be gone.
  * code-review item 3 — the baseline_carried_to_day0 marker.
  * code-review item 9 — the confidence multiplier is derived, not hard-coded.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

import step0a_samplesize_analysis as step0a  # noqa: E402
from analysis_config import CONF_LEVEL, z_critical  # noqa: E402
from persistence import (  # noqa: E402
    adherence_flags,
    censor_day_for_patient,
    censor_days,
    normalize_patient_id,
)

FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


# ── item 1: identifier dtype ──────────────────────────────────────────────────
def test_integer_patient_ids_still_count():
    """Integer-parseable patient IDs must not collapse post-baseline counts to 0.

    Builds a small cohort with numeric identifiers, writes it as a CSV so the
    identifiers round-trip through read_csv as int64 exactly as a real export
    would, and runs the real aggregate_counts. In the submitted code the
    str-vs-int64 membership test made every month after month 0 zero.
    """
    print("\nitem 1 — patient-identifier dtype")
    gap = 120
    rows = []
    # 5 patients, monthly GLP-1 mentions out to day 540, so every month is covered.
    for pid in (1001, 1002, 1003, 1004, 1005):
        for day in range(0, 541, 30):
            rows.append(
                {
                    "patient_id": pid,
                    "days_from_baseline": day,
                    "glp1_event_for_adherance": 1,
                    "glp1_days_from_baseline": day,
                    "pct_weight_change": -1.0 * day / 100.0,
                }
            )
    frame = pd.DataFrame(rows)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        frame.to_csv(tmp / f"analysis_ready_gap{gap}.csv", index=False)

        reread = pd.read_csv(tmp / f"analysis_ready_gap{gap}.csv")
        check(
            "patient_id round-trips as an integer dtype (precondition)",
            reread["patient_id"].dtype.kind in "iu",
            f"dtype={reread['patient_id'].dtype}",
        )

        counts = step0a.aggregate_counts(
            {gap: tmp / f"analysis_ready_gap{gap}.csv"},
            n_months=18,
            censor_from="gapfiles",
        )

    col = counts[f"gap_{gap}"].tolist()
    check("month 0 counts all 5 patients", col[0] == 5, f"got {col[0]}")
    check(
        "every month 1..18 is non-zero",
        all(c == 5 for c in col),
        f"counts={col}",
    )


def test_normalize_patient_id_is_the_single_point():
    print("\nitem 1 — identifier normalisation is centralised")
    check("int and str forms normalise equal", normalize_patient_id(1001) == normalize_patient_id("1001"))
    check("whitespace is stripped", normalize_patient_id("  abc  ") == "abc")
    series = pd.Series([1001, 1002])
    check("Series normalises to str", list(normalize_patient_id(series)) == ["1001", "1002"])
    cmap = censor_days(
        pd.DataFrame(
            {
                "patient_id": [1001, 1001],
                "glp1_event_for_adherance": [1, 1],
                "glp1_days_from_baseline": [0, 100],
            }
        ),
        120,
    )
    check("censor_days keys are normalised strings", set(cmap) == {"1001"}, f"keys={set(cmap)}")


# ── item 2: one follow-up rule ────────────────────────────────────────────────
def test_removed_variants_are_gone():
    print("\nitem 2 — removed rule variants")
    import step1_prepare_analysis_dataset as s1
    import step1_prepare_analysis_dataset_a1c as s1a

    for mod, name in [
        (s1, "_apply_adherence_censoring"),
        (s1, "_compute_censor_map_from_step8f"),
        (s1, "_first_nonadherence_gap"),
        (s1, "_compute_adherence_flags"),
        (s1a, "_compute_adherence_flag"),
        (s1a, "_compute_adherence_flags"),
        (s1, "GLP1_INJECTABLE_NAMES"),
    ]:
        check(f"{mod.__name__}.{name} removed", not hasattr(mod, name))


def test_censor_day_matches_flags_rule():
    """censor_day_for_patient must be the per-patient reading of adherence_flags.

    Tested under the invariant the pipeline guarantees: every patient has a
    mention on day 0, because step1 anchors every trajectory there and that
    created row carries glp1_event_for_adherance = 1. Without a day-0 mention the
    two readings diverge by construction — rows before the first mention are not
    within persistence row-wise, so step1 would censor the patient at day 0
    entirely, while a per-patient coverage day would report the later block. That
    case cannot arise after anchoring, and asserting equivalence outside the
    invariant would be asserting something untrue. See censor_day_for_patient.

    Mention sets are also generated without abutting intervals (no mention exactly
    gap + 1 days after the previous one), which is the one boundary where the
    projection and the row-level reading differ by design. That boundary is pinned
    separately in test_abutting_interval_boundary, and the reason the projection
    keeps the published construction there is in the persistence module docstring.
    """
    print("\nitem 2 — censor_days agrees with adherence_flags row by row")
    rng = np.random.default_rng(20260817)
    mismatches = 0
    cases = 0
    skipped = 0
    for _ in range(300):
        n = int(rng.integers(1, 8))
        mention_days = sorted(set([0] + [int(d) for d in rng.integers(0, 600, size=n)]))
        # exercise the stop branch too, even though this dataset has no value 2
        use_stop = bool(rng.integers(0, 2))
        stop_day = int(rng.integers(0, 700)) if use_stop else None
        pairs = [(1.0, float(d)) for d in mention_days]
        if stop_day is not None:
            pairs.append((2.0, float(stop_day)))
        for gap in (30, 120, 365):
            # Skip the abutting boundary, which differs by design and is pinned in
            # test_abutting_interval_boundary.
            if any(
                (b - a) == gap + 1
                for a, b in zip(mention_days, mention_days[1:])
            ):
                skipped += 1
                continue
            cases += 1
            expected = censor_day_for_patient(pairs, gap)
            # Evaluate the row-level rule on a dense day grid and take the last
            # day of the first contiguous within-persistence run from day 0. The
            # grid must extend past the widest possible coverage end, or the
            # comparison would measure the grid rather than the rule.
            n_days = max(mention_days) + gap + 5
            grid = pd.DataFrame(
                {
                    "patient_id": ["p"] * n_days,
                    "days_from_baseline": list(range(n_days)),
                    "glp1_event_for_adherance": [np.nan] * n_days,
                    "glp1_days_from_baseline": [np.nan] * n_days,
                }
            )
            ev = pd.DataFrame(
                {
                    "patient_id": ["p"] * len(pairs),
                    "days_from_baseline": [np.nan] * len(pairs),
                    "glp1_event_for_adherance": [v for v, _ in pairs],
                    "glp1_days_from_baseline": [d for _, d in pairs],
                }
            )
            flagged = adherence_flags(pd.concat([grid, ev], ignore_index=True), [gap])
            obs = flagged.dropna(subset=["days_from_baseline"]).sort_values("days_from_baseline")
            flags = obs[f"adherence_{gap}"].to_numpy()
            days = obs["days_from_baseline"].to_numpy()
            if flags[0] == 0:
                last_covered = None
            else:
                first_zero = np.argmax(flags == 0) if (flags == 0).any() else len(flags)
                last_covered = int(days[first_zero - 1])
            if expected != last_covered:
                mismatches += 1
    check(
        "per-patient censor day == last day of first within-persistence run",
        mismatches == 0,
        f"{cases} cases, {mismatches} mismatches, {skipped} abutting cases skipped",
    )
    check("the randomised sweep actually ran", cases > 500, f"cases={cases}")


def test_stop_day_is_excluded_strictly():
    print("\nitem 2 — explicit-stop semantics (not exercised by this dataset)")
    pairs = [(1.0, 0.0), (1.0, 100.0), (2.0, 150.0)]
    check(
        "stop day excluded strictly (censor = stop - 1)",
        censor_day_for_patient(pairs, 120) == 149,
        f"got {censor_day_for_patient(pairs, 120)}",
    )
    check(
        "no mentions -> no persistence day",
        censor_day_for_patient([(0.0, 5.0)], 120) is None,
    )
    check(
        "gap longer than g ends coverage at prev + g",
        censor_day_for_patient([(1.0, 0.0), (1.0, 400.0)], 120) == 120,
    )


def test_abutting_interval_boundary():
    """Pin the projection boundary that reproduces the published attrition table.

    A mention landing exactly gap + 1 days after the previous one abuts the earlier
    coverage interval. The projection breaks there (coverage ends at prev + gap),
    which is the construction behind the published table; a day-by-day reading
    would carry on. Breaking on > gap + 1 instead reproduces the row-level rule
    exactly but changes 79 of the 171 published cells, so the published
    construction is kept and pinned here. See persistence.py, "ONE RULE, TWO
    READINGS".
    """
    print("\nitem 2 — abutting-interval projection boundary")
    check(
        "mention at exactly prev + g + 1 ends coverage at prev + g (published)",
        censor_day_for_patient([(1.0, 0.0), (1.0, 121.0)], 120) == 120,
        f"got {censor_day_for_patient([(1.0, 0.0), (1.0, 121.0)], 120)}",
    )
    check(
        "mention at prev + g continues coverage",
        censor_day_for_patient([(1.0, 0.0), (1.0, 120.0)], 120) == 240,
        f"got {censor_day_for_patient([(1.0, 0.0), (1.0, 120.0)], 120)}",
    )


# ── item 3: the baseline_carried_to_day0 marker ───────────────────────────────
def test_baseline_carried_marker():
    print("\nitem 3 — baseline_carried_to_day0 marker")
    import step1_prepare_analysis_dataset as s1

    check("marker name is baseline_carried_to_day0",
          s1.BASELINE_CARRIED_COL == "baseline_carried_to_day0")
    frame = pd.DataFrame(
        {
            "patient_id": ["a", "a", "b"],
            "days_from_baseline": [0, 90, 90],
            "baseline_weight_final": [200.0, 200.0, 180.0],
            "weight_in_pounds_final": [200.0, 190.0, 170.0],
            "pct_weight_change": [0.0, -5.0, -5.55],
            "glp1_event_for_adherance": [1, 1, 1],
            "glp1_days_from_baseline": [0, 90, 90],
        }
    )
    out = s1._ensure_baseline_rows(frame)
    check("a day-0 row was created for the patient lacking one", len(out) == 4, f"rows={len(out)}")
    created = out[out[s1.BASELINE_CARRIED_COL] == 1]
    check("exactly one created row", len(created) == 1, f"got {len(created)}")
    check("created row is patient b at day 0", set(created["patient_id"]) == {"b"}
          and created["days_from_baseline"].tolist() == [0])
    check("created row carries the recorded baseline weight",
          created["weight_in_pounds_final"].tolist() == [180.0])
    check("created row outcome is 0 by definition",
          created["pct_weight_change"].tolist() == [0.0])
    check("observed rows are marked 0",
          out.loc[out[s1.BASELINE_CARRIED_COL] == 0, "patient_id"].tolist() == ["a", "a", "b"])
    check("marker is integer dtype", out[s1.BASELINE_CARRIED_COL].dtype.kind in "iu",
          f"dtype={out[s1.BASELINE_CARRIED_COL].dtype}")

    # A frame where nobody needs anchoring: marker present and everywhere zero,
    # which is the HbA1c case.
    already = pd.DataFrame(
        {
            "patient_id": ["a", "b"],
            "days_from_baseline": [0, 0],
            "glp1_event_for_adherance": [1, 1],
            "glp1_days_from_baseline": [0, 0],
        }
    )
    out2 = s1._ensure_baseline_rows(already)
    check("marker present when nothing is created", s1.BASELINE_CARRIED_COL in out2.columns)
    check("marker sums to zero when nothing is created",
          int(out2[s1.BASELINE_CARRIED_COL].sum()) == 0)


# ── item 9: derived confidence multiplier ─────────────────────────────────────
def test_z_critical():
    print("\nitem 9 — confidence multiplier is derived")
    z = z_critical()
    check("default confidence level is 0.95", CONF_LEVEL == 0.95, f"got {CONF_LEVEL}")
    check("z rounds to the 1.96 the submitted code hard-coded",
          round(z, 2) == 1.96, f"z={z!r}")
    check("z is the exact normal quantile, not a literal",
          abs(z - 1.959963984540054) < 1e-12, f"z={z!r}")
    check("99% level gives a different multiplier",
          abs(z_critical(0.99) - 2.5758293035489004) < 1e-12)
    for bad in (0.0, 1.0, -0.5, 1.5):
        try:
            z_critical(bad)
            check(f"conf_level={bad} rejected", False)
        except ValueError:
            check(f"conf_level={bad} rejected", True)


# ── item 10: fail-closed metformin helper ─────────────────────────────────────
def test_metformin_helper_fails_closed():
    print("\nmiscellaneous — _has_metformin_near_baseline fails closed")
    import json as _json

    import step1_prepare_analysis_dataset as s1

    good = _json.dumps([{"medication_name": "Metformin HCl", "medication_date": "2024-01-10"}])
    check("in-window metformin is found",
          s1._has_metformin_near_baseline(good, "2024-01-01") is True)
    far = _json.dumps([{"medication_name": "metformin", "medication_date": "2019-01-10"}])
    check("out-of-window metformin is not counted",
          s1._has_metformin_near_baseline(far, "2024-01-01") is False)
    check("unparseable JSON no longer falls back to a substring match",
          s1._has_metformin_near_baseline("metformin somewhere in free text", "2024-01-01") is False)
    undated = _json.dumps([{"medication_name": "metformin", "medication_date": "not-a-date"}])
    check("undatable entry no longer returns True",
          s1._has_metformin_near_baseline(undated, "2024-01-01") is False)
    check("missing baseline date no longer returns True",
          s1._has_metformin_near_baseline(good, None) is False)
    check("absent metformin is False", s1._has_metformin_near_baseline("[]", "2024-01-01") is False)


# ── miscellaneous: absent exclusion column ────────────────────────────────────
def test_exclusion_flag_default():
    print("\nmiscellaneous — df.get(...).fillna(0) AttributeError")
    import step1_prepare_analysis_dataset as s1

    frame = pd.DataFrame({"patient_id": ["a"]})
    s1._fill_exclusion_flag(frame, "pregnant_during_glp1")
    check("absent column defaults to 0 instead of raising AttributeError",
          frame["pregnant_during_glp1"].tolist() == [0])
    frame2 = pd.DataFrame({"pregnant_during_glp1": [np.nan, 1.0]})
    s1._fill_exclusion_flag(frame2, "pregnant_during_glp1")
    check("present column has NaN filled",
          frame2["pregnant_during_glp1"].tolist() == [0.0, 1.0])


def main():
    print("=" * 78)
    print("Part II v2 regression tests")
    print("=" * 78)
    test_integer_patient_ids_still_count()
    test_normalize_patient_id_is_the_single_point()
    test_removed_variants_are_gone()
    test_censor_day_matches_flags_rule()
    test_stop_day_is_excluded_strictly()
    test_abutting_interval_boundary()
    test_baseline_carried_marker()
    test_z_critical()
    test_metformin_helper_fails_closed()
    test_exclusion_flag_default()
    print("\n" + "=" * 78)
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        print("=" * 78)
        return 1
    print("All checks passed.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
