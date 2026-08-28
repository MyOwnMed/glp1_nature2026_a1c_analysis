#!/usr/bin/env python3
"""
Mention frequency summary — unstructured assessment outcomes.

For each of the five NLP-extracted outcome domains, computes per-patient
observation counts split by period (pre-GLP-1 / post-GLP-1 / total), then
reports the average per patient across the study cohort.

Runs for two scenarios:
  1. No adherence restriction — all patients with any assessment observation
  2. Per-gap cohort (gap30…gap730) — post-GLP-1 observations censored at each
     patient's adherence cutoff derived from the step1 analysis-ready files.
     The cutoff is the max days_from_baseline that patient appears in the
     step1 file for that gap threshold.

Output:
  output/add_unstructured/freq/mention_frequency.csv
"""

import pandas as pd
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[3]
DATADIR = ROOT / "output" / "submitted_analysis" / "1_no_adherence" / "data"
STEP1   = ROOT / "output" / "step1_prepare_analysis_dataset"
OUTDIR  = ROOT / "output" / "submitted_analysis"

DOMAINS = {
    "PHQ-9":               ("phq9_prepared.csv",               "phq9_n_rows"),
    "Pain Score":          ("pain_score_prepared.csv",          "pain_score_n_rows"),
    "Waist Circumference": ("waist_circumference_prepared.csv", "waist_circumference_n_rows"),
    "Alcohol Use":         ("alcohol_prepared.csv",             "alcohol_n_rows"),
    "Muscle Strength":     ("muscle_strength_prepared.csv",     "muscle_strength_n_rows"),
}

GAPS = [30, 60, 90, 120, 150, 180, 365, 548, 730]

# Change-from-baseline window constants (matches run_baseline_anchor_analysis.py)
BASELINE_WINDOW = 30  # +/-30 days around GLP-1 start
FOLLOWUP_START  = 31  # post-GLP-1 follow-up starts day 31


def compute_stats(df, nlp_col, cohort_label, outcome_label):
    """Return one dict of frequency stats for a single domain x cohort slice."""
    df = df.copy()
    df["post"]               = pd.to_numeric(df["post"],               errors="coerce").fillna(0).astype(int)
    df[nlp_col]              = pd.to_numeric(df[nlp_col],              errors="coerce").fillna(1)
    df["days_from_baseline"] = pd.to_numeric(df["days_from_baseline"], errors="coerce")

    n_total     = df["patient_id"].nunique()
    n_with_pre  = df[df["post"] == 0]["patient_id"].nunique()
    n_with_post = df[df["post"] == 1]["patient_id"].nunique()

    # ITS paired: patient has at least one pre AND one post observation (capped at day 365)
    pre_ids  = set(df[df["post"] == 0]["patient_id"])
    post_ids = set(df[(df["post"] == 1) & (df["days_from_baseline"] <= 365)]["patient_id"])
    its_pts  = len(pre_ids & post_ids)

    # Change From Baseline: >=1 obs within +/-30d of start AND >=1 obs in days 31-365
    anchor_ids   = set(df[df["days_from_baseline"].between(-BASELINE_WINDOW, BASELINE_WINDOW)]["patient_id"])
    followup_ids = set(df[df["days_from_baseline"].between(FOLLOWUP_START, 365)]["patient_id"])
    cfb_pts      = len(anchor_ids & followup_ids)

    def avg_enc_nlp(mask):
        sub = df[mask].groupby("patient_id").agg(enc=(nlp_col, "count"), nlp=(nlp_col, "sum"))
        return round(sub["enc"].mean(), 2), round(sub["nlp"].mean(), 2)

    pre_enc,   pre_nlp   = avg_enc_nlp(df["post"] == 0)
    post_enc,  post_nlp  = avg_enc_nlp(df["post"] == 1)
    total_enc, total_nlp = avg_enc_nlp(pd.Series(True, index=df.index))

    return {
        "Outcome": outcome_label,
        "Cohort":  cohort_label,
        "N Patients (any observation)": n_total,
        "N Interrupted Time Series (requires both pre AND post obs)": its_pts,
        "N Change From Baseline (anchor +/-30d + follow-up)":         cfb_pts,
        # NOTE: pre + post can exceed total; overlap = pre + post - total = ITS count
        "N with any pre-GLP1 observation":  n_with_pre,
        "N with any post-GLP1 observation": n_with_post,
        "Avg Encounters Pre-GLP1 (among patients with pre obs)":    pre_enc,
        "Avg Encounters Post-GLP1 (among patients with post obs)":  post_enc,
        "Avg Encounters Total (among all patients with any obs)":    total_enc,
        "Avg NLP Extractions Pre-GLP1 (among patients with pre obs)":   pre_nlp,
        "Avg NLP Extractions Post-GLP1 (among patients with post obs)": post_nlp,
        "Avg NLP Extractions Total (among all patients with any obs)":   total_nlp,
    }


# -- Pre-load per-patient adherence censoring cutoffs for each gap ---------
print("Loading adherence censoring cutoffs from step1 files ...")
gap_cutoffs = {}  # gap_int -> Series(patient_id -> max_days_from_baseline)
for g in GAPS:
    fpath = STEP1 / f"analysis_ready_gap{g}.csv"
    if fpath.exists():
        tmp = pd.read_csv(fpath, usecols=["patient_id", "days_from_baseline"])
        gap_cutoffs[g] = tmp.groupby("patient_id")["days_from_baseline"].max()
        print(f"  gap{g}: {len(gap_cutoffs[g])} patients")
    else:
        print(f"  gap{g}: file not found, skipping")

# -- Main loop -------------------------------------------------------------
all_rows = []

for outcome_label, (fname, nlp_col) in DOMAINS.items():
    print(f"\nProcessing {outcome_label} ...")
    base_df = pd.read_csv(DATADIR / fname)
    base_df["days_from_baseline"] = pd.to_numeric(base_df["days_from_baseline"], errors="coerce")

    # No restriction
    all_rows.append(compute_stats(base_df, nlp_col, "No adherence restriction", outcome_label))

    # Per gap threshold
    for g in GAPS:
        if g not in gap_cutoffs:
            continue
        cutoff = gap_cutoffs[g]

        # Restrict to patients in the step1 cohort
        cohort_ids = set(cutoff.index)
        df_gap = base_df[base_df["patient_id"].isin(cohort_ids)].copy()

        # Censor post-GLP1 rows beyond each patient's adherence cutoff.
        # Pre-GLP1 rows (days_from_baseline < 0) are never censored.
        cutoff_map = cutoff.to_dict()
        df_gap["_cutoff"] = df_gap["patient_id"].map(cutoff_map)
        df_gap = df_gap[
            (df_gap["days_from_baseline"] < 0) |
            (df_gap["days_from_baseline"] <= df_gap["_cutoff"])
        ].drop(columns=["_cutoff"])

        all_rows.append(compute_stats(df_gap, nlp_col, f"Gap {g}d", outcome_label))

# -- Write output ----------------------------------------------------------
OUTDIR.mkdir(parents=True, exist_ok=True)
out = pd.DataFrame(all_rows)
out_path = OUTDIR / "add_unstructured_adherence_freq.csv"
out.to_csv(out_path, index=False)

print(f"\nSaved --> {out_path}  ({len(out)} rows x {len(out.columns)} columns)")
preview_cols = ["Outcome", "Cohort",
                "N Patients (any observation)",
                "N Interrupted Time Series (requires both pre AND post obs)",
                "N with any post-GLP1 observation"]
print(out[preview_cols].to_string(index=False))
