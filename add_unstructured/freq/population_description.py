#!/usr/bin/env python3
"""
Population description for unstructured assessment domains.

For each of the three primary NLP-extracted outcome domains (PHQ-9, Pain Score,
Waist Circumference), describes the population of patients who have any observation
by the characteristics used in Table 1 of the main manuscript:

  - Mean (SD): Baseline HbA1c (%), Age (years), Weight (lbs), BMI (kg/m²)
  - Age group: <40 years, ≥40 years
  - Baseline BMI category: Underweight, Normal weight, Overweight,
                           Obese class I, Obese class II, Obese class III
  - Sex: Female, Male
  - Race: Caucasian, Other
  - Clinical response – Achieved:
      ≥5%  / ≥10% / ≥15% weight loss
      ≥0.5% / ≥1.0% / ≥1.5% / ≥2.0% / ≥2.5% HbA1c reduction

Population used: all patients with ANY observation in each domain
(no adherence restriction), matching the "1_no_adherence" cohort.

Clinical response denominators are patients with available structured
outcomes in the step1 files (gap=120d), so may be smaller than the
domain patient count.

Output:
  output/submitted_analysis/add_unstructured_population_description.csv
"""

import os
import pandas as pd
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[3]
_CI     = Path(os.environ.get('CONF_INT_DIR', str(ROOT / "output" / "submitted_analysis")))
DATA    = Path(os.environ.get('AU_DATADIR',   str(_CI / "1_no_adherence" / "data")))
STEP1_W = Path(os.environ.get('AU_STEP1_W',   str(ROOT / "output" / "step1_prepare_analysis_dataset" / "analysis_ready_gap120.csv")))
STEP1_A = Path(os.environ.get('AU_STEP1_A',   str(ROOT / "output" / "step1_prepare_analysis_dataset_a1c" / "analysis_ready_a1c_gap120.csv")))
OUTDIR  = Path(os.environ.get('AU_OUTDIR',    str(_CI)))

DOMAINS = {
    "PHQ-9":               "phq9_prepared.csv",
    "Pain Score":          "pain_score_prepared.csv",
    "Waist Circumference": "waist_circumference_prepared.csv",
}

BMI_LABELS = {
    "Underweight": "Underweight",
    "Normal":      "Normal weight",
    "Overweight":  "Overweight",
    "Obese I":     "Obese class I",
    "Obese II":    "Obese class II",
    "Obese III":   "Obese class III",
}
BMI_ORDER = ["Underweight", "Normal weight", "Overweight",
             "Obese class I", "Obese class II", "Obese class III"]


def fmt_mean_sd(series):
    s = series.dropna()
    if len(s) == 0:
        return ""
    return f"{s.mean():.1f} ({s.std():.1f})"


def fmt_n_pct(n, denom):
    if denom == 0:
        return ""
    return f"{n} ({100 * n / denom:.1f}%)"


# ── Load step1 structured data (best-achieved outcomes per patient) ─────────
print("Loading step1 structured data ...")
df_w = pd.read_csv(STEP1_W)
df_a = pd.read_csv(STEP1_A)

# Per-patient baseline characteristics from step1_weight
demo_w = (df_w.sort_values("days_from_baseline")
              .groupby("patient_id")
              .first()
              .reset_index())

# Per-patient baseline A1c value from step1_a1c
demo_a = (df_a.sort_values("days_from_baseline")
              .groupby("patient_id")
              .first()
              .reset_index()
              [["patient_id", "baseline_a1c_final"]])

# Best-achieved clinical outcomes (minimum = most negative = greatest improvement)
best_w = (df_w.groupby("patient_id")["pct_weight_change"]
              .min()
              .reset_index()
              .rename(columns={"pct_weight_change": "best_pct_weight_change"}))

best_a = (df_a.groupby("patient_id")["abs_a1c_change"]
              .min()
              .reset_index()
              .rename(columns={"abs_a1c_change": "best_abs_a1c_change"}))

print(f"  step1_weight: {len(demo_w)} patients")
print(f"  step1_a1c:    {len(demo_a)} patients")


# ── Build description for one domain ────────────────────────────────────────
def describe_domain(domain_label, fname):
    print(f"\nDescribing {domain_label} ...")
    raw = pd.read_csv(DATA / fname)

    # One row per patient — take first non-null for demographics
    pts = raw.groupby("patient_id").first().reset_index()
    N = len(pts)
    print(f"  {N} unique patients with any observation")

    # ── Merge baseline numeric values ────────────────────────────────────
    pts = pts.merge(demo_w[["patient_id", "weight_in_pounds_final", "baseline_bmi_final"]],
                    on="patient_id", how="left")
    pts = pts.merge(demo_a, on="patient_id", how="left")

    # ── Merge clinical response ──────────────────────────────────────────
    pts = pts.merge(best_w, on="patient_id", how="left")
    pts = pts.merge(best_a, on="patient_id", how="left")

    rows = []

    def add(category, characteristic, value):
        rows.append({"Category": category, "Characteristic": characteristic,
                     domain_label: value})

    # ── Mean (SD) continuous ─────────────────────────────────────────────
    add("Mean (SD)", "Baseline HbA1c (%)",
        fmt_mean_sd(pts["baseline_a1c_final"]))
    add("Mean (SD)", "Age (years)",
        fmt_mean_sd(pts["age"]))
    add("Mean (SD)", "Weight (lbs)",
        fmt_mean_sd(pts["weight_in_pounds_final"]))
    add("Mean (SD)", "BMI (kg/m²)",
        fmt_mean_sd(pts["baseline_bmi_final"]))

    # ── Age group ────────────────────────────────────────────────────────
    age_lt40  = (pts["age"] < 40).sum()
    age_ge40  = (pts["age"] >= 40).sum()
    age_known = pts["age"].notna().sum()
    add("Age group", "< 40 years",  fmt_n_pct(age_lt40, age_known))
    add("Age group", "≥ 40 years",  fmt_n_pct(age_ge40, age_known))

    # ── Baseline BMI category ────────────────────────────────────────────
    bmi_raw = pts["baseline_bmi_final_category"].map(BMI_LABELS)
    bmi_denom = bmi_raw.notna().sum()
    for label in BMI_ORDER:
        n = (bmi_raw == label).sum()
        add("Baseline BMI category", label, fmt_n_pct(n, bmi_denom))

    # ── Sex ──────────────────────────────────────────────────────────────
    gender_map = {"F": "Female", "M": "Male"}
    gen = pts["gender"].map(gender_map)
    gen_denom = gen.notna().sum()
    add("Sex", "Female", fmt_n_pct((gen == "Female").sum(), gen_denom))
    add("Sex", "Male",   fmt_n_pct((gen == "Male").sum(),   gen_denom))

    # ── Race ─────────────────────────────────────────────────────────────
    race_caucasian = (pts["race"] == "Caucasian").sum()
    race_other     = (pts["race"] != "Caucasian").sum()
    race_denom     = pts["race"].notna().sum()
    add("Race", "Caucasian", fmt_n_pct(race_caucasian, race_denom))
    add("Race", "Other",     fmt_n_pct(race_other,     race_denom))

    # ── Clinical response – Achieved ─────────────────────────────────────
    # Weight loss denominators: patients with any weight-loss data
    w_denom = pts["best_pct_weight_change"].notna().sum()
    for thr in [5, 10, 15]:
        n = (pts["best_pct_weight_change"] <= -thr).sum()
        add("Clinical response – Achieved", f"≥{thr}% weight loss",
            fmt_n_pct(n, w_denom))

    # A1c reduction denominators: patients with any A1c data
    a_denom = pts["best_abs_a1c_change"].notna().sum()
    for thr in [0.5, 1.0, 1.5, 2.0, 2.5]:
        n = (pts["best_abs_a1c_change"] <= -thr).sum()
        add("Clinical response – Achieved", f"≥{thr:.1f}% HbA1c reduction",
            fmt_n_pct(n, a_denom))

    print(f"  N for weight loss denom: {w_denom}, A1c denom: {a_denom}")
    return pd.DataFrame(rows)


# ── Run all domains ──────────────────────────────────────────────────────────
all_dfs = []
for domain, fname in DOMAINS.items():
    all_dfs.append(describe_domain(domain, fname))

# ── Merge on Category + Characteristic ──────────────────────────────────────
from functools import reduce
merged = reduce(lambda a, b: pd.merge(a, b, on=["Category", "Characteristic"],
                                      how="outer"), all_dfs)

# Define row order to match Table 1 layout
ROW_ORDER = [
    ("N",                            "Total patients"),
    ("Mean (SD)",                    "Baseline HbA1c (%)"),
    ("Mean (SD)",                    "Age (years)"),
    ("Mean (SD)",                    "Weight (lbs)"),
    ("Mean (SD)",                    "BMI (kg/m²)"),
    ("Age group",                    "< 40 years"),
    ("Age group",                    "≥ 40 years"),
    ("Baseline BMI category",        "Underweight"),
    ("Baseline BMI category",        "Normal weight"),
    ("Baseline BMI category",        "Overweight"),
    ("Baseline BMI category",        "Obese class I"),
    ("Baseline BMI category",        "Obese class II"),
    ("Baseline BMI category",        "Obese class III"),
    ("Sex",                          "Female"),
    ("Sex",                          "Male"),
    ("Race",                         "Caucasian"),
    ("Race",                         "Other"),
    ("Clinical response – Achieved", "≥5% weight loss"),
    ("Clinical response – Achieved", "≥10% weight loss"),
    ("Clinical response – Achieved", "≥15% weight loss"),
    ("Clinical response – Achieved", "≥0.5% HbA1c reduction"),
    ("Clinical response – Achieved", "≥1.0% HbA1c reduction"),
    ("Clinical response – Achieved", "≥1.5% HbA1c reduction"),
    ("Clinical response – Achieved", "≥2.0% HbA1c reduction"),
    ("Clinical response – Achieved", "≥2.5% HbA1c reduction"),
]

# Insert an N row at the top
n_row = pd.DataFrame([{
    "Category": "N",
    "Characteristic": "Total patients",
}])
for domain, fname in DOMAINS.items():
    raw = pd.read_csv(DATA / fname)
    n = raw["patient_id"].nunique()
    n_row[domain] = str(n)

merged = pd.concat([n_row, merged], ignore_index=True)

# Sort by defined order
order_df = pd.DataFrame(ROW_ORDER, columns=["Category", "Characteristic"])
order_df["_order"] = range(len(order_df))
merged = merged.merge(order_df, on=["Category", "Characteristic"], how="left")
merged = merged.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)

# ── Write output ─────────────────────────────────────────────────────────────
OUTDIR.mkdir(parents=True, exist_ok=True)
out_path = OUTDIR / "add_unstructured_population_description.csv"
merged.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}")

out_xlsx = OUTDIR / "add_unstructured_population_description.xlsx"
merged.to_excel(out_xlsx, index=False)
print(f"Saved → {out_xlsx}")
print(f"  {len(merged)} rows × {len(merged.columns)} columns")
print()
print(merged.to_string(index=False))
