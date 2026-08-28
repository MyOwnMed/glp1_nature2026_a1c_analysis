#!/usr/bin/env python3
"""Step 1: Prepare assessment domain data for within-subject ITS analysis.

Reads merged step8g+assessments CSV, extracts per-domain observations,
derives antidepressant baseline flag, filters PHQ sources.

Window: 6 months pre-GLP-1 (−180 d) through 12 months post-GLP-1 (+365 d).
Output: domain-specific CSVs in output/add_unstructured/data/
"""

import json, logging, os
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = ROOT / "root_data" / "merged" / \
    "step8g_with_unstructured_flags_with_assessments_weightcleaned.csv"
# AU_DATADIR allows targeting a different output directory (e.g. 1_no_adherence_full/data)
_AU_DIR = os.environ.get('AU_DATADIR')
OUTDIR = Path(_AU_DIR) if _AU_DIR else ROOT / "output" / "submitted_analysis" / "1_no_adherence" / "data"

# ── Constants ─────────────────────────────────────────────────────────────
PRE_DAYS  = -180
POST_DAYS =  365

ANTIDEP_KEYWORDS = [
    'sertraline','fluoxetine','escitalopram','citalopram','paroxetine',
    'fluvoxamine','venlafaxine','duloxetine','desvenlafaxine',
    'levomilnacipran','bupropion','mirtazapine','amitriptyline',
    'nortriptyline','trazodone','doxepin',
    'zoloft','prozac','lexapro','celexa','paxil','effexor',
    'cymbalta','pristiq','wellbutrin','remeron',
]

DOMAINS = ['phq9', 'pain_score', 'waist_circumference', 'alcohol',
           'muscle_strength']

COVARIATES = ['age', 'gender', 'race', 'baseline_a1c_category',
              'baseline_bmi_final_category', 'glp1_user_group',
              'baseline_glp1_date']


# ═══════════════════════════════════════════════════════════════════════════
def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load assessment + covariate columns (skip medication_history) ──
    assess_cols = []
    for d in DOMAINS:
        for suffix in ('present', 'value', 'min', 'max',
                        'source_names', 'n_rows'):
            assess_cols.append(f"{d}_{suffix}")

    use_cols = (['patient_id', 'days_from_baseline', 'date']
                + COVARIATES + assess_cols)

    log.info("Loading %s …", INPUT_CSV.name)
    df = pd.read_csv(INPUT_CSV, usecols=use_cols)
    log.info("Loaded %d rows, %d patients", len(df), df.patient_id.nunique())

    # ── 2. Filter to analysis window ──────────────────────────────────────
    df['days_from_baseline'] = pd.to_numeric(
        df['days_from_baseline'], errors='coerce')
    df = df[(df.days_from_baseline >= PRE_DAYS) &
            (df.days_from_baseline <= POST_DAYS)].copy()
    log.info("Window [%d, %d]: %d rows, %d patients",
             PRE_DAYS, POST_DAYS, len(df), df.patient_id.nunique())

    # ── 3. Per-domain extraction ──────────────────────────────────────────
    domain_frames = {}
    for domain in DOMAINS:
        obs = _extract_domain(df, domain)
        domain_frames[domain] = obs

    # ── 4. PHQ-9: filter to PHQ-9 sources (exclude PHQ-2 only) ───────────
    phq = domain_frames.get('phq9')
    if phq is not None and not phq.empty:
        phq = _filter_phq_sources(phq)
        domain_frames['phq9'] = phq

    # ── 5. Antidepressant flag for PHQ-9 patients ─────────────────────────
    phq = domain_frames.get('phq9')
    if phq is not None and not phq.empty:
        phq = _derive_antidep_flag(phq)
        domain_frames['phq9'] = phq

    # ── 6. Save ───────────────────────────────────────────────────────────
    summary = {}
    for domain, obs in domain_frames.items():
        out = OUTDIR / f"{domain}_prepared.csv"
        obs.to_csv(out, index=False)
        n_both = 0
        if not obs.empty and 'has_both_periods' in obs.columns:
            n_both = int(obs.loc[obs.has_both_periods == 1,
                                 'patient_id'].nunique())
        summary[domain] = {
            'total_obs': int(len(obs)),
            'total_patients': int(obs.patient_id.nunique()) if not obs.empty
                              else 0,
            'patients_both': n_both,
        }
        log.info("  Saved %s  (%d rows, %d patients, %d with both periods)",
                 out.name, len(obs),
                 summary[domain]['total_patients'],
                 summary[domain]['patients_both'])

    with open(OUTDIR / "domain_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    log.info("Done — prepared data in %s", OUTDIR)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _extract_domain(df, domain):
    """Extract deduplicated patient × day observations for one domain."""
    pcol = f"{domain}_present"
    vcol = f"{domain}_value"
    mask = df[pcol].fillna(0).astype(bool) & df[vcol].notna()
    sub = df[mask].copy()
    if sub.empty:
        log.warning("  %s: no observations in window", domain)
        return sub

    # Deduplicate by patient + day
    agg = {vcol: 'mean'}
    for suffix in ('min', 'max', 'n_rows', 'source_names'):
        c = f"{domain}_{suffix}"
        if c in sub.columns:
            agg[c] = 'first'
    for c in COVARIATES:
        if c in sub.columns:
            agg[c] = 'first'
    if 'date' in sub.columns:
        agg['date'] = 'first'

    obs = sub.groupby(['patient_id', 'days_from_baseline'],
                       as_index=False).agg(agg)

    # Derived time variables
    obs['post'] = (obs.days_from_baseline >= 0).astype(int)
    obs['time_months'] = obs.days_from_baseline / 30.44
    obs['time_post'] = obs.post * obs.time_months

    # Flag patients with both periods
    pre_ids  = set(obs.loc[obs.post == 0, 'patient_id'])
    post_ids = set(obs.loc[obs.post == 1, 'patient_id'])
    both     = pre_ids & post_ids
    obs['has_both_periods'] = obs.patient_id.isin(both).astype(int)

    # COVID-19 pandemic era flag (WHO declaration to end of US emergency)
    _COVID_START = pd.Timestamp('2020-03-11')
    _COVID_END   = pd.Timestamp('2022-05-11')
    if 'baseline_glp1_date' in obs.columns:
        bdate = pd.to_datetime(obs['baseline_glp1_date'], errors='coerce')
        obs['covid_era'] = ((bdate >= _COVID_START) & (bdate <= _COVID_END)).astype(int)
    else:
        obs['covid_era'] = 0

    log.info("  %s: %d obs, %d patients, %d with both pre+post",
             domain, len(obs), obs.patient_id.nunique(), len(both))
    return obs


def _filter_phq_sources(phq):
    """Keep PHQ-9 specific sources; exclude rows that are PHQ-2 only."""
    src = phq['phq9_source_names'].fillna('').str.lower()

    is_phq2_only = (src.str.contains('phq-2|phq2') &
                    ~src.str.contains('phq-9|phq9'))
    n_excluded = is_phq2_only.sum()
    phq = phq[~is_phq2_only].copy()

    log.info("  PHQ source filter: excluded %d PHQ-2-only rows; "
             "%d rows remain (%d patients)",
             n_excluded, len(phq), phq.patient_id.nunique())

    # Refresh has_both_periods flag after filtering
    pre_ids  = set(phq.loc[phq.post == 0, 'patient_id'])
    post_ids = set(phq.loc[phq.post == 1, 'patient_id'])
    both     = pre_ids & post_ids
    phq['has_both_periods'] = phq.patient_id.isin(both).astype(int)
    log.info("  After PHQ filter: %d patients with both periods", len(both))
    return phq


def _derive_antidep_flag(phq):
    """Chunked read of medication_history → antidepressant near baseline."""
    needed_pids = set(phq.patient_id.unique())
    baseline_dates = (phq.drop_duplicates('patient_id')
                         .set_index('patient_id')['baseline_glp1_date']
                         .to_dict())

    log.info("  Parsing medication_history for %d PHQ-9 patients …",
             len(needed_pids))

    antidep_map = {}
    for chunk in pd.read_csv(INPUT_CSV,
                             usecols=['patient_id', 'medication_history'],
                             chunksize=100_000):
        remaining = needed_pids - set(antidep_map.keys())
        if not remaining:
            break
        hits = chunk[chunk.patient_id.isin(remaining)].drop_duplicates(
            'patient_id')
        for _, row in hits.iterrows():
            pid = row['patient_id']
            antidep_map[pid] = _check_antidep(
                row['medication_history'], baseline_dates.get(pid))

    # Fill missed patients
    for pid in needed_pids:
        antidep_map.setdefault(pid, 0)

    n_pos = sum(antidep_map.values())
    log.info("  Antidepressant at baseline: %d / %d (%.1f%%)",
             n_pos, len(antidep_map),
             100 * n_pos / max(len(antidep_map), 1))

    phq = phq.copy()
    phq['antidepressant_baseline'] = (phq.patient_id
                                         .map(antidep_map)
                                         .fillna(0)
                                         .astype(int))
    return phq


def _check_antidep(mh, baseline_date_str):
    """Return 1 if medication_history contains antidepressant near baseline."""
    if pd.isna(mh) or not mh:
        return 0
    mh_lower = str(mh).lower()
    if not any(kw in mh_lower for kw in ANTIDEP_KEYWORDS):
        return 0
    try:
        entries = json.loads(mh)
        baseline_dt = pd.to_datetime(baseline_date_str, errors='coerce')
        for entry in entries:
            name = str(entry.get('medication_name', '')).lower()
            if any(kw in name for kw in ANTIDEP_KEYWORDS):
                if pd.isna(baseline_dt):
                    return 1            # can't check timing → presence is enough
                med_dt = pd.to_datetime(
                    entry.get('medication_date'), errors='coerce')
                if pd.notna(med_dt):
                    diff = (med_dt - baseline_dt).days
                    if -180 <= diff <= 90:
                        return 1
        return 0
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0


if __name__ == '__main__':
    main()
