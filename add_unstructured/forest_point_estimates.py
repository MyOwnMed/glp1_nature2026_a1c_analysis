#!/usr/bin/env python3
"""
Forest plots of GEE point estimates at 3, 6, 9, 12 months — one plot per outcome.

Each outcome gets its own forest plot where:
  - Y-axis rows    = timepoints (3 mo, 6 mo, 9 mo, 12 mo)
  - Colour/marker  = subgroup (All, Elevated >=T1, Elevated >=T2)
  - Two panels     = ITS (left), CFB (right)

Elevated subgroups:
  PHQ-9:  All, >=5 (mild+), >=10 (mod-severe)
  Pain:   All, >=4 (moderate+), >=7 (severe)
  Alcohol: All, >=8.6 (top 25th percentile, ~drinks/day)
  Waist / Muscle: All only

Output -> <OUT>/forest_point_estimates/
"""

import logging, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from patsy import dmatrices, build_design_matrices
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.cov_struct import Independence

# Confidence multiplier derived from a single configurable confidence level
# (code-review item 9); replaces a hard-coded 1.96. See analysis_config.py.
import sys as _sys
from pathlib import Path as _Path

for _p in [_Path(__file__).resolve().parent, *_Path(__file__).resolve().parents]:
    if (_p / "analysis_config.py").exists():
        _sys.path.insert(0, str(_p))
        break
from analysis_config import z_critical

Z_CRIT = z_critical()

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
_AU  = os.environ.get('AU_OUTROOT')
DATA = Path(os.environ.get('AU_DATADIR',
            str(ROOT / "output" / "submitted_analysis" / "1_no_adherence" / "data")))
OUT  = (Path(_AU) / "Forest") if _AU \
       else ROOT / "output" / "submitted_analysis" / "1_no_adherence" / "Forest"
FIGS = OUT / "figures"
TABS = OUT / "tables"
for _d in [FIGS, TABS]:
    _d.mkdir(parents=True, exist_ok=True)

# -- Domain metadata -------------------------------------------------------
DOMAINS = {
    'phq9': {
        'label': 'Depression (PHQ-9, 0–27 scale)',
        'val': 'phq9_value',
        'spline_df': 3,
        'unit': 'points',
        'extra_cov': ['antidepressant_baseline'],
        'subgroups': [
            {'name': 'All patients', 'threshold': None,
             'color': '#7BAFD4', 'marker': 'o'},
            {'name': 'Baseline >=5 (mild+)', 'threshold': 5,
             'color': '#1A5C8A', 'marker': 's'},
            {'name': 'Baseline >=10 (mod-severe)', 'threshold': 10,
             'color': '#0A2340', 'marker': 'D'},
        ],
    },
    'pain_score': {
        'label': 'General Pain Intensity (0–10 scale)',
        'val': 'pain_score_value',
        'spline_df': 3,
        'unit': 'points',
        'subgroups': [
            {'name': 'All patients', 'threshold': None,
             'color': '#F5A673', 'marker': 'o'},
            {'name': 'Baseline >=4 (moderate+)', 'threshold': 4,
             'color': '#C0392B', 'marker': 's'},
            {'name': 'Baseline >=7 (severe)', 'threshold': 7,
             'color': '#7B0000', 'marker': 'D'},
        ],
    },
    'waist_circumference': {
        'label': 'Waist Circumference (Inches)',
        'val': 'waist_circumference_value',
        'spline_df': 3,
        'unit': 'inches',
        'subgroups': [
            {'name': 'All patients', 'threshold': None,
             'color': '#5DAD6F', 'marker': 'o'},
        ],
    },
    'alcohol': {
        'label': 'Alcohol Use (Drinks/day)',
        'val': 'alcohol_value',
        'spline_df': 3,
        'unit': 'drinks/day',
        'subgroups': [
            {'name': 'All patients', 'threshold': None,
             'color': '#C39BD3', 'marker': 'o'},
            {'name': 'Baseline >=8.6 (top 25%)', 'threshold': 8.6,
             'color': '#6C3483', 'marker': 's'},
        ],
    },
    'muscle_strength': {
        'label': 'Muscle Strength (MRC, 0–5 scale)',
        'val': 'muscle_strength_value',
        'spline_df': 3,
        'unit': 'score',
        'subgroups': [
            {'name': 'All patients', 'threshold': None,
             'color': '#27AE60', 'marker': 'o'},
        ],
    },
}

TIMEPOINTS = [
    (91,  '3 mo'),
    (183, '6 mo'),
    (274, '9 mo'),
    (365, '12 mo'),
]

BASELINE_WINDOW = 30
FOLLOWUP_START  = 30
FOLLOWUP_END    = 365

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
    'axes.spines.top': False, 'axes.spines.right': False,
})

# Demographic covariates added to all GEE models
BASE_COV_COLS = [
    'age', 'gender', 'race',
    'baseline_a1c_category', 'baseline_bmi_final_category',
]
BASE_COV_FORMULA = (
    "age + C(gender) + C(race) + "
    "C(baseline_a1c_category) + C(baseline_bmi_final_category)"
)


def get_ref_covars(df):
    """Return reference covariate values (mean age, modal categories) for prediction."""
    ref = {}
    if 'age' in df.columns:
        ref['age'] = float(df['age'].mean())
    for c in ['gender', 'race', 'baseline_a1c_category', 'baseline_bmi_final_category']:
        if c in df.columns and not df[c].dropna().empty:
            ref[c] = df[c].mode().iloc[0]
    return ref


# ===========================================================================
# GEE fitting and prediction
# ===========================================================================

def fit_gee(df, outcome_col, spline_df=3, extra_cov=None):
    """Fit GEE with B-spline and demographic covariates.

    Returns (result, design_info, d_min, d_max, ref_covars).
    extra_cov: list of additional covariate column names (e.g. antidepressant_baseline).
    """
    extra_cov = extra_cov or []

    df = df.dropna(subset=[outcome_col, 'days_from_baseline']).copy()
    # Clean categoricals
    for c in ['gender', 'race', 'baseline_a1c_category', 'baseline_bmi_final_category']:
        if c in df.columns:
            df[c] = df[c].fillna('Unknown').astype(str)
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(df['age'].median())

    n_pts = df.patient_id.nunique()
    if n_pts < 15 or len(df) < 30:
        return None, None, None, None, None

    d_min = float(df.days_from_baseline.min())
    d_max = float(df.days_from_baseline.max())
    ref_covars = get_ref_covars(df)

    cov_str = BASE_COV_FORMULA
    xtra = [c for c in extra_cov if c in df.columns]
    if xtra:
        cov_str += ' + ' + ' + '.join(xtra)
        for c in xtra:
            ref_covars[c] = df[c].mode().iloc[0] if df[c].dtype == object else float(df[c].mean())

    formula = (f"{outcome_col} ~ bs(days_from_baseline, df={spline_df}, include_intercept=False)"
               + (f" + {cov_str}" if cov_str else ""))
    try:
        y, X = dmatrices(formula, df, return_type='dataframe')
    except Exception as e:
        log.error("  dmatrices failed: %s", e)
        return None, None, None, None, None

    ids = df.loc[y.index, 'patient_id']
    try:
        result = GEE(y, X, groups=ids,
                     family=Gaussian(), cov_struct=Independence()).fit()
        return result, X.design_info, d_min, d_max, ref_covars
    except Exception as e:
        log.error("  GEE failed: %s", e)
        return None, None, None, None, None


def predict_at(result, design_info, days, d_min, d_max, ref_covars=None):
    """Predict at specific days (clamped to data range). Returns (mean, se).

    ref_covars: dict of covariate name → reference value for prediction.
    """
    days_arr = np.array(days, dtype=float)
    mean_v = np.full(len(days_arr), np.nan)
    se_v   = np.full(len(days_arr), np.nan)

    mask = (days_arr >= d_min - 3) & (days_arr <= d_max + 3)
    if not mask.any():
        return mean_v, se_v

    clamped = np.clip(days_arr[mask], d_min, d_max)
    pred_df = pd.DataFrame({'days_from_baseline': clamped})
    if ref_covars:
        for col, val in ref_covars.items():
            pred_df[col] = val
    try:
        X_pred = np.asarray(build_design_matrices([design_info], pred_df)[0])
        mv     = np.asarray(result.predict(X_pred))
        V      = np.asarray(result.cov_params())
        var    = np.clip(np.einsum('ij,jk,ik->i', X_pred, V, X_pred), 0, None)
        mean_v[mask] = mv
        se_v[mask]   = np.sqrt(var)
    except Exception as e:
        log.error("  predict_at failed: %s", e)

    return mean_v, se_v


# ===========================================================================
# ITS estimate: delta = y-hat(t) - y-hat(0)  (patients with both pre + post)
# ===========================================================================

def compute_its_estimates(df_raw, val_col, spline_df, subgroup_name, extra_cov=None):
    """ITS estimates for one domain+subgroup."""
    extra_cov = extra_cov or []
    df = df_raw.copy()
    df['days_from_baseline'] = pd.to_numeric(df['days_from_baseline'], errors='coerce')

    # Restrict to patients with both pre and post
    if 'has_both_periods' in df.columns:
        df = df[df['has_both_periods'] == 1]
    else:
        pre  = set(df[df['post'] == 0]['patient_id'])
        post = set(df[(df['post'] == 1) & (df['days_from_baseline'] <= 365)]['patient_id'])
        both = pre & post
        df = df[df['patient_id'].isin(both)]

    n_pts = df.patient_id.nunique()
    if n_pts < 15:
        log.warning("  ITS %s: too few patients (%d)", subgroup_name, n_pts)
        return []

    log.info("  ITS %s: %d patients, %d obs", subgroup_name, n_pts, len(df))

    result, design_info, d_min, d_max, ref = fit_gee(df, val_col, spline_df,
                                                      extra_cov=extra_cov)
    if result is None:
        return []

    all_days = [0.0] + [float(d) for d, _ in TIMEPOINTS]
    mean_v, se_v = predict_at(result, design_info, all_days, d_min, d_max, ref_covars=ref)

    y0 = mean_v[0]
    if np.isnan(y0):
        log.warning("  ITS %s: day-0 prediction unavailable", subgroup_name)
        return []

    rows = []
    for i, (day, label) in enumerate(TIMEPOINTS, start=1):
        if np.isnan(mean_v[i]):
            continue
        delta    = mean_v[i] - y0
        delta_se = np.sqrt(se_v[i]**2 + se_v[0]**2)
        rows.append({
            'analysis':  'ITS',
            'subgroup':  subgroup_name,
            'timepoint': label,
            'day':       day,
            'n':         n_pts,
            'estimate':  delta,
            'se':        delta_se,
            'ci_lo':     delta - Z_CRIT * delta_se,
            'ci_hi':     delta + Z_CRIT * delta_se,
        })
    return rows


# ===========================================================================
# CFB estimate: change from baseline directly
# ===========================================================================

def compute_cfb_estimates(df_raw, val_col, spline_df, subgroup_name, extra_cov=None):
    """CFB estimates for one domain+subgroup."""
    extra_cov = extra_cov or []
    df = df_raw.copy()
    df['days_from_baseline'] = pd.to_numeric(df['days_from_baseline'], errors='coerce')
    df = df.dropna(subset=[val_col, 'days_from_baseline'])

    # Baseline: mean score within +/-30 days
    base_mask = df['days_from_baseline'].between(-BASELINE_WINDOW, BASELINE_WINDOW)
    base_df = (df[base_mask]
               .groupby('patient_id')[val_col]
               .mean()
               .rename('baseline_score')
               .reset_index())

    # Follow-up — include demographic cols so GEE can use them
    demo_cols_avail = [c for c in BASE_COV_COLS + extra_cov if c in df.columns]
    fu_mask = df['days_from_baseline'].between(FOLLOWUP_START, FOLLOWUP_END)
    fu_df = df[fu_mask][['patient_id', 'days_from_baseline', val_col]
                        + demo_cols_avail].copy()
    fu_df = fu_df[fu_df.patient_id.isin(set(base_df.patient_id))]
    fu_df = fu_df.merge(base_df, on='patient_id', how='inner')
    fu_df['change_score'] = fu_df[val_col] - fu_df['baseline_score']

    n_pts = fu_df.patient_id.nunique()
    if n_pts < 15:
        log.warning("  CFB %s: too few patients (%d)", subgroup_name, n_pts)
        return []

    log.info("  CFB %s: %d patients, %d obs", subgroup_name, n_pts, len(fu_df))

    # Day-0 anchor — carry demographics per patient
    if demo_cols_avail:
        pt_demos = (fu_df.drop_duplicates('patient_id')
                        .set_index('patient_id')[demo_cols_avail])
        anchor = pd.DataFrame({
            'patient_id':         fu_df['patient_id'].unique(),
            'days_from_baseline': 0.0,
            'change_score':       0.0,
        })
        anchor = anchor.join(pt_demos, on='patient_id')
    else:
        anchor = pd.DataFrame({
            'patient_id':         fu_df['patient_id'].unique(),
            'days_from_baseline': 0.0,
            'change_score':       0.0,
        })
    keep_cols = ['patient_id', 'days_from_baseline', 'change_score'] + demo_cols_avail
    gee_df = pd.concat([anchor, fu_df[keep_cols]], ignore_index=True)

    result, design_info, d_min, d_max, ref = fit_gee(gee_df, 'change_score', spline_df,
                                                      extra_cov=extra_cov)
    if result is None:
        return []

    # Predict at day 0 + each timepoint, then centre by subtracting day-0
    all_days = [0.0] + [float(d) for d, _ in TIMEPOINTS]
    mean_v, se_v = predict_at(result, design_info, all_days, d_min, d_max, ref_covars=ref)

    y0 = mean_v[0]
    if np.isnan(y0):
        log.warning("  CFB %s: day-0 prediction unavailable — using raw", subgroup_name)
        y0 = 0.0
        se0 = 0.0
    else:
        se0 = se_v[0]

    rows = []
    for i, (day, label) in enumerate(TIMEPOINTS):
        j = i + 1  # offset by 1 because index 0 is day-0
        if np.isnan(mean_v[j]):
            continue
        delta    = mean_v[j] - y0
        delta_se = np.sqrt(se_v[j]**2 + se0**2)
        rows.append({
            'analysis':  'CFB',
            'subgroup':  subgroup_name,
            'timepoint': label,
            'day':       day,
            'n':         n_pts,
            'estimate':  delta,
            'se':        delta_se,
            'ci_lo':     delta - Z_CRIT * delta_se,
            'ci_hi':     delta + Z_CRIT * delta_se,
        })
    return rows


# ===========================================================================
# Build all estimates for one domain (all subgroups)
# ===========================================================================

def process_domain(domain, meta):
    """For one domain, compute ITS + CFB estimates for all subgroups."""
    p = DATA / f"{domain}_prepared.csv"
    if not p.exists():
        log.warning("  %s: prepared CSV not found -- skipping", domain)
        return []

    df_full = pd.read_csv(p)
    df_full['days_from_baseline'] = pd.to_numeric(
        df_full['days_from_baseline'], errors='coerce')
    val = meta['val']
    df_full = df_full.dropna(subset=[val])

    all_rows = []

    for sg in meta['subgroups']:
        name  = sg['name']
        thr   = sg['threshold']

        if thr is not None:
            # Filter to patients whose baseline mean >= threshold
            base_mask = df_full['days_from_baseline'].between(
                -BASELINE_WINDOW, BASELINE_WINDOW)
            pat_means = (df_full[base_mask]
                         .groupby('patient_id')[val]
                         .mean())
            elev_pids = set(pat_means[pat_means >= thr].index)
            df_sg = df_full[df_full['patient_id'].isin(elev_pids)]
            log.info("  Subgroup '%s': %d patients (baseline >= %.1f)",
                     name, len(elev_pids), thr)
        else:
            df_sg = df_full

        its_rows = compute_its_estimates(df_sg, val, meta['spline_df'], name,
                                         extra_cov=meta.get('extra_cov', []))
        cfb_rows = compute_cfb_estimates(df_sg, val, meta['spline_df'], name,
                                         extra_cov=meta.get('extra_cov', []))
        all_rows.extend(its_rows)
        all_rows.extend(cfb_rows)

    return all_rows


# ===========================================================================
# Forest plot -- one per outcome
# ===========================================================================

def make_outcome_forest(domain, meta, est_df):
    """One forest plot per outcome: rows = timepoints, colored by subgroup.
    Two panels: ITS (left), CFB (right).
    """
    timepoint_labels = [t for _, t in TIMEPOINTS]
    subgroups = meta['subgroups']
    n_sg  = len(subgroups)
    n_tp  = len(timepoint_labels)

    fig_h = max(3.5, n_tp * (n_sg * 0.4 + 0.6) + 1.5)
    fig, axes = plt.subplots(1, 2, figsize=(14, fig_h), sharey=True)

    # Build y-positions: group by timepoint, within each offset by subgroup
    # Timepoints from top (12 mo) to bottom (3 mo) so 3 mo is at top visually
    y_gap = 0.4   # space between subgroups within a timepoint
    tp_gap = 1.3  # space between timepoint groups

    y_positions = {}  # (tp_label, sg_name) -> y
    y_tick_pos = {}   # tp_label -> centre y
    y_current = 0

    for tp_label in reversed(timepoint_labels):
        sg_ys = []
        for j, sg in enumerate(subgroups):
            y_positions[(tp_label, sg['name'])] = y_current
            sg_ys.append(y_current)
            y_current += y_gap
        y_tick_pos[tp_label] = np.mean(sg_ys)
        y_current += tp_gap - y_gap

    for ax, analysis in zip(axes, ['ITS', 'CFB']):
        sub = est_df[est_df.analysis == analysis]

        for _, row in sub.iterrows():
            key = (row.timepoint, row.subgroup)
            if key not in y_positions:
                continue
            yi = y_positions[key]
            sg_meta = next((s for s in subgroups if s['name'] == row.subgroup), None)
            if sg_meta is None:
                continue
            color  = sg_meta['color']
            marker = sg_meta['marker']

            ax.errorbar(row.estimate, yi,
                        xerr=[[row.estimate - row.ci_lo],
                              [row.ci_hi - row.estimate]],
                        fmt=marker, color=color, markersize=8,
                        capsize=4, capthick=1.3, linewidth=1.5, zorder=3)

            # Annotation
            sig = ' *' if (row.ci_hi < 0 or row.ci_lo > 0) else ''
            ann = f"{row.estimate:+.2f} [{row.ci_lo:+.2f}, {row.ci_hi:+.2f}]{sig}  n={row.n}"
            x_range = ax.get_xlim()
            x_off = abs(x_range[1] - x_range[0]) * 0.03 if x_range[1] != x_range[0] else 0.1
            ax.text(row.ci_hi + x_off, yi, ann,
                    va='center', fontsize=7.5, color='#333333')

        ax.axvline(0, color='#555555', linewidth=1.0, linestyle='--', alpha=0.7)
        ax.set_title(analysis, fontweight='bold', fontsize=12)
        ax.set_xlabel(f'Change ({meta["unit"]})')
        ax.grid(True, alpha=0.2, axis='x')

    # Y-axis: timepoint labels at centre of each group
    tp_ticks = [y_tick_pos[t] for t in reversed(timepoint_labels)]
    tp_labels_rev = list(reversed(timepoint_labels))
    axes[0].set_yticks(tp_ticks)
    axes[0].set_yticklabels(tp_labels_rev, fontsize=11, fontweight='bold')

    # Faint dividers between timepoint groups
    for ax in axes:
        for i in range(len(tp_labels_rev) - 1):
            tp_a = tp_labels_rev[i]
            tp_b = tp_labels_rev[i + 1]
            mid_y = (y_tick_pos[tp_a] + y_tick_pos[tp_b]) / 2
            ax.axhline(mid_y, color='#cccccc', linewidth=0.5, linestyle='-', alpha=0.5)

    # Re-adjust x-limits after annotation
    for ax in axes:
        ax.relim()
        ax.autoscale_view(scalex=True, scaley=False)

    # Legend for subgroups
    legend_handles = [
        Line2D([0], [0], marker=sg['marker'], color=sg['color'],
               label=sg['name'], markersize=8, linestyle='None')
        for sg in subgroups
    ]
    axes[1].legend(handles=legend_handles, loc='lower right', fontsize=8,
                   framealpha=0.9)

    fig.suptitle(f'{meta["label"]}: GEE Point Estimates Post-GLP-1',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()

    safe_name = domain.replace(' ', '_')
    for ext in ('png', 'pdf'):
        fig.savefig(FIGS / f"forest_{safe_name}.{ext}", bbox_inches='tight', dpi=300)
    plt.close(fig)
    log.info("Saved forest_%s.png/pdf", safe_name)


# ===========================================================================
# Main
# ===========================================================================

def main():
    grand_rows = []

    for domain, meta in DOMAINS.items():
        log.info("=== %s ===", meta['label'])
        rows = process_domain(domain, meta)
        if rows:
            for r in rows:
                r['domain'] = meta['label']
            grand_rows.extend(rows)

            est_df = pd.DataFrame(rows)
            make_outcome_forest(domain, meta, est_df)

    if not grand_rows:
        log.error("No estimates computed -- check data.")
        return

    # Save combined CSV
    df = pd.DataFrame(grand_rows)
    csv_path = TABS / "point_estimates_3_6_9_12mo.csv"
    df.to_csv(csv_path, index=False)
    log.info("Saved point estimates -> %s", csv_path)

    # Print summary
    print("\n" + "=" * 90)
    print("POINT ESTIMATES AT 3, 6, 9, 12 MONTHS")
    print("=" * 90)
    for analysis in ['ITS', 'CFB']:
        print(f"\n{'_' * 40} {analysis} {'_' * 40}")
        sub = df[df.analysis == analysis]
        if sub.empty:
            print("  (no data)")
            continue
        for sg_name in sub['subgroup'].unique():
            sg_sub = sub[sub['subgroup'] == sg_name]
            print(f"\n  {sg_name}:")
            for _, row in sg_sub.iterrows():
                sig = '*' if (row.ci_hi < 0 or row.ci_lo > 0) else ' '
                print(f"    {row.timepoint:>6s}  {row.estimate:+7.3f} [{row.ci_lo:+7.3f}, {row.ci_hi:+7.3f}] {sig}  n={row.n}")

    log.info("Done. %d total rows across %d domains.", len(df), df.domain.nunique())


if __name__ == '__main__':
    main()
