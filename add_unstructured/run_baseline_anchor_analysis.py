#!/usr/bin/env python3
"""Baseline-anchored change-from-baseline sensitivity analysis.

Instead of requiring a pre-GLP-1 observation, this anchors each patient's
starting score to their assessment nearest to GLP-1 initiation (within ±30 days).
Change from baseline is then tracked forward over the post-GLP-1 follow-up period.

Rationale:
  Many patients receive their first PHQ-9/pain assessment concurrent with starting
  GLP-1 (explaining the large post-only population in the ITS analysis). This
  approach captures those patients and asks: "Given where you started at GLP-1
  initiation, how did your score change over the following 12 months?"

Design:
  - Baseline window:  −30 to +30 days of GLP-1 start
  - Follow-up window: +30 to +365 days (months 1–12)
  - Outcome:          change_score = follow_up_score − baseline_score (per patient)
  - Model:            mixed-effects LM, change_score ~ time_months + (1|patient_id)
                      also GEE with B-spline for smooth trajectory
  - Elevated subgroup: baseline score ≥ primary threshold (same thresholds as ITS)

Output → output/add_unstructured/baseline_anchored/
"""

import logging, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from scipy import stats
from patsy import dmatrices, build_design_matrices, bs  # noqa: F401
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.cov_struct import Independence
import statsmodels.formula.api as smf

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

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT  = Path(__file__).resolve().parents[2]
_AU   = os.environ.get('AU_OUTROOT')
DATA  = Path(os.environ.get('AU_DATADIR', str(ROOT / "output" / "submitted_analysis" / "1_no_adherence" / "data")))
OUT   = (Path(_AU) / "CFB") if _AU else ROOT / "output" / "submitted_analysis" / "1_no_adherence" / "CFB"
FIGS  = OUT / "figures"
TABS  = OUT / "tables"
for d in [FIGS, TABS]:
    d.mkdir(parents=True, exist_ok=True)

# ── Domain config ──────────────────────────────────────────────────────────
DOMAIN_META = {
    'phq9': {
        'label':     'Depression (PHQ-9, 0–27 scale)',
        'ylabel':    'Change from baseline (PHQ-9 points)',
        'val':       'phq9_value',
        'elev_threshold': 5,
        'elev_label':     'Elevated baseline ≥5 (mild+)',
        'elev_threshold_2': 10,
        'elev_label_2':     'Elevated baseline ≥10 (mod-severe)',
        'color_all':   '#7BAFD4',
        'color_elev':  '#1A5C8A',
        'color_elev_2': '#0A2340',
        'spline_df': 3,
        'extra_cov': ['antidepressant_baseline', 'covid_era'],
    },
    'pain_score': {
        'label':     'General Pain Intensity (0–10 scale)',
        'ylabel':    'Change from baseline (pain points, 0–10)',
        'val':       'pain_score_value',
        'elev_threshold': 4,
        'elev_label':     'Elevated baseline ≥4 (moderate+)',
        'elev_threshold_2': 7,
        'elev_label_2':     'Elevated baseline ≥7 (severe)',
        'color_all':   '#F5A673',
        'color_elev':  '#C0392B',
        'color_elev_2': '#7B0000',
        'spline_df': 3,
        'extra_cov': ['covid_era'],
    },
    'waist_circumference': {
        'label':     'Waist Circumference (Inches)',
        'ylabel':    'Change from baseline (inches)',
        'val':       'waist_circumference_value',
        'elev_threshold': None,
        'color_all':   '#5DAD6F',
        'spline_df': 3,
    },
    'alcohol': {
        'label':     'Alcohol Use (Drinks/day)',
        'ylabel':    'Change from baseline (score)',
        'val':       'alcohol_value',
        'elev_threshold': None,
        'color_all':   '#9B59B6',
        'spline_df': 3,
        'extra_cov': ['covid_era'],
    },
    'muscle_strength': {
        'label':     'Muscle Strength (MRC, 0–5 scale)',
        'ylabel':    'Change from baseline (score)',
        'val':       'muscle_strength_value',
        'elev_threshold': None,
        'color_all':   '#27AE60',
        'spline_df': 3,
        'extra_cov': ['covid_era'],
    },
}

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'figure.dpi': 150, 'savefig.dpi': 600, 'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
    'axes.spines.top': False, 'axes.spines.right': False,
})

BASELINE_WINDOW = 30   # ±30 days around GLP-1 start
FOLLOWUP_START  = 30   # start tracking change after day +30
FOLLOWUP_END    = 365  # 12 months post

# Demographic covariates added to all models
BASE_COV_COLS = [
    'age', 'gender', 'race',
    'baseline_a1c_category', 'baseline_bmi_final_category',
]
BASE_COV_FORMULA = (
    "age + C(gender) + C(race) + "
    "C(baseline_a1c_category) + C(baseline_bmi_final_category)"
)


def get_ref_covars(df, extra_cov=None):
    """Return reference covariate values (mean age, modal categories) for prediction."""
    ref = {}
    if 'age' in df.columns:
        ref['age'] = float(df['age'].mean())
    for c in ['gender', 'race', 'baseline_a1c_category', 'baseline_bmi_final_category']:
        if c in df.columns and not df[c].dropna().empty:
            ref[c] = df[c].mode().iloc[0]
    for c in (extra_cov or []):
        if c in df.columns and not df[c].dropna().empty:
            ref[c] = df[c].mode().iloc[0]
    return ref


# ═══════════════════════════════════════════════════════════════════════════
# Data preparation
# ═══════════════════════════════════════════════════════════════════════════

def build_change_dataset(domain, meta):
    """Build change-from-baseline dataset for one domain.

    Returns a dataframe with columns:
      patient_id, days_from_baseline, time_months, change_score, baseline_score
    One row per follow-up observation per patient.
    """
    p = DATA / f"{domain}_prepared.csv"
    if not p.exists():
        log.warning("  %s: prepared CSV not found", domain)
        return pd.DataFrame()

    df = pd.read_csv(p)
    df['days_from_baseline'] = pd.to_numeric(df['days_from_baseline'],
                                             errors='coerce')
    val = meta['val']
    df = df.dropna(subset=[val, 'days_from_baseline'])

    # ── Step 1: identify baseline score per patient ──────────────────────
    base_mask = df['days_from_baseline'].between(-BASELINE_WINDOW, BASELINE_WINDOW)
    base_df = (df[base_mask]
               .groupby('patient_id')[val]
               .mean()
               .rename('baseline_score')
               .reset_index())

    log.info("  %s: %d patients with baseline score (±%d d)",
             domain, len(base_df), BASELINE_WINDOW)

    # ── Step 2: follow-up observations after FOLLOWUP_START ──────────────
    fu_mask = df['days_from_baseline'].between(FOLLOWUP_START, FOLLOWUP_END)
    extra_cov = meta.get('extra_cov', [])
    extra_cols = [c for c in extra_cov if c in df.columns]
    demo_cols = [c for c in BASE_COV_COLS if c in df.columns] + extra_cols
    fu_df = df[fu_mask][['patient_id', 'days_from_baseline', val] + demo_cols].copy()

    # Keep only patients who also have a baseline score
    fu_df = fu_df[fu_df.patient_id.isin(set(base_df.patient_id))]

    # ── Step 3: merge baseline score ──────────────────────────────────────
    fu_df = fu_df.merge(base_df, on='patient_id', how='inner')
    fu_df['change_score'] = fu_df[val] - fu_df['baseline_score']
    fu_df['time_months']  = fu_df['days_from_baseline'] / 30.44

    n_pts = fu_df.patient_id.nunique()
    log.info("  %s: %d patients with baseline + follow-up; %d obs",
             domain, n_pts, len(fu_df))

    if n_pts == 0:
        return pd.DataFrame()

    return fu_df, base_df


def filter_elevated(fu_df, base_df, threshold):
    """Keep patients whose baseline score >= threshold."""
    if threshold is None:
        return None, None
    elev_pids = set(base_df.loc[base_df.baseline_score >= threshold, 'patient_id'])
    fu_elev   = fu_df[fu_df.patient_id.isin(elev_pids)].copy()
    base_elev = base_df[base_df.patient_id.isin(elev_pids)].copy()
    log.info("    Elevated ≥%s: %d patients", threshold, len(elev_pids))
    return (fu_elev if len(elev_pids) > 0 else None,
            base_elev if len(elev_pids) > 0 else None)


# ═══════════════════════════════════════════════════════════════════════════
# Statistical tests
# ═══════════════════════════════════════════════════════════════════════════

def timepoint_tests(fu_df, time_windows):
    """Paired t-test at specified time windows (start_day, end_day, label)."""
    results = []
    for start, end, label in time_windows:
        window_obs = fu_df[fu_df.days_from_baseline.between(start, end)]
        per_pt = window_obs.groupby('patient_id')['change_score'].mean()
        n = len(per_pt)
        if n < 5:
            continue
        mean_ch = per_pt.mean()
        sem     = per_pt.sem()
        ci_lo   = mean_ch - Z_CRIT * sem
        ci_hi   = mean_ch + Z_CRIT * sem
        _, t_p  = stats.ttest_1samp(per_pt, 0)
        w_p     = stats.wilcoxon(per_pt).pvalue if n >= 10 else np.nan
        results.append({
            'timepoint': label,
            'n': n,
            'mean_change': mean_ch,
            'ci_lo': ci_lo,
            'ci_hi': ci_hi,
            'ttest_p': t_p,
            'wilcoxon_p': w_p,
        })
    return pd.DataFrame(results)


def fit_lmm(fu_df, extra_cov=None):
    """Mixed-effects model: change_score ~ time_months + demographics + (1|patient_id)."""
    df = fu_df.dropna(subset=['change_score', 'time_months']).copy()
    n_pts = df.patient_id.nunique()
    if n_pts < 20 or len(df) < 40:
        return None
    demo_cols = [c for c in BASE_COV_COLS if c in df.columns]
    cov_terms = BASE_COV_FORMULA if demo_cols else ''
    xtra = ' + '.join(extra_cov or [])
    if xtra:
        cov_terms = (cov_terms + ' + ' + xtra) if cov_terms else xtra
    formula = "change_score ~ time_months" + (f" + {cov_terms}" if cov_terms else "")
    try:
        model = smf.mixedlm(formula, df, groups=df['patient_id'])
        return model.fit(reml=True)
    except Exception as e:
        log.warning("  LMM failed: %s", e)
        return None


def fit_gee_spline(fu_df, spline_df=3, extra_cov=None):
    """GEE with B-spline on time → smooth change-from-baseline trajectory.

    Returns (result, design_info, d_min, d_max, ref_covars).
    """
    extra_cov = extra_cov or []
    df = fu_df.dropna(subset=['change_score', 'days_from_baseline']).copy()

    # Clean covariates
    for c in ['gender', 'race', 'baseline_a1c_category', 'baseline_bmi_final_category']:
        if c in df.columns:
            df[c] = df[c].fillna('Unknown').astype(str)
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(df['age'].median())

    # Inject day-0 anchor: change_score = 0 by definition at baseline.
    # Demographics come from each patient's follow-up rows (time-invariant).
    demo_cols = [c for c in BASE_COV_COLS + extra_cov if c in df.columns]
    if demo_cols:
        pt_demos = (df.drop_duplicates('patient_id')
                       .set_index('patient_id')[demo_cols])
        anchor = pd.DataFrame({
            'patient_id':         df['patient_id'].unique(),
            'days_from_baseline': 0.0,
            'change_score':       0.0,
        })
        anchor = anchor.join(pt_demos, on='patient_id')
    else:
        anchor = pd.DataFrame({
            'patient_id':         df['patient_id'].unique(),
            'days_from_baseline': 0.0,
            'change_score':       0.0,
        })
    df = pd.concat([anchor, df], ignore_index=True).sort_values('days_from_baseline')

    n_pts = df.patient_id.nunique()
    if n_pts < 15 or len(df) < 30:
        return None, None, None, None, None

    ref_covars = get_ref_covars(df, extra_cov=extra_cov)

    cov_str = " + ".join(
        [BASE_COV_FORMULA] + [c for c in extra_cov if c in df.columns])
    formula = (f"change_score ~ bs(days_from_baseline, df={spline_df}, "
               f"include_intercept=False)"
               + (f" + {cov_str}" if cov_str else ""))
    try:
        y, X = dmatrices(formula, df, return_type='dataframe')
    except Exception as e:
        log.error("  dmatrices failed: %s", e)
        return None, None, None, None, None

    ids   = df.loc[y.index, 'patient_id']
    d_min = float(df.days_from_baseline.min())
    d_max = float(df.days_from_baseline.max())

    try:
        result = GEE(y, X, groups=ids,
                     family=Gaussian(), cov_struct=Independence()).fit()
        log.info("    GEE (%d obs, %d pts, df=%d)", len(df), n_pts, spline_df)
        return result, X.design_info, d_min, d_max, ref_covars
    except Exception as e:
        log.error("  GEE failed: %s", e)
        return None, None, None, None, None


def predict_gee(result, design_info, days_grid, d_min, d_max, ref_covars=None):
    """Generate GEE predictions + 95% CI, clipped to data support.

    ref_covars: dict of covariate name → reference value for prediction.
    """
    out_mean = np.full(len(days_grid), np.nan)
    out_lo   = np.full(len(days_grid), np.nan)
    out_hi   = np.full(len(days_grid), np.nan)

    eps  = 0.5
    mask = (days_grid >= d_min) & (days_grid <= d_max - eps)
    if not mask.any():
        return out_mean, out_lo, out_hi

    pred_df = pd.DataFrame({'days_from_baseline': days_grid[mask]})
    if ref_covars:
        for col, val in ref_covars.items():
            pred_df[col] = val
    try:
        X_pred  = np.asarray(build_design_matrices([design_info], pred_df)[0])
        mean_v  = np.asarray(result.predict(X_pred))
        V       = np.asarray(result.cov_params())
        var     = np.clip(np.einsum('ij,jk,ik->i', X_pred, V, X_pred), 0, None)
        se      = np.sqrt(var)
        out_mean[mask] = mean_v
        out_lo[mask]   = mean_v - Z_CRIT * se
        out_hi[mask]   = mean_v + Z_CRIT * se
    except Exception as e:
        log.error("  GEE predict failed: %s", e)

    return out_mean, out_lo, out_hi


def support_mask(fu_df, days_grid, window_days=30, min_pts=10):
    """Suppress CI where fewer than min_pts patients have nearby data."""
    mask = np.zeros(len(days_grid), dtype=bool)
    for i, d in enumerate(days_grid):
        n = fu_df[np.abs(fu_df.days_from_baseline - d) <= window_days
                  ].patient_id.nunique()
        mask[i] = n >= min_pts
    return mask


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

TIME_WINDOWS = [
    ( 30,  90,  "1–3 mo"),
    ( 91, 180,  "3–6 mo"),
    (181, 270,  "6–9 mo"),
    (271, 365,  "9–12 mo"),
]

DAYS_GRID = np.arange(0, FOLLOWUP_END + 1, 7)  # start from day 0 (change = 0 by definition at baseline)


def plot_domain(domain, meta, cohorts, fname_stem=None):
    """One plot per domain — smooth GEE change trajectory + binned monthly means.

    cohorts: list of (label, color, fu_df) tuples
    fname_stem: override output filename stem (default: cfb_{domain})
    """
    n_valid = sum(1 for _, _, df in cohorts if df is not None and
                  not df.empty and df.patient_id.nunique() >= 15)
    if n_valid == 0:
        log.warning("  %s: no cohort passes minimum n — skip", domain)
        return

    spline_df = meta['spline_df']
    extra_cov = meta.get('extra_cov', [])

    fig, ax = plt.subplots(1, 1, figsize=(11, 6))
    handles = []
    any_patchy = False

    for clabel, color, fu_df in cohorts:
        if fu_df is None or fu_df.empty or fu_df.patient_id.nunique() < 15:
            continue
        n_pts = fu_df.patient_id.nunique()

        # ── GEE smooth curve ──────────────────────────────────────────────
        result, design_info, d_min, d_max, ref = fit_gee_spline(fu_df, spline_df,
                                                                 extra_cov=extra_cov)
        if result is not None:
            mean, lo, hi = predict_gee(result, design_info, DAYS_GRID, d_min, d_max,
                                       ref_covars=ref)
            smask = support_mask(fu_df, DAYS_GRID, min_pts=10)
            # Centre at time 0: subtract model-predicted value at day 0
            if DAYS_GRID[0] == 0:
                m0 = mean[0] if not np.isnan(mean[0]) else 0.0
                mean, lo, hi = mean - m0, lo - m0, hi - m0
                smask[0] = True
            if not smask.all():
                any_patchy = True
            # Apply mask only to CI band; main line stays continuous within data range
            m_lo   = np.where(smask, lo,   np.nan)
            m_hi   = np.where(smask, hi,   np.nan)
            ax.fill_between(DAYS_GRID, m_lo, m_hi, alpha=0.15, color=color)
            line, = ax.plot(DAYS_GRID, mean, '-', color=color,
                            linewidth=2.5, label=f'{clabel} (n={n_pts:,})', zorder=3)
        else:
            # Fallback: 30-day-binned observed means
            fu_df = fu_df.copy()
            fu_df['day_bin'] = (fu_df.days_from_baseline // 30).astype(int) * 30
            grp   = fu_df.groupby('day_bin')['change_score']
            means = grp.mean()
            sems  = grp.sem().fillna(0)
            ax.fill_between(means.index, means - Z_CRIT*sems,
                            means + Z_CRIT*sems, alpha=0.15, color=color)
            line, = ax.plot(means.index, means, 'o-', color=color,
                            linewidth=2, label=f'{clabel} (n={n_pts:,}, binned)',
                            zorder=3)
        handles.append(line)

    # ── Reference lines ───────────────────────────────────────────────────
    ax.axhline(0, color='#888888', linewidth=1.0, linestyle=':',
               label='No change from baseline')
    ax.axvline(30, color='#999999', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.text(32, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.1,
            'Day 30', fontsize=8, color='#777777', va='top')

    ax.set_xlim(-10, FOLLOWUP_END + 10)
    ax.set_ylabel(meta['ylabel'])
    ax.set_xlabel('Days from baseline')
    ax.xaxis.set_major_locator(mticker.MultipleLocator(60))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: '0' if x == 0 else (f'+{int(x)}' if x > 0 else str(int(x)))))
    ax.set_title(
        f'GLP-1 and {meta["label"]} — Change from Baseline\n'
        f'(Baseline = score at GLP-1 start ±30 days; GEE spline smooth)',
        pad=10)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.92)
    ax.grid(True, alpha=0.2, which='major')

    if any_patchy:
        fig.text(0.01, -0.02,
                 '† CI band not shown where <10 patients have data within ±30 days.',
                 fontsize=7.5, color='#666666', ha='left', va='top')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(FIGS / f"{fname_stem or f'cfb_{domain}'}.{ext}")
    plt.close(fig)
    log.info("  Saved cfb_%s.png/pdf", domain)


def plot_timepoint_summary(all_results):
    """Forest-style plot of mean change at each time window, all domains."""
    rows = []
    for domain, label, color, tp_df in all_results:
        for _, row in tp_df.iterrows():
            rows.append({
                'domain': label,
                'color': color,
                'timepoint': row['timepoint'],
                'n': row['n'],
                'mean_change': row['mean_change'],
                'ci_lo': row['ci_lo'],
                'ci_hi': row['ci_hi'],
                'p': row['wilcoxon_p'],
            })
    if not rows:
        return
    df = pd.DataFrame(rows)

    domains = df['domain'].unique()
    tps     = df['timepoint'].unique()
    colors  = {d: df.loc[df.domain == d, 'color'].iloc[0] for d in domains}
    n_tp    = len(tps)
    n_dom   = len(domains)

    fig, axes = plt.subplots(1, n_tp, figsize=(3.5 * n_tp, max(3, n_dom * 0.8 + 1)),
                             sharey=True)
    if n_tp == 1:
        axes = [axes]

    for ax, tp in zip(axes, tps):
        sub = df[df.timepoint == tp].copy()
        y_pos = range(len(sub) - 1, -1, -1)
        for i, (_, row) in zip(y_pos, sub.iterrows()):
            color = colors[row['domain']]
            ax.barh(i, row['mean_change'], xerr=0, color=color, alpha=0.0)
            ax.errorbar(row['mean_change'], i,
                        xerr=[[row['mean_change'] - row['ci_lo']],
                              [row['ci_hi'] - row['mean_change']]],
                        fmt='o', color=color, markersize=7,
                        capsize=4, capthick=1.5, linewidth=1.5)
            # Significance annotation
            p = row['p']
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else
                  ('*' if p < 0.05 else ''))
            ann = f"n={row['n']}{' '+sig if sig else ''}"
            ax.text(row['ci_hi'], i, f'  {ann}', va='center', fontsize=7.5,
                    color='#333333')
        ax.axvline(0, color='#555555', linewidth=1.0, linestyle='--')
        ax.set_xlabel('Mean change from baseline')
        ax.set_title(tp, fontweight='bold')
        ax.set_yticks(list(range(len(sub) - 1, -1, -1)))
        ax.set_yticklabels([r.domain for _, r in sub.iterrows()], fontsize=9)
        ax.grid(True, alpha=0.2, axis='x')

    fig.suptitle('Change from GLP-1 Baseline by Domain and Timepoint\n'
                 '(mean ± 95% CI; *p<0.05 **p<0.01 ***p<0.001 vs. zero change)',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(FIGS / f"cfb_timepoint_summary.{ext}", bbox_inches='tight')
    plt.close(fig)
    log.info("Saved cfb_timepoint_summary.png/pdf")


def plot_panel(all_domain_plots):
    """Single panel figure — all domains stacked."""
    valid = [(d, m, clist) for d, m, clist in all_domain_plots
             if any(df is not None and not df.empty and
                    df.patient_id.nunique() >= 15
                    for _, _, df in clist)]
    n = len(valid)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(11, 3.8 * n), sharex=True)
    if n == 1:
        axes = [axes]
    any_patchy = False
    for ax, (domain, meta, cohorts) in zip(axes, valid):
        spline_df = meta['spline_df']
        extra_cov = meta.get('extra_cov', [])
        for clabel, color, fu_df in cohorts:
            if fu_df is None or fu_df.empty or fu_df.patient_id.nunique() < 15:
                continue
            n_pts = fu_df.patient_id.nunique()
            result, design_info, d_min, d_max, ref = fit_gee_spline(fu_df, spline_df,
                                                                     extra_cov=extra_cov)
            if result is None:
                continue
            mean, lo, hi = predict_gee(result, design_info, DAYS_GRID, d_min, d_max,
                                       ref_covars=ref)
            smask = support_mask(fu_df, DAYS_GRID, min_pts=10)
            # Centre at time 0: subtract model-predicted value at day 0
            if DAYS_GRID[0] == 0:
                m0 = mean[0] if not np.isnan(mean[0]) else 0.0
                mean, lo, hi = mean - m0, lo - m0, hi - m0
                smask[0] = True
            if not smask.all():
                any_patchy = True
            # Apply mask only to CI band; main line stays continuous
            m_lo   = np.where(smask, lo,   np.nan)
            m_hi   = np.where(smask, hi,   np.nan)
            ax.fill_between(DAYS_GRID, m_lo, m_hi, alpha=0.15, color=color)
            ax.plot(DAYS_GRID, mean, '-', color=color, linewidth=2.0,
                    label=f'{clabel} (n={n_pts:,})')

        ax.axhline(0, color='#888888', linewidth=1.0, linestyle=':')
        ax.set_ylabel(meta['ylabel'], fontsize=9)
        ax.set_title(meta['label'], fontsize=10, fontweight='bold', pad=4)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel('Days from baseline')
    axes[-1].xaxis.set_major_locator(mticker.MultipleLocator(60))
    axes[-1].xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: '0' if x == 0 else (f'+{int(x)}' if x > 0 else str(int(x)))))

    fig.suptitle('GLP-1 and Patient-Reported Outcomes: Change from Baseline',
                 fontsize=13, fontweight='bold', y=1.01)
    footnote = (
        'GEE-estimated change from baseline (score at GLP-1 initiation ±30 days) '
        'over 12 months, adjusted for age, sex, race, baseline HbA1c, and BMI. '
        'Shaded bands = 95% CI'
        + (' († suppressed where n<10).' if any_patchy else '.')
    )
    fig.text(0.01, -0.02, footnote, fontsize=8, color='#444444',
             ha='left', va='top', wrap=True)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(FIGS / f"cfb_panel_all_domains.{ext}", bbox_inches='tight')
    plt.close(fig)
    log.info("Saved cfb_panel_all_domains.png/pdf")


def plot_panel_wide(all_domain_plots):
    """Wide panel figure — 3 panels on the left, 2 on the right, all square."""
    valid = [(d, m, clist) for d, m, clist in all_domain_plots
             if any(df is not None and not df.empty and
                    df.patient_id.nunique() >= 15
                    for _, _, df in clist)]
    n = len(valid)
    if n == 0:
        return

    # Layout: 3 rows × 2 cols; left column gets rows 0-2, right gets rows 0-1
    n_rows = 3
    sq = 4.5  # side length of each square panel
    fig = plt.figure(figsize=(sq * 2 + 1.2, sq * n_rows + 1.0))
    gs = fig.add_gridspec(n_rows, 2, wspace=0.35, hspace=0.35)

    # Map panels: first 3 → left column, next 2 → right column
    positions = []
    for r in range(min(n, 3)):
        positions.append((r, 0))
    for r in range(max(0, n - 3)):
        positions.append((r, 1))

    any_patchy = False
    axes = []
    for idx, (domain, meta, cohorts) in enumerate(valid):
        row, col = positions[idx]
        ax = fig.add_subplot(gs[row, col])
        axes.append(ax)
        ax.set_aspect('auto')
        ax.set_box_aspect(1)  # force square

        spline_df = meta['spline_df']
        extra_cov = meta.get('extra_cov', [])
        for clabel, color, fu_df in cohorts:
            if fu_df is None or fu_df.empty or fu_df.patient_id.nunique() < 15:
                continue
            n_pts = fu_df.patient_id.nunique()
            result, design_info, d_min, d_max, ref = fit_gee_spline(
                fu_df, spline_df, extra_cov=extra_cov)
            if result is None:
                continue
            mean, lo, hi = predict_gee(result, design_info, DAYS_GRID,
                                       d_min, d_max, ref_covars=ref)
            smask = support_mask(fu_df, DAYS_GRID, min_pts=10)
            if DAYS_GRID[0] == 0:
                m0 = mean[0] if not np.isnan(mean[0]) else 0.0
                mean, lo, hi = mean - m0, lo - m0, hi - m0
                smask[0] = True
            if not smask.all():
                any_patchy = True
            m_lo = np.where(smask, lo, np.nan)
            m_hi = np.where(smask, hi, np.nan)
            ax.fill_between(DAYS_GRID, m_lo, m_hi, alpha=0.15, color=color)
            ax.plot(DAYS_GRID, mean, '-', color=color, linewidth=2.0,
                    label=f'{clabel} (n={n_pts:,})')

        ax.axhline(0, color='#888888', linewidth=1.0, linestyle=':')
        ax.set_ylabel(meta['ylabel'], fontsize=8)
        ax.set_title(meta['label'], fontsize=9, fontweight='bold', pad=4)
        ax.legend(fontsize=6.5, loc='best')
        ax.grid(True, alpha=0.2)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(60))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: '0' if x == 0 else (f'+{int(x)}' if x > 0 else str(int(x)))))
        ax.set_xlabel('Days from baseline', fontsize=8)

    fig.suptitle('GLP-1 and Patient-Reported Outcomes: Change from Baseline',
                 fontsize=13, fontweight='bold', y=1.01)
    footnote = (
        'GEE-estimated change from baseline (score at GLP-1 initiation ±30 days) '
        'over 12 months, adjusted for age, sex, race, baseline HbA1c, and BMI. '
        'Shaded bands = 95% CI'
        + (' († suppressed where n<10).' if any_patchy else '.')
    )
    fig.text(0.01, -0.01, footnote, fontsize=8, color='#444444',
             ha='left', va='top', wrap=True)
    for ext in ('png', 'pdf'):
        fig.savefig(FIGS / f"cfb_panel_all_domains_wide.{ext}", bbox_inches='tight')
    plt.close(fig)
    log.info("Saved cfb_panel_all_domains_wide.png/pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def _pstar(p):
    if pd.isna(p):
        return ''
    return '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))


def main():
    report_lines = [
        "# GLP-1 — Baseline-Anchored Change-from-Baseline Analysis",
        "",
        "## Design",
        f"- **Baseline window**: ±{BASELINE_WINDOW} days of GLP-1 start date",
        f"- **Follow-up window**: +{FOLLOWUP_START} to +{FOLLOWUP_END} days",
        "- **Outcome**: change_score = follow-up score − patient's baseline score",
        "- **Model**: mixed-effects LM (change_score ~ time_months) + ",
        "  GEE B-spline smooth trajectory",
        "- **Elevated subgroup**: baseline score ≥ primary threshold",
        "",
        "This approach uses patients whose **first assessment was concurrent with",
        "GLP-1 initiation** — not captured by the ITS analysis (which requires",
        "pre-period data). All analysis is *prospective forward from baseline*.",
        "",
        "## Sample Sizes",
        "",
        "| Domain | Baseline-anchored (+follow-up) | ITS (both pre+post) | New patients |",
        "|--------|-------------------------------|--------------------|----|",
    ]

    all_timepoint_results = []
    all_domain_plots      = []

    for domain, meta in DOMAIN_META.items():
        log.info("═══ %s ═══", meta['label'])

        result = build_change_dataset(domain, meta)
        if not result or (isinstance(result, pd.DataFrame) and result.empty):
            log.warning("  No data.")
            continue
        fu_df, base_df = result

        n_ba = fu_df.patient_id.nunique()

        # Load ITS both-period count for comparison
        try:
            df_raw = pd.read_csv(DATA / f"{domain}_prepared.csv")
            n_its = df_raw[df_raw.has_both_periods == 1].patient_id.nunique()
        except Exception:
            n_its = 0

        # Patients unique to BA (not in ITS)
        its_pids = (set(pd.read_csv(DATA / f"{domain}_prepared.csv")
                        .query("has_both_periods==1").patient_id)
                    if (DATA / f"{domain}_prepared.csv").exists() else set())
        n_new = len(set(fu_df.patient_id) - its_pids)

        report_lines.append(
            f"| {meta['label']} | {n_ba} | {n_its} | +{n_new} |")

        # ── Sub-cohorts ───────────────────────────────────────────────────
        cohorts = [(f"All (n={n_ba})", meta['color_all'], fu_df)]

        fu_e1, _ = filter_elevated(fu_df, base_df,
                                   meta.get('elev_threshold'))
        if fu_e1 is not None and fu_e1.patient_id.nunique() >= 15:
            n_e1 = fu_e1.patient_id.nunique()
            cohorts.append((meta.get('elev_label', f'Elev≥{meta["elev_threshold"]}'),
                            meta['color_elev'], fu_e1))
            log.info("  Elevated ≥%s: %d pts", meta.get('elev_threshold'), n_e1)

        fu_e2, _ = filter_elevated(fu_df, base_df,
                                   meta.get('elev_threshold_2'))
        if fu_e2 is not None and fu_e2.patient_id.nunique() >= 15:
            n_e2 = fu_e2.patient_id.nunique()
            cohorts.append((meta.get('elev_label_2', f'Elev≥{meta["elev_threshold_2"]}'),
                            meta['color_elev_2'], fu_e2))
            log.info("  Elevated ≥%s: %d pts", meta.get('elev_threshold_2'), n_e2)

        # ── Timepoint tests ───────────────────────────────────────────────
        log.info("  Timepoint tests (all patients)…")
        tp_df = timepoint_tests(fu_df, TIME_WINDOWS)
        if not tp_df.empty:
            all_timepoint_results.append(
                (domain, meta['label'], meta['color_all'], tp_df))

        # ── LMM trend ─────────────────────────────────────────────────────
        log.info("  LMM: change_score ~ time_months…")
        lmm = fit_lmm(fu_df, extra_cov=meta.get('extra_cov', []))

        # ── Plots ─────────────────────────────────────────────────────────
        plot_domain(domain, meta, cohorts)
        all_domain_plots.append((domain, meta, cohorts))

        # Per-subgroup CFB plots (separated so CIs are clearly visible)
        if fu_e1 is not None and fu_e1.patient_id.nunique() >= 15:
            thr1 = meta.get('elev_threshold')
            if thr1 is not None:
                tag = f"gte{int(thr1)}"
                plot_domain(domain, meta,
                            [(meta.get('elev_label', f'Elev\u2265{thr1}'),
                              meta['color_elev'], fu_e1)],
                            fname_stem=f"cfb_{domain}_{tag}")
        if fu_e2 is not None and fu_e2.patient_id.nunique() >= 15:
            thr2 = meta.get('elev_threshold_2')
            if thr2 is not None:
                tag = f"gte{int(thr2)}"
                plot_domain(domain, meta,
                            [(meta.get('elev_label_2', f'Elev\u2265{thr2}'),
                              meta['color_elev_2'], fu_e2)],
                            fname_stem=f"cfb_{domain}_{tag}")

        # ── Summary numbers for report ────────────────────────────────────
        report_lines.append("")
        report_lines.append(f"### {meta['label']}")
        report_lines.append("")

        if not tp_df.empty:
            report_lines.append(
                "**Mean change from baseline (all patients with follow-up):**")
            report_lines.append("")
            report_lines.append(
                "| Timepoint | n | Mean Δ [95% CI] | p (Wilcoxon) |")
            report_lines.append(
                "|-----------|---|-----------------|--------------|")
            for _, row in tp_df.iterrows():
                p_str = f"{row['wilcoxon_p']:.4f}{_pstar(row['wilcoxon_p'])}"
                report_lines.append(
                    f"| {row['timepoint']} | {row['n']} | "
                    f"{row['mean_change']:.3f} [{row['ci_lo']:.3f}, "
                    f"{row['ci_hi']:.3f}] | {p_str} |")
            report_lines.append("")

        if lmm is not None:
            slope = lmm.params.get('time_months', np.nan)
            slope_p = lmm.pvalues.get('time_months', np.nan)
            report_lines.append(
                f"**LMM slope**: {slope:.4f} points/month "
                f"(p={slope_p:.4f}{_pstar(slope_p)})")
            report_lines.append(
                "*Negative slope = scores declining (improving) over time*")
            report_lines.append("")

        # Elevated subgroup tables
        for elev_key, elev_label_key in [('elev_threshold', 'elev_label'),
                                          ('elev_threshold_2', 'elev_label_2')]:
            thr = meta.get(elev_key)
            if thr is None:
                continue
            fu_e, _ = filter_elevated(fu_df, base_df, thr)
            if fu_e is None or fu_e.patient_id.nunique() < 10:
                continue
            n_e = fu_e.patient_id.nunique()
            elev_tp = timepoint_tests(fu_e, TIME_WINDOWS)
            elev_lmm = fit_lmm(fu_e, extra_cov=meta.get('extra_cov', []))
            lbl = meta.get(elev_label_key, f'≥{thr}')
            report_lines.append(f"**Elevated: {lbl} (n={n_e})**")
            report_lines.append("")
            if not elev_tp.empty:
                report_lines.append(
                    "| Timepoint | n | Mean Δ [95% CI] | p (Wilcoxon) |")
                report_lines.append(
                    "|-----------|---|-----------------|--------------|")
                for _, row in elev_tp.iterrows():
                    p_str = f"{row['wilcoxon_p']:.4f}{_pstar(row['wilcoxon_p'])}"
                    report_lines.append(
                        f"| {row['timepoint']} | {row['n']} | "
                        f"{row['mean_change']:.3f} [{row['ci_lo']:.3f}, "
                        f"{row['ci_hi']:.3f}] | {p_str} |")
                report_lines.append("")
            if elev_lmm is not None:
                slope   = elev_lmm.params.get('time_months', np.nan)
                slope_p = elev_lmm.pvalues.get('time_months', np.nan)
                report_lines.append(
                    f"LMM slope: {slope:.4f} pts/month "
                    f"(p={slope_p:.4f}{_pstar(slope_p)})")
                report_lines.append("")

    # ── Panel and timepoint summary ───────────────────────────────────────
    plot_panel(all_domain_plots)
    plot_panel_wide(all_domain_plots)
    plot_timepoint_summary(all_timepoint_results)

    # ── Write report ──────────────────────────────────────────────────────
    report_lines += [
        "",
        "## Notes",
        "",
        "- Baseline-anchored patients overlap partially with the ITS cohort.",
        "  Patients in both analyses received their baseline-window assessment",
        "  AND had prior (pre-period) observations. Patients unique to this",
        "  analysis are those who had no assessment before GLP-1 start.",
        "",
        "- By design, mean change at time ~0 is zero. Meaningful change",
        "  should not be expected in the first month; clinical response to",
        "  GLP-1 typically emerges over months 2–6.",
        "",
        "- The LMM slope estimates the average change per month across the",
        "  full 1–12 month follow-up. A significant negative slope indicates",
        "  a sustained downward trend in scores after GLP-1 initiation.",
        "",
        "- Elevated subgroups restrict to patients whose **baseline** (±30 d)",
        "  score was at or above the clinical threshold, ensuring we are",
        "  studying patients with meaningful symptom burden at GLP-1 start.",
    ]

    report_path = OUT / "baseline_anchored_report.md"
    report_path.write_text("\n".join(report_lines))
    log.info("Saved report → %s", report_path)
    log.info("═══ Baseline-anchored analysis complete → %s ═══", OUT)


if __name__ == '__main__':
    main()
