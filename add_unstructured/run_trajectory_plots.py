#!/usr/bin/env python3
"""GEE smooth trajectory plots for assessment domains.

Fits a GEE with B-spline basis on days_from_baseline to produce smooth
model-predicted trajectories — mirroring the style of step4_predictive_plots.py
(weight/A1c outcome trajectories).

Three lines per domain plot:
  1. All patients with any data in the window
  2. Elevated-baseline subgroup (primary threshold)

Output → output/add_unstructured/trajectories/
"""

import logging, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from patsy import dmatrices, build_design_matrices, bs  # noqa: F401
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

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT  = Path(__file__).resolve().parents[2]
_AU   = os.environ.get('AU_OUTROOT')
DATA  = Path(os.environ.get('AU_DATADIR', str(ROOT / "output" / "submitted_analysis" / "1_no_adherence" / "data")))
OUT   = (Path(_AU) / "GEE") if _AU else ROOT / "output" / "submitted_analysis" / "1_no_adherence" / "GEE"
FIGS  = OUT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# ── Domain config ─────────────────────────────────────────────────────────
DOMAINS = {
    'a1c': {
        'label': 'HbA1c',
        'ylabel': 'HbA1c (%)',
        'val': 'a1c_value',
        'elev_threshold':   6.5,
        'elev_label':       'Elevated baseline ≥6.5% (diabetic)',
        'elev_threshold_2': 9.0,
        'elev_label_2':     'Elevated baseline ≥9.0% (poorly controlled)',
        'spline_df': 4,
        'color_all':    '#3498DB',   # blue
        'color_elev':   '#E67E22',   # orange (diabetic)
        'color_elev_2': '#C0392B',   # red (poorly controlled)
        # Restrict GEE to patients with BOTH pre and post observations
        'both_periods_only': True,
    },
    'phq9': {
        'label': 'Depression (PHQ-9, 0–27 scale)',
        'ylabel': 'PHQ-9 Score (0–27)',
        'val': 'phq9_value',
        'elev_threshold':   5,
        'elev_label':       'Elevated baseline ≥5 (mild+)',
        'elev_threshold_2': 10,
        'elev_label_2':     'Elevated baseline ≥10 (moderate-severe)',
        'spline_df': 3,
        'color_all':    '#7BAFD4',   # light blue
        'color_elev':   '#1A5C8A',   # dark blue
        'color_elev_2': '#0A2340',   # very dark navy
        # Restrict GEE to patients with BOTH pre and post observations
        # so the trajectory is comparable to the ITS forest plot
        'both_periods_only': True,
        'extra_cov': ['antidepressant_baseline', 'covid_era'],
    },
    'pain_score': {
        'label': 'General Pain Intensity (0–10 scale)',
        'ylabel': 'Pain Score (0–10)',
        'val': 'pain_score_value',
        'elev_threshold':   4,
        'elev_label':       'Elevated baseline ≥4 (moderate+)',
        'elev_threshold_2': 7,
        'elev_label_2':     'Elevated baseline ≥7 (severe)',
        'spline_df': 3,
        'color_all':    '#F5A673',
        'color_elev':   '#C0392B',
        'color_elev_2': '#7B0000',
        # Restrict GEE to patients with BOTH pre and post observations
        'both_periods_only': True,
        'extra_cov': ['covid_era'],
    },
    'waist_circumference': {
        'label': 'Waist Circumference (Inches)',
        'ylabel': 'Waist Circumference (inches)',
        'val': 'waist_circumference_value',
        'elev_threshold': None,
        'elev_label': None,
        'spline_df': 3,
        'color_all':   '#5DAD6F',
        'color_elev':  None,
        'both_periods_only': True,
    },
    'alcohol': {
        'label': 'Alcohol Use (Drinks/day)',
        'ylabel': 'Alcohol Score',
        'val': 'alcohol_value',
        'elev_threshold': 12,
        'elev_label':     'Top 25% baseline drinkers (score ≥12)',
        'elev_threshold_2': None,
        'elev_label_2':     None,
        'spline_df': 3,
        'color_all':   '#9B59B6',    # medium purple
        'color_elev':  '#6C1A8A',    # dark purple
        'color_elev_2': None,
        'both_periods_only': True,
        'extra_cov': ['covid_era'],
    },
    'muscle_strength': {
        'label': 'Muscle Strength (MRC, 0–5 scale)',
        'ylabel': 'Strength Score',
        'val': 'muscle_strength_value',
        'elev_threshold': None,
        'elev_label': None,
        'spline_df': 3,
        'color_all':   '#95A5A6',
        'color_elev':  None,
        'both_periods_only': True,
        'extra_cov': ['covid_era'],
    },
}

# ── Plot style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
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
# GEE fitting
# ═══════════════════════════════════════════════════════════════════════════

def fit_gee(df, val_col, spline_df=4, extra_cov=None):
    """Fit GEE with B-spline basis on time and demographic covariates.

    Returns (result, design_info, d_min, d_max, ref_covars) where ref_covars
    is a dict of reference covariate values used for prediction.
    """
    extra_cov = extra_cov or []
    demo_cols = [c for c in BASE_COV_COLS + extra_cov if c in df.columns]
    sub = df[[val_col, 'days_from_baseline', 'patient_id'] + demo_cols].dropna(
        subset=[val_col, 'days_from_baseline']).copy()

    # Clean categorical and numeric covariates
    for c in ['gender', 'race', 'baseline_a1c_category', 'baseline_bmi_final_category']:
        if c in sub.columns:
            sub[c] = sub[c].fillna('Unknown').astype(str)
    if 'age' in sub.columns:
        sub['age'] = pd.to_numeric(sub['age'], errors='coerce')
        sub['age'] = sub['age'].fillna(sub['age'].median())

    if len(sub) < 30 or sub.patient_id.nunique() < 10:
        return None, None, None, None, None

    ref_covars = get_ref_covars(sub, extra_cov=extra_cov)

    base_demo = [c for c in BASE_COV_COLS if c in sub.columns]
    cov_str = " + ".join(
        f"C({c})" if c != 'age' else 'age' for c in base_demo)
    xtra_str = " + ".join(c for c in extra_cov if c in sub.columns)
    if xtra_str:
        cov_str = (cov_str + " + " + xtra_str) if cov_str else xtra_str
    formula = (f"{val_col} ~ bs(days_from_baseline, df={spline_df}, include_intercept=False)"
               + (f" + {cov_str}" if cov_str else ""))
    try:
        y, X = dmatrices(formula, sub, return_type='dataframe')
    except Exception as e:
        log.error("  dmatrices failed: %s", e)
        return None, None, None, None, None

    ids = sub.loc[y.index, 'patient_id']
    d_min = float(sub.days_from_baseline.min())
    d_max = float(sub.days_from_baseline.max())
    try:
        model = GEE(y, X, groups=ids, family=Gaussian(), cov_struct=Independence())
        result = model.fit()
        log.info("  GEE fit: %d obs, %d patients, df=%d",
                 len(sub), sub.patient_id.nunique(), spline_df)
        return result, X.design_info, d_min, d_max, ref_covars
    except Exception as e:
        log.error("  GEE fit failed: %s", e)
        return None, None, None, None, None


def predict_trajectory(result, design_info, days_grid, d_min=None, d_max=None,
                       ref_covars=None):
    """Generate predictions + 95% CI on a day grid.

    Clips the grid to the training-data range to avoid patsy bs() out-of-knot
    errors; slots outside that range are returned as NaN.
    ref_covars: dict of covariate name → reference value for prediction.
    """
    out_mean = np.full(len(days_grid), np.nan)
    out_lo   = np.full(len(days_grid), np.nan)
    out_hi   = np.full(len(days_grid), np.nan)

    eps = 0.5  # half-day buffer inside boundary
    valid_mask = np.ones(len(days_grid), dtype=bool)
    if d_min is not None:
        valid_mask &= days_grid >= (d_min + eps)
    if d_max is not None:
        valid_mask &= days_grid <= (d_max - eps)

    days_valid = days_grid[valid_mask]
    if len(days_valid) == 0:
        return out_mean, out_lo, out_hi

    pred_df = pd.DataFrame({'days_from_baseline': days_valid})
    if ref_covars:
        for col, val in ref_covars.items():
            pred_df[col] = val
    try:
        X_pred = build_design_matrices([design_info], pred_df)[0]
        X_pred = np.asarray(X_pred, dtype=float)
        mean_v = np.asarray(result.predict(X_pred), dtype=float)
        V = np.asarray(result.cov_params(), dtype=float)
        var = np.einsum('ij,jk,ik->i', X_pred, V, X_pred, optimize=True)
        var = np.clip(var, 0.0, None)
        se = np.sqrt(var)
        out_mean[valid_mask] = mean_v
        out_lo[valid_mask]   = mean_v - Z_CRIT * se
        out_hi[valid_mask]   = mean_v + Z_CRIT * se
        return out_mean, out_lo, out_hi
    except Exception as e:
        log.error("  Prediction failed: %s", e)
        return None, None, None


def support_mask(df, days_grid, val_col, window_days=30, min_patients=10):
    """Mask grid points where fewer than min_patients have data nearby."""
    sub = df[[val_col, 'days_from_baseline', 'patient_id']].dropna()
    if sub.empty:
        return np.zeros(len(days_grid), dtype=bool)
    mask = np.zeros(len(days_grid), dtype=bool)
    for i, d in enumerate(days_grid):
        sel = sub.loc[np.abs(sub.days_from_baseline - d) <= window_days]
        mask[i] = sel.patient_id.nunique() >= min_patients
    return mask


# ═══════════════════════════════════════════════════════════════════════════
# Patient count annotation helper
# ═══════════════════════════════════════════════════════════════════════════

def add_n_ribbon(ax, df, val_col, days_grid, color, y_pos_frac=0.02, window_days=30):
    """Add a thin strip below the plot showing patient counts over time."""
    sub = df[[val_col, 'days_from_baseline', 'patient_id']].dropna()
    ns = []
    for d in days_grid:
        sel = sub.loc[np.abs(sub.days_from_baseline - d) <= window_days]
        ns.append(sel.patient_id.nunique())
    # Thin bar proportional to sample size
    max_n = max(ns) if max(ns) > 0 else 1
    months = days_grid / 30.44
    return months, ns, max_n


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_domain(domain, meta, df_all, df_elev, df_elev2=None):
    """Generate the smooth GEE trajectory plot for one domain."""
    val_col = meta['val']
    spline_df = meta['spline_df']
    label = meta['label']
    both_only = meta.get('both_periods_only', False)

    # Day grid: −6 to +12 months
    days_pre  = np.arange(-180, 0, 7)
    days_post = np.arange(0, 366, 7)
    days_grid = np.concatenate([days_pre, days_post])

    fig, axes = plt.subplots(2, 1, figsize=(11, 7),
                             gridspec_kw={'height_ratios': [7, 1]})
    ax, ax_n = axes

    handles = []

    # ── Fit and plot for each sub-cohort ──
    all_label = 'All patients (pre & post obs)' if both_only else 'All patients'
    cohorts = [('all', df_all, meta['color_all'], all_label)]
    if df_elev is not None and meta.get('elev_threshold') is not None:
        cohorts.append(('elev', df_elev, meta['color_elev'], meta['elev_label']))
    if df_elev2 is not None and meta.get('elev_threshold_2') is not None:
        cohorts.append(('elev2', df_elev2, meta['color_elev_2'], meta['elev_label_2']))

    n_traces_plotted = 0
    any_patchy = False
    for cohort_key, df, color, clabel in cohorts:
        if df is None or df.empty:
            continue

        n_pts = df.patient_id.nunique()
        result, design_info, d_min, d_max, ref = fit_gee(df, val_col, spline_df,
                                                          extra_cov=meta.get('extra_cov', []))
        if result is None:
            # Fall back to binned observed means
            log.warning("  GEE failed for %s/%s — using binned means", domain, cohort_key)
            df2 = df.copy()
            df2['week_bin'] = (df2.days_from_baseline // 14) * 14
            grp = df2.groupby('week_bin')[val_col]
            means = grp.mean()
            m = means.index.values / 30.44
            ax.plot(m, means.values, '-', color=color, linewidth=2, alpha=0.8,
                    label=f'{clabel} (n={n_pts:,}) [observed]')
            handles.append(ax.lines[-1])
            continue

        mean, lo, hi = predict_trajectory(result, design_info, days_grid, d_min, d_max, ref)
        if mean is None:
            continue

        mask = support_mask(df, days_grid, val_col,
                            window_days=21, min_patients=3)
        if not mask.all():
            any_patchy = True
        # CI band: suppressed where data is sparse; main line stays continuous
        m_lo = np.where(mask, lo, np.nan)
        m_hi = np.where(mask, hi, np.nan)
        ax.fill_between(days_grid, m_lo, m_hi,
                        alpha=0.15, color=color)

        # Predicted line: continuous within data range (mean is NaN outside d_min/d_max)
        line, = ax.plot(days_grid, mean, '-', color=color,
                        linewidth=2.5, label=f'{clabel} (n={n_pts:,})',
                        zorder=3)
        handles.append(line)
        n_traces_plotted += 1

    if n_traces_plotted == 0:
        plt.close(fig)
        log.warning("  %s: nothing to plot", domain)
        return

    # ── Vertical line at GLP-1 initiation ──
    ax.axvline(0, color='#CC3333', linestyle='--', linewidth=1.5,
               alpha=0.8, label='GLP-1 initiation')
    ax.axhline(0, color='#888888', linestyle=':', linewidth=0.8, alpha=0.4)

    # ── Axes labels ──
    ax.set_ylabel(meta['ylabel'])
    ax.set_xlabel('Days from GLP-1 Initiation')
    ax.set_xlim(days_grid[0], days_grid[-1])
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(30))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(60))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: '0' if x == 0 else (f'+{int(x)}' if x > 0 else str(int(x)))))

    subtitle = ('patients with pre & post observations, ' if both_only else '')
    ax.set_title(f'GLP-1 and {label} — GEE Smooth Trajectory\n'
                 f'(GEE B-spline model, {subtitle}95% CI)',
                 pad=10)
    ax.legend(handles=handles, loc='upper right', fontsize=9,
              framealpha=0.92)
    ax.grid(True, alpha=0.2, which='major')

    # ── Patient-count strip ──
    ax_n.set_facecolor('white')
    ax_n.spines[:].set_visible(False)

    # For the "all patients" cohort, show N over time
    if not df_all.empty:
        sub = df_all[[val_col, 'days_from_baseline', 'patient_id']].dropna()
        bins = np.arange(-180, 380, 30)
        bin_labels = (bins[:-1] + bins[1:]) / 2  # day-unit centres
        ns_all = []
        for left, right in zip(bins[:-1], bins[1:]):
            sel = sub[(sub.days_from_baseline >= left) &
                      (sub.days_from_baseline < right)]
            ns_all.append(sel.patient_id.nunique())

        ax_n.bar(bin_labels, ns_all, width=26, color=meta['color_all'],
                 alpha=0.5, label='N (all)')
        if df_elev is not None and not df_elev.empty:
            sub_e = df_elev[[val_col, 'days_from_baseline', 'patient_id']].dropna()
            ns_elev = []
            for left, right in zip(bins[:-1], bins[1:]):
                sel = sub_e[(sub_e.days_from_baseline >= left) &
                            (sub_e.days_from_baseline < right)]
                ns_elev.append(sel.patient_id.nunique())
            ax_n.bar(bin_labels, ns_elev, width=26, color=meta['color_elev'],
                     alpha=0.7, label='N (elevated)')

        ax_n.set_xlabel('Days from GLP-1 Initiation')
        ax_n.set_ylabel('N', fontsize=8)
        ax_n.tick_params(labelsize=7)
        ax_n.set_xlim(days_grid[0], days_grid[-1])
        ax_n.xaxis.set_major_locator(mticker.MultipleLocator(60))
        ax_n.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: '0' if x == 0 else (f'+{int(x)}' if x > 0 else str(int(x)))))
        ax_n.axvline(0, color='#CC3333', linestyle='--',
                     linewidth=1.2, alpha=0.6)
        ax_n.legend(fontsize=7, loc='upper right')

    if any_patchy:
        fig.text(0.01, -0.02,
                 '† CI band not shown where <3 patients have data within ±21 days.',
                 fontsize=7.5, color='#666666', ha='left', va='top')
    fig.subplots_adjust(hspace=0.08)

    for ext in ('png', 'pdf'):
        fig.savefig(FIGS / f"trajectory_gee_{domain}.{ext}")
    plt.close(fig)
    log.info("  Saved trajectory_gee_%s.png/pdf", domain)


def plot_subgroup(domain, meta, df_sub, color, label, fname_stem):
    """Single-cohort GEE trajectory plot for one elevated subgroup."""
    if df_sub is None or df_sub.empty:
        return
    val_col   = meta['val']
    spline_df = meta['spline_df']

    days_pre  = np.arange(-180, 0, 7)
    days_post = np.arange(0, 366, 7)
    days_grid = np.concatenate([days_pre, days_post])

    fig, axes = plt.subplots(2, 1, figsize=(11, 7),
                             gridspec_kw={'height_ratios': [7, 1]})
    ax, ax_n = axes

    n_pts  = df_sub.patient_id.nunique()
    result, design_info, d_min, d_max, ref = fit_gee(df_sub, val_col, spline_df,
                                                      extra_cov=meta.get('extra_cov', []))
    if result is None:
        plt.close(fig)
        log.warning("  %s subgroup GEE failed for %s — skip", domain, fname_stem)
        return

    mean, lo, hi = predict_trajectory(result, design_info, days_grid, d_min, d_max, ref)
    if mean is None:
        plt.close(fig)
        return

    mask = support_mask(df_sub, days_grid, val_col, window_days=21, min_patients=3)
    has_gap = not mask.all()
    m_lo = np.where(mask, lo, np.nan)
    m_hi = np.where(mask, hi, np.nan)
    ax.fill_between(days_grid, m_lo, m_hi, alpha=0.20, color=color)
    ax.plot(days_grid, mean, '-', color=color, linewidth=2.5,
            label=f'{label} (n={n_pts:,})', zorder=3)

    ax.axvline(0, color='#CC3333', linestyle='--', linewidth=1.5,
               alpha=0.8, label='GLP-1 initiation')
    ax.axhline(0, color='#888888', linestyle=':', linewidth=0.8, alpha=0.4)
    ax.set_ylabel(meta['ylabel'])
    ax.set_xlabel('Days from GLP-1 Initiation')
    ax.set_xlim(days_grid[0], days_grid[-1])
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(30))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(60))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: '0' if x == 0 else (f'+{int(x)}' if x > 0 else str(int(x)))))
    ax.set_title(f'GLP-1 and {meta["label"]} — GEE Smooth Trajectory\n'
                 f'({label}, GEE B-spline, 95% CI)', pad=10)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.92)
    ax.grid(True, alpha=0.2, which='major')

    # N ribbon
    ax_n.set_facecolor('white')
    ax_n.spines[:].set_visible(False)
    sub = df_sub[[val_col, 'days_from_baseline', 'patient_id']].dropna()
    bins = np.arange(-180, 380, 30)
    bin_labels = (bins[:-1] + bins[1:]) / 2
    ns = []
    for left_b, right_b in zip(bins[:-1], bins[1:]):
        sel = sub[(sub.days_from_baseline >= left_b) &
                  (sub.days_from_baseline < right_b)]
        ns.append(sel.patient_id.nunique())
    ax_n.bar(bin_labels, ns, width=26, color=color, alpha=0.6)
    ax_n.set_xlabel('Days from GLP-1 Initiation')
    ax_n.set_ylabel('N', fontsize=8)
    ax_n.tick_params(labelsize=7)
    ax_n.set_xlim(days_grid[0], days_grid[-1])
    ax_n.xaxis.set_major_locator(mticker.MultipleLocator(60))
    ax_n.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: '0' if x == 0 else (f'+{int(x)}' if x > 0 else str(int(x)))))
    ax_n.axvline(0, color='#CC3333', linestyle='--', linewidth=1.2, alpha=0.6)

    if has_gap:
        fig.text(0.01, -0.02,
                 '† CI band not shown where <3 patients have data within ±21 days.',
                 fontsize=7.5, color='#666666', ha='left', va='top')
    fig.subplots_adjust(hspace=0.08)
    for ext in ('png', 'pdf'):
        fig.savefig(FIGS / f"{fname_stem}.{ext}")
    plt.close(fig)
    log.info("  Saved %s.png/pdf", fname_stem)


def plot_panel_all(domain_data):
    """Multi-panel figure — all domains in one figure."""
    items = [(d, m, data) for (d, m, data) in domain_data
             if data[0] is not None and len(data) > 0 and not data[0].empty]
    n = len(items)
    if n == 0:
        return

    days_pre  = np.arange(-180, 0, 14)
    days_post = np.arange(0, 366, 14)
    days_grid = np.concatenate([days_pre, days_post])

    fig, axes = plt.subplots(n, 1, figsize=(11, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    any_patchy = False
    for ax, (domain, meta, data) in zip(axes, items):
        df_all   = data[0]
        df_elev  = data[1] if len(data) > 1 else None
        df_elev2 = data[2] if len(data) > 2 else None
        val_col = meta['val']
        spline_df = meta['spline_df']

        both_only = meta.get('both_periods_only', False)
        all_label = 'All (pre & post obs)' if both_only else 'All patients'
        cohorts = [('all', df_all, meta['color_all'], all_label)]
        if df_elev is not None and meta.get('elev_threshold') is not None:
            cohorts.append(('elev', df_elev, meta['color_elev'], meta['elev_label']))
        if df_elev2 is not None and meta.get('elev_threshold_2') is not None:
            cohorts.append(('elev2', df_elev2, meta['color_elev_2'], meta['elev_label_2']))

        for cohort_key, df, color, clabel in cohorts:
            if df is None or df.empty:
                continue
            n_pts = df.patient_id.nunique()
            result, design_info, d_min, d_max, ref = fit_gee(df, val_col, spline_df,
                                                              extra_cov=meta.get('extra_cov', []))
            if result is None:
                continue
            mean, lo, hi = predict_trajectory(result, design_info, days_grid, d_min, d_max, ref)
            if mean is None:
                continue
            mask = support_mask(df, days_grid, val_col,
                                window_days=21, min_patients=3)
            if not mask.all():
                any_patchy = True
            m_lo = np.where(mask, lo, np.nan)
            m_hi = np.where(mask, hi, np.nan)
            ax.fill_between(days_grid, m_lo, m_hi, alpha=0.15, color=color)
            ax.plot(days_grid, mean, '-', color=color, linewidth=2.0,
                    label=f'{clabel} (n={n_pts:,})')

        ax.axvline(0, color='#CC3333', linestyle='--', linewidth=1.2, alpha=0.7)
        ax.set_ylabel(meta['ylabel'], fontsize=9)
        ax.set_title(meta['label'], fontsize=10, fontweight='bold', pad=4)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel('Days from GLP-1 Initiation')
    axes[-1].xaxis.set_major_locator(mticker.MultipleLocator(60))
    axes[-1].xaxis.set_minor_locator(mticker.MultipleLocator(30))
    axes[-1].xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: '0' if x == 0 else (f'+{int(x)}' if x > 0 else str(int(x)))))

    fig.suptitle('GLP-1 Assessment Trajectories — GEE Model-Predicted',
                 fontsize=13, y=1.01)
    if any_patchy:
        fig.text(0.01, -0.02,
                 '† CI band not shown where <3 patients have data within ±21 days.',
                 fontsize=7.5, color='#666666', ha='left', va='top')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(FIGS / f"panel_gee_all_domains.{ext}",
                    bbox_inches='tight')
    plt.close(fig)
    log.info("Saved panel_gee_all_domains.png/pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def load_domain(domain, meta):
    p = DATA / f"{domain}_prepared.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df['days_from_baseline'] = pd.to_numeric(
        df['days_from_baseline'], errors='coerce')
    if meta.get('both_periods_only', False) and 'has_both_periods' in df.columns:
        n_before = df.patient_id.nunique()
        df = df[df['has_both_periods'] == 1].copy()
        log.info("  both_periods filter: %d → %d patients",
                 n_before, df.patient_id.nunique())
    return df


def filter_elevated(df, val_col, threshold):
    if threshold is None or df.empty:
        return None
    # Pre-period mean per patient
    pre = df[df['post'] == 0]
    means = pre.groupby('patient_id')[val_col].mean()
    pids  = set(means[means >= threshold].index)
    return df[df.patient_id.isin(pids)].copy() if pids else None


def main():
    domain_data = []

    for domain, meta in DOMAINS.items():
        log.info("═══ %s ═══", meta['label'])
        df_all = load_domain(domain, meta)

        if df_all.empty:
            log.warning("  No data — skip")
            domain_data.append((domain, meta, (None, None, None)))
            continue

        df_elev  = filter_elevated(df_all, meta['val'], meta.get('elev_threshold'))
        df_elev2 = filter_elevated(df_all, meta['val'], meta.get('elev_threshold_2'))

        log.info("  All: %d patients | Elev≥%s: %s | Elev≥%s: %s",
                 df_all.patient_id.nunique(),
                 meta.get('elev_threshold', '-'),
                 f"{df_elev.patient_id.nunique()}" if df_elev is not None else "n/a",
                 meta.get('elev_threshold_2', '-'),
                 f"{df_elev2.patient_id.nunique()}" if df_elev2 is not None else "n/a")

        plot_domain(domain, meta, df_all, df_elev, df_elev2=df_elev2)

        # Per-subgroup GEE plots (separated so CIs are clearly visible)
        if df_elev is not None and meta.get('elev_threshold') is not None:
            tag = f"gte{int(meta['elev_threshold'])}"
            plot_subgroup(domain, meta, df_elev,
                          meta['color_elev'], meta['elev_label'],
                          f"trajectory_gee_{domain}_{tag}")
        if df_elev2 is not None and meta.get('elev_threshold_2') is not None:
            tag = f"gte{int(meta['elev_threshold_2'])}"
            plot_subgroup(domain, meta, df_elev2,
                          meta['color_elev_2'], meta['elev_label_2'],
                          f"trajectory_gee_{domain}_{tag}")

        domain_data.append((domain, meta, (df_all, df_elev, df_elev2)))

    # Combined panel
    plot_panel_all(domain_data)
    log.info("═══ Trajectory plots complete → %s ═══", OUT)


if __name__ == '__main__':
    main()
