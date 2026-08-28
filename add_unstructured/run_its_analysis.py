#!/usr/bin/env python3
"""Step 2: Run ITS mixed-effects models, paired tests, and figures.

PHQ-9 and Pain Score: full ITS mixed model + trajectory + forest plot
Waist Circumference, Alcohol, Muscle Strength: paired tests + trajectory
All: descriptive summary table
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

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT  = Path(__file__).resolve().parents[2]
_AU   = os.environ.get('AU_OUTROOT')
DATA  = Path(os.environ.get('AU_DATADIR', str(ROOT / "output" / "submitted_analysis" / "1_no_adherence" / "data")))
OUT   = (Path(_AU) / "ITS") if _AU else ROOT / "output" / "submitted_analysis" / "1_no_adherence" / "ITS"
FIGS  = OUT / "figures"
TABS  = OUT / "tables"
MODS  = OUT / "models"
for d in [FIGS, TABS, MODS]:
    d.mkdir(parents=True, exist_ok=True)

# ── Domain config ─────────────────────────────────────────────────────────
DOMAIN_META = {
    'phq9': {
        'label': 'Depression (PHQ-9, 0–27 scale)',
        'ylabel': 'PHQ-9 Score',
        'model': True,
        'val': 'phq9_value',
        'extra_cov': ['antidepressant_baseline', 'covid_era'],
    },
    'pain_score': {
        'label': 'General Pain Intensity (0–10 scale)',
        'ylabel': 'Pain Score (0–10)',
        'model': True,
        'val': 'pain_score_value',
        'extra_cov': ['covid_era'],
    },
    'waist_circumference': {
        'label': 'Waist Circumference (Inches)',
        'ylabel': 'Inches',
        'model': False,
        'val': 'waist_circumference_value',
        'extra_cov': [],
    },
    'alcohol': {
        'label': 'Alcohol Use (Drinks/day)',
        'ylabel': 'Score',
        'model': False,
        'val': 'alcohol_value',
        'extra_cov': ['covid_era'],
    },
    'muscle_strength': {
        'label': 'Muscle Strength (MRC, 0–5 scale)',
        'ylabel': 'Score',
        'model': False,
        'val': 'muscle_strength_value',
        'extra_cov': ['covid_era'],
    },
}

# ── Plot styling ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})

COLORS = {
    'main': '#3474A7',      # steelblue
    'ci':   '#3474A7',
    'pre':  '#3474A7',
    'post': '#D64541',      # red-orange
    'vline':'#CC3333',
    'grid': '#CCCCCC',
}

# Demographic covariates added to all fixed-effects models
BASE_COV_COLS = [
    'age', 'gender', 'race',
    'baseline_a1c_category', 'baseline_bmi_final_category',
]
BASE_COV_FORMULA = (
    "age + C(gender) + C(race) + "
    "C(baseline_a1c_category) + C(baseline_bmi_final_category)"
)


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load(domain):
    """Load prepared domain CSV and clean covariates."""
    p = DATA / f"{domain}_prepared.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    for c in ['gender', 'race', 'baseline_a1c_category',
              'baseline_bmi_final_category', 'glp1_user_group']:
        if c in df.columns:
            df[c] = df[c].fillna('Unknown').astype(str)
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df['age'] = df['age'].fillna(df['age'].median())
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Paired pre/post test
# ═══════════════════════════════════════════════════════════════════════════

def paired_test(df, val_col):
    """Per-patient pre-mean vs post-mean paired tests."""
    both = df[df.has_both_periods == 1]
    pre  = both[both.post == 0].groupby('patient_id')[val_col].mean()
    post = both[both.post == 1].groupby('patient_id')[val_col].mean()
    common = pre.index.intersection(post.index)
    if len(common) < 5:
        return None
    pv, pp = pre.loc[common], post.loc[common]
    diff = pp - pv

    w_stat, w_p = (stats.wilcoxon(pv, pp)
                   if len(common) >= 10 else (np.nan, np.nan))
    t_stat, t_p = (stats.ttest_rel(pv, pp)
                   if len(common) >= 10 else (np.nan, np.nan))

    se = diff.std() / np.sqrt(len(diff))
    return {
        'n': len(common),
        'pre_mean': pv.mean(), 'pre_sd': pv.std(),
        'post_mean': pp.mean(), 'post_sd': pp.std(),
        'diff_mean': diff.mean(), 'diff_sd': diff.std(),
        'diff_ci_lo': diff.mean() - Z_CRIT * se,
        'diff_ci_hi': diff.mean() + Z_CRIT * se,
        'wilcoxon_p': w_p, 'ttest_p': t_p,
    }


# ═══════════════════════════════════════════════════════════════════════════
# ITS Mixed Model
# ═══════════════════════════════════════════════════════════════════════════

def fit_its(domain, df, val_col, extra_cov):
    """Within-person ITS mixed-effects model (random intercept per patient).

    Fixed effects include ITS terms (time_months, post, time_post), demographic
    covariates (age, sex, race, baseline BMI/A1c), and any domain-specific
    modifiers supplied via extra_cov (e.g., antidepressant_baseline for PHQ-9).
    """
    both = df[df.has_both_periods == 1].copy()
    model_vars = ([val_col, 'time_months', 'post', 'time_post', 'patient_id']
                  + extra_cov
                  + [c for c in BASE_COV_COLS if c in both.columns])

    mdf = both.dropna(
        subset=[c for c in model_vars if c in both.columns]).copy()

    n_pts = mdf.patient_id.nunique()
    if n_pts < 20 or len(mdf) < 50:
        log.warning("  %s: too sparse (%d pts, %d obs) — skip model",
                     domain, n_pts, len(mdf))
        return None

    formula = f"{val_col} ~ time_months + post + time_post + {BASE_COV_FORMULA}"
    for c in extra_cov:
        formula += f" + {c}"

    log.info("  ITS model: %d obs, %d patients", len(mdf), n_pts)
    log.info("  Formula: %s", formula)

    try:
        model = smf.mixedlm(formula, mdf, groups=mdf['patient_id'])
        result = model.fit(reml=True)
        log.info("  Converged=%s, AIC=%.1f", result.converged,
                 2 * result.k_fe - 2 * result.llf)
        return result
    except Exception as e:
        log.error("  Model failed: %s", e)
        # Fallback: minimal model without covariates
        log.info("  Trying minimal model (no covariates)…")
        try:
            formula_min = f"{val_col} ~ time_months + post + time_post"
            model = smf.mixedlm(formula_min, mdf, groups=mdf['patient_id'])
            result = model.fit(reml=True)
            log.info("  Minimal model converged=%s", result.converged)
            return result
        except Exception as e2:
            log.error("  Minimal model also failed: %s", e2)
            return None


# ═══════════════════════════════════════════════════════════════════════════
# Figures
# ═══════════════════════════════════════════════════════════════════════════

def plot_trajectory(domain, df, val_col, label, ylabel, model_result=None):
    """Monthly-binned trajectory with CI and piecewise trend lines."""
    both = df[df.has_both_periods == 1].copy()
    if both.empty or both.patient_id.nunique() < 5:
        log.warning("  %s: too few patients for trajectory plot", domain)
        return

    both['day_bin'] = (both.days_from_baseline // 30).astype(int) * 30
    grp = both.groupby('day_bin')[val_col]
    means  = grp.mean()
    sems   = grp.sem().fillna(0)  # fill NaN sems to prevent CI band gaps
    days = means.index.values  # 30-day bin centres

    fig, ax = plt.subplots(figsize=(10, 5))

    # CI band + observed means
    lo = means - Z_CRIT * sems
    hi = means + Z_CRIT * sems
    ax.fill_between(days, lo, hi, alpha=0.15, color=COLORS['ci'])
    ax.plot(days, means, 'o-', color=COLORS['main'], markersize=5,
            linewidth=2, label='Monthly mean ± 95% CI', zorder=3)

    # Piecewise linear fits
    pre_m = means[means.index < 0]
    post_m = means[means.index >= 0]
    if len(pre_m) >= 2:
        cp, bp = np.polyfit(pre_m.index, pre_m.values, 1)
        t = np.linspace(pre_m.index.min(), -1, 50)
        ax.plot(t, cp * t + bp, '--', color=COLORS['pre'],
                linewidth=1.5, alpha=0.6, label='Pre-GLP-1 trend')
    if len(post_m) >= 2:
        cp, bp = np.polyfit(post_m.index, post_m.values, 1)
        t = np.linspace(1, post_m.index.max(), 50)
        ax.plot(t, cp * t + bp, '--', color=COLORS['post'],
                linewidth=1.5, alpha=0.6, label='Post-GLP-1 trend')

    # GLP-1 initiation line
    ax.axvline(0, color=COLORS['vline'], linestyle='--', linewidth=1.5,
               alpha=0.7, label='GLP-1 initiation')

    # Formatting
    ax.set_xlabel('Days from GLP-1 Initiation')
    ax.set_ylabel(ylabel)
    n_pts = both.patient_id.nunique()
    ax.set_title(f'{label} — Within-Subject Trajectory\n'
                 f'(n = {n_pts:,} patients with pre + post data)')
    ax.xaxis.set_major_locator(mticker.MultipleLocator(60))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: '0' if x == 0 else (f'+{int(x)}' if x > 0 else str(int(x)))))
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.25, color=COLORS['grid'])

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(FIGS / f"trajectory_{domain}.{ext}")
    plt.close(fig)
    log.info("  Saved trajectory_%s.png/pdf", domain)


def plot_forest(domain, model_result):
    """Forest plot of key ITS coefficients."""
    if model_result is None:
        return

    fe = model_result.fe_params
    se = model_result.bse
    pv = model_result.pvalues

    keys   = ['time_months', 'post', 'time_post']
    labels = ['Pre-trend\n(per month)',
              'Level change\nat GLP-1 start',
              'Slope change\n(per month post)']

    available = [(k, l) for k, l in zip(keys, labels) if k in fe.index]
    if not available:
        return

    coefs  = [fe[k] for k, _ in available]
    errors = [Z_CRIT * se[k] for k, _ in available]
    labs   = [l for _, l in available]
    pvals  = [pv[k] for k, _ in available]

    fig, ax = plt.subplots(figsize=(8, 2.5 + 0.6 * len(available)))
    y_pos = list(range(len(available)))

    ax.errorbar(coefs, y_pos, xerr=errors, fmt='o',
                color=COLORS['main'], markersize=9, capsize=5, linewidth=2)
    ax.axvline(0, color='gray', linestyle='-', linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labs)

    for i, (c, p) in enumerate(zip(coefs, pvals)):
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        ax.annotate(f'{c:.3f} (p={p:.4f}){sig}',
                    (c, i), textcoords="offset points",
                    xytext=(12, 0), fontsize=9)

    meta = DOMAIN_META[domain]
    ax.set_title(f'{meta["label"]} — ITS Model Coefficients')
    ax.set_xlabel('Coefficient Estimate (95% CI)')
    ax.grid(True, alpha=0.25, axis='x')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(FIGS / f"forest_{domain}.{ext}")
    plt.close(fig)
    log.info("  Saved forest_%s.png/pdf", domain)


def plot_panel(all_data):
    """Combined panel figure — all domains with data."""
    items = [(d, m) for d, m in DOMAIN_META.items()
             if d in all_data and not all_data[d].empty
             and 'has_both_periods' in all_data[d].columns
             and all_data[d].has_both_periods.sum() > 0]
    n = len(items)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(10, 3.8 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (domain, meta) in zip(axes, items):
        df = all_data[domain]
        vcol = meta['val']
        both = df[df.has_both_periods == 1].copy()
        both['month_bin'] = np.floor(
            both.days_from_baseline / 30.44).astype(int)
        grp = both.groupby('month_bin')[vcol]
        means = grp.mean()
        sems  = grp.sem()
        months = means.index.values

        ax.fill_between(months, means - Z_CRIT * sems,
                        means + Z_CRIT * sems,
                        alpha=0.15, color=COLORS['ci'])
        ax.plot(months, means, 'o-', color=COLORS['main'],
                markersize=4, linewidth=1.5)
        ax.axvline(0, color=COLORS['vline'], linestyle='--',
                   linewidth=1.2, alpha=0.7)
        ax.set_ylabel(meta['ylabel'], fontsize=10)
        ax.set_title(f"{meta['label']}  (n={both.patient_id.nunique():,})",
                     fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel('Months from GLP-1 Initiation')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(FIGS / f"panel_all_trajectories.{ext}")
    plt.close(fig)
    log.info("Saved panel_all_trajectories.png/pdf")


def plot_pre_post_bars(desc_rows):
    """Grouped bar chart: pre vs post mean for each domain."""
    if not desc_rows:
        return
    rows = [r for r in desc_rows if r['n'] >= 5]
    if not rows:
        return

    labels = [r['domain'] for r in rows]
    pre    = [r['pre_mean'] for r in rows]
    post   = [r['post_mean'] for r in rows]
    pre_e  = [r['pre_sd'] / np.sqrt(r['n']) * Z_CRIT for r in rows]
    post_e = [r['post_sd'] / np.sqrt(r['n']) * Z_CRIT for r in rows]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, pre, w, yerr=pre_e, capsize=4,
           color=COLORS['pre'], alpha=0.7, label='Pre-GLP-1')
    ax.bar(x + w/2, post, w, yerr=post_e, capsize=4,
           color=COLORS['post'], alpha=0.7, label='Post-GLP-1')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Score (mean ± 95% CI)')
    ax.set_title('Pre vs Post GLP-1 Assessment Scores')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.25)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(FIGS / f"pre_post_bars.{ext}")
    plt.close(fig)
    log.info("Saved pre_post_bars.png/pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    all_data = {}
    desc_rows = []
    model_results = {}

    for domain, meta in DOMAIN_META.items():
        log.info("═══ %s ═══", meta['label'])
        df = load(domain)
        all_data[domain] = df

        if df.empty:
            log.warning("  No data — skipping")
            continue

        val_col = meta['val']

        # ── Descriptive + paired test ─────────────────────────────────────
        result = paired_test(df, val_col)
        if result:
            result['domain'] = meta['label']
            desc_rows.append(result)
            log.info("  n=%d  pre=%.2f±%.2f  post=%.2f±%.2f  "
                     "Δ=%.2f [%.2f,%.2f]  Wilcoxon p=%.4f",
                     result['n'], result['pre_mean'], result['pre_sd'],
                     result['post_mean'], result['post_sd'],
                     result['diff_mean'], result['diff_ci_lo'],
                     result['diff_ci_hi'], result['wilcoxon_p'])

        # ── ITS model ────────────────────────────────────────────────────
        model_result = None
        if meta['model']:
            model_result = fit_its(domain, df, val_col,
                                    meta.get('extra_cov', []))
            if model_result:
                model_results[domain] = model_result
                # Save summary text
                with open(MODS / f"its_{domain}_summary.txt", 'w') as f:
                    f.write(str(model_result.summary()))
                # Save coefficients
                coef_df = pd.DataFrame({
                    'coef': model_result.fe_params,
                    'se': model_result.bse,
                    'z': model_result.tvalues,
                    'p': model_result.pvalues,
                    'ci_lo': model_result.conf_int()[0],
                    'ci_hi': model_result.conf_int()[1],
                })
                coef_df.to_csv(TABS / f"model_coef_{domain}.csv")
                log.info("  Saved model summary + coefficients")

        # ── Trajectory figure ────────────────────────────────────────────
        plot_trajectory(domain, df, val_col,
                        meta['label'], meta['ylabel'], model_result)
        if model_result:
            plot_forest(domain, model_result)

    # ── Panel figure ──────────────────────────────────────────────────────
    plot_panel(all_data)
    plot_pre_post_bars(desc_rows)

    # ── Save descriptive table ────────────────────────────────────────────
    if desc_rows:
        desc_df = pd.DataFrame(desc_rows)
        desc_df.to_csv(TABS / "descriptive_summary.csv", index=False)

        with open(TABS / "descriptive_summary.md", 'w') as f:
            f.write("# Pre/Post GLP-1 Assessment Summary\n\n")
            f.write("| Domain | n | Pre (mean±SD) | Post (mean±SD) "
                    "| Δ [95% CI] | Wilcoxon p | t-test p |\n")
            f.write("|--------|---|---------------|----------------"
                    "|------------|-----------|----------|\n")
            for r in desc_rows:
                f.write(
                    f"| {r['domain']} | {r['n']} | "
                    f"{r['pre_mean']:.2f}±{r['pre_sd']:.2f} | "
                    f"{r['post_mean']:.2f}±{r['post_sd']:.2f} | "
                    f"{r['diff_mean']:.2f} "
                    f"[{r['diff_ci_lo']:.2f}, {r['diff_ci_hi']:.2f}] | "
                    f"{r['wilcoxon_p']:.4f} | {r['ttest_p']:.4f} |\n"
                )
        log.info("Saved descriptive_summary.csv/.md")

    # ── Combined markdown report ──────────────────────────────────────────
    _write_report(desc_rows, model_results)

    log.info("═══ Analysis complete ═══")
    log.info("Figures → %s", FIGS)
    log.info("Tables  → %s", TABS)
    log.info("Models  → %s", MODS)


def _write_report(desc_rows, model_results):
    """Write combined markdown report."""
    lines = [
        "# GLP-1 Assessment Analysis Report",
        "",
        "## Study Design",
        "Within-subject interrupted time series (ITS) analysis assessing",
        "the effect of GLP-1 initiation on five clinical assessment domains.",
        "",
        "- **Window**: 6 months pre through 12 months post GLP-1 start",
        "- **Model**: Mixed-effects with random intercept per patient",
        "- **Covariates**: age, gender, race, baseline A1c category, "
        "baseline BMI category, GLP-1 user group",
        "- **PHQ-9 additional covariate**: antidepressant at baseline",
        "",
    ]

    if desc_rows:
        lines.append("## Descriptive Summary")
        lines.append("")
        lines.append("| Domain | n | Pre | Post | Δ | p (Wilcoxon) |")
        lines.append("|--------|---|-----|------|---|-------------|")
        for r in desc_rows:
            sig = '***' if r['wilcoxon_p'] < 0.001 else \
                  '**' if r['wilcoxon_p'] < 0.01 else \
                  '*' if r['wilcoxon_p'] < 0.05 else ''
            lines.append(
                f"| {r['domain']} | {r['n']} | "
                f"{r['pre_mean']:.2f} | {r['post_mean']:.2f} | "
                f"{r['diff_mean']:+.2f} | "
                f"{r['wilcoxon_p']:.4f}{sig} |"
            )
        lines.append("")

    for domain, result in model_results.items():
        meta = DOMAIN_META[domain]
        fe = result.fe_params
        pv = result.pvalues
        lines.append(f"## ITS Model: {meta['label']}")
        lines.append("")
        lines.append(f"- **Observations**: {result.nobs}")
        lines.append(f"- **Groups**: {result.nobs}")  # approx
        lines.append(f"- **Converged**: {result.converged}")
        lines.append("")
        for key, label in [('post', 'Level change at GLP-1 start'),
                           ('time_post', 'Slope change post-GLP-1'),
                           ('time_months', 'Pre-GLP-1 trend')]:
            if key in fe.index:
                sig = '***' if pv[key] < 0.001 else \
                      '**' if pv[key] < 0.01 else \
                      '*' if pv[key] < 0.05 else ''
                lines.append(
                    f"- **{label}**: β = {fe[key]:.4f} "
                    f"(p = {pv[key]:.4f}){sig}"
                )
        lines.append("")

    lines.extend([
        "## Figures",
        "",
        "- `trajectory_<domain>.png` — Monthly-binned means ± CI",
        "- `forest_<domain>.png` — ITS coefficient forest plot",
        "- `panel_all_trajectories.png` — Combined panel",
        "- `pre_post_bars.png` — Grouped bar chart",
        "",
        "## Notes",
        "",
        "- PHQ-9 analysis excludes entries sourced from PHQ-2 only",
        "- Pain scores are on a 0–10 scale (no rescaling needed)",
        "- Antidepressant flag = any antidepressant prescribed within "
        "180 d pre to 90 d post baseline",
        "- Rare covariate levels (< 5 patients) collapsed to 'Other'",
    ])

    with open(OUT / "analysis_report.md", 'w') as f:
        f.write('\n'.join(lines))
    log.info("Saved analysis_report.md")


if __name__ == '__main__':
    main()
