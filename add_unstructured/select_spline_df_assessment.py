#!/usr/bin/env python3
"""QIC-based spline df selection for add_unstructured GEE trajectory models.

For each assessment domain (phq9, pain_score, waist_circumference, alcohol,
muscle_strength), sweep df in [2..6], fit the same GEE used in
run_trajectory_plots.py, and record QIC.  Results are printed as a summary
table and saved as JSON per gap.

Usage:
    python3 select_spline_df_assessment.py --gap 120 --data-dir <path>
  or use the companion shell wrapper (run_spline_select_all.sh).
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import pandas as pd
from patsy import dmatrices
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.cov_struct import Independence

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

DOMAINS = {
    "phq9":               {"file": "phq9_prepared.csv",               "col": "phq9_value"},
    "pain_score":         {"file": "pain_score_prepared.csv",         "col": "pain_score_value"},
    "waist_circumference":{"file": "waist_circumference_prepared.csv","col": "waist_circumference_value"},
    "alcohol":            {"file": "alcohol_prepared.csv",            "col": "alcohol_value"},
    "muscle_strength":    {"file": "muscle_strength_prepared.csv",    "col": "muscle_strength_value"},
}

DF_GRID = [2, 3, 4, 5, 6]


def qic_for_df(df_data: pd.DataFrame, val_col: str, spline_df: int):
    """Fit GEE and return QICu (smaller is better)."""
    sub = df_data[["patient_id", "days_from_baseline", val_col]].dropna().copy()
    if len(sub) < 30 or sub.patient_id.nunique() < 10:
        return None, None

    formula = (f"{val_col} ~ bs(days_from_baseline, df={spline_df}, "
               f"include_intercept=False)")
    try:
        y, X = dmatrices(formula, sub, return_type="dataframe")
    except Exception as e:
        log.warning("  dmatrices failed df=%d: %s", spline_df, e)
        return None, None

    ids = sub.loc[y.index, "patient_id"]
    try:
        model = GEE(y, X, groups=ids, family=Gaussian(), cov_struct=Independence())
        result = model.fit()
        qic_val, qicu_val = result.qic()
        return float(qic_val), float(qicu_val)
    except Exception as e:
        log.warning("  GEE failed df=%d: %s", spline_df, e)
        return None, None


def select_for_gap(data_dir: Path, out_dir: Path, gap: int) -> dict:
    """Run QIC selection for all domains for a single gap."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"gap": gap, "domains": {}}

    for domain, meta in DOMAINS.items():
        csv_path = data_dir / meta["file"]
        if not csv_path.exists():
            log.warning("  %s: data file not found (%s) — skip", domain, csv_path)
            continue

        df_data = pd.read_csv(csv_path)
        val_col = meta["col"]
        if val_col not in df_data.columns:
            log.warning("  %s: column %s not found — skip", domain, val_col)
            continue

        n_patients = df_data["patient_id"].nunique() if "patient_id" in df_data.columns else 0
        n_obs = df_data[val_col].notna().sum()
        log.info("gap=%d domain=%s  n_patients=%d  n_obs=%d", gap, domain, n_patients, n_obs)

        rows = []
        for df_spline in DF_GRID:
            qic, qicu = qic_for_df(df_data, val_col, df_spline)
            rows.append({"df": df_spline, "qic": qic, "qicu": qicu})
            log.info("  df=%d  QIC=%.4f  QICu=%.4f",
                     df_spline, qic or float("nan"), qicu or float("nan"))

        valid = [(r["df"], r["qicu"]) for r in rows if r["qicu"] is not None]
        if not valid:
            log.warning("  %s: no valid QIC values", domain)
            best_df = None
        else:
            best_df = min(valid, key=lambda t: t[1])[0]
        log.info("  → best df=%s", best_df)

        results["domains"][domain] = {
            "n_patients": int(n_patients),
            "n_obs": int(n_obs),
            "sweep": rows,
            "best_df": best_df,
        }

    out_path = out_dir / f"spline_selection_gap{gap}.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    log.info("Saved %s", out_path)
    return results


def summarise(all_results: list) -> pd.DataFrame:
    """Build a domain × gap matrix of best_df values."""
    rows = []
    for res in all_results:
        gap = res["gap"]
        for domain, info in res["domains"].items():
            rows.append({"gap": gap, "domain": domain, "best_df": info["best_df"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    pivot = df.pivot(index="domain", columns="gap", values="best_df")
    pivot["mode"] = pivot.mode(axis=1).iloc[:, 0]
    return pivot


# ────────────────────────────────────────────────────────────────── main ──

def main():
    ROOT = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser()
    parser.add_argument("--gaps", nargs="+", type=int,
                        default=[30, 60, 90, 120, 150, 180, 365, 548, 730])
    parser.add_argument("--conf-int-dir",
                        default=str(ROOT / "output" / "submitted_analysis"),
                        help="Directory containing gap_N subdirs")
    parser.add_argument("--out-dir",
                        default=str(ROOT / "output" / "submitted_analysis" / "spline_selection"),
                        help="Where to write JSON results")
    args = parser.parse_args()

    conf_int_dir = Path(args.conf_int_dir)
    out_dir = Path(args.out_dir)

    all_results = []
    for gap in args.gaps:
        data_dir = conf_int_dir / f"gap_{gap}" / "add_unstructured" / "data"
        if not data_dir.exists():
            log.warning("gap_%d: data dir not found (%s)", gap, data_dir)
            continue
        log.info("═══ gap=%d ═══", gap)
        res = select_for_gap(data_dir, out_dir, gap)
        all_results.append(res)

    if not all_results:
        log.error("No results — exiting")
        return

    pivot = summarise(all_results)
    print("\n═══ Best df per domain × gap (QICu-based) ═══")
    print(pivot.to_string())

    # Save summary CSV
    pivot.to_csv(out_dir / "spline_selection_summary.csv")
    print(f"\nSaved summary → {out_dir / 'spline_selection_summary.csv'}")


if __name__ == "__main__":
    main()
