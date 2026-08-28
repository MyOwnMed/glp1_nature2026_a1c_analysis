#!/usr/bin/env python

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from patsy import dmatrices
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.cov_struct import Independence


def _to_categorical(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df


def _build_formula(df: pd.DataFrame, df_spline: int) -> str:
    covariate_candidates = [
        "age_group",
        "gender",
        "baseline_a1c_category",
        "baseline_bmi_final_category",
        "race",
        "weight_change_med",
    ]
    covariates: List[str] = []
    for c in covariate_candidates:
        if c in df.columns:
            if df[c].nunique(dropna=True) > 1:
                covariates.append(c)
    rhs_terms = [f"bs(days_from_baseline, df={df_spline})"] + covariates
    rhs = " + ".join(rhs_terms)
    return f"pct_weight_change ~ {rhs}"


def select_spline_df(
    input_csv: Path,
    outdir: Path,
    df_grid: Iterable[int],
) -> None:
    logging.info("Reading analysis-ready CSV %s", input_csv)
    df = pd.read_csv(input_csv)

    keep_cols = [
        "patient_id",
        "days_from_baseline",
        "pct_weight_change",
        "baseline_a1c_category",
        "baseline_bmi_final_category",
        "age_group",
        "gender",
        "race",
        "weight_change_med",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    cat_cols = [
        "gender",
        "baseline_a1c_category",
        "baseline_bmi_final_category",
        "race",
        "age_group",
    ]
    df = _to_categorical(df, cat_cols)

    outdir.mkdir(parents=True, exist_ok=True)

    rows = []

    for df_spline in df_grid:
        logging.info("Fitting GEE with df=%s", df_spline)
        try:
            formula = _build_formula(df, df_spline)
            logging.debug("Formula: %s", formula)
            y, X = dmatrices(formula, data=df, return_type="dataframe")

            if "patient_id" not in df.columns:
                raise ValueError("patient_id column is required for groups")

            groups = df["patient_id"]
            model = GEE(y, X, groups=groups, family=Gaussian(), cov_struct=Independence())
            res = model.fit()
            qic, qicu = res.qic()
            rows.append({"df": df_spline, "QIC": float(qic), "QICu": float(qicu)})
        except Exception as e:  # noqa: BLE001
            logging.exception("Error fitting GEE for df=%s", df_spline)
            rows.append({"df": df_spline, "QIC": np.nan, "QICu": np.nan, "error": str(e)})

    df_results = pd.DataFrame(rows)
    out_csv = outdir / "df_selection.csv"
    logging.info("Writing df selection results to %s", out_csv)
    df_results.to_csv(out_csv, index=False)

    best_df = None
    best_qicu = None

    finite = df_results[np.isfinite(df_results["QICu"])] if not df_results.empty else df_results
    if not finite.empty:
        idx = finite["QICu"].idxmin()
        best_row = finite.loc[idx]
        best_df = int(best_row["df"])
        best_qicu = float(best_row["QICu"])

    config = {"best_df": best_df, "best_qicu": best_qicu}
    out_json = outdir / "model_config.json"
    logging.info("Writing best model config to %s", out_json)
    out_json.write_text(json.dumps(config))


def parse_df_grid(value: str) -> List[int]:
    parts = [v.strip() for v in value.split(",") if v.strip()]
    out: List[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            logging.warning("Could not parse df value '%s'", p)
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Step 2: Choose spline df for days_from_baseline using GEE QIC/QICu",
    )
    parser.add_argument(
        "--input-csv",
        default="output/step1_prepare_analysis_dataset/analysis_ready_gap90.csv",
        help="Input analysis-ready CSV (default: output/step1_prepare_analysis_dataset/analysis_ready_gap90.csv)",
    )
    parser.add_argument(
        "--outdir",
        default="output/step2_select_spline_df",
        help="Output directory for spline df selection results",
    )
    parser.add_argument(
        "--df-grid",
        default="3,4,5,6",
        help="Comma-separated list of spline degrees of freedom to evaluate",
    )
    parser.add_argument(
        "--adherence-gap-days",
        type=int,
        default=None,
        help="Adherence gap in days for gap-specific output folder (e.g., 90)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
    )

    input_csv = Path(args.input_csv)
    # Gap-aware outdir routing
    import re
    gap = args.adherence_gap_days
    if gap is None:
        m = re.search(r"gap[_]?(\d+)", str(args.input_csv))
        if m:
            try:
                gap = int(m.group(1))
            except Exception:
                gap = None
    outdir = Path(args.outdir)
    if gap is not None and "gap_" not in str(outdir):
        outdir = Path("output") / f"gap_{gap}" / outdir.name

    df_grid = parse_df_grid(args.df_grid)
    if not df_grid:
        logging.error("df-grid is empty after parsing; nothing to do")
        return

    select_spline_df(input_csv=input_csv, outdir=outdir, df_grid=df_grid)


if __name__ == "__main__":
    main()
