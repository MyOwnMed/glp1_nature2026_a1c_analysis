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


A1C_CATEGORY_ORDER = [
    "Normal Glycemia",
    "Prediabetes",
    "Type 2 Diabetes",
    "Poorly Controlled Diabetes",
    "Unknown",
]


def _to_categorical(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df


def _prepare_a1c_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "baseline_a1c_category" in df.columns:
        cat_type = pd.CategoricalDtype(A1C_CATEGORY_ORDER, ordered=True)
        df["baseline_a1c_category"] = df["baseline_a1c_category"].astype(cat_type)
        df = df[df["baseline_a1c_category"] != "Unknown"]
    return df


def _build_formula(df: pd.DataFrame, df_spline: int) -> str:
    covariate_candidates = [
        "age_group",
        "gender",
        "baseline_a1c_category",
        "baseline_bmi_final_category",
        "race",
        "metformin_with_glp1_baseline",
        "weight_change_med",
    ]
    covariates: List[str] = []
    for c in covariate_candidates:
        if c in df.columns:
            if df[c].nunique(dropna=True) > 1:
                covariates.append(c)
    rhs_terms = [f"bs(days_from_baseline, df={df_spline})"] + covariates
    rhs = " + ".join(rhs_terms)
    return f"abs_a1c_change ~ {rhs}"


def select_spline_df_a1c(
    input_csv: Path,
    outdir: Path,
    df_grid: Iterable[int],
) -> None:
    logging.info("Reading A1c analysis-ready CSV %s", input_csv)
    df = pd.read_csv(input_csv)

    if "a1c_value" not in df.columns and {
        "baseline_a1c_final",
        "abs_a1c_change",
    }.issubset(set(df.columns)):
        df["a1c_value"] = df["baseline_a1c_final"] + df["abs_a1c_change"]

    keep_cols = [
        "patient_id",
        "days_from_baseline",
        "abs_a1c_change",
        "baseline_a1c_category",
        "baseline_bmi_final_category",
        "age_group",
        "gender",
        "race",
        "metformin_with_glp1_baseline",
        "weight_change_med",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    df = _prepare_a1c_categories(df)

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
        logging.info("Fitting A1c GEE with df=%s", df_spline)
        try:
            # Drop rows with missing outcome or time before building design matrices
            df_model = df.dropna(subset=["abs_a1c_change", "days_from_baseline"]).copy()

            if df_model.empty:
                raise ValueError("No rows with non-missing abs_a1c_change and days_from_baseline")

            formula = _build_formula(df_model, df_spline)
            logging.debug("Formula: %s", formula)
            y, X = dmatrices(formula, data=df_model, return_type="dataframe")

            if "patient_id" not in df_model.columns:
                raise ValueError("patient_id column is required for groups")

            # Align groups with the rows used by patsy
            groups = df_model.loc[y.index, "patient_id"]
            model = GEE(y, X, groups=groups, family=Gaussian(), cov_struct=Independence())
            res = model.fit()
            qic, qicu = res.qic()
            rows.append({"df": df_spline, "QIC": float(qic), "QICu": float(qicu)})
        except Exception as e:  # noqa: BLE001
            logging.exception("Error fitting A1c GEE for df=%s", df_spline)
            rows.append({"df": df_spline, "QIC": np.nan, "QICu": np.nan, "error": str(e)})

    df_results = pd.DataFrame(rows)
    out_csv = outdir / "df_selection_a1c.csv"
    logging.info("Writing A1c df selection results to %s", out_csv)
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
    out_json = outdir / "model_config_a1c.json"
    logging.info("Writing best A1c model config to %s", out_json)
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
        description="Step 2 (A1c): Choose spline df for days_from_baseline using GEE QIC/QICu",
    )
    parser.add_argument(
        "--input-csv",
        default="output/step1_prepare_analysis_dataset_a1c/analysis_ready_a1c_gap90.csv",
        help=(
            "Input A1c analysis-ready CSV "
            "(default: output/step1_prepare_analysis_dataset_a1c/analysis_ready_a1c_gap90.csv)"
        ),
    )
    parser.add_argument(
        "--outdir",
        default="output/step2_select_spline_df_a1c",
        help="Output directory for A1c spline df selection results",
    )
    parser.add_argument(
        "--df-grid",
        default="3,4,5,6",
        help="Comma-separated list of spline degrees of freedom to evaluate",
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
    outdir = Path(args.outdir)

    df_grid = parse_df_grid(args.df_grid)
    if not df_grid:
        logging.error("df-grid is empty after parsing; nothing to do")
        return

    select_spline_df_a1c(input_csv=input_csv, outdir=outdir, df_grid=df_grid)


if __name__ == "__main__":
    main()
