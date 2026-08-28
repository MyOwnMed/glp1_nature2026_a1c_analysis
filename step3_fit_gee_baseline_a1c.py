#!/usr/bin/env python

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd
from patsy import dmatrices
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


def _select_covariates(df: pd.DataFrame, drop: Optional[Sequence[str]] = None) -> List[str]:
    candidates = [
        "age_group",
        "gender",
        "baseline_a1c_category",
        "baseline_bmi_final_category",
        "race",
        "metformin_with_glp1_baseline",
        "weight_change_med",
    ]
    drop_set = set(drop or [])
    covariates: List[str] = []
    # Code-review item: this helper silently dropped any covariate with
    # nunique <= 1, so models compared across gaps and strata were not necessarily
    # adjusted for the same variable set and nothing recorded the difference. The
    # drops still happen — a single-valued covariate is not estimable and would
    # make the design matrix singular — but each one is now logged with its
    # reason, so the contents of every fitted model are visible in the run log.
    #
    # This matters only inside small subgroups, where a covariate can collapse to
    # one level; in those cases it could not have been included in any case.
    explicitly_dropped: List[str] = []
    absent: List[str] = []
    single_valued: List[str] = []
    for c in candidates:
        if c in drop_set:
            explicitly_dropped.append(c)
            continue
        if c not in df.columns:
            absent.append(c)
            continue
        n_unique = df[c].nunique(dropna=True)
        if n_unique <= 1:
            observed = df[c].dropna().unique()
            single_valued.append(
                f"{c} (nunique={n_unique}"
                + (f", value={observed[0]!r}" if len(observed) else ", all missing")
                + ")"
            )
            continue
        covariates.append(c)

    logging.info("Covariates included in the model: %s", covariates or "(none)")
    if explicitly_dropped:
        logging.info("Covariates dropped by request (--drop-covariate): %s",
                     explicitly_dropped)
    if absent:
        logging.info("Covariates absent from the input data: %s", absent)
    if single_valued:
        logging.warning(
            "Covariates dropped as single-valued (not estimable in this subset): %s",
            single_valued,
        )
    return covariates


def _build_formula(df: pd.DataFrame, df_spline: int, drop: Optional[Sequence[str]] = None) -> str:
    covariates = _select_covariates(df, drop=drop)
    rhs_terms = [f"bs(days_from_baseline, df={df_spline})"] + covariates
    rhs = " + ".join(rhs_terms)
    return f"abs_a1c_change ~ {rhs}"


def fit_gee_baseline_a1c(
    input_csv: Path,
    config_json: Path,
    outdir: Path,
    drop_covariates: Optional[Sequence[str]] = None,
) -> None:
    logging.info("Reading A1c analysis CSV %s", input_csv)
    df = pd.read_csv(input_csv)

    with config_json.open() as f:
        config = json.load(f)
    best_df = config.get("best_df")
    if best_df is None:
        raise ValueError("best_df not found in config JSON")

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

    # Ensure a1c_value exists if reconstructable (for future plotting; outcome uses abs_a1c_change)
    if "a1c_value" not in df.columns and {
        "baseline_a1c_final",
        "abs_a1c_change",
    }.issubset(set(df.columns)):
        df["a1c_value"] = df["baseline_a1c_final"] + df["abs_a1c_change"]

    # Drop rows with missing outcome or time
    df_model = df.dropna(subset=["abs_a1c_change", "days_from_baseline"]).copy()
    if df_model.empty:
        raise ValueError("No rows with non-missing abs_a1c_change and days_from_baseline")

    from patsy import bs  # noqa: F401  # needed in eval environment for patsy

    formula = _build_formula(df_model, int(best_df), drop=drop_covariates)
    logging.info("Using formula: %s", formula)
    y, X = dmatrices(formula, data=df_model, return_type="dataframe")

    if "patient_id" not in df_model.columns:
        raise ValueError("patient_id column is required for groups")

    groups = df_model.loc[y.index, "patient_id"]

    model = GEE(y, X, groups=groups, family=Gaussian(), cov_struct=Independence())
    res = model.fit()

    outdir.mkdir(parents=True, exist_ok=True)

    # Coefficients table
    params = res.params
    bse = res.bse
    z = Z_CRIT
    ci_low = params - z * bse
    ci_high = params + z * bse
    pvalues = res.pvalues

    coef_df = pd.DataFrame(
        {
            "term": params.index,
            "estimate": params.values,
            "std_error": bse.values,
            "ci_lower": ci_low.values,
            "ci_upper": ci_high.values,
            "p_value": pvalues.values,
        }
    )
    coef_path = outdir / "coefficients.csv"
    coef_df.to_csv(coef_path, index=False)
    logging.info("Wrote coefficients to %s", coef_path)

    # Text summary
    summary_path = outdir / "model_summary.txt"
    summary_path.write_text(str(res.summary()))
    logging.info("Wrote model summary to %s", summary_path)

    # Config used
    cfg_used_path = outdir / "model_config_used.json"
    cfg_used_path.write_text(json.dumps({"best_df": best_df}))
    logging.info("Wrote model config used to %s", cfg_used_path)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Step 3 (A1c): Fit baseline GEE model for abs_a1c_change",
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="A1c analysis CSV (e.g., output/step1_prepare_analysis_dataset_a1c/analysis_ready_a1c_gap90.csv)",
    )
    parser.add_argument(
        "--config-json",
        required=True,
        help="A1c model config JSON from Step 2 (e.g., output/step2_select_spline_df_a1c/model_config_a1c.json)",
    )
    parser.add_argument(
        "--outdir",
        required=False,
        default="output/step3_fit_gee_baseline_a1c",
        help="Output directory for fitted A1c GEE model results",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--drop-covariate",
        action="append",
        default=None,
        help="Covariate name to exclude from the model (repeatable), e.g. metformin_with_glp1_baseline",
    )
    parser.add_argument("--adherence-gap-days", type=int, default=None, help="Adherence gap for gap-specific subfolder")

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
    )

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

    fit_gee_baseline_a1c(
        input_csv=Path(args.input_csv),
        config_json=Path(args.config_json),
        outdir=outdir,
        drop_covariates=args.drop_covariate,
    )


if __name__ == "__main__":
    main()
