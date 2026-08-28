#!/usr/bin/env python3
"""Model-specification inputs that are never silently defaulted.

Part II v2 (2026-08-17). Addresses code-review item 8.

The reviewer identified two silent handlers on methodologically load-bearing
paths in ``step5_forest_contrasts_weight.py``:

  * lines 168-177 fell back to ``df_spline = 3`` if the step2 config could not be
    read, substituting a different model specification for the selected one;
  * lines 148-153 swallowed a failure to set the ordered HbA1c categorical, which
    fixes the reference level for every contrast in the file.

Both patterns turned out to be copy-pasted across the package: the spline-df
fallback in ten scripts, the swallowed categorical in nine. Fixing only the two
instances the reviewer happened to open would have left the same defect live
everywhere else, so every instance now routes through this module and stops the
run with a clear message.

Neither ever altered a published result:

  * 3 degrees of freedom is what the a priori QICu selection procedure chose, and
    the production runs passed the value explicitly, so the fallback and the
    selection agreed. The defect was that nothing would have said otherwise.
  * The categorical assignment does not fail on the study data; the labels are
    exactly the four expected categories. Had it failed, contrasts would have been
    reported against an unknown reference level.
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Optional, Sequence

import pandas as pd

__all__ = ["A1C_ORDER", "REF_CATEGORY", "load_spline_df", "enforce_a1c_order"]

#: Baseline glycemic categories, in the order the manuscript reports them. The
#: first entry is the reference level for every contrast.
A1C_ORDER: List[str] = [
    "Normal Glycemia",
    "Prediabetes",
    "Type 2 Diabetes",
    "Poorly Controlled Diabetes",
]

#: Reference level for every HbA1c-category contrast.
REF_CATEGORY: str = A1C_ORDER[0]


def load_spline_df(config_json, key: str = "best_df") -> int:
    """Return the spline degrees of freedom selected by step2. Never defaulted.

    Raises rather than falling back, because the degrees of freedom *are* the model
    specification: continuing with a guess would report one model's estimates under
    another model's name.
    """
    path = os.fspath(config_json)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Spline config not found: {path}. Run step2 for this cohort and gap "
            "first. The spline degrees of freedom are a model specification and are "
            "not defaulted."
        )
    try:
        with open(path) as fh:
            cfg = json.load(fh)
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Could not read the spline config {path}: {exc}. The spline degrees of "
            "freedom are a model specification and are not defaulted."
        ) from exc

    if not isinstance(cfg, dict) or key not in cfg:
        keys = sorted(cfg) if isinstance(cfg, dict) else type(cfg).__name__
        raise KeyError(
            f"{key!r} is missing from {path} (found: {keys}). The spline degrees of "
            "freedom are a model specification and are not defaulted."
        )
    try:
        value = int(cfg[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{key!r} in {path} is {cfg[key]!r}, which is not an integer number of "
            "degrees of freedom."
        ) from exc
    if value < 1:
        raise ValueError(f"{key!r} in {path} is {value}; expected a positive integer.")

    logging.info("Spline df = %d (from %s)", value, path)
    return value


def enforce_a1c_order(
    df: pd.DataFrame,
    column: str = "baseline_a1c_category",
    order: Optional[Sequence[str]] = None,
    ref_category: Optional[str] = None,
    require_ref: bool = True,
    context: str = "",
) -> pd.DataFrame:
    """Set *column* to an ordered categorical with the manuscript's level order.

    Raises if the assignment fails, if no value matches the expected categories, or
    if the reference level is absent — each of which would silently re-base every
    contrast computed downstream.

    Mutates and returns *df*.
    """
    levels = list(order if order is not None else A1C_ORDER)
    ref = ref_category if ref_category is not None else (levels[0] if levels else None)
    where = f" [{context}]" if context else ""

    if column not in df.columns:
        raise KeyError(
            f"{column!r} is absent from the input{where}; it is the effect modifier "
            "for every reported contrast."
        )

    try:
        df[column] = pd.Categorical(df[column], categories=levels, ordered=True)
    except Exception as exc:
        raise RuntimeError(
            f"Could not set {column!r} as an ordered categorical with levels "
            f"{levels}{where}: {exc}. This fixes the reference level for every "
            "contrast, so the run is stopped rather than continued against an "
            "unknown reference."
        ) from exc

    observed = set(pd.Series(df[column]).dropna().unique())
    if not observed:
        raise RuntimeError(
            f"No value of {column!r} matches the expected levels {levels}{where}; "
            "every contrast would be undefined. Check the category labels in the "
            "input CSV."
        )
    missing = [lvl for lvl in levels if lvl not in observed]
    if missing:
        logging.info("Levels of %s absent from this subset%s: %s", column, where, missing)
    if require_ref and ref is not None and ref not in observed:
        raise RuntimeError(
            f"The reference level {ref!r} is absent from the input{where} "
            f"(present: {sorted(observed)}). Contrasts would be reported against a "
            "different reference than the manuscript's."
        )
    return df
