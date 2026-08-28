#!/usr/bin/env python3
"""Covariate selection with the drops recorded.

Part II v2 (2026-08-17). Addresses the code-review point that
``_select_covariates`` silently dropped any covariate with ``nunique <= 1``, so
models compared across gaps and strata were not necessarily adjusted for the same
variable set and nothing in the output recorded the difference.

The same pattern appeared as an inline list comprehension in twelve other places
(step4 predictive plots, step6/6b/6c/6cc stratified models, step6d group
comparisons). All of them now route through :func:`filter_estimable`, so every
drop is logged with its reason and the contents of every fitted model are visible
in the run log.

The drops themselves are correct and unchanged: a covariate with a single
observed level is not estimable and would make the design matrix singular. This
only ever bites inside small strata, where such a covariate could not have been
included in any case. What was missing was the record.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Sequence

import pandas as pd

__all__ = ["filter_estimable"]


def filter_estimable(
    candidates: Sequence[str],
    df: pd.DataFrame,
    *,
    exclude: Optional[Iterable[str]] = None,
    context: str = "",
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """Return the candidates that can actually enter a model, logging every drop.

    A candidate is kept when it is not excluded, is present in *df*, and has more
    than one distinct non-missing value.

    Args:
        candidates: covariate names to consider, in model order.
        df: the frame the model will be fitted on (the *subset*, for stratified
            fits — that is the point: estimability is a property of the subset).
        exclude: names to drop deliberately, e.g. the stratification variable,
            which is constant within a stratum by construction.
        context: short label identifying the fit, so the log line says which model
            the drops belong to (e.g. "step6b weight, strat_var=gender, Prediabetes").
        logger: logger to use; defaults to the root logger.

    Returns:
        The retained covariate names, in the order given.
    """
    log = logger or logging.getLogger()
    excluded = set(exclude or ())
    where = f" [{context}]" if context else ""

    kept: List[str] = []
    dropped_excluded: List[str] = []
    dropped_absent: List[str] = []
    dropped_single: List[str] = []

    for name in candidates:
        if name in excluded:
            dropped_excluded.append(name)
            continue
        if name not in df.columns:
            dropped_absent.append(name)
            continue
        n_unique = df[name].nunique(dropna=True)
        if n_unique <= 1:
            observed = df[name].dropna().unique()
            detail = (
                f"{name} (nunique={n_unique}, value={observed[0]!r})"
                if len(observed)
                else f"{name} (all missing)"
            )
            dropped_single.append(detail)
            continue
        kept.append(name)

    log.info("Covariates in model%s: %s", where, kept or "(none)")
    if dropped_excluded:
        log.info("Covariates excluded by design%s: %s", where, dropped_excluded)
    if dropped_absent:
        log.info("Covariates absent from the data%s: %s", where, dropped_absent)
    if dropped_single:
        log.warning(
            "Covariates dropped as single-valued (not estimable in this subset)%s: %s",
            where,
            dropped_single,
        )
    return kept
