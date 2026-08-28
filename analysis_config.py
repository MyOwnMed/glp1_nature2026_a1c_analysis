#!/usr/bin/env python3
"""Shared analysis configuration for the GLP-1 weight / HbA1c pipeline.

Part II v2 (2026-08-17). Addresses code-review item 9: the normal-quantile
multiplier used to build confidence intervals was hard-coded as ``1.96`` in 61
places, which fixed the confidence level at 95% and made it un-parameterizable.
Every script now derives the multiplier from a single confidence level defined
here.

At the default confidence level of 0.95 the multiplier is 1.959963984540054,
so every interval in the package is numerically unchanged to well beyond the
precision at which results are reported. ``1.96`` was the right number; it was
simply not derived from anything.

Override the confidence level for a whole run without editing code:

    CI_CONF_LEVEL=0.99 python3 code/step5_forest_contrasts_weight.py ...

Import from anywhere in the package (any subdirectory depth) with::

    import sys as _sys, pathlib as _pl
    for _p in _pl.Path(__file__).resolve().parents:
        if (_p / "analysis_config.py").exists():
            _sys.path.insert(0, str(_p)); break
    from analysis_config import z_critical
"""

from __future__ import annotations

import os

from scipy.stats import norm

__all__ = ["CONF_LEVEL", "z_critical", "conf_level_label", "DEFAULT_CONF_LEVEL"]

DEFAULT_CONF_LEVEL = 0.95


def _read_conf_level() -> float:
    raw = os.getenv("CI_CONF_LEVEL")
    if raw is None or str(raw).strip() == "":
        return DEFAULT_CONF_LEVEL
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"CI_CONF_LEVEL={raw!r} is not a number. Give a two-sided confidence "
            "level strictly between 0 and 1, e.g. CI_CONF_LEVEL=0.95"
        )
    if not (0.0 < value < 1.0):
        raise ValueError(
            f"CI_CONF_LEVEL={value} is out of range. Give a two-sided confidence "
            "level strictly between 0 and 1, e.g. CI_CONF_LEVEL=0.95"
        )
    return value


#: Two-sided confidence level used for every interval in the package.
CONF_LEVEL: float = _read_conf_level()


def z_critical(conf_level: float | None = None) -> float:
    """Two-sided normal critical value for *conf_level* (default: CONF_LEVEL).

    >>> round(z_critical(0.95), 6)
    1.959964
    """
    level = CONF_LEVEL if conf_level is None else float(conf_level)
    if not (0.0 < level < 1.0):
        raise ValueError(
            f"conf_level={level} is out of range; expected 0 < conf_level < 1"
        )
    return float(norm.ppf(0.5 + level / 2.0))


def conf_level_label(conf_level: float | None = None) -> str:
    """Human-readable label for plot/table captions, e.g. ``"95%"``."""
    level = CONF_LEVEL if conf_level is None else float(conf_level)
    pct = level * 100.0
    return f"{pct:g}%"


if __name__ == "__main__":
    print(f"CONF_LEVEL   = {CONF_LEVEL}")
    print(f"z_critical() = {z_critical()!r}")
    print(f"label        = {conf_level_label()}")
