#!/usr/bin/env python3
import argparse
import os
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

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


WEIGHT_THRESHOLDS = [-5.0, -10.0, -15.0]  # percent change (negative = loss)
# ── Follow-up horizon cap (code-review point: horizons look inconsistent) ─────
# 540 days = 18 x 30-day months. This is the event-time cap for the time-to-event
# analyses: a threshold crossing observed later than this is censored at 540, so
# no event time exceeds the paper's stated 18-month follow-up limit.
#
# The three caps in the package are all "18 months or beyond" under different
# month conventions, not three different analysis windows:
#
#   step1 --max-days                   730  outer bound on the data step1 emits,
#                                           wide enough for the 730-day
#                                           persistence sensitivity analysis
#   step6b/step6c --max-days           548  ~18 months as a calendar figure,
#                                           contrast and plot horizon
#   step6d --truncate-days             548  ~18 months, fitting horizon
#   MAX_FOLLOWUP_DAYS (here)           540  18 x 30 days, event-time cap
#
# 540 is the tightest of them and applies only to event times. Changing none of
# them changes any reported estimate; the paper reports nothing past 18 months.
MAX_FOLLOWUP_DAYS = 18 * 30.0


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Time-to-event survival-style dataset for first achievement of "
            "5%, 10%, and 15% weight loss, stratified by baseline A1C category."
        ),
    )
    p.add_argument(
        "--input-csv",
        required=True,
        help=(
            "Path to weight analysis_ready.csv from Step 1 "
            "(e.g., output/step1_prepare_analysis_dataset/analysis_ready_gap90.csv)."
        ),
    )
    p.add_argument(
        "--outdir",
        required=False,
        default=os.path.join("output", "step8_survival_time_to_weight_loss"),
        help="Directory to write event-level and summary CSVs",
    )
    p.add_argument(
        "--adherence-gap-days",
        type=int,
        default=None,
        help="Adherence gap in days for gap-specific output folder (e.g., 90)",
    )
    return p.parse_args(argv)


def _first_time_below_threshold(
    days: Iterable[float], pct_change: Iterable[float], threshold: float
) -> Tuple[float, int]:
    """Return (time, event) for first time pct_change <= threshold.

    If the threshold is never reached, returns (max_observed_time, 0).
    """

    days_arr = np.asarray(list(days), dtype=float)
    pct_arr = np.asarray(list(pct_change), dtype=float)

    if days_arr.size == 0:
        return (np.nan, 0)

    order = np.argsort(days_arr)
    days_arr = days_arr[order]
    pct_arr = pct_arr[order]

    mask = pct_arr <= threshold
    if mask.any():
        idx = np.argmax(mask)
        return (float(days_arr[idx]), 1)
    else:
        return (float(days_arr.max()), 0)


def build_events(df: pd.DataFrame) -> pd.DataFrame:
    if "patient_id" not in df.columns:
        raise ValueError("Expected column 'patient_id' in input dataset")
    if "days_from_baseline" not in df.columns:
        raise ValueError("Expected column 'days_from_baseline' in input dataset")
    if "pct_weight_change" not in df.columns:
        raise ValueError("Expected column 'pct_weight_change' in input dataset")
    if "baseline_a1c_category" not in df.columns:
        raise ValueError("Expected column 'baseline_a1c_category' in input dataset")

    rows: List[dict] = []
    for (pid, group), gdf in df.groupby(["patient_id", "baseline_a1c_category"]):
        g_days = gdf["days_from_baseline"].astype(float)
        g_pct = gdf["pct_weight_change"].astype(float)
        for thr in WEIGHT_THRESHOLDS:
            t, event = _first_time_below_threshold(g_days, g_pct, thr)
            if not np.isfinite(t):
                # Skip completely missing follow-up
                continue

            # Right-censor at MAX_FOLLOWUP_DAYS: events after this window
            # are treated as non-events with time truncated to the window.
            if t > MAX_FOLLOWUP_DAYS:
                t = MAX_FOLLOWUP_DAYS
                event = 0

            rows.append(
                {
                    "patient_id": pid,
                    "baseline_a1c_category": group,
                    "threshold_pct": thr,
                    "time_days": t,
                    "event": int(event),
                }
            )

    return pd.DataFrame(rows)


def km_summary(events_df: pd.DataFrame) -> pd.DataFrame:
    """Kaplan–Meier median time and 95% CI by baseline A1C and weight threshold.

    Uses the product-limit (KM) estimator with Greenwood's formula and a
    log(-log(S)) transformation to obtain 95% confidence intervals for
    survival, then inverts these to derive CIs for the median time to
    achieving each percent weight-loss threshold. If the KM curve never
    crosses 0.5, the median and CIs are left as NA.
    """

    if events_df.empty:
        return pd.DataFrame(
            columns=[
                "baseline_a1c_category",
                "threshold_pct",
                "n",
                "n_events",
                "median_time_days",
                "median_time_days_lower_ci",
                "median_time_days_upper_ci",
            ]
        )

    out_rows: List[dict] = []
    z = Z_CRIT
    for (group, thr), gdf in events_df.groupby(["baseline_a1c_category", "threshold_pct"]):
        gdf = gdf.dropna(subset=["time_days"])
        if gdf.empty:
            continue

        # Sort by time
        gdf = gdf.sort_values("time_days")
        times = gdf["time_days"].to_numpy()
        events = gdf["event"].to_numpy().astype(int)

        uniq_times = np.unique(times)
        n = len(times)
        n_events = int(events.sum())

        # Kaplan–Meier product-limit with Greenwood-based CI for S(t)
        at_risk = n
        surv = 1.0
        greenwood_sum = 0.0
        median_time = np.nan
        lower_ci = np.nan
        upper_ci = np.nan

        t_vals: List[float] = []
        s_vals: List[float] = []
        s_lower_vals: List[float] = []
        s_upper_vals: List[float] = []

        for t in uniq_times:
            mask_t = times == t
            d_i = int(events[mask_t].sum())
            n_i = int(mask_t.sum())

            if at_risk > 0 and d_i > 0:
                surv *= (1.0 - d_i / at_risk)
                if at_risk - d_i > 0:
                    greenwood_sum += d_i / (at_risk * (at_risk - d_i))

            at_risk -= n_i

            if surv <= 0.5 and np.isnan(median_time):
                median_time = float(t)

            # CI for S(t) via log(-log(S)) transform, where defined
            s_lower = np.nan
            s_upper = np.nan
            if 0.0 < surv < 1.0 and greenwood_sum > 0.0:
                var_s = (surv ** 2) * greenwood_sum
                denom = surv * abs(np.log(surv))
                if denom > 0:
                    se_loglog_s = np.sqrt(var_s) / denom
                    y = np.log(-np.log(surv))
                    y_lo = y - z * se_loglog_s
                    y_hi = y + z * se_loglog_s
                    s_lower = float(np.exp(-np.exp(y_hi)))
                    s_upper = float(np.exp(-np.exp(y_lo)))

            t_vals.append(float(t))
            s_vals.append(float(surv))
            s_lower_vals.append(s_lower)
            s_upper_vals.append(s_upper)

        # Invert survival CIs to get median time CIs when median is defined
        if np.isfinite(median_time):
            for t, s_u, s_l in zip(t_vals, s_upper_vals, s_lower_vals):
                if np.isnan(lower_ci) and np.isfinite(s_u) and s_u <= 0.5:
                    lower_ci = float(t)
                if np.isnan(upper_ci) and np.isfinite(s_l) and s_l <= 0.5:
                    upper_ci = float(t)

        out_rows.append(
            {
                "baseline_a1c_category": group,
                "threshold_pct": thr,
                "n": n,
                "n_events": n_events,
                "median_time_days": median_time,
                "median_time_days_lower_ci": lower_ci,
                "median_time_days_upper_ci": upper_ci,
            }
        )

    return pd.DataFrame(out_rows)


def main(argv=None):
    args = parse_args(argv)

    df = pd.read_csv(args.input_csv)
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
    outdir = args.outdir
    if gap is not None and "/gap_" not in outdir:
        outdir = os.path.join("output", f"gap_{gap}", os.path.basename(args.outdir))
    os.makedirs(outdir, exist_ok=True)

    events_df = build_events(df)
    events_out = os.path.join(outdir, "step8_weight_time_to_threshold_events.csv")
    events_df.to_csv(events_out, index=False)

    summary_df = km_summary(events_df)
    summary_out = os.path.join(outdir, "step8_weight_time_to_threshold_summary.csv")
    summary_df.to_csv(summary_out, index=False)


if __name__ == "__main__":
    main()
