#!/usr/bin/env python3
"""Step 4: Observed summaries and plots for weight and A1C.

Outputs:
- 90-day binned summaries (centered at first bin per group)
- Month-window summaries at selected months (1,3,6,12,15,18,24)
- Trajectory-style observed plots mirroring the predictive plot style
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
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


TIME_MONTHS = [1, 3, 6, 9, 12, 15, 18, 21, 24]
WINDOW_DAYS = 30  # +/- 30 day window around each month (matches 90-day bins)


def configure_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _bin_index(days: np.ndarray, max_days: int, bin_width: int) -> np.ndarray:
    days = np.asarray(days, dtype=float)
    valid = (days >= 0) & (days < max_days)
    idx = np.full(days.shape, -1, dtype=int)
    idx[valid] = (days[valid] // bin_width).astype(int)
    return idx


def summarize_weight(df: pd.DataFrame, max_days: int, bin_width: int) -> pd.DataFrame:
    df = df.copy()
    df = df[(df["days_from_baseline"] >= 0) & (df["days_from_baseline"] < max_days)]
    df = df.dropna(subset=["pct_weight_change", "baseline_a1c_category"])

    df["bin"] = _bin_index(df["days_from_baseline"].to_numpy(), max_days, bin_width)
    df = df[df["bin"] >= 0]

    grouped = (
        df.groupby(["baseline_a1c_category", "bin"], observed=True)["pct_weight_change"]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    grouped.rename(
        columns={
            "count": "n",
            "mean": "mean_pct_weight_change",
            "std": "sd_pct_weight_change",
        },
        inplace=True,
    )

    out_list = []
    for cat, g in grouped.groupby("baseline_a1c_category", observed=True):
        g = g.sort_values("bin").copy()
        if not g.empty:
            anchor = g.loc[g["bin"].idxmin(), "mean_pct_weight_change"]
            g["mean_pct_weight_change"] = g["mean_pct_weight_change"] - anchor
        g["month"] = (g["bin"] * bin_width) / 30.0
        g["bin_center_days"] = g["bin"] * bin_width + bin_width / 2.0
        out_list.append(g)

    if not out_list:
        return pd.DataFrame(
            columns=[
                "baseline_a1c_category",
                "bin",
                "n",
                "mean_pct_weight_change",
                "sd_pct_weight_change",
                "month",
                "bin_center_days",
            ]
        )

    return pd.concat(out_list, ignore_index=True)


def summarize_a1c(df: pd.DataFrame, max_days: int, bin_width: int) -> pd.DataFrame:
    df = df.copy()
    df = df[(df["days_from_baseline"] >= 0) & (df["days_from_baseline"] < max_days)]
    df = df.dropna(subset=["abs_a1c_change", "baseline_a1c_category"])

    if "baseline_a1c_final" in df.columns:
        base_col = "baseline_a1c_final"
    elif "baseline_a1c" in df.columns:
        base_col = "baseline_a1c"
    else:
        base_col = None

    if base_col is not None:
        df["a1c_value_followup"] = df[base_col] + df["abs_a1c_change"]
    else:
        df["a1c_value_followup"] = np.nan

    df["bin"] = _bin_index(df["days_from_baseline"].to_numpy(), max_days, bin_width)
    df = df[df["bin"] >= 0]

    grouped = (
        df.groupby(["baseline_a1c_category", "bin"], observed=True)
        .agg(
            n=("abs_a1c_change", "count"),
            mean_a1c_value=("a1c_value_followup", "mean"),
            sd_a1c_value=("a1c_value_followup", "std"),
            mean_abs_a1c_change=("abs_a1c_change", "mean"),
            sd_abs_a1c_change=("abs_a1c_change", "std"),
        )
        .reset_index()
    )

    out_list = []
    for cat, g in grouped.groupby("baseline_a1c_category", observed=True):
        g = g.sort_values("bin").copy()
        if not g.empty:
            anchor = g.loc[g["bin"].idxmin(), "mean_abs_a1c_change"]
            g["mean_abs_a1c_change"] = g["mean_abs_a1c_change"] - anchor
        g["month"] = (g["bin"] * bin_width) / 30.0
        g["bin_center_days"] = g["bin"] * bin_width + bin_width / 2.0
        out_list.append(g)

    if not out_list:
        return pd.DataFrame(
            columns=[
                "baseline_a1c_category",
                "bin",
                "n",
                "mean_a1c_value",
                "sd_a1c_value",
                "mean_abs_a1c_change",
                "sd_abs_a1c_change",
                "month",
                "bin_center_days",
            ]
        )

    return pd.concat(out_list, ignore_index=True)


def summarize_weight_months(df: pd.DataFrame) -> pd.DataFrame:
    if "baseline_a1c_category" not in df.columns:
        raise ValueError("Expected column 'baseline_a1c_category' in weight dataset")

    rows: List[dict] = []
    for group, gdf in df.groupby("baseline_a1c_category", observed=True):
        for m in TIME_MONTHS:
            center = m * 30
            lo = center - WINDOW_DAYS
            hi = center + WINDOW_DAYS
            sub = gdf[(gdf["days_from_baseline"] >= lo) & (gdf["days_from_baseline"] <= hi)]
            if sub.empty:
                continue
            vals = sub["pct_weight_change"].astype(float)
            rows.append(
                {
                    "baseline_a1c_category": group,
                    "month": m,
                    "n": vals.shape[0],
                    "mean_pct_weight_change": vals.mean(),
                    "sd_pct_weight_change": vals.std(ddof=1),
                }
            )
    out_df = pd.DataFrame(rows)
    if out_df.empty:
        return out_df
    out_df.sort_values(["baseline_a1c_category", "month"], inplace=True)
    return out_df


def summarize_a1c_months(df: pd.DataFrame) -> pd.DataFrame:
    if "baseline_a1c_category" not in df.columns:
        raise ValueError("Expected column 'baseline_a1c_category' in A1C dataset")

    has_value = "a1c_value" in df.columns
    if "abs_a1c_change" not in df.columns:
        raise ValueError("Expected column 'abs_a1c_change' in A1C dataset")
    if not has_value:
        base_col = None
        if "baseline_a1c_final" in df.columns:
            base_col = "baseline_a1c_final"
        elif "baseline_a1c" in df.columns:
            base_col = "baseline_a1c"
        if base_col is None:
            raise ValueError("Expected either 'a1c_value' or a baseline A1C column in A1C dataset")
        df = df.copy()
        df["a1c_value"] = df[base_col].astype(float) + df["abs_a1c_change"].astype(float)

    rows: List[dict] = []
    for group, gdf in df.groupby("baseline_a1c_category", observed=True):
        for m in TIME_MONTHS:
            center = m * 30
            lo = center - WINDOW_DAYS
            hi = center + WINDOW_DAYS
            sub = gdf[(gdf["days_from_baseline"] >= lo) & (gdf["days_from_baseline"] <= hi)]
            if sub.empty:
                continue
            vals_level = sub["a1c_value"].astype(float)
            vals_change = sub["abs_a1c_change"].astype(float)
            rows.append(
                {
                    "baseline_a1c_category": group,
                    "month": m,
                    "n": vals_change.shape[0],
                    "mean_a1c_value": vals_level.mean(),
                    "sd_a1c_value": vals_level.std(ddof=1),
                    "mean_abs_a1c_change": vals_change.mean(),
                    "sd_abs_a1c_change": vals_change.std(ddof=1),
                }
            )
    out_df = pd.DataFrame(rows)
    if out_df.empty:
        return out_df
    out_df.sort_values(["baseline_a1c_category", "month"], inplace=True)
    return out_df


def summarize_weight_level_months(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly windows (±30d) for absolute weight by baseline A1C group.
    Expects columns: baseline_a1c_category, weight_in_pounds_final
    """
    if "baseline_a1c_category" not in df.columns:
        raise ValueError("Expected column 'baseline_a1c_category' in weight dataset")
    if "weight_in_pounds_final" not in df.columns:
        raise ValueError("Expected column 'weight_in_pounds_final' in weight dataset")
    rows: List[dict] = []
    for group, gdf in df.groupby("baseline_a1c_category", observed=True):
        for m in TIME_MONTHS:
            center = m * 30
            lo = center - WINDOW_DAYS
            hi = center + WINDOW_DAYS
            sub = gdf[(gdf["days_from_baseline"] >= lo) & (gdf["days_from_baseline"] <= hi)]
            if sub.empty:
                continue
            vals = sub["weight_in_pounds_final"].astype(float)
            rows.append(
                {
                    "baseline_a1c_category": group,
                    "month": m,
                    "n": vals.shape[0],
                    "mean_weight_lbs": vals.mean(),
                    "sd_weight_lbs": vals.std(ddof=1),
                }
            )
    out_df = pd.DataFrame(rows)
    if out_df.empty:
        return out_df
    out_df.sort_values(["baseline_a1c_category", "month"], inplace=True)
    return out_df


def _plot_observed_trajectories(summary: pd.DataFrame, value_col: str, group_label: str, ylabel: str, title_prefix: str, out_path: Path, max_days: int) -> None:
    """Grouped observed trajectories over days, matching predictive axis style."""
    if summary.empty:
        return
    # Derive the sd column name from value_col
    sd_col = value_col.replace("mean_", "sd_")
    plt.figure(figsize=(12, 7))
    groups = list(summary[group_label].unique())
    try:
        import matplotlib as mpl
        cmap = mpl.colormaps.get_cmap("tab10")
    except Exception:
        cmap = None

    for idx, grp in enumerate(groups):
        g = summary[summary[group_label] == grp].copy()
        x = g["bin_center_days"].to_numpy(dtype=float)
        y = g[value_col].to_numpy(dtype=float)
        color = cmap(idx % 10) if cmap is not None else None
        line, = plt.plot(x, y, label=str(grp), color=color)
        # Add 95% CI band if sd and n columns exist
        if sd_col in g.columns and "n" in g.columns:
            sd = g[sd_col].to_numpy(dtype=float)
            n = g["n"].to_numpy(dtype=float)
            se = np.where(n > 1, sd / np.sqrt(n), 0.0)
            ci_lo = y - Z_CRIT * se
            ci_hi = y + Z_CRIT * se
            valid = np.isfinite(y) & np.isfinite(ci_lo)
            if np.any(valid):
                plt.fill_between(x[valid], ci_lo[valid], ci_hi[valid], alpha=0.12, color=line.get_color())

    # Use solid baseline and cap x-axis to requested horizon
    plt.axhline(0, color="black", linestyle="-", linewidth=1)
    plt.xlabel("Days from Baseline")
    plt.ylabel(ylabel)
    plt.title(f"{title_prefix} by {group_label} (observed)")
    plt.xlim(0, int(max_days))
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_monthly_points(
    summary: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
    y_limits: Optional[tuple] = None,
    max_days: Optional[int] = None,
) -> None:
    """Plot month-specific observed means as points/lines by baseline A1C.

    Expects a month-summary DataFrame with columns:
    - baseline_a1c_category
    - month
    - value_col (mean column to plot)
    """
    if summary.empty:
        return
    plt.figure(figsize=(12, 7))
    groups = list(summary["baseline_a1c_category"].unique())
    try:
        import matplotlib as mpl
        cmap = mpl.colormaps.get_cmap("tab10")
    except Exception:
        cmap = None

    for idx, grp in enumerate(groups):
        g = summary[summary["baseline_a1c_category"] == grp].copy()
        x = (g["month"] * 30.0).to_numpy(dtype=float)
        y = g[value_col].to_numpy(dtype=float)
        color = cmap(idx % 10) if cmap is not None else None
        line, = plt.plot(x, y, marker="o", label=str(grp), color=color)
        # Add 95% CI band if sd and n columns exist
        sd_col = value_col.replace("mean_", "sd_")
        if sd_col in g.columns and "n" in g.columns:
            sd = g[sd_col].to_numpy(dtype=float)
            n = g["n"].to_numpy(dtype=float)
            se = np.where(n > 1, sd / np.sqrt(n), 0.0)
            ci_lo = y - Z_CRIT * se
            ci_hi = y + Z_CRIT * se
            valid = np.isfinite(y) & np.isfinite(ci_lo)
            if np.any(valid):
                plt.fill_between(x[valid], ci_lo[valid], ci_hi[valid], alpha=0.12, color=line.get_color())

    # Solid baseline
    plt.axhline(0, color="black", linestyle="-", linewidth=1)
    plt.xlabel("Days from Baseline")
    # Cap x-axis (default to 1.5 years if not provided)
    if max_days is None:
        max_days = 548
    plt.xlim(0, int(max_days))
    # Y-axis handling
    if y_limits is not None:
        plt.ylim(*y_limits)
    else:
        if value_col == "mean_a1c_value":
            plt.ylim(4, 12)
        else:
            vals = summary[value_col].astype(float)
            if vals.notna().any():
                vmin = np.nanpercentile(vals, 2)
                vmax = np.nanpercentile(vals, 98)
                if np.isfinite(vmin) and np.isfinite(vmax) and vmin != vmax:
                    pad = 0.05 * (vmax - vmin)
                    plt.ylim(vmin - pad, vmax + pad)
    
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def parse_args(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(
        description="Step 4: 90-day observed summaries for weight and A1C",
    )
    p.add_argument(
        "--weight-csv",
        default=os.path.join(
            "output", "step1_prepare_analysis_dataset", "analysis_ready_gap90.csv"
        ),
        help="Path to weight analysis-ready CSV from Step 1 (weight)",
    )
    p.add_argument(
        "--a1c-csv",
        default=os.path.join(
            "output",
            "step1_prepare_analysis_dataset_a1c",
            "analysis_ready_a1c_gap90.csv",
        ),
        help="Path to A1C analysis-ready CSV from Step 1 (A1C)",
    )
    p.add_argument(
        "--outdir",
        default=os.path.join("output", "step4_observed_summary_plots"),
        help="Directory to write summary CSVs",
    )
    p.add_argument(
        "--adherence-gap-days",
        type=int,
        default=None,
        help="Adherence gap in days for gap-specific output folder (e.g., 90)",
    )
    p.add_argument(
        "--max-days",
        type=int,
        default=548,
        help="Maximum days from baseline to include (default: 548)",
    )
    p.add_argument(
        "--bin-width",
        type=int,
        default=90,
        help="Bin width in days (default: 90)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    configure_logging(args.log_level)

    # Infer gap for output routing
    import re
    gap = args.adherence_gap_days
    if gap is None:
        for s in [args.weight_csv, args.a1c_csv]:
            m = re.search(r"gap[_]?(\d+)", str(s))
            if m:
                try:
                    gap = int(m.group(1))
                    break
                except Exception as exc:
                    logging.debug(
                        "could not parse a gap number from the input path (%s); falling back to the --adherence-gap-days value",
                        exc,
                    )
    outdir = Path(args.outdir)
    if gap is not None and "gap_" not in str(outdir):
        outdir = Path("output") / f"gap_{gap}" / Path(args.outdir).name
    outdir.mkdir(parents=True, exist_ok=True)

    logging.info("Reading weight dataset from %s", args.weight_csv)
    w = pd.read_csv(args.weight_csv)
    weight_summary = summarize_weight(w, max_days=args.max_days, bin_width=args.bin_width)
    weight_path = outdir / "summary_weight_by_baseline_a1c_90day.csv"
    weight_summary.to_csv(weight_path, index=False)
    logging.info("Wrote weight 90-day summary to %s", weight_path)

    logging.info("Reading A1C dataset from %s", args.a1c_csv)
    a = pd.read_csv(args.a1c_csv)
    a1c_summary = summarize_a1c(a, max_days=args.max_days, bin_width=args.bin_width)
    a1c_path = outdir / "summary_a1c_by_baseline_a1c_90day.csv"
    a1c_summary.to_csv(a1c_path, index=False)
    logging.info("Wrote A1C 90-day summary to %s", a1c_path)

    # Month-window summaries
    weight_months = summarize_weight_months(w)
    weight_months_path = outdir / "summary_weight_by_baseline_a1c_months.csv"
    weight_months.to_csv(weight_months_path, index=False)
    logging.info("Wrote weight month-window summary to %s", weight_months_path)

    # Absolute weight monthly summary
    weight_level_months = summarize_weight_level_months(w)
    weight_level_months_path = outdir / "summary_weight_level_by_baseline_a1c_months.csv"
    weight_level_months.to_csv(weight_level_months_path, index=False)
    logging.info("Wrote weight level month-window summary to %s", weight_level_months_path)

    # Compute baseline mean weights per A1C group and set y-axis upper bound
    try:
        base_df = w[w["days_from_baseline"] == 0]
        baseline_means = (
            base_df.groupby("baseline_a1c_category", observed=True)["weight_in_pounds_final"].mean()
            if not base_df.empty and "weight_in_pounds_final" in base_df.columns
            else pd.Series([], dtype=float)
        )
        baseline_max = float(baseline_means.max()) if not baseline_means.empty else float(np.nan)
    except Exception:
        baseline_max = float(np.nan)
    # Determine dynamic lower bound from monthly means and apply small padding
    monthly_vals = weight_level_months["mean_weight_lbs"].astype(float)
    if monthly_vals.notna().any():
        y_min = float(np.nanmin(monthly_vals))
    else:
        y_min = float("nan")
    pad = 2.0
    y_limits_weight_level = None
    if np.isfinite(baseline_max):
        upper = baseline_max + pad
        lower = y_min - pad if np.isfinite(y_min) else None
        if lower is not None:
            y_limits_weight_level = (lower, upper)
        else:
            y_limits_weight_level = (0, upper)

    a1c_months = summarize_a1c_months(a)
    a1c_months_path = outdir / "summary_a1c_by_baseline_a1c_months.csv"
    a1c_months.to_csv(a1c_months_path, index=False)
    logging.info("Wrote A1C month-window summary to %s", a1c_months_path)

    # Observed trajectory-style plots (mirroring predictive style)
    plots_dir = outdir / "plots"
    _plot_observed_trajectories(
        weight_summary,
        value_col="mean_pct_weight_change",
        group_label="baseline_a1c_category",
        ylabel="Percent Weight Change (Centered)",
        title_prefix="Observed Percent Weight Change",
        out_path=plots_dir / "observed_weight_centered_by_a1c.png",
        max_days=args.max_days,
    )

    _plot_observed_trajectories(
        a1c_summary,
        value_col="mean_abs_a1c_change",
        group_label="baseline_a1c_category",
        ylabel="Absolute A1C Change (Centered)",
        title_prefix="Observed Absolute A1C Change",
        out_path=plots_dir / "observed_a1c_centered_by_a1c.png",
        max_days=args.max_days,
    )

    # Monthly observed plots (weight % change and A1C level + change)
    _plot_monthly_points(
        weight_months,
        value_col="mean_pct_weight_change",
        ylabel="Percent Weight Change",
        title="Observed Percent Weight Change at Selected Months by Baseline A1C",
        out_path=plots_dir / "observed_weight_monthly_by_a1c.png",
        y_limits=None,
        max_days=args.max_days,
    )

    # New absolute weight monthly plot
    _plot_monthly_points(
        weight_level_months,
        value_col="mean_weight_lbs",
        ylabel="Weight (lbs)",
        title="Observed Weight at Selected Months by Baseline A1C",
        out_path=plots_dir / "observed_weight_level_monthly_by_a1c.png",
        y_limits=y_limits_weight_level,
        max_days=args.max_days,
    )

    _plot_monthly_points(
        a1c_months,
        value_col="mean_a1c_value",
        ylabel="A1C Level",
        title="Observed A1C Level at Selected Months by Baseline A1C",
        out_path=plots_dir / "observed_a1c_level_monthly_by_a1c.png",
        y_limits=(4, 12),
        max_days=args.max_days,
    )

    _plot_monthly_points(
        a1c_months,
        value_col="mean_abs_a1c_change",
        ylabel="Absolute A1C Change",
        title="Observed Absolute A1C Change at Selected Months by Baseline A1C",
        out_path=plots_dir / "observed_a1c_change_monthly_by_a1c.png",
        y_limits=None,
        max_days=args.max_days,
    )


if __name__ == "__main__":
    main()
