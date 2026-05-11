#!/usr/bin/env python3
"""Compute descriptive stats for ED811 M22 enrichment data."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

TARGET_NUM_MUT = (5, 6, 7, 8, 9, 10, 11)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("../data/raw/ED811_M22_enrichment_full.csv"),
        help="Path to ED811_M22_enrichment_full.csv",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help=(
            "Optional output path for M22_binding_enrichment_adj distribution plot "
            "(default: <input_stem>_M22_enrichment_distribution.png)."
        ),
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("../plots"),
        help="Directory where all plot files are saved (default: ../plots/).",
    )
    return parser.parse_args()


def summarize_distribution(values: pd.Series) -> dict[str, Any]:
    clean = values.dropna()
    if clean.empty:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "max": None,
        }

    desc = clean.describe(percentiles=[0.25, 0.5, 0.75])
    summary = {
        "positives": int((clean > 5.190013461).sum()),
        "count": int(desc["count"]),
        "mean": float(desc["mean"]),
        "std": float(desc["std"]) if pd.notna(desc["std"]) else 0.0,
        "min": float(desc["min"]),
        "p25": float(desc["25%"]),
        "p50": float(desc["50%"]),
        "p75": float(desc["75%"]),
        "max": float(desc["max"]),
    }
    return summary


def compute_stats(csv_path: Path) -> dict[str, Any]:
    df = pd.read_csv(csv_path)
    required_cols = {
        "num_mut",
        "M22_binding_enrichment_adj",
        "count_ED811M22r1pos",
        "count_ED811M22neg",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result: dict[str, Any] = {"input_csv": str(csv_path), "num_mut_stats": {}}
    if {"count_ED811M22r1pos", "count_ED811M22neg"}.issubset(df.columns):
        has_count_ED811M22r1pos = df["count_ED811M22r1pos"].notna()
        has_count_ED811M22neg = df["count_ED811M22neg"].notna()
        both_present = has_count_ED811M22r1pos & has_count_ED811M22neg
        result["rows_with_both_count_ED811M22r1pos_and_count_ED811M22neg"] = int(both_present.sum())
        result["rows_missing_at_least_one_of_count_ED811M22r1pos_count_ED811M22neg"] = int((~both_present).sum())
    else:
        result["rows_with_both_count_ED811M22r1pos_and_count_ED811M22neg"] = None
        result["rows_missing_at_least_one_of_count_ED811M22r1pos_count_ED811M22neg"] = None

    for num_mut in TARGET_NUM_MUT:
        subset = df[df["num_mut"] == num_mut]
        # enilinate duplicate sqeuences
        subset = subset.drop_duplicates(subset=["aa"])


        has_enrichment = subset["M22_binding_enrichment_adj"].notna()
        has_pos = subset["count_ED811M22r1pos"].notna()
        has_neg = subset["count_ED811M22neg"].notna()

        result["num_mut_stats"][str(num_mut)] = {
            "total_rows": int(len(subset)),
            "num_with_all_three_columns": int((has_enrichment & has_pos & has_neg).sum()),
            "num_with_enrichment_only": int((has_enrichment).sum()),
            "distribution_M22_binding_enrichment_adj": summarize_distribution(
                subset["M22_binding_enrichment_adj"]
            ),
        }

    return result


def plot_m22_enrichment_distribution(df: pd.DataFrame, output_path: Path) -> bool:
    df = df[df["num_mut"].isin(TARGET_NUM_MUT)]
    if "M22_binding_enrichment_adj" not in df.columns:
        return False

    values = df["M22_binding_enrichment_adj"].dropna()
    if values.empty:
        return False

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=50, edgecolor="black", alpha=0.8)
    plt.title("Distribution of M22_binding_enrichment_adj")
    plt.xlabel("M22_binding_enrichment_adj")
    plt.ylabel("Count")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def plot_log_ratio_distribution(df: pd.DataFrame, output_path: Path) -> bool:
    df = df[df["num_mut"].isin(TARGET_NUM_MUT)]
    required = {"count_ED811M22r1pos", "count_ED811M22neg"}
    if not required.issubset(df.columns):
        return False

    pos = df["count_ED811M22r1pos"]
    neg = df["count_ED811M22neg"]
    valid = pos.notna() & neg.notna()
    if not valid.any():
        return False

    values = ((pos[valid] + 0.5) / (neg[valid] + 0.5)).map(math.log).dropna()
    if values.empty:
        return False

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=50, edgecolor="black", alpha=0.8)
    plt.title("Distribution of log(count_ED811M22r1pos / count_ED811M22neg)")
    plt.xlabel("log(count_ED811M22r1pos / count_ED811M22neg)")
    plt.ylabel("Count")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def plot_column_distribution(df: pd.DataFrame, column: str, output_path: Path) -> bool:
    df = df[df["num_mut"].isin(TARGET_NUM_MUT)]
    if column not in df.columns:
        return False

    values = df[column].dropna()
    if values.empty:
        return False

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=50, edgecolor="black", alpha=0.8)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def main() -> None:
    args = parse_args()
    stats = compute_stats(args.input)
    print(json.dumps(stats, indent=2))
    df = pd.read_csv(args.input)
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    plot_output = (
        args.plot_output
        if args.plot_output is not None
        else args.plots_dir / f"{args.input.stem}_M22_enrichment_distribution.png"
    )
    plotted = plot_m22_enrichment_distribution(df, plot_output)
    if plotted:
        print(f"Saved M22_enrichment distribution plot to: {plot_output}", file=sys.stderr)
    else:
        print(
            "Skipped M22_enrichment plot: column missing or no non-null values.",
            file=sys.stderr,
        )
    ratio_plot_output = args.plots_dir / (
        f"{args.input.stem}_log_count_ED811M22r1pos_over_count_ED811M22neg_distribution.png"
    )
    ratio_plotted = plot_log_ratio_distribution(df, ratio_plot_output)
    if ratio_plotted:
        print(
            f"Saved log(count_ED811M22r1pos/count_ED811M22neg) distribution plot to: {ratio_plot_output}",
            file=sys.stderr,
        )
    else:
        print(
            "Skipped log(count_ED811M22r1pos/count_ED811M22neg) plot: required columns missing or no valid values.",
            file=sys.stderr,
        )
    count_ED811M22r1pos_output = args.plots_dir / f"{args.input.stem}_count_ED811M22r1pos_distribution.png"
    count_ED811M22r1pos_plotted = plot_column_distribution(df, "count_ED811M22r1pos", count_ED811M22r1pos_output)
    if count_ED811M22r1pos_plotted:
        print(
            f"Saved count_ED811M22r1pos distribution plot to: {count_ED811M22r1pos_output}",
            file=sys.stderr,
        )
    else:
        print(
            "Skipped count_ED811M22r1pos plot: column missing or no non-null values.",
            file=sys.stderr,
        )
    count_ed811ed811_output = args.plots_dir / f"{args.input.stem}_count_ED811Ed811_distribution.png"
    count_ed811ed811_plotted = plot_column_distribution(df, "count_ED811Ed811", count_ed811ed811_output)
    if count_ed811ed811_plotted:
        print(
            f"Saved count_ED811Ed811 distribution plot to: {count_ed811ed811_output}",
            file=sys.stderr,
        )
    else:
        print(
            "Skipped count_ED811Ed811 plot: column missing or no non-null values.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
