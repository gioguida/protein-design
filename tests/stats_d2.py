#!/usr/bin/env python3
"""Compute descriptive stats for ED5 M22 enrichment data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

TARGET_NUM_MUT = (2, 3, 4, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("../data/raw/ED2_M22_binding_enrichment.csv"),
        help="Path to ED2_M22_binding_enrichment.csv",
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
        "count_ED2M22pos",
        "count_ED2M22neg",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result: dict[str, Any] = {"input_csv": str(csv_path), "num_mut_stats": {}}
    for num_mut in TARGET_NUM_MUT:
        subset = df[df["num_mut"] == num_mut]
        # enilinate duplicate sqeuences
        subset = subset.drop_duplicates(subset=["aa"])


        has_enrichment = subset["M22_binding_enrichment_adj"].notna()
        has_pos = subset["count_ED2M22pos"].notna()
        has_neg = subset["count_ED2M22neg"].notna()

        result["num_mut_stats"][str(num_mut)] = {
            "total_rows": int(len(subset)),
            "num_with_all_three_columns": int((has_enrichment & has_pos & has_neg).sum()),
            "num_with_enrichment_only": int((has_enrichment).sum()),
            "distribution_M22_binding_enrichment_adj": summarize_distribution(
                subset["M22_binding_enrichment_adj"]
            ),
        }

    return result


def main() -> None:
    args = parse_args()
    stats = compute_stats(args.input)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
