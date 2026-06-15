#!/usr/bin/env python3
"""Compute descriptive stats for ED2 M22 enrichment data."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protein_design.analysis.entropy import position_entropy  # noqa: E402
from protein_design.constants import C05_CDRH3  # noqa: E402
from protein_design.dms_splitting import dataset_spec, resolve_dataset_split  # noqa: E402

TARGET_NUM_MUT = (2, 3, 4, 5)
DATASET_KEY = "ed2_m22"
DMS_CONFIG_PATH = REPO_ROOT / "conf" / "data" / "dms" / "default.yaml"
LOCAL_RUNTIME_DIR = REPO_ROOT / "data" / "processed" / "stats_d2"
LOCAL_RAW_FILENAME = "ED2_M22_binding_enrichment.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "ED2_M22_binding_enrichment.csv",
        help="Path to ED2_M22_binding_enrichment.csv",
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
        default=REPO_ROOT / "plots",
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
        "count_ED2M22pos",
        "count_ED2M22neg",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result: dict[str, Any] = {"input_csv": str(csv_path), "num_mut_stats": {}}
    if {"count_ED2M22pos", "count_ED2M22neg"}.issubset(df.columns):
        has_count_ED2M22pos = df["count_ED2M22pos"].notna()
        has_count_ED2M22neg = df["count_ED2M22neg"].notna()
        both_present = has_count_ED2M22pos & has_count_ED2M22neg
        result["rows_with_both_count_ED2M22pos_and_count_ED2M22neg"] = int(both_present.sum())
        result["rows_missing_at_least_one_of_count_ED2M22pos_count_ED2M22neg"] = int((~both_present).sum())
    else:
        result["rows_with_both_count_ED2M22pos_and_count_ED2M22neg"] = None
        result["rows_missing_at_least_one_of_count_ED2M22pos_count_ED2M22neg"] = None

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


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _build_local_dms_config() -> Path:
    dms_cfg = _load_yaml(DMS_CONFIG_PATH)
    local_raw_path = REPO_ROOT / "data" / "raw" / LOCAL_RAW_FILENAME
    datasets = dms_cfg.get("datasets", {}) or {}
    if DATASET_KEY in datasets and local_raw_path.exists():
        datasets[DATASET_KEY]["path"] = str(local_raw_path)
    split_cfg = dms_cfg.get("split", {}) or {}
    split_cfg["output_dir"] = str(REPO_ROOT / "data" / "dms_splits")

    LOCAL_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    local_config_path = LOCAL_RUNTIME_DIR / "dms.local.yaml"
    with local_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dms_cfg, handle, sort_keys=False)
    return local_config_path


def plot_train_entropy_heatmap(plots_dir: Path) -> Path:
    dms_config_path = _build_local_dms_config()
    split_path = resolve_dataset_split(DATASET_KEY, "train", dms_config_path, force=False)
    spec = dataset_spec(DATASET_KEY, dms_config_path)
    train_df = pd.read_csv(split_path)
    sequences = [
        seq for seq in train_df[spec.sequence_col].astype(str).tolist()
        if len(seq) == len(C05_CDRH3)
    ]
    entropies = position_entropy(sequences, expected_length=len(C05_CDRH3))

    positions = np.arange(1, len(entropies) + 1)
    output_path = plots_dir / "ED2_train_temp_entropy_heatmap.png"
    fig, ax = plt.subplots(figsize=(11.5, 2.6))
    image = ax.imshow(entropies[np.newaxis, :], aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xticks(np.arange(len(entropies)))
    ax.set_xticklabels([f"{idx}\n{aa}" for idx, aa in zip(positions, C05_CDRH3)], fontsize=8)
    ax.set_yticks([0])
    ax.set_yticklabels(["train"])
    ax.set_xlabel("CDR-H3 position (WT residue)")
    ax.set_title("ED2 train split: position-wise entropy")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Shannon entropy (bits)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_m22_enrichment_distribution(df: pd.DataFrame, output_path: Path) -> bool:
    df = df[df["num_mut"].isin(TARGET_NUM_MUT)]
    if "M22_binding_enrichment_adj" not in df.columns:
        return False

    values = df["M22_binding_enrichment_adj"].dropna()
    if values.empty:
        return False

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=50, edgecolor="black", alpha=0.8)
    plt.title("ED2 Distribution of M22_binding_enrichment_adj", fontsize=20)
    plt.xlabel("M22_binding_enrichment_adj", fontsize=16)
    plt.ylabel("Count", fontsize=16)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def plot_log_ratio_distribution(df: pd.DataFrame, output_path: Path) -> bool:
    df = df[df["num_mut"].isin(TARGET_NUM_MUT)]
    required = {"count_ED2M22pos", "count_ED2M22neg"}
    if not required.issubset(df.columns):
        return False

    pos = df["count_ED2M22pos"]
    neg = df["count_ED2M22neg"]
    valid = pos.notna() & neg.notna()
    if not valid.any():
        return False

    values = ((pos[valid] + 0.5) / (neg[valid] + 0.5)).map(math.log).dropna()
    if values.empty:
        return False

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=50, edgecolor="black", alpha=0.8)
    plt.title("Distribution of log(count_ED2M22pos / count_ED2M22neg)")
    plt.xlabel("log(count_ED2M22pos / count_ED2M22neg)")
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
    plt.title(f"Distribution of {column}", fontsize=20)
    plt.xlabel(column, fontsize=16)
    plt.ylabel("Count", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
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
        f"{args.input.stem}_log_count_ED2M22pos_over_count_ED2M22neg_distribution.png"
    )
    ratio_plotted = plot_log_ratio_distribution(df, ratio_plot_output)
    if ratio_plotted:
        print(
            f"Saved log(count_ED2M22pos/count_ED2M22neg) distribution plot to: {ratio_plot_output}",
            file=sys.stderr,
        )
    else:
        print(
            "Skipped log(count_ED2M22pos/count_ED2M22neg) plot: required columns missing or no valid values.",
            file=sys.stderr,
        )
    count_ed2m22pos_output = args.plots_dir / f"{args.input.stem}_count_ED2M22pos_distribution.png"
    count_ed2m22pos_plotted = plot_column_distribution(df, "count_ED2M22pos", count_ed2m22pos_output)
    if count_ed2m22pos_plotted:
        print(
            f"Saved count_ED2M22pos distribution plot to: {count_ed2m22pos_output}",
            file=sys.stderr,
        )
    else:
        print(
            "Skipped count_ED2M22pos plot: column missing or no non-null values.",
            file=sys.stderr,
        )
    count_ed2ed2_output = args.plots_dir / f"{args.input.stem}_count_ED2Ed5_distribution.png"
    count_ed2ed2_plotted = plot_column_distribution(df, "count_ED2pre", count_ed2ed2_output)
    if count_ed2ed2_plotted:
        print(
            f"Saved count_ED2pre distribution plot to: {count_ed2ed2_output}",
            file=sys.stderr,
        )
    else:
        print(
            "Skipped count_ED2pre plot: column missing or no non-null values.",
            file=sys.stderr,
        )
    entropy_output = plot_train_entropy_heatmap(args.plots_dir)
    print(f"Saved train entropy heatmap to: {entropy_output}", file=sys.stderr)


if __name__ == "__main__":
    main()
