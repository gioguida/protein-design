#!/usr/bin/env python
"""Plot CDRH3 length distribution from oas_filtered.csv.gz.

Usage:
    python scripts/plot_cdrh3_lengths.py [csv_path] [--output PATH] [--highlight N]

Reads only the cdr3_aa column (memory-efficient) and produces a histogram with
KDE overlay. A vertical line marks the highlighted length (default: 24, C05's CDRH3).
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless backend for compute nodes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import gaussian_kde

load_dotenv()

_PROJECT_DIR = os.environ.get("PROJECT_DIR", ".")
DEFAULT_CSV = os.path.join(_PROJECT_DIR, "datasets", "oas_filtered.csv.gz")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CDRH3 length distribution")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=DEFAULT_CSV,
        help=f"Path to oas_filtered.csv.gz (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output PNG path (default: cdrh3_length_distribution.png next to the CSV)",
    )
    parser.add_argument(
        "--highlight",
        type=int,
        default=24,
        help="CDRH3 length to annotate (default: 24, C05)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"Error: {args.csv_path} not found.", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or os.path.join(
        os.path.dirname(args.csv_path), "cdrh3_length_distribution.png"
    )

    print(f"Reading {args.csv_path} ...")
    df = pd.read_csv(args.csv_path, usecols=["cdr3_aa"], compression="gzip")
    lengths = df["cdr3_aa"].dropna().str.len()
    del df  # free memory

    if lengths.empty:
        print("Error: no valid cdr3_aa entries found.", file=sys.stderr)
        sys.exit(1)

    # ── Summary stats ──────────────────────────────────────────────────────────
    n_total = len(lengths)
    n_highlight = int((lengths == args.highlight).sum())
    pct_highlight = 100.0 * n_highlight / n_total
    mode_val = int(lengths.mode().iloc[0])

    print(f"\nCDRH3 length statistics ({n_total:,} sequences):")
    print(f"  Min:    {lengths.min()}")
    print(f"  Max:    {lengths.max()}")
    print(f"  Mean:   {lengths.mean():.2f}")
    print(f"  Median: {int(lengths.median())}")
    print(f"  Mode:   {mode_val}")
    print(f"\n  Length {args.highlight} (C05): {n_highlight:,} sequences ({pct_highlight:.2f}%)")

    # ── Plot ───────────────────────────────────────────────────────────────────
    counts = lengths.value_counts().sort_index()
    freqs = counts / n_total

    x_min = max(1, int(lengths.min()))
    x_max = min(40, int(lengths.max()))

    fig, ax = plt.subplots(figsize=(13, 5))

    # Histogram bars
    ax.bar(
        counts.index, freqs.values,
        width=0.8, color="#4C72B0", alpha=0.65, label="Relative frequency",
    )

    # KDE overlay (evaluated on a fine grid, then scaled to match bar heights)
    kde = gaussian_kde(lengths, bw_method=0.4)
    x_fine = np.linspace(x_min, x_max, 500)
    kde_vals = kde(x_fine)
    # Scale KDE so its area over integer bins ≈ 1 (matches bar heights)
    ax.plot(x_fine, kde_vals, color="#DD8452", linewidth=2, label="KDE")

    # Vertical line at highlighted length
    hl = args.highlight
    ax.axvline(hl, color="crimson", linestyle="--", linewidth=1.8,
               label=f"C05 CDRH3 (len={hl})")

    # Annotation at the highlighted bar
    if hl in freqs.index:
        bar_top = float(freqs[hl])
        ax.annotate(
            f"n={n_highlight:,}\n({pct_highlight:.1f}%)",
            xy=(hl, bar_top),
            xytext=(hl + 1.5, bar_top + 0.004),
            fontsize=9,
            color="crimson",
            arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="crimson", alpha=0.8),
        )

    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    ax.set_xlabel("CDRH3 length (aa)", fontsize=12)
    ax.set_ylabel("Relative frequency", fontsize=12)
    ax.set_title(
        f"CDRH3 Length Distribution — OAS filtered corpus (n={n_total:,})",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")


if __name__ == "__main__":
    main()
