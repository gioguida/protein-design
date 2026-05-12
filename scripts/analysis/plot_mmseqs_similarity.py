#!/usr/bin/env python3
"""Plot MMseqs2 pident distribution between OAS sequences and C05.

Reads a MMseqs2 results.tsv produced by either:
  - scripts/data_prep/search_c05.py           (full VH chain)
  - scripts/data_prep/extract_c05_cdrh3_mmseqs.py  (CDR-H3 only)

NOTE: MMseqs2 only returns hits above --min-seq-id, so the distribution is
truncated on the left — it does not represent all OAS sequences, only those
above the search threshold.

Produces three PDFs:
  1. <output>                      — histogram of pident among all hits
  2. <stem>_cumulative.<ext>       — cumulative count vs threshold
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

_PROJECT_DIR = os.environ.get("PROJECT_DIR", ".")
_SCRATCH_DIR = os.environ.get("SCRATCH_DIR", ".")

MMSEQS_COLS_FULL = [
    "query", "target", "pident", "alnlen", "mismatch", "gapopen",
    "qstart", "qend", "tstart", "tend", "evalue", "bits", "qcov", "tcov", "tseq",
]
MMSEQS_COLS_CDR = [
    "query", "target", "pident", "alnlen", "mismatch", "gapopen",
    "qstart", "qend", "tstart", "tend", "evalue", "bits",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--tsv",
        required=True,
        help="Path to MMseqs2 results.tsv (from search_c05.py or extract_c05_cdrh3_mmseqs.py).",
    )
    p.add_argument(
        "--label",
        default="VH chain",
        help="Short description used in plot titles, e.g. 'full VH chain' or 'CDR-H3'.",
    )
    p.add_argument(
        "--output", "-o",
        default=None,
        help="Output PDF path for the histogram. Cumulative PDF gets a _cumulative suffix. "
             "Default: $PROJECT_DIR/plots/oas_similarity/mmseqs_<label>.pdf",
    )
    p.add_argument(
        "--bins", type=int, default=60,
        help="Number of histogram bins (default: 60).",
    )
    return p.parse_args()


def load_tsv(path: Path) -> pd.DataFrame:
    """Load MMseqs2 results.tsv, auto-detecting column count."""
    # Peek at first line to count columns
    with open(path) as f:
        first = f.readline()
    ncols = len(first.split("\t"))
    cols = MMSEQS_COLS_FULL if ncols >= 15 else MMSEQS_COLS_CDR
    df = pd.read_csv(path, sep="\t", header=None, names=cols[:ncols])
    # MMseqs2 reports pident as percentage (0-100); keep as-is for plotting
    return df


def main() -> None:
    args = parse_args()
    tsv_path = Path(args.tsv)

    if not tsv_path.exists():
        sys.exit(
            f"Error: results file not found: {tsv_path}\n"
            "Run the corresponding MMseqs2 search sbatch job first."
        )

    print(f"Loading {tsv_path} ...", flush=True)
    df = load_tsv(tsv_path)

    if df.empty:
        sys.exit("Error: results file is empty — no MMseqs2 hits found.")

    pident = df["pident"].dropna().values  # already in percent (0-100)
    n_hits = len(pident)

    print(f"\nMMseqs2 pident — {n_hits:,} hits ({args.label}):")
    print(f"  Min   : {pident.min():.1f}%")
    print(f"  Max   : {pident.max():.1f}%")
    print(f"  Mean  : {pident.mean():.1f}%")
    print(f"  Median: {float(np.median(pident)):.1f}%")
    for t in [30, 40, 50, 60, 70, 80, 90]:
        n = int((pident >= t).sum())
        print(f"  Hits ≥ {t}%: {n:,}  ({100*n/n_hits:.1f}%)")

    # Output paths
    label_slug = args.label.lower().replace(" ", "_")
    default_out = os.path.join(
        _PROJECT_DIR, "plots", "oas_similarity", f"mmseqs_{label_slug}.pdf"
    )
    out = Path(args.output or default_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_cumul = out.with_stem(out.stem + "_cumulative")

    x_lo = float(np.floor(pident.min() / 5) * 5)
    x_hi = min(105.0, float(np.ceil(pident.max() / 5) * 5 + 5))

    # ── Plot 1: histogram ─────────────────────────────────────────────────────
    bin_edges = np.linspace(x_lo, x_hi, args.bins + 1)
    counts, edges = np.histogram(pident, bins=bin_edges)
    bar_w = edges[1] - edges[0]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        edges[:-1], counts, width=bar_w * 0.9,
        align="edge", color="#4C72B0", alpha=0.75,
    )
    ax.axvline(100.0, color="crimson", linestyle="--", linewidth=1.8, label="C05 WT (100%)")
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel(f"MMseqs2 sequence identity to C05 {args.label} (%)", fontsize=12)
    ax.set_ylabel("Number of OAS hits", fontsize=12)
    ax.set_title(
        f"MMseqs2 Identity Distribution — C05 {args.label} vs OAS\n"
        f"({n_hits:,} hits above search threshold; distribution is left-truncated)",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot 1 saved to: {out}")

    # ── Plot 2: cumulative count vs threshold ─────────────────────────────────
    thresholds = np.linspace(x_lo, pident.max(), 300)
    cumul = np.array([(pident >= t).sum() for t in thresholds])

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.plot(thresholds, cumul, color="#4C72B0", linewidth=2)
    ax2.fill_between(thresholds, cumul, alpha=0.15, color="#4C72B0")

    for t_mark in [30, 40, 50, 60, 70, 80, 90]:
        if x_lo <= t_mark <= pident.max():
            n_mark = int((pident >= t_mark).sum())
            ax2.axvline(t_mark, color="#888888", linestyle=":", linewidth=0.9)
            ax2.annotate(
                f"{n_mark:,}",
                xy=(t_mark, n_mark),
                xytext=(t_mark + 0.5, n_mark * 1.06),
                fontsize=8, color="#333333",
            )

    ax2.set_xlim(x_lo, x_hi)
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel("Identity threshold (%)", fontsize=12)
    ax2.set_ylabel("OAS hits ≥ threshold", fontsize=12)
    ax2.set_title(
        f"Cumulative hit count above MMseqs2 identity threshold\n"
        f"C05 {args.label} vs OAS  ({n_hits:,} total hits)",
        fontsize=12,
    )
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    plt.tight_layout()
    fig2.savefig(out_cumul, bbox_inches="tight")
    plt.close(fig2)
    print(f"Plot 2 saved to: {out_cumul}")


if __name__ == "__main__":
    main()
