#!/usr/bin/env python3
"""Plot BLOSUM62 CDR-H3 similarity distribution between OAS sequences and C05.

Reads the cached per-H3 alignment scores produced by
``scripts/data_prep/extract_c05_cdrh3_blosum.py`` (expected at
``$SCRATCH_DIR/c05_cdrh3_blosum/h3_scores.parquet``).

If the H3-to-seq-id mapping pickle is also present, the y-axis reflects total
OAS sequence counts (many seq_ids can share the same H3); otherwise it counts
unique H3 sequences.

Produces two PDF files:
  1. Full distribution (all sequences)
  2. Tail ≥ 20% similarity zoom
"""
from __future__ import annotations

import argparse
import os
import pickle
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

# C05 CDRH3 in OAS format (used only for annotation)
C05_CDRH3_OAS = "AKHMSMQQVVSAGWERADLVGDAFDV"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--scores",
        default=os.path.join(_SCRATCH_DIR, "c05_cdrh3_blosum", "h3_scores.parquet"),
        help="Path to h3_scores.parquet (default: $SCRATCH_DIR/c05_cdrh3_blosum/h3_scores.parquet)",
    )
    p.add_argument(
        "--mapping",
        default=os.path.join(_SCRATCH_DIR, "c05_cdrh3_blosum", "h3_mapping.pkl"),
        help="Path to h3_mapping.pkl for per-seq-id counts (optional).",
    )
    p.add_argument(
        "--output", "-o",
        default=os.path.join(_PROJECT_DIR, "plots", "oas_similarity", "cdrh3_blosum_similarity.pdf"),
        help="Output path for the full-distribution PDF (tail PDF gets a _tail20 suffix).",
    )
    p.add_argument(
        "--bins", type=int, default=80,
        help="Number of histogram bins (default: 80).",
    )
    p.add_argument(
        "--xmin", type=float, default=None,
        help="Left x-axis limit (default: auto).",
    )
    p.add_argument(
        "--xmax", type=float, default=None,
        help="Right x-axis limit (default: auto, capped at 105).",
    )
    return p.parse_args()


def load_seq_counts(mapping_path: Path, h3s: list[str]) -> np.ndarray:
    """Return per-H3 sequence count array aligned to h3s."""
    with open(mapping_path, "rb") as f:
        mapping: dict[str, list[str]] = pickle.load(f)
    return np.array([len(mapping.get(h, [])) for h in h3s], dtype=np.int64)


def main() -> None:
    args = parse_args()
    scores_path = Path(args.scores)
    mapping_path = Path(args.mapping)

    if not scores_path.exists():
        sys.exit(
            f"Error: scores file not found: {scores_path}\n"
            "Run  sbatch bash_scripts/utils/extract_c05_cdrh3_blosum.sbatch preview  first."
        )

    print(f"Loading {scores_path} ...", flush=True)
    df = pd.read_parquet(scores_path)

    # Drop NaN (sequences with characters outside BLOSUM62 alphabet)
    n_total = len(df)
    df = df.dropna(subset=["norm_score"]).reset_index(drop=True)
    n_valid = len(df)
    n_nan = n_total - n_valid
    print(f"  {n_valid:,} valid scores, {n_nan:,} NaN dropped.", flush=True)

    # Convert to percent
    scores_pct = df["norm_score"].values * 100.0

    # Determine y values (sequence counts vs unique H3 counts)
    use_seq_counts = mapping_path.exists()
    if use_seq_counts:
        print(f"Loading mapping from {mapping_path} ...", flush=True)
        weights = load_seq_counts(mapping_path, df["h3"].tolist())
        y_label = "OAS sequence count"
        total_seqs = int(weights.sum())
        print(f"  Total sequences represented: {total_seqs:,}", flush=True)
    else:
        weights = np.ones(n_valid, dtype=np.int64)
        y_label = "Unique CDR-H3 count"
        total_seqs = n_valid
        print("  Mapping not found — counting unique H3 sequences.", flush=True)

    # Summary stats
    print(f"\nNormalized BLOSUM62 score (%) — {n_valid:,} unique H3:")
    print(f"  Min   : {scores_pct.min():.1f}%")
    print(f"  Max   : {scores_pct.max():.1f}%")
    print(f"  Mean  : {scores_pct.mean():.1f}%")
    print(f"  Median: {float(np.median(scores_pct)):.1f}%")
    n_above50 = int((weights[scores_pct >= 50]).sum())
    n_above20 = int((weights[scores_pct >= 20]).sum())
    print(f"\n  Sequences with score ≥ 50%: {n_above50:,}  ({100*n_above50/total_seqs:.2f}%)")
    print(f"  Sequences with score ≥ 20%: {n_above20:,}  ({100*n_above20/total_seqs:.2f}%)")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_tail = out.with_stem(out.stem + "_tail20")

    x_lo = float(args.xmin) if args.xmin is not None else float(np.floor(scores_pct.min() / 5) * 5)
    x_hi = float(args.xmax) if args.xmax is not None else min(105.0, float(np.ceil(scores_pct.max() / 5) * 5 + 5))

    # ── Plot 1: full distribution ─────────────────────────────────────────────
    bin_edges = np.linspace(x_lo, x_hi, args.bins + 1)
    counts, edges = np.histogram(scores_pct, bins=bin_edges, weights=weights.astype(float))
    bar_w = edges[1] - edges[0]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(
        edges[:-1], counts, width=bar_w * 0.9,
        align="edge", color="#4C72B0", alpha=0.75, label="OAS CDR-H3 sequences",
    )
    ax.axvline(100.0, color="crimson", linestyle="--", linewidth=1.8, label="C05 WT (100%)")
    ax.axvline(20.0, color="#888888", linestyle=":", linewidth=1.2, label="≥20% threshold")
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel("BLOSUM62 normalized similarity to C05 CDR-H3 (%)", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(
        f"CDR-H3 BLOSUM62 Similarity to C05 — OAS filtered corpus\n"
        f"(n = {n_valid:,} unique H3 | {total_seqs:,} total sequences)",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot 1 saved to: {out}")

    # ── Plot 2: tail ≥ 20% ───────────────────────────────────────────────────
    TAIL = 20.0
    mask_tail = scores_pct >= TAIL
    n_tail = int(weights[mask_tail].sum())

    counts_tail, edges_tail = np.histogram(
        scores_pct[mask_tail], bins=40,
        weights=weights[mask_tail].astype(float),
        range=(TAIL, x_hi),
    )
    bw = edges_tail[1] - edges_tail[0]

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.bar(
        edges_tail[:-1], counts_tail, width=bw * 0.9,
        align="edge", color="#4C72B0", alpha=0.75, label="OAS CDR-H3 sequences",
    )
    ax2.axvline(100.0, color="crimson", linestyle="--", linewidth=1.8, label="C05 WT (100%)")
    ax2.set_xlim(TAIL, x_hi)
    ax2.set_xlabel("BLOSUM62 normalized similarity to C05 CDR-H3 (%)", fontsize=12)
    ax2.set_ylabel(y_label, fontsize=12)
    ax2.set_title(
        f"CDR-H3 BLOSUM62 Similarity to C05 — tail ≥ {int(TAIL)}%\n"
        f"({n_tail:,} sequences out of {total_seqs:,} total)",
        fontsize=12,
    )
    ax2.legend(fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    plt.tight_layout()
    fig2.savefig(out_tail, bbox_inches="tight")
    plt.close(fig2)
    print(f"Plot 2 saved to: {out_tail}")

    # ── Plot 3: cumulative tail count from ≥ 20% ─────────────────────────────
    # For each threshold t, how many sequences have similarity ≥ t?
    out_cumul = out.with_stem(out.stem + "_cumulative_tail20")

    thresholds = np.linspace(TAIL, scores_pct[mask_tail].max(), 300)
    w_tail = weights[mask_tail].astype(float)
    s_tail = scores_pct[mask_tail]
    cumul_counts = np.array([w_tail[s_tail >= t].sum() for t in thresholds])

    fig3, ax3 = plt.subplots(figsize=(9, 5))
    ax3.plot(thresholds, cumul_counts, color="#4C72B0", linewidth=2)
    ax3.fill_between(thresholds, cumul_counts, alpha=0.15, color="#4C72B0")

    # Annotate a few round thresholds for quick reading
    for t_mark in [20, 25, 30, 35, 40]:
        if t_mark < thresholds[-1]:
            n_mark = float(w_tail[s_tail >= t_mark].sum())
            ax3.axvline(t_mark, color="#888888", linestyle=":", linewidth=0.9)
            ax3.annotate(
                f"{int(n_mark):,}",
                xy=(t_mark, n_mark),
                xytext=(t_mark + 0.4, n_mark * 1.06),
                fontsize=8, color="#333333",
            )

    ax3.set_xlim(TAIL, thresholds[-1] + 1)
    ax3.set_ylim(bottom=0)
    ax3.set_xlabel("Similarity threshold (%)", fontsize=12)
    ax3.set_ylabel(f"{y_label} ≥ threshold", fontsize=12)
    ax3.set_title(
        f"Cumulative sequence count above BLOSUM62 similarity threshold\n"
        f"(tail ≥ {int(TAIL)}%, total corpus: {total_seqs:,} sequences)",
        fontsize=12,
    )
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    plt.tight_layout()
    fig3.savefig(out_cumul, bbox_inches="tight")
    plt.close(fig3)
    print(f"Plot 3 saved to: {out_cumul}")


if __name__ == "__main__":
    main()
