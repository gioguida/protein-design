#!/usr/bin/env python3
"""Compare c05_cdrh3_blosum25 and c05_cdrh3_mmseqs20 datasets.

Produces a single 2×2 figure with:
  1. Venn diagram   — overlap between the two datasets
  2. CDR-H3 length  — length distribution per dataset
  3. BLOSUM scores  — norm_score distribution coloured by membership
  4. MMseqs pident  — pident distribution for the 135 CDR-H3 MMseqs hits
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(".env.local")
load_dotenv()

_PROJECT_DIR = os.environ.get("PROJECT_DIR", ".")
_SCRATCH_DIR = os.environ.get("SCRATCH_DIR", ".")

C05_H3_LEN = 24  # CDR-H3 length of C05 (without OAS flanks)

BLOSUM_FASTA   = os.path.join(_PROJECT_DIR, "data", "c05", "c05_cdrh3_blosum25.fasta")
MMSEQS_FASTA   = os.path.join(_PROJECT_DIR, "data", "c05", "c05_cdrh3_mmseqs20.fasta")
BLOSUM_PARQUET = os.path.join(_SCRATCH_DIR, "c05_cdrh3_blosum", "h3_scores.parquet")
MMSEQS_RESULTS = os.path.join(_SCRATCH_DIR, "c05_cdrh3_mmseqs", "results.tsv")
TARGET_H3_FASTA= os.path.join(_SCRATCH_DIR,  "c05_cdrh3_mmseqs",  "target_h3.fasta")

MMSEQS_COLS = ["query", "target", "pident", "alnlen", "mismatch", "gapopen",
               "qstart", "qend", "tstart", "tend", "evalue", "bits"]

COLORS = {
    "blosum_only": "#4C72B0",
    "shared":      "#55A868",
    "mmseqs_only": "#C44E52",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_fasta_ids(path: str) -> set[str]:
    ids: set[str] = set()
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                ids.add(line[1:].strip())
    return ids


def parse_target_h3_fasta(fasta_path: str) -> dict[str, str]:
    """Return {h3_N: sequence} from target_h3.fasta."""
    result: dict[str, str] = {}
    key: str | None = None
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                key = line[1:]
            elif key is not None:
                result[key] = line
    return result


# ── panels ────────────────────────────────────────────────────────────────────

def plot_venn(ax: plt.Axes, n_blosum_only: int, n_shared: int, n_mmseqs_only: int) -> None:
    total_b = n_blosum_only + n_shared
    total_m = n_mmseqs_only + n_shared

    r_b = 0.42
    r_m = r_b * (total_m / total_b) ** 0.5  # area proportional to seq count

    # With ~98% of MMseqs shared with BLOSUM, the small circle should sit almost
    # entirely inside the large one.  d = r_b - r_m is the internal-tangent distance;
    # adding a small offset creates a thin crescent for the MMseqs-only sequences.
    d = (r_b - r_m) + 0.015
    cx_b = -d * 0.35        # keep the figure centred
    cx_m = cx_b + d
    cy = 0.0

    circ_b = mpatches.Circle((cx_b, cy), r_b, color=COLORS["blosum_only"], alpha=0.35, lw=0)
    circ_m = mpatches.Circle((cx_m, cy), r_m, color=COLORS["mmseqs_only"], alpha=0.55, lw=0)
    ax.add_patch(circ_b)
    ax.add_patch(circ_m)

    # BLOSUM-only count — left region of the large circle
    ax.text(cx_b - r_b * 0.50, cy, f"{n_blosum_only:,}",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=COLORS["blosum_only"])
    # Shared count — inside the MMseqs circle, which sits inside BLOSUM
    ax.text(cx_m, cy + r_m * 0.25, f"{n_shared:,}",
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=COLORS["shared"])
    # MMseqs-only crescent is tiny — annotate with a leader line
    ax.annotate(
        f"{n_mmseqs_only}  MMseqs20-only",
        xy=(cx_m + r_m * 0.98, cy),
        xytext=(cx_m + r_m + 0.12, cy + 0.14),
        ha="left", va="center", fontsize=8, fontweight="bold",
        color=COLORS["mmseqs_only"],
        arrowprops=dict(arrowstyle="-", color=COLORS["mmseqs_only"], lw=0.8),
    )

    # Category labels
    ax.text(cx_b - r_b * 0.42, cy + r_b + 0.07, "BLOSUM25\nonly",
            ha="center", va="bottom", fontsize=9, color=COLORS["blosum_only"])
    ax.text(cx_m, cy + r_m + 0.07, "MMseqs20",
            ha="center", va="bottom", fontsize=9, color=COLORS["mmseqs_only"])
    ax.text(cx_m, cy - r_m - 0.07, "shared",
            ha="center", va="top", fontsize=9, color=COLORS["shared"])

    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(-0.65, 0.75)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Dataset overlap", fontsize=11)


def plot_length_dist(ax: plt.Axes,
                     blosum_h3s: list[str], mmseqs_h3s: list[str]) -> None:
    bl_lens = [len(h) for h in blosum_h3s]
    mm_lens = [len(h) for h in mmseqs_h3s]

    lo = min(min(bl_lens), min(mm_lens))
    hi = max(max(bl_lens), max(mm_lens))
    bins = np.arange(lo - 0.5, hi + 1.5, 1)

    ax.hist(bl_lens, bins=bins, alpha=0.55, color=COLORS["blosum_only"],
            label=f"BLOSUM25 (n={len(bl_lens):,})", density=True)
    ax.hist(mm_lens, bins=bins, alpha=0.55, color=COLORS["mmseqs_only"],
            label=f"MMseqs20 (n={len(mm_lens):,})", density=True)
    ax.axvline(C05_H3_LEN, color="crimson", linestyle="--", linewidth=1.6,
               label=f"C05 H3 (len={C05_H3_LEN})")

    ax.set_xlabel("CDR-H3 length (aa)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("CDR-H3 length distribution", fontsize=11)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_blosum_scores(ax: plt.Axes,
                       df: pd.DataFrame,
                       blosum_h3_set: set[str],
                       mmseqs_h3_set: set[str]) -> None:
    pct = df["norm_score"] * 100.0
    in_b = df["h3"].isin(blosum_h3_set)
    in_m = df["h3"].isin(mmseqs_h3_set)

    shared_mask     = in_b & in_m
    blosum_only_mask = in_b & ~in_m
    mmseqs_only_mask = ~in_b & in_m

    bins = np.linspace(20, 45, 40)

    for mask, label, color in [
        (blosum_only_mask, f"BLOSUM only ({int(blosum_only_mask.sum()):,})", COLORS["blosum_only"]),
        (shared_mask,      f"shared ({int(shared_mask.sum()):,})",           COLORS["shared"]),
        (mmseqs_only_mask, f"MMseqs only ({int(mmseqs_only_mask.sum()):,})", COLORS["mmseqs_only"]),
    ]:
        ax.hist(pct[mask], bins=bins, alpha=0.6, color=color, label=label)

    ax.axvline(25.0, color="#888888", linestyle=":", linewidth=1.1,
               label="BLOSUM25 threshold (25%)")
    ax.set_xlabel("BLOSUM62 normalized score (%)", fontsize=10)
    ax.set_ylabel("Unique H3 count", fontsize=10)
    ax.set_title("BLOSUM score distribution (tail ≥ 20%)", fontsize=11)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_mmseqs_pident(ax: plt.Axes, results_tsv: str, target_fasta: str) -> None:
    df = pd.read_csv(results_tsv, sep="\t", header=None, names=MMSEQS_COLS)
    pident = df["pident"].values  # already in percent (0–100)

    target_seqs = parse_target_h3_fasta(target_fasta)
    id_to_h3 = target_seqs  # h3_N -> h3_seq
    h3_lens = [len(id_to_h3.get(t, "")) for t in df["target"]]

    bins = np.linspace(20, 80, 25)
    ax.hist(pident, bins=bins, color=COLORS["mmseqs_only"], alpha=0.7,
            label=f"MMseqs hits (n={len(pident):,} H3s)")
    ax.axvline(20.0, color="#888888", linestyle=":", linewidth=1.1,
               label="pident ≥ 20% threshold")

    ax.set_xlabel("MMseqs2 pident (%)", fontsize=10)
    ax.set_ylabel("Unique H3 count", fontsize=10)
    ax.set_title("MMseqs pident distribution\n(135 CDR-H3 hits)", fontsize=11)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", "-o",
                   default=os.path.join(_PROJECT_DIR, "plots",
                                        "c05_dataset_comparison",
                                        "c05_dataset_comparison.pdf"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    for path in [BLOSUM_FASTA, MMSEQS_FASTA, BLOSUM_PARQUET, MMSEQS_RESULTS, TARGET_H3_FASTA]:
        if not Path(path).exists():
            sys.exit(f"Missing file: {path}")

    print("Parsing FASTA IDs ...", flush=True)
    blosum_ids = parse_fasta_ids(BLOSUM_FASTA)
    mmseqs_ids = parse_fasta_ids(MMSEQS_FASTA)
    shared_ids      = blosum_ids & mmseqs_ids
    blosum_only_ids = blosum_ids - mmseqs_ids
    mmseqs_only_ids = mmseqs_ids - blosum_ids
    print(f"  BLOSUM: {len(blosum_ids):,}  MMseqs: {len(mmseqs_ids):,}  "
          f"shared: {len(shared_ids):,}  "
          f"BLOSUM-only: {len(blosum_only_ids):,}  MMseqs-only: {len(mmseqs_only_ids):,}")

    # H3 sequences come directly from their respective sources, avoiding the expensive
    # seq_id→h3 reverse-lookup through the h3_mapping pickle.
    print("Loading BLOSUM parquet ...", flush=True)
    df_parquet = pd.read_parquet(BLOSUM_PARQUET)
    # blosum_h3_set: H3s in the actual BLOSUM25 dataset (25% threshold used to build the FASTA).
    # This must be separate from the broader ≥20% display range so that MMseqs H3s with
    # 20–25% BLOSUM score correctly appear as "MMseqs-only" in the score distribution panel.
    blosum_h3_set = set(df_parquet.loc[df_parquet["norm_score"] >= 0.25, "h3"])
    blosum_h3s = list(blosum_h3_set)
    # Broader tail for the score distribution panel
    df_scores = df_parquet[df_parquet["norm_score"] >= 0.20].reset_index(drop=True)
    print(f"  BLOSUM25 dataset: {len(blosum_h3_set):,} unique H3s  |  tail ≥20%: {len(df_scores):,}")

    print("Loading MMseqs results ...", flush=True)
    target_seqs = parse_target_h3_fasta(TARGET_H3_FASTA)
    df_mmseqs = pd.read_csv(MMSEQS_RESULTS, sep="\t", header=None, names=MMSEQS_COLS)
    mmseqs_h3s = [target_seqs[t] for t in df_mmseqs["target"] if t in target_seqs]
    mmseqs_h3_set = set(mmseqs_h3s)
    print(f"  {len(mmseqs_h3s):,} MMseqs H3 hits")

    print("Rendering figure ...", flush=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("C05 CDR-H3 dataset comparison: BLOSUM25 vs MMseqs20", fontsize=13)

    plot_venn(axes[0, 0], len(blosum_only_ids), len(shared_ids), len(mmseqs_only_ids))
    plot_length_dist(axes[0, 1], blosum_h3s, mmseqs_h3s)
    plot_blosum_scores(axes[1, 0], df_scores, blosum_h3_set, mmseqs_h3_set)
    plot_mmseqs_pident(axes[1, 1], MMSEQS_RESULTS, TARGET_H3_FASTA)

    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
