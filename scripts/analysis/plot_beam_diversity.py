"""Beam search diversity diagnostics: three-panel figure.

Subplots (final-step beam sequences only):
  1. Pairwise Hamming distance histogram across all CDR-H3 sequences.
  2. Position-wise Shannon entropy bar chart (bits) for each mutable position.
  3. Number of unique positions mutated vs WT histogram (= n_mutations histogram).

NOTE: Subplots 1 and 3 are also individually available via gibbs_diagnostics.py
(plot_pairwise_hamming and plot_edit_distance respectively). This script composes
all three diversity metrics into a single figure for beam search evaluation.

Inputs
------
--beam-csv PATH       Beam search output CSV
--model-variant STR   Model label (for title)
--wt-cdrh3 STR        WT CDR-H3 (default: C05_CDRH3)
--max-pairs INT       Cap on sequences for O(N^2) Hamming computation (default: 1000)
--output-dir PATH

Output
------
<output-dir>/beam_diversity_diagnostics.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from protein_design.constants import C05_CDRH3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("plot_beam_diversity")

SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--beam-csv", type=Path, required=True)
    p.add_argument("--model-variant", required=True)
    p.add_argument("--wt-cdrh3", default=C05_CDRH3)
    p.add_argument("--max-pairs", type=int, default=1000,
                   help="Cap sequences for pairwise Hamming (O(N^2)) computation.")
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def _load_beam_final_step(csv_path: Path, cdrh3_len: int) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(csv_path)
    df = df[df["cdrh3"].astype(str).str.len() == cdrh3_len].copy()
    if df.empty:
        return [], np.array([], dtype=np.int32)
    final_step = df["gibbs_step"].max()
    df = df[df["gibbs_step"] == final_step]
    seqs = df["cdrh3"].astype(str).tolist()
    n_mut = df["n_mutations"].to_numpy(dtype=np.int32)
    return seqs, n_mut


def _position_entropy(chars: np.ndarray) -> np.ndarray:
    """Shannon entropy in bits at each CDR position. chars shape: (N, P)."""
    P = chars.shape[1]
    entropies = np.zeros(P, dtype=np.float32)
    for p_idx in range(P):
        col = chars[:, p_idx]
        counts = Counter(col.tolist())
        total = sum(counts.values())
        freqs = np.array([c / total for c in counts.values()], dtype=np.float64)
        freqs = freqs[freqs > 0]
        entropies[p_idx] = float(-np.sum(freqs * np.log2(freqs)))
    return entropies


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cdrh3_len = len(args.wt_cdrh3)
    wt_arr = np.array(list(args.wt_cdrh3))

    log.info("Loading beam CSV: %s", args.beam_csv)
    seqs, n_mutations = _load_beam_final_step(args.beam_csv, cdrh3_len)
    if not seqs:
        log.error("No beam sequences found at the final step.")
        return 1
    log.info("Final-step beam sequences: %d", len(seqs))

    chars = np.array([list(s) for s in seqs])  # (N, P)
    P = cdrh3_len
    n = len(seqs)

    # --- subplot 1: pairwise Hamming ---
    rng = np.random.default_rng(SEED)
    chars_h = chars
    if n > args.max_pairs:
        keep = rng.choice(n, size=args.max_pairs, replace=False)
        chars_h = chars[keep]
        log.info("Capped to %d sequences for pairwise Hamming.", args.max_pairs)
    nh = chars_h.shape[0]
    eq = chars_h[:, None, :] == chars_h[None, :, :]
    dist_mat = P - eq.sum(axis=2)
    iu = np.triu_indices(nh, k=1)
    hamming_flat = dist_mat[iu]

    # --- subplot 2: position-wise Shannon entropy ---
    entropies = _position_entropy(chars)

    # --- subplot 3: n_mutations histogram (unique positions mutated vs WT) ---
    # n_mutations is pre-computed in the beam CSV (Hamming vs WT CDR-H3)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))

    # Subplot 1
    ax0 = axes[0]
    ax0.hist(hamming_flat, bins=range(0, P + 2), align="left",
             color="tab:purple", edgecolor="white", linewidth=0.4)
    ax0.set_xlabel("Pairwise Hamming distance")
    ax0.set_ylabel("Pair count")
    ax0.set_title("Pairwise Hamming (CDR-H3)")
    ax0.set_xlim(-0.5, P + 0.5)

    # Subplot 2
    ax1 = axes[1]
    x_pos = np.arange(P)
    bars = ax1.bar(x_pos, entropies, color="tab:cyan", edgecolor="white", linewidth=0.4)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(
        [f"{i + 1}\n{a}" for i, a in enumerate(args.wt_cdrh3)],
        fontsize=7,
    )
    ax1.set_xlabel("CDR-H3 position (WT residue below)")
    ax1.set_ylabel("Shannon entropy (bits)")
    ax1.set_title("Per-position amino acid entropy")

    # Subplot 3
    ax2 = axes[2]
    max_mut = int(n_mutations.max()) if len(n_mutations) else P
    ax2.hist(n_mutations, bins=range(0, max_mut + 2), align="left",
             color="tab:green", edgecolor="white", linewidth=0.4)
    ax2.set_xlabel("Positions mutated vs WT")
    ax2.set_ylabel("Sequence count")
    ax2.set_title("Unique positions mutated vs WT")
    ax2.set_xlim(left=-0.5)

    fig.suptitle(
        f"Beam search diversity diagnostics (model: {args.model_variant})\n"
        f"n={len(seqs)} sequences (final step)",
        fontsize=12,
    )
    fig.tight_layout()

    out_path = args.output_dir / "beam_diversity_diagnostics.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
