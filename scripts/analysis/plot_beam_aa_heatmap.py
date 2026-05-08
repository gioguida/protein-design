"""Amino acid frequency heatmaps: SBS-generated vs DMS (per CDR-H3 position).

Two side-by-side heatmaps (20 AAs × 24 mutable positions):
  Left:  AA frequency at each position across SBS final-step sequences.
  Right: AA frequency at each position across DMS sequences.

NOTE: gibbs_diagnostics.py has plot_position_mutation_freq (P(non-WT) per
position, one scalar per position) and plot_sequence_logo (stacked bars).
This script produces full 20×24 frequency matrices for both SBS and DMS,
enabling direct side-by-side AA composition comparison.

Inputs
------
--beam-csv PATH     Beam search output CSV (final step used)
--dms-m22 PATH      M22 DMS CSV (must contain 'aa' column)
--dms-si06 PATH     SI06 DMS CSV (optional; merged for sequence diversity)
--dms-m22-col COL   M22 enrichment column (default: M22_binding_enrichment_adj)
--dms-si06-col COL  SI06 enrichment column (default: SI06_binding_enrichment_adj)
--max-dms INT       Max DMS sequences to include (default: 500)
--model-variant STR Model label (for title)
--output-dir PATH

Output
------
<output-dir>/beam_aa_heatmap.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gibbs_diagnostics import load_dms
from protein_design.constants import C05_CDRH3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("plot_beam_aa_heatmap")

STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--beam-csv", type=Path, required=True)
    p.add_argument("--dms-m22", type=Path, default=None)
    p.add_argument("--dms-si06", type=Path, default=None)
    p.add_argument("--dms-m22-col", default="M22_binding_enrichment_adj")
    p.add_argument("--dms-si06-col", default="SI06_binding_enrichment_adj")
    p.add_argument("--max-dms", type=int, default=500)
    p.add_argument("--model-variant", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def _load_beam_final_step(csv_path: Path, cdrh3_len: int) -> list[str]:
    df = pd.read_csv(csv_path)
    df = df[df["cdrh3"].astype(str).str.len() == cdrh3_len].copy()
    if df.empty:
        return []
    final_step = df["gibbs_step"].max()
    return df.loc[df["gibbs_step"] == final_step, "cdrh3"].astype(str).tolist()


def _aa_frequency_matrix(seqs: list[str], P: int) -> np.ndarray:
    """Return (20, P) matrix of amino acid frequencies (each column sums to 1)."""
    chars = np.array([list(s) for s in seqs])  # (N, P)
    matrix = np.zeros((20, P), dtype=np.float32)
    for i, aa in enumerate(STANDARD_AAS):
        matrix[i] = (chars == aa).mean(axis=0)
    return matrix


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dms_m22 is None:
        log.error("--dms-m22 is required for this plot.")
        return 1

    P = len(C05_CDRH3)

    log.info("Loading beam CSV: %s", args.beam_csv)
    beam_seqs = _load_beam_final_step(args.beam_csv, P)
    if not beam_seqs:
        log.error("No beam sequences found at the final step.")
        return 1
    log.info("Final-step beam sequences: %d", len(beam_seqs))

    log.info("Loading DMS sequences (max %d) ...", args.max_dms)
    dms_cdrh3, _, _ = load_dms(
        args.dms_m22,
        args.dms_si06,
        args.max_dms,
        m22_col=args.dms_m22_col,
        si06_col=args.dms_si06_col,
    )
    if not dms_cdrh3:
        log.error("No DMS sequences loaded.")
        return 1
    log.info("DMS sequences: %d", len(dms_cdrh3))

    beam_freq = _aa_frequency_matrix(beam_seqs, P)
    dms_freq = _aa_frequency_matrix(dms_cdrh3, P)

    x_ticks = [f"{i + 1}\n{a}" for i, a in enumerate(C05_CDRH3)]

    fig, axes = plt.subplots(
        1, 2, figsize=(max(10.0, 0.55 * P + 4.0) * 2, 0.6 * 20 + 2.5),
        constrained_layout=True,
    )

    for ax, freq_mat, label, n in [
        (axes[0], beam_freq, f"SBS-generated (n={len(beam_seqs)})", len(beam_seqs)),
        (axes[1], dms_freq, f"DMS (n={len(dms_cdrh3)})", len(dms_cdrh3)),
    ]:
        im = ax.imshow(freq_mat, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Frequency")
        ax.set_yticks(range(20))
        ax.set_yticklabels(list(STANDARD_AAS), fontsize=9)
        ax.set_xticks(range(P))
        ax.set_xticklabels(x_ticks, fontsize=7)
        ax.set_xlabel("CDR-H3 position (WT residue below)")
        ax.set_ylabel("Amino acid")
        ax.set_title(label, fontsize=11)

    fig.suptitle(
        f"Amino acid frequency: SBS-generated vs DMS (model: {args.model_variant})",
        fontsize=13,
    )

    out_path = args.output_dir / "beam_aa_heatmap.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
