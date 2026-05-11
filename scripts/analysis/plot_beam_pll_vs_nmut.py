"""Scatter plot of SBS-generated sequence PLL vs number of mutations from WT.

Uses the final-step beam sequences. Computes CDR-H3 PLL under the given
model checkpoint and plots PLL (y) vs n_mutations (x), one dot per sequence.
Spearman correlation is annotated on the figure.

Inputs
------
--beam-csv PATH         Beam search output CSV
--checkpoint-path PATH  Model checkpoint (HF dir, .pt file, or HF model ID)
--model-variant STR     Model label (for title)
--batch-size INT        Inference batch size (default: 32)
--clip-pll FLOAT        Clip lower PLL axis bound and mark clipped points
--output-dir PATH

Output
------
<output-dir>/beam_pll_vs_nmut.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer

from gibbs_diagnostics import (
    ESM2_MODEL_ID,
    load_esm_for_mlm,
    per_position_cdr_log_probs,
    sequence_pll,
)
from protein_design.constants import C05_CDRH3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("plot_beam_pll_vs_nmut")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--beam-csv", type=Path, required=True)
    p.add_argument("--checkpoint-path", default="")
    p.add_argument("--model-variant", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--clip-pll", type=float, default=-100.0)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def _load_beam_final_step(csv_path: Path) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(csv_path)
    df = df[df["cdrh3"].astype(str).str.len() == len(C05_CDRH3)].copy()
    if df.empty:
        return [], np.array([], dtype=np.int32)
    final_step = df["gibbs_step"].max()
    df = df[df["gibbs_step"] == final_step]
    return (
        df["cdrh3"].astype(str).tolist(),
        df["n_mutations"].to_numpy(dtype=np.int32),
    )


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading beam CSV: %s", args.beam_csv)
    seqs, n_mutations = _load_beam_final_step(args.beam_csv)
    if not seqs:
        log.error("No beam sequences found at the final step.")
        return 1
    log.info("Final-step beam sequences: %d", len(seqs))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    log.info("Loading model %s ...", args.model_variant)
    model = load_esm_for_mlm(args.checkpoint_path).eval().to(device)
    for param in model.parameters():
        param.requires_grad = False
    if device.type == "cuda":
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_ID)

    log.info("Computing PLL (%d sequences) ...", len(seqs))
    pll = sequence_pll(
        per_position_cdr_log_probs(model, tokenizer, seqs, device, args.batch_size)
    )

    rho, pval = spearmanr(n_mutations, pll)

    clipped_mask = pll < args.clip_pll
    pll_plot = np.maximum(pll, args.clip_pll)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.scatter(
        n_mutations[~clipped_mask], pll_plot[~clipped_mask],
        s=22, alpha=0.65, color="tab:blue",
        edgecolors="black", linewidths=0.25, label="PLL",
    )
    if clipped_mask.any():
        ax.scatter(
            n_mutations[clipped_mask], pll_plot[clipped_mask],
            s=34, alpha=0.9, color="tab:red", marker="v",
            edgecolors="black", linewidths=0.3,
            label=f"Clipped at {args.clip_pll:.1f}",
        )

    ax.set_xlabel("Number of mutations from WT (n_mutations)")
    ax.set_ylabel("CDR-H3 PLL")
    ax.set_title(
        f"PLL vs number of mutations from seed (model: {args.model_variant})"
    )
    ax.set_ylim(bottom=args.clip_pll - 2.0)
    ax.text(
        0.97, 0.97,
        f"Spearman rho = {rho:.3f}\n(p = {pval:.2e}, n = {len(seqs)})",
        transform=ax.transAxes,
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="grey"),
    )
    if clipped_mask.any():
        ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    out_path = args.output_dir / "beam_pll_vs_nmut.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
