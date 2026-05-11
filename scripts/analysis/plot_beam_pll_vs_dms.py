"""PLL distribution histogram: SBS-generated vs DMS test sequences.

Loads beam CSV (final step) and DMS sequences, computes CDR-H3 PLL for both
under the same model checkpoint, and overlays their distributions.

NOTE: gibbs_diagnostics.py produces violin plots comparing DMS vs sampler PLL
(plot_pll_violin_grid). This script produces a dedicated histogram overlay for
beam sequences vs DMS test set under a single model.

Inputs
------
--beam-csv PATH         Beam search output CSV
--checkpoint-path PATH  Model checkpoint (HF dir, .pt file, or HF model ID)
--model-variant STR     Model label used in title
--dms-m22 PATH          M22 DMS CSV (must contain 'aa' column)
--dms-si06 PATH         SI06 DMS CSV (optional)
--dms-m22-col COL       M22 enrichment column (default: M22_binding_enrichment_adj)
--dms-si06-col COL      SI06 enrichment column (default: SI06_binding_enrichment_adj)
--max-dms INT           Max DMS sequences to score (default: 500)
--batch-size INT        Inference batch size (default: 32)
--output-dir PATH

Output
------
<output-dir>/beam_pll_vs_dms_histogram.png
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
from transformers import AutoTokenizer

from gibbs_diagnostics import (
    ESM2_MODEL_ID,
    load_dms,
    load_esm_for_mlm,
    per_position_cdr_log_probs,
    sequence_pll,
)
from protein_design.constants import C05_CDRH3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("plot_beam_pll_vs_dms")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--beam-csv", type=Path, required=True)
    p.add_argument("--checkpoint-path", default="")
    p.add_argument("--model-variant", required=True)
    p.add_argument("--dms-m22", type=Path, default=None)
    p.add_argument("--dms-si06", type=Path, default=None)
    p.add_argument("--dms-m22-col", default="M22_binding_enrichment_adj")
    p.add_argument("--dms-si06-col", default="SI06_binding_enrichment_adj")
    p.add_argument("--max-dms", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--xlim-low", type=float, default=-100.0)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def _load_beam_final_step(csv_path: Path) -> list[str]:
    df = pd.read_csv(csv_path)
    df = df[df["cdrh3"].astype(str).str.len() == len(C05_CDRH3)].copy()
    if df.empty:
        return []
    final_step = df["gibbs_step"].max()
    return df.loc[df["gibbs_step"] == final_step, "cdrh3"].astype(str).tolist()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dms_m22 is None:
        log.error("--dms-m22 is required for this plot.")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    log.info("Loading model %s ...", args.model_variant)
    model = load_esm_for_mlm(args.checkpoint_path).eval().to(device)
    for param in model.parameters():
        param.requires_grad = False
    if device.type == "cuda":
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_ID)

    log.info("Loading DMS sequences (max %d) ...", args.max_dms)
    dms_cdrh3, _, _ = load_dms(
        args.dms_m22,
        args.dms_si06,
        args.max_dms,
        m22_col=args.dms_m22_col,
        si06_col=args.dms_si06_col,
    )
    if not dms_cdrh3:
        log.error("No DMS sequences loaded — cannot produce comparison plot.")
        return 1

    log.info("Loading beam CSV: %s", args.beam_csv)
    beam_cdrh3 = _load_beam_final_step(args.beam_csv)
    if not beam_cdrh3:
        log.error("No beam sequences found at the final step.")
        return 1

    log.info("Computing DMS PLL (%d sequences) ...", len(dms_cdrh3))
    dms_pll = sequence_pll(
        per_position_cdr_log_probs(model, tokenizer, dms_cdrh3, device, args.batch_size)
    )

    log.info("Computing beam PLL (%d sequences) ...", len(beam_cdrh3))
    beam_pll = sequence_pll(
        per_position_cdr_log_probs(model, tokenizer, beam_cdrh3, device, args.batch_size)
    )

    lo = min(float(dms_pll.min()), float(beam_pll.min()))
    hi = max(float(dms_pll.max()), float(beam_pll.max()))
    bins = np.linspace(lo, hi, 40)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.hist(
        dms_pll, bins=bins, color="tab:blue", alpha=0.55,
        label=f"DMS test (n={len(dms_pll)})", density=True,
    )
    ax.hist(
        beam_pll, bins=bins, color="tab:orange", alpha=0.65,
        label=f"SBS final step (n={len(beam_pll)})", density=True,
    )
    ax.set_xlabel("CDR-H3 PLL")
    ax.set_ylabel("Density")
    dms_below = int((dms_pll < args.xlim_low).sum())
    dms_below_pct = 100.0 * dms_below / max(1, len(dms_pll))
    ax.set_xlim(args.xlim_low, 0.0)
    ax.set_title(
        f"PLL distribution: SBS-generated vs DMS test sequences (model: {args.model_variant})"
    )
    ax.text(
        0.98, 0.97,
        f"DMS below {args.xlim_low:.1f}: {dms_below} ({dms_below_pct:.1f}%)",
        transform=ax.transAxes,
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="grey"),
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()

    out_path = args.output_dir / "beam_pll_vs_dms_histogram.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
