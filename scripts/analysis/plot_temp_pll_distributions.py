"""Overlaid PLL histograms by temperature.

For each temperature, computes CDR-H3 PLL of final-step beam sequences and
overlays histograms with one color per temperature.

Inputs
------
--temp-csv T=CSV_PATH   Repeated once per temperature
--model-variant STR     Model label (for title)
--checkpoint-path PATH  Model checkpoint
--batch-size INT        Inference batch size (default: 32)
--output-dir PATH

Output
------
<output-dir>/temp_pll_distributions.png
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
    load_esm_for_mlm,
    per_position_cdr_log_probs,
    sequence_pll,
)
from protein_design.constants import C05_CDRH3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("plot_temp_pll_distributions")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--temp-csv", action="append", required=True,
        help="T=CSV_PATH; repeat once per temperature.",
    )
    p.add_argument("--model-variant", required=True)
    p.add_argument("--checkpoint-path", default="")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--xlim-low", type=float, default=-80.0)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def _parse_temp_csv(spec: str) -> tuple[float, Path]:
    parts = spec.split("=", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"--temp-csv must be T=CSV_PATH, got {spec!r}"
        )
    return float(parts[0]), Path(parts[1])


def _load_final_step(csv_path: Path) -> list[str]:
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    df = df[df["cdrh3"].astype(str).str.len() == len(C05_CDRH3)].copy()
    if df.empty:
        return []
    final_step = df["gibbs_step"].max()
    return df.loc[df["gibbs_step"] == final_step, "cdrh3"].astype(str).tolist()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    temp_csv_pairs = [_parse_temp_csv(s) for s in args.temp_csv]
    temp_csv_pairs.sort(key=lambda x: x[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    log.info("Loading model %s ...", args.model_variant)
    model = load_esm_for_mlm(args.checkpoint_path).eval().to(device)
    for param in model.parameters():
        param.requires_grad = False
    if device.type == "cuda":
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_ID)

    pll_by_temp: list[tuple[float, np.ndarray]] = []

    for temp, csv_path in temp_csv_pairs:
        seqs = _load_final_step(csv_path)
        if not seqs:
            log.warning("No final-step sequences for T=%s (%s) — skipping.", temp, csv_path)
            continue
        log.info("T=%s: %d sequences, computing PLL ...", temp, len(seqs))
        pll_vals = sequence_pll(
            per_position_cdr_log_probs(model, tokenizer, seqs, device, args.batch_size)
        )
        pll_by_temp.append((temp, pll_vals))

    if not pll_by_temp:
        log.error("No temperature points with valid data — cannot produce plot.")
        return 1

    all_plls = np.concatenate([p for _, p in pll_by_temp])
    lo, hi = float(all_plls.min()), float(all_plls.max())
    bins = np.linspace(lo, hi, 40)

    cmap = plt.get_cmap("plasma")
    n = len(pll_by_temp)
    colors = [cmap(i / max(1, n - 1)) for i in range(n)]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for i, (temp, pll_vals) in enumerate(pll_by_temp):
        ax.hist(
            pll_vals, bins=bins, density=True,
            color=colors[i], alpha=0.55,
            label=f"T={temp} (n={len(pll_vals)})",
        )

    below = int((all_plls < args.xlim_low).sum())
    below_pct = 100.0 * below / max(1, len(all_plls))
    ax.set_xlim(args.xlim_low, 0.0)
    ax.set_xlabel("CDR-H3 PLL")
    ax.set_ylabel("Density")
    ax.set_title(
        f"PLL distribution by temperature (model: {args.model_variant})"
    )
    ax.text(
        0.98, 0.97,
        f"N sequences below {args.xlim_low:.1f}: {below} ({below_pct:.1f}%)",
        transform=ax.transAxes,
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="grey"),
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    out_path = args.output_dir / "temp_pll_distributions.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
