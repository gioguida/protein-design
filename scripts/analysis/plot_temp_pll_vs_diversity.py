"""PLL vs diversity tradeoff across temperatures.

For each temperature, computes:
  x = mean pairwise Hamming distance of final-step CDR-H3 sequences
  y = mean PLL of the top-k sequences (by PLL) at the final step

Produces a scatter plot with one labeled point per temperature.

Inputs
------
--temp-csv T=CSV_PATH   Repeated once per temperature; T is the float label
--model-variant STR     Model label (for title)
--checkpoint-path PATH  Model checkpoint
--top-k INT             Top sequences by PLL used for mean PLL (default: 50)
--batch-size INT        Inference batch size (default: 32)
--output-dir PATH

Output
------
<output-dir>/temp_pll_vs_diversity.png
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
log = logging.getLogger("plot_temp_pll_vs_diversity")

SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--temp-csv", action="append", required=True,
        help="T=CSV_PATH; repeat once per temperature.",
    )
    p.add_argument("--model-variant", required=True)
    p.add_argument("--checkpoint-path", default="")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
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


def _mean_pairwise_hamming(seqs: list[str], max_n: int = 1000) -> float:
    rng = np.random.default_rng(SEED)
    chars = np.array([list(s) for s in seqs])
    if len(seqs) > max_n:
        keep = rng.choice(len(seqs), size=max_n, replace=False)
        chars = chars[keep]
    P = chars.shape[1]
    n = chars.shape[0]
    if n < 2:
        return 0.0
    eq = chars[:, None, :] == chars[None, :, :]
    dist_mat = P - eq.sum(axis=2)
    iu = np.triu_indices(n, k=1)
    return float(dist_mat[iu].mean())


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

    temps: list[float] = []
    mean_hammings: list[float] = []
    mean_plls: list[float] = []

    for temp, csv_path in temp_csv_pairs:
        seqs = _load_final_step(csv_path)
        if not seqs:
            log.warning("No final-step sequences for T=%s (%s) — skipping.", temp, csv_path)
            continue

        log.info("T=%s: %d sequences, computing PLL ...", temp, len(seqs))
        pll_vals = sequence_pll(
            per_position_cdr_log_probs(model, tokenizer, seqs, device, args.batch_size)
        )

        k = min(args.top_k, len(pll_vals))
        top_pll = float(np.sort(pll_vals)[-k:].mean())

        hamming = _mean_pairwise_hamming(seqs)

        temps.append(temp)
        mean_hammings.append(hamming)
        mean_plls.append(top_pll)
        log.info("T=%s: mean_hamming=%.3f mean_top%d_pll=%.3f", temp, hamming, k, top_pll)

    if not temps:
        log.error("No temperature points with valid data — cannot produce plot.")
        return 1

    cmap = plt.get_cmap("plasma")
    colors = [cmap(i / max(1, len(temps) - 1)) for i in range(len(temps))]

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    for i, (t, x, y) in enumerate(zip(temps, mean_hammings, mean_plls)):
        ax.scatter(x, y, s=80, color=colors[i], edgecolors="black",
                   linewidths=0.5, zorder=3)
        ax.annotate(
            f"T={t}",
            (x, y),
            textcoords="offset points", xytext=(7, 4),
            fontsize=9, color=colors[i],
        )
    ax.set_xlabel("Mean pairwise Hamming distance")
    ax.set_ylabel(f"Mean PLL (top-{args.top_k})")
    ax.set_title(
        f"PLL vs diversity tradeoff across temperatures (model: {args.model_variant})"
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()

    out_path = args.output_dir / "temp_pll_vs_diversity.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
