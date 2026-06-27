#!/usr/bin/env python
"""Score an arbitrary CSV of CDR-H3 sequences with a model's CDR-H3 PLL.

Companion to ``score_sequences_with_esme.py`` / ``score_sequences_with_uncertainty.py``
but the "scorer" here is a masked-language model itself: for every unique CDR-H3
we compute the sum of one-at-a-time masked log-probs over the 24 CDR-H3 positions
(the same pseudo-log-likelihood used by ``compute_pll.py`` / ``gibbs_diagnostics.py``).

Used by report/meetings/pssm_vs_dpo_650m_comparison.ipynb to get the DPO 650M
model's PLL for both the DPO-generated and the PSSM-generated sequences, so the
notebook can correlate the generator's own likelihood with the two external
scorers (M22 esme, UQ ensemble).

Example:
    sbatch bash_scripts/utils/score_generated_with_pll.sbatch \
        --input-csv  /path/to/generated.csv \
        --output-csv /path/to/pll.csv \
        --checkpoint /path/to/just_dpo_650M/step_8250.pt

Writes columns: <seq_col>, pll  (one row per unique sequence).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the single implementation of the PLL loader + batched forward pass.
from compute_pll import ESM2_650M_ID, compute_pll, load_esm_for_mlm  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("score_generated_with_pll")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-csv", required=True,
                   help="CSV of generated sequences (must contain --seq-col).")
    p.add_argument("--seq-col", default="cdrh3",
                   help="Name of the sequence column in --input-csv (default: cdrh3).")
    p.add_argument("--output-csv", required=True,
                   help="Where to write the <seq_col>,pll CSV.")
    p.add_argument("--checkpoint", default=None,
                   help="Model checkpoint (HF dir, .pt state-dict, or HF id). "
                        "Empty string forces the vanilla --base-model.")
    p.add_argument("--base-model", default=ESM2_650M_ID,
                   help=f"Base model for architecture + tokenizer (default {ESM2_650M_ID}).")
    p.add_argument("--batch-size", type=int, default=256,
                   help="(sequence, masked-position) pairs per forward pass.")
    p.add_argument("--force", action="store_true",
                   help="Re-score even if --output-csv already exists.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_csv = Path(args.output_csv)
    if out_csv.exists() and not args.force:
        log.info("[skip] %s exists — pass --force to recompute", out_csv)
        return

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise SystemExit(f"Input CSV not found: {input_csv}")
    df = pd.read_csv(input_csv)
    if args.seq_col not in df.columns:
        raise SystemExit(f"{input_csv} missing column {args.seq_col!r}. "
                         f"Columns: {list(df.columns)}")
    seqs = df[args.seq_col].astype(str).str.strip().drop_duplicates()
    seqs = [s for s in seqs.tolist() if s]
    log.info("Computing PLL on %d unique sequences from %s (col=%s)",
             len(seqs), input_csv, args.seq_col)

    checkpoint = args.checkpoint or ""
    device = torch.device(args.device)
    log.info("Loading model for PLL: %s (base=%s)",
             checkpoint or "vanilla", args.base_model)
    model = load_esm_for_mlm(checkpoint, args.base_model).eval().to(device)
    if device.type == "cuda":
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    pll = compute_pll(model, tokenizer, seqs, device, args.batch_size)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({args.seq_col: seqs, "pll": pll}).to_csv(out_csv, index=False)
    log.info("Wrote %s  (n=%d)", out_csv, len(seqs))


if __name__ == "__main__":
    main()
