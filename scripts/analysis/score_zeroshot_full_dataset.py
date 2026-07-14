"""Score every row of a DMS dataset's RAW file (train+val+test pooled) with one
model's zero-shot cdr_pll pseudo-log-likelihood, and cache the per-row scores.

Zero-shot is deterministic (checkpoint + sequence -> score, no training
randomness: num_epochs=0, no gradient steps), so this needs to run once per
(model, dataset) -- NOT once per split/test-set. Downstream consumers can join
these cached per-row scores against whichever rows a given split's test set
(canonical or an external split.seed variant, see
protein_design.dms_splitting's `<dataset>_splitseed<N>` keys) actually
contains, entirely locally with no further GPU jobs:
  - per-split-seed zero-shot baselines for the external-split-seed sweep
  - bootstrap-CI resampling of the zero-shot rho (raw per-row predictions
    are exactly what this script produces)
  - the "zero-shot on the full pooled dataset" robustness check (just use
    every row instead of filtering to one split's test set)

Uses the exact same scoring path as the DPO training loop's val/test Spearman
tracking (protein_design.eval.score_sequences_cdr_pll via ESM2Model,
scoring_mode="cdr_pll"), so restricting the output to a prior run's test-split
rows reproduces that run's summary.json test_spearman_avg exactly.

Output: $ANALYSIS_DIR/<model>/zeroshot_full/<dataset>.csv
  columns: <sequence_col>, <key_metric_col>, score

Usage:
  uv run python scripts/analysis/score_zeroshot_full_dataset.py --model vanilla_35m --dataset ed1_m22
  uv run python scripts/analysis/score_zeroshot_full_dataset.py --model evo_35m --dataset cetuximab_h

Prefer running via: sbatch bash_scripts/extract.sbatch --what zeroshot_full --model … --dataset …
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from protein_design.analysis import registry  # noqa: E402
from protein_design.config import ModelConfig  # noqa: E402
from protein_design.dms_splitting import dataset_spec  # noqa: E402
from protein_design.eval import score_sequences_cdr_pll  # noqa: E402
from protein_design.model import ESM2Model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# use_context=True wraps a bare CDR-H3 sequence in the fixed C05 VH framework
# before scoring (protein_design.constants.add_context) -- the DPO pipeline's
# default. use_context=False scores the dataset's `aa` column as a complete
# sequence directly (mirrors model.use_context=false in the DPO Hydra configs
# for datasets whose rows already ARE the full chain, e.g. cetuximab_h).
USE_CONTEXT = {
    "ed1_m22": True,
    "ed2_m22": True,
    "ed5_m22": True,
    "ed811_m22": True,
    "cetuximab_h": False,
}


def _load_state_dict(checkpoint: str) -> dict:
    ckpt = torch.load(checkpoint, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise KeyError(f"Checkpoint {checkpoint} does not contain 'model_state_dict'.")
    return ckpt["model_state_dict"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Model key from conf/analysis/models.yaml")
    p.add_argument("--dataset", required=True,
                   help=f"Base DMS dataset key from conf/data/dms/default.yaml. Known use_context: {sorted(USE_CONTEXT)}")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset not in USE_CONTEXT:
        raise KeyError(
            f"Unknown dataset {args.dataset!r}; add its use_context convention to "
            f"USE_CONTEXT in this script. Known: {sorted(USE_CONTEXT)}"
        )
    use_context = USE_CONTEXT[args.dataset]

    spec = registry.resolve_model(args.model)
    checkpoint = spec["checkpoint"] or ""
    base_model = spec["base_model"]

    ds_spec = dataset_spec(args.dataset)
    out_csv = registry.artifact_path(args.model, "zeroshot_full", f"{args.dataset}.csv")
    expected = {
        "checkpoint": checkpoint,
        "base_model": base_model,
        "dataset_path": str(ds_spec.path),
        "use_context": use_context,
    }
    if not registry.needs_recompute(out_csv, expected, force=args.force):
        log.info("[skip] %s is up to date — pass --force to recompute", out_csv)
        return

    df = pd.read_csv(ds_spec.path)
    df[ds_spec.sequence_col] = df[ds_spec.sequence_col].astype(str).str.strip()
    df = df[df[ds_spec.sequence_col] != ""].reset_index(drop=True)
    sequences = df[ds_spec.sequence_col].tolist()
    log.info("[%s/%s] scoring %d rows (use_context=%s)", args.model, args.dataset, len(sequences), use_context)

    device = torch.device(args.device)
    model_cfg = ModelConfig(esm_model_path=base_model, device=str(device), use_context=use_context)
    scorer = ESM2Model(model_cfg)
    if checkpoint:
        scorer.load_state_dict(_load_state_dict(checkpoint))
        log.info("Loaded checkpoint: %s", checkpoint)
    scorer.to(scorer.device)
    scorer.model.eval()

    scores = score_sequences_cdr_pll(scorer, sequences, batch_size=args.batch_size)

    out_df = df[[ds_spec.sequence_col, ds_spec.key_metric_col]].copy()
    out_df["score"] = scores
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    registry.write_meta(
        out_csv, n=len(out_df), seq_col=ds_spec.sequence_col,
        metric_col=ds_spec.key_metric_col, **expected,
    )
    log.info("Wrote %s (n=%d)", out_csv, len(out_df))


if __name__ == "__main__":
    main()
