#!/usr/bin/env python
"""Standalone evaluation: score a checkpoint against scoring datasets.

Useful for re-scoring a completed run (or any .pt checkpoint) without
retraining. Writes metrics.json next to the checkpoint (or in +out_dir).

Example:
    python scripts/eval.py +checkpoint=/path/to/best.pt scoring=d2
    python scripts/eval.py +checkpoint=/path/to/best.pt +out_dir=/tmp/eval_out
    python scripts/eval.py +checkpoint=/path/to/best.pt scoring=d2 +output_csv_dir=/tmp/scores
"""

import json
import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer

from protein_design.eval import compute_cdr_pseudo_perplexity, load_scoring_datasets, run_multi_scoring_evaluation
from protein_design.model import ESM2Model
from protein_design.config import build_model_config, build_scoring_config

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ckpt_path = cfg.get("checkpoint")
    if not ckpt_path:
        raise ValueError("Pass `+checkpoint=/path/to/model.pt` on the CLI.")
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir = Path(cfg.get("out_dir") or ckpt_path.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    scoring_cfg = build_scoring_config(cfg)
    if not scoring_cfg.datasets:
        raise ValueError("No scoring datasets configured (scoring.datasets is empty).")

    model_cfg = build_model_config(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model = ESM2Model(model_cfg)
    logger.info("Loading checkpoint: %s", ckpt_path)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.esm_model_path)
    scoring_datasets = load_scoring_datasets(
        scoring_cfg.datasets, n_samples=scoring_cfg.n_samples, seed=int(cfg.seed),
    )
    results = run_multi_scoring_evaluation(
        model, tokenizer, scoring_datasets,
        device=device, batch_size=scoring_cfg.batch_size, seed=int(cfg.seed),
        scores_csv_dir=cfg.get("output_csv_dir"),
        flank_ks=scoring_cfg.flank_ks,
    )

    cdr_ppl = compute_cdr_pseudo_perplexity(model, tokenizer, device)
    logger.info("CDR-H3 pseudo-perplexity: %.2f", cdr_ppl)

    out_path = out_dir / "eval_metrics.json"
    payload = {"checkpoint": str(ckpt_path), "cdr_ppl": cdr_ppl, "scoring": results}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote %s", out_path)
    for k, v in results.items():
        logger.info("  %s = %s", k, v)


if __name__ == "__main__":
    main()
