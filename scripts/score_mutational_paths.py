"""Standalone script to score ED2 double mutants and compute Spearman correlation.

Usage:
    python scripts/score_mutational_paths.py scoring=d2
    python scripts/score_mutational_paths.py scoring=d2 +checkpoint=path/to/checkpoint.pt
    python scripts/score_mutational_paths.py scoring=d2 +checkpoint=path/to/best.pt +output_csv=scores.csv
"""

import logging

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer

from protein_design.eval import (
    evaluate_spearman,
    load_scoring_data,
    score_double_mutants,
)
from protein_design.model import ESM2Model
from protein_design.utils import C05_CDRH3, build_model_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _score_model(model, tokenizer, df, enrichment_col, device, batch_size, seed, label):
    """Score with both strategies and print results."""
    enrichment = df[enrichment_col].values

    scores_avg = score_double_mutants(
        model, tokenizer, C05_CDRH3, df, device,
        strategy="average", batch_size=batch_size, seed=seed,
    )
    rho_avg, pval_avg = evaluate_spearman(scores_avg, enrichment)

    scores_rnd = score_double_mutants(
        model, tokenizer, C05_CDRH3, df, device,
        strategy="random", batch_size=batch_size, seed=seed,
    )
    rho_rnd, pval_rnd = evaluate_spearman(scores_rnd, enrichment)

    print(f"  {label:20s} | average  | rho={rho_avg:+.4f} | p={pval_avg:.2e}")
    print(f"  {'':20s} | random   | rho={rho_rnd:+.4f} | p={pval_rnd:.2e}")

    return scores_avg, scores_rnd


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    checkpoint = cfg.get("checkpoint", None)
    data_path = cfg.get("data_path", "data/processed/D2.csv")
    enrichment_col = cfg.get("enrichment_col", "M22_binding_enrichment_adj")
    n_samples = cfg.get("n_samples", cfg.scoring.n_samples)
    output_csv = cfg.get("output_csv", None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    mcfg = build_model_config(cfg, device=str(device))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    df = load_scoring_data(data_path, n_samples, enrichment_col, cfg.seed)
    batch_size = cfg.scoring.batch_size

    print(f"\n{'Model':>22s} | Strategy | Spearman")
    print("-" * 60)

    # Base model
    model = ESM2Model(mcfg)
    model.to(device)
    base_avg, base_rnd = _score_model(
        model, tokenizer, df, enrichment_col, device, batch_size, cfg.seed, "Base ESM2"
    )

    # Evotuned model
    evo_avg, evo_rnd = None, None
    if checkpoint:
        del model
        torch.cuda.empty_cache()

        model = ESM2Model(mcfg)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        evo_avg, evo_rnd = _score_model(
            model, tokenizer, df, enrichment_col, device, batch_size, cfg.seed, "Evotuned"
        )

    print("-" * 60)

    # Save scores
    if output_csv:
        out = df[["aa", "mut", enrichment_col]].copy()
        out["base_score_avg"] = base_avg
        out["base_score_random"] = base_rnd
        if evo_avg is not None:
            out["evotuned_score_avg"] = evo_avg
            out["evotuned_score_random"] = evo_rnd
        out.to_csv(output_csv, index=False)
        logger.info("Saved scores to %s", output_csv)


if __name__ == "__main__":
    main()
