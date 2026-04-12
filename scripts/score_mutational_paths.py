"""Standalone script to score ED2 double mutants and compute Spearman correlation.

Usage:
    python scripts/score_mutational_paths.py --config configs/scoring.yaml
    python scripts/score_mutational_paths.py --config configs/scoring.yaml --checkpoint path/to/checkpoint.pt
    python scripts/score_mutational_paths.py --config configs/scoring.yaml --checkpoint path/to/checkpoint.pt --output-csv scores.csv
"""

import argparse
import logging

import pandas as pd
import torch
import yaml
from transformers import AutoTokenizer

from protein_design.model import EvotuningModel
from protein_design.scoring import (
    evaluate_spearman,
    load_scoring_data,
    score_double_mutants,
    C05_CDRH3,
)

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


def main():
    parser = argparse.ArgumentParser(description="Score ED2 double mutants with ESM2")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", default=None, help="Path to evotuned checkpoint.pt")
    parser.add_argument("--data-path", default="data/processed/D2.csv")
    parser.add_argument("--enrichment-col", default="M22_binding_enrichment_adj")
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--output-csv", default=None, help="Save scores to CSV")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    df = load_scoring_data(args.data_path, args.n_samples, args.enrichment_col, args.seed)
    batch_size = config.get("batch_size", 512)

    print(f"\n{'Model':>22s} | Strategy | Spearman")
    print("-" * 60)

    # Base model
    model = EvotuningModel(config)
    model.to(device)
    base_avg, base_rnd = _score_model(
        model, tokenizer, df, args.enrichment_col, device, batch_size, args.seed, "Base ESM2"
    )

    # Evotuned model
    evo_avg, evo_rnd = None, None
    if args.checkpoint:
        del model
        torch.cuda.empty_cache()

        model = EvotuningModel(config)
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        evo_avg, evo_rnd = _score_model(
            model, tokenizer, df, args.enrichment_col, device, batch_size, args.seed, "Evotuned"
        )

    print("-" * 60)

    # Save scores
    if args.output_csv:
        out = df[["aa", "mut", args.enrichment_col]].copy()
        out["base_score_avg"] = base_avg
        out["base_score_random"] = base_rnd
        if evo_avg is not None:
            out["evotuned_score_avg"] = evo_avg
            out["evotuned_score_random"] = evo_rnd
        out.to_csv(args.output_csv, index=False)
        logger.info("Saved scores to %s", args.output_csv)


if __name__ == "__main__":
    main()
