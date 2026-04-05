"""Sanity checks for ESM2 PLL scoring.

Run with:
1) pytest src/tests/test_model.py
2) python -m src.tests.test_model
"""

import torch
from functools import lru_cache
from pathlib import Path

from src.model import ESM2PLLScorer
from src.dataset import (
    create_train_val_test_split
    )
from src.train_dpo import (
     load_dpo_pair_dataframe, 
     _load_checkpoint,
     _build_dataloader
)
from src.utils import LEFT_CONTEXT, ModelConfig
from eval import sequence_perplexity



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ESM_MODEL_PATH = "/cluster/project/krause/flohmann/mgm/oracle_assets/esm2_8m.safetensors"
FINE_TUNED_CHECKPOINT_PATH = "/cluster/home/gguidarini/protein-design/outputs/2026-04-05/08-24-51/checkpoints/best.pt"

TEST_CDR_SEQUENCES = [
    "HMSMQQVVSAGWERADLVGDAFDV",     # wild type 
    "AASMQQVRSAGWERADLVGDAFEV",
    "ACSMQQVVSAGWSRADLVGDDFDV",
]

class dataset_config:
    def __init__(self):
        self.raw_csv = "data/raw/M22_binding_enrichment.csv"
        self.processed_dir = "data/processed"
        self.force_rebuild = False
        self.pairing_strategy = "positive_vs_tail"  # positive_vs_tail | positive_only_extremes
        self.min_positive_delta = 1.0
        self.min_delta_margin = 2.0
        self.include_views = ["mut1", "mut2"]
        self.deduplicate_across_views = True
        self.train_frac = 0.8
        self.val_frac = 0.1
        self.test_frac = 0.1


class policy_config:
    def __init__(self):
        self.esm_model_path=FINE_TUNED_CHECKPOINT_PATH
        self.device="cuda" if torch.cuda.is_available() else "cpu",
        self.use_context=True,


class reference_config:
    def __init__(self):
        self.esm_model_path="/cluster/project/krause/flohmann/mgm/oracle_assets/esm2_8m.safetensors"
        self.device="cuda" if torch.cuda.is_available() else "cpu",
        self.use_context=True,


def main():   
    cfg_data = dataset_config()
    policy_cfg = policy_config()
    reference_cfg = reference_config()

    pairs_df = load_dpo_pair_dataframe(
        pairing_strategy=cfg_data.pairing_strategy,
        include_views=[str(v) for v in cfg_data.include_views],
        raw_csv_path=cfg_data.raw_csv,
        processed_dir=cfg_data.processed_dir,
        force_rebuild=False,
        min_positive_delta=cfg_data.min_positive_delta,
        min_delta_margin=cfg_data.min_delta_margin,
        deduplicate_across_views=cfg_data.deduplicate_across_views,
    )
    train_df, val_df, test_df = create_train_val_test_split(
        pairs_df,
        train_frac=float(cfg_data.train_frac),
        val_frac=float(cfg_data.val_frac),
        test_frac=float(cfg_data.test_frac),
        seed=int(cfg_data.seed),
    )

    policy = ESM2PLLScorer(policy_cfg)
    reference = ESM2PLLScorer(reference_cfg)

    for param in reference.model.parameters():
        param.requires_grad_(False)

    test_loader = _build_dataloader(
        pairs_df=test_df,
        batch_size=64,
        shuffle=False,
        seed=42,
        num_workers=0,
    )

    with torch.no_grad():
        for batch in test_loader:
            chosen_seqs = [pair[0] for pair in batch]
            try:
                perplexities_finetuned = sequence_perplexity(chosen_seqs, scorer=policy, cdr_only=True)
                total_perplexity_finetuned += float(perplexities_finetuned.sum().item())

                perplexities_reference = sequence_perplexity(chosen_seqs, scorer=reference, cdr_only=True)
                total_perplexity_reference += float(perplexities_reference.sum().item())

                num_chosen += len(chosen_seqs)
            except ValueError:
                continue

    avg_test_perplexity_finetuned = total_perplexity_finetuned / max(1, num_chosen)
    print(f"Test Chosen Perplexity (Finetuned): {avg_test_perplexity_finetuned:.4f}")

    avg_test_perplexity_reference = total_perplexity_reference / max(1, num_chosen)
    print(f"Test Chosen Perplexity (Reference): {avg_test_perplexity_reference:.4f}")


if __name__ == "__main__":
    main()

