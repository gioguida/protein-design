"""Sanity checks for ESM2 PLL scoring.

Run with:
1) pytest src/tests/test_model.py
2) python -m src.tests.test_model
"""

import torch
from pathlib import Path
from typing import List

from src.model import ESM2PLLScorer
from src.dataset import (
    create_train_val_test_split
    )
from src.train_dpo import (
     load_dpo_pair_dataframe, 
     _load_checkpoint,
     _build_dataloader
)
from src.utils import ModelConfig
from src.eval import sequence_perplexity



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ESM_MODEL_PATH = "facebook/esm2_t12_35M_UR50D"
FINE_TUNED_CHECKPOINT_PATH = "/cluster/home/gguidarini/protein-design/outputs/2026-04-05/08-24-51/checkpoints/best.pt"

TEST_CDR_SEQUENCES = [
    "HMSMQQVVSAGWERADLVGDAFDV",     # wild type 
    "AASMQQVRSAGWERADLVGDAFEV",
    "ACSMQQVVSAGWSRADLVGDDFDV",
]


def _diff_positions(winner: str, loser: str) -> List[int]:
    if len(winner) != len(loser):
        raise ValueError("Winner and loser must have equal length.")
    return [idx for idx, (w_char, l_char) in enumerate(zip(winner, loser)) if w_char != l_char]


def _assert_mask_position_mapping(sequence: str, diff_positions: List[int], scorer: ESM2PLLScorer) -> None:
    if not diff_positions:
        return

    token_positions = scorer.cdr_to_token_positions(diff_positions)

    pll_from_cdr_positions = scorer.masked_pseudo_log_likelihood(
        [sequence],
        diff_positions,
        use_grad=False,
        positions_are_cdr=True,
    )
    pll_from_token_positions = scorer.masked_pseudo_log_likelihood(
        [sequence],
        token_positions,
        use_grad=False,
        positions_are_cdr=False,
    )

    if not torch.allclose(pll_from_cdr_positions, pll_from_token_positions, atol=1e-6):
        raise AssertionError("CDR-position masking and token-position masking do not match.")


def _assert_perplexity_formula(chosen_seqs: List[str], scorer: ESM2PLLScorer) -> None:
    perplexities = sequence_perplexity(chosen_seqs, scorer=scorer, cdr_only=True)
    pll_scores = scorer.pseudo_log_likelihood(chosen_seqs, cdr_only=True, use_grad=False)
    cdr_len = float(len(chosen_seqs[0]))
    manual_perplexities = torch.exp(-pll_scores / cdr_len)

    if not torch.allclose(perplexities, manual_perplexities, atol=1e-6):
        raise AssertionError("Perplexity does not match exp(-PLL / N).")

    if not torch.isfinite(perplexities).all().item():
        raise AssertionError("Perplexity contains non-finite values.")

class dataset_config:
    def __init__(self):
        self.raw_csv = "data/raw/M22_binding_enrichment.csv"
        self.processed_dir = "data/processed"
        self.force_rebuild = False
        self.pairing_strategy = "positive_vs_tail"  # positive_vs_tail | positive_only_extremes | both_structured | delta_based
        self.min_positive_delta = 1.0
        self.min_delta_margin = 2.0
        self.gap = 0.5
        self.wt_pairs_frac = 0.1
        self.cross_pairs_frac = 0.1
        self.strong_pos_threshold = 1.0
        self.strong_neg_threshold = -5.0
        self.min_score_margin = 0.1
        self.include_views = ["mut1", "mut2"]
        self.deduplicate_across_views = True
        self.train_frac = 0.8
        self.val_frac = 0.1
        self.test_frac = 0.1


def main():   
    cfg_data = dataset_config()
    policy_cfg = ModelConfig(
        esm_model_path=ESM_MODEL_PATH,
        device=DEVICE,
        use_context=True,
    )
    reference_cfg = ModelConfig(
        esm_model_path=ESM_MODEL_PATH,
        device=DEVICE,
        use_context=True,
    )

    pairs_df = load_dpo_pair_dataframe(
        pairing_strategy=cfg_data.pairing_strategy,
        include_views=[str(v) for v in cfg_data.include_views],
        raw_csv_path=cfg_data.raw_csv,
        processed_dir=cfg_data.processed_dir,
        force_rebuild=False,
        min_positive_delta=cfg_data.min_positive_delta,
        min_delta_margin=cfg_data.min_delta_margin,
        gap=cfg_data.gap,
        wt_pairs_frac=cfg_data.wt_pairs_frac,
        cross_pairs_frac=cfg_data.cross_pairs_frac,
        strong_pos_threshold=cfg_data.strong_pos_threshold,
        strong_neg_threshold=cfg_data.strong_neg_threshold,
        min_score_margin=cfg_data.min_score_margin,
        deduplicate_across_views=cfg_data.deduplicate_across_views,
    )
    train_df, val_df, test_df = create_train_val_test_split(
        pairs_df,
        train_frac=float(cfg_data.train_frac),
        val_frac=float(cfg_data.val_frac),
        test_frac=float(cfg_data.test_frac),
        seed=42
    )

    if test_df.empty:
        raise ValueError("Test split is empty; cannot verify model/perplexity behavior.")

    policy = ESM2PLLScorer(policy_cfg)
    _load_checkpoint(
        checkpoint_path=Path(FINE_TUNED_CHECKPOINT_PATH),
        policy=policy,
        optimizer=None,
        scheduler=None,
    )
    reference = ESM2PLLScorer(reference_cfg)

    for param in reference.model.parameters():
        param.requires_grad_(False)

    for param in policy.model.parameters():
        param.requires_grad_(False)

    test_loader = _build_dataloader(
        pairs_df=test_df,
        batch_size=64,
        shuffle=False,
        seed=42,
        num_workers=0,
    )

    first_chosen = str(test_df.iloc[0]["chosen_sequence"])
    first_rejected = str(test_df.iloc[0]["rejected_sequence"])
    first_diff_positions = _diff_positions(first_chosen, first_rejected)
    _assert_mask_position_mapping(first_chosen, first_diff_positions, policy)
    _assert_mask_position_mapping(first_chosen, first_diff_positions, reference)

    total_perplexity_finetuned = 0.0
    total_perplexity_reference = 0.0
    num_chosen = 0
    formula_checked = False

    with torch.no_grad():
        for batch in test_loader:
            chosen_seqs = [str(pair[0]["aa"]) for pair in batch]
            try:
                if not formula_checked:
                    _assert_perplexity_formula(chosen_seqs, policy)
                    _assert_perplexity_formula(chosen_seqs, reference)
                    formula_checked = True

                perplexities_finetuned = sequence_perplexity(chosen_seqs, scorer=policy, cdr_only=True)
                total_perplexity_finetuned += float(perplexities_finetuned.sum().item())

                perplexities_reference = sequence_perplexity(chosen_seqs, scorer=reference, cdr_only=True)
                total_perplexity_reference += float(perplexities_reference.sum().item())

                num_chosen += len(chosen_seqs)
            except ValueError:
                continue

    if num_chosen == 0:
        raise ValueError("No valid chosen sequences were evaluated.")

    avg_test_perplexity_finetuned = total_perplexity_finetuned / max(1, num_chosen)
    print(f"Test Chosen Perplexity (Finetuned): {avg_test_perplexity_finetuned:.4f}")

    avg_test_perplexity_reference = total_perplexity_reference / max(1, num_chosen)
    print(f"Test Chosen Perplexity (Reference): {avg_test_perplexity_reference:.4f}")
    print("Verification checks passed: perplexity formula and masking position mapping are consistent.")


if __name__ == "__main__":
    main()

