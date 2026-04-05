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
     _build_pairs_dataframe, 
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

    

@lru_cache(maxsize=1)
def build_scorer() -> ESM2PLLScorer:
    cfg = ModelConfig(
        esm_model_path=ESM_MODEL_PATH,
        device=DEVICE,
        use_context=True,
    )
    scorer = ESM2PLLScorer(cfg)

    ckpt_path = Path(FINE_TUNED_CHECKPOINT_PATH)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Fine-tuned checkpoint not found: {ckpt_path}")

    _load_checkpoint(ckpt_path, policy=scorer, optimizer=None, scheduler=None)
    return scorer


def test_cdr_positions_mapping():
    scorer = build_scorer()
    cdr = TEST_CDR_SEQUENCES[0]

    positions = scorer._cdr_positions(len(cdr))
    tokens = scorer.tokenize_sequences([cdr])

    assert len(positions) == len(cdr)
    assert positions[0] == len(LEFT_CONTEXT)
    assert positions[-1] == len(LEFT_CONTEXT) + len(cdr) - 1
    assert positions[-1] < tokens.shape[1]
    assert positions[0] > 0


def test_logits_shape_is_batch_len_vocab():
    scorer = build_scorer()
    tokens = scorer.tokenize_sequences(TEST_CDR_SEQUENCES)

    with torch.no_grad():
        logits = scorer.forward_logits(tokens)

    assert logits.ndim == 3
    assert logits.shape[0] == len(TEST_CDR_SEQUENCES)
    assert logits.shape[1] == tokens.shape[1]
    assert logits.shape[2] > 0


def test_pll_shape_and_finite_values():
    scorer = build_scorer()

    pll = scorer.pseudo_log_likelihood(
        TEST_CDR_SEQUENCES,
        cdr_only=True,
        use_grad=False,
    )

    assert pll.shape == (len(TEST_CDR_SEQUENCES),)
    assert torch.isfinite(pll).all().item()


def test_use_grad_toggle_behavior():
    scorer = build_scorer()

    pll_no_grad = scorer.pseudo_log_likelihood(
        TEST_CDR_SEQUENCES,
        cdr_only=True,
        use_grad=False,
    )
    pll_with_grad = scorer.pseudo_log_likelihood(
        TEST_CDR_SEQUENCES,
        cdr_only=True,
        use_grad=True,
    )

    assert not pll_no_grad.requires_grad
    assert pll_with_grad.requires_grad


def test_masked_pll_shape_finite_and_grad_behavior():
    scorer = build_scorer()
    cdr_positions = scorer._cdr_positions(len(TEST_CDR_SEQUENCES[0]))
    subset_positions = cdr_positions[:4]

    masked_pll_no_grad = scorer.masked_pseudo_log_likelihood(
        TEST_CDR_SEQUENCES,
        mask_positions=subset_positions,
        use_grad=False,
    )
    masked_pll_with_grad = scorer.masked_pseudo_log_likelihood(
        TEST_CDR_SEQUENCES,
        mask_positions=subset_positions,
        use_grad=True,
    )

    assert masked_pll_no_grad.shape == (len(TEST_CDR_SEQUENCES),)
    assert torch.isfinite(masked_pll_no_grad).all().item()
    assert not masked_pll_no_grad.requires_grad
    assert masked_pll_with_grad.requires_grad


def test_masked_pll_matches_full_pll_on_cdr_positions():
    scorer = build_scorer()
    cdr_positions = scorer._cdr_positions(len(TEST_CDR_SEQUENCES[0]))

    pll_full_cdr = scorer.pseudo_log_likelihood(
        TEST_CDR_SEQUENCES,
        cdr_only=True,
        use_grad=False,
    )
    pll_masked_cdr = scorer.masked_pseudo_log_likelihood(
        TEST_CDR_SEQUENCES,
        mask_positions=cdr_positions,
        use_grad=False,
    )

    assert torch.allclose(pll_full_cdr, pll_masked_cdr, atol=1e-6)


def main():   
    cfg_data = dataset_config()
    policy_cfg = policy_config()
    reference_cfg = reference_config()

    pairs_df = _build_pairs_dataframe(cfg_data)
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

