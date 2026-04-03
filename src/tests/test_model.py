"""Sanity checks for ESM2 PLL scoring.

Run with:
1) pytest src/tests/test_model.py
2) python -m src.tests.test_model
"""

import torch
from functools import lru_cache

from src.model import ESM2Config, ESM2PLLScorer, LEFT_CONTEXT


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ESM_MODEL_PATH = "/cluster/project/krause/flohmann/mgm/oracle_assets/esm2_8m.safetensors"

TEST_CDR_SEQUENCES = [
    "HMSMQQVVSAGWERADLVGDAFDV",     # wild type 
    "AASMQQVRSAGWERADLVGDAFEV",
    "ACSMQQVVSAGWSRADLVGDDFDV",
]


@lru_cache(maxsize=1)
def build_scorer() -> ESM2PLLScorer:
    cfg = ESM2Config(
        esm_model_path=ESM_MODEL_PATH,
        device=DEVICE,
        use_context=True,
    )
    return ESM2PLLScorer(cfg)


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
    test_cdr_positions_mapping()
    print("[OK] CDR position mapping sanity check passed.")

    test_logits_shape_is_batch_len_vocab()
    print("[OK] Logits shape sanity check passed.")

    test_pll_shape_and_finite_values()
    print("[OK] PLL finite/shape sanity check passed.")

    test_use_grad_toggle_behavior()
    print("[OK] use_grad toggle sanity check passed.")

    test_masked_pll_shape_finite_and_grad_behavior()
    print("[OK] masked PLL finite/shape/grad sanity check passed.")

    test_masked_pll_matches_full_pll_on_cdr_positions()
    print("[OK] masked PLL matches full CDR PLL sanity check passed.")


if __name__ == "__main__":
    main()

