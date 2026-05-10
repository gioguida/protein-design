import pandas as pd
import pytest

from protein_design.dpo.dataset import build_dpo_pairs_from_clustered_dataframe


def test_legacy_dpo_pairing_strategy_is_rejected() -> None:
    df = pd.DataFrame(
        {
            "aa": ["AAAA", "BBBB"],
            "delta_M22_binding_enrichment_adj": [1.0, -1.0],
        }
    )
    with pytest.raises(ValueError, match="Only data.pairing_strategy='delta_based'"):
        build_dpo_pairs_from_clustered_dataframe(df, pairing_strategy="positive_vs_tail")


def test_delta_based_pairing_accepts_mixed_num_mut_rows() -> None:
    df = pd.DataFrame(
        {
            "aa": ["AAAA", "BBBB", "CCCC", "DDDD"],
            "num_mut": [1, 2, 4, 7],
            "delta_M22_binding_enrichment_adj": [2.0, -2.0, 1.5, -1.5],
        }
    )
    pairs = build_dpo_pairs_from_clustered_dataframe(
        df,
        pairing_strategy="delta_based",
        delta_components=["cross"],
        cross_pairs_frac=1.0,
        min_score_margin=0.1,
        random_seed=1,
    )
    assert not pairs.empty
    assert set(pairs["pairing_strategy"]) == {"delta_based"}
