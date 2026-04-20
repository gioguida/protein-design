import pandas as pd

from protein_design.dpo.dataset import (
    build_dpo_pairs_from_clustered_dataframe,
    create_train_val_test_split,
)


def test_positive_vs_tail_pairing_builds_expected_pairs() -> None:
    clustered = pd.DataFrame(
        {
            "cluster_idx": [0, 0, 0, 0],
            "aa": ["AAA", "AAB", "AAC", "AAD"],
            "delta_M22_binding_enrichment_adj": [3.0, 2.0, -1.0, -2.0],
        }
    )

    pairs = build_dpo_pairs_from_clustered_dataframe(
        clustered_df=clustered,
        pairing_strategy="positive_vs_tail",
        min_positive_delta=0.0,
        min_delta_margin=0.1,
        source_view="mut1",
    )

    assert len(pairs) == 2
    assert list(pairs["chosen_sequence"]) == ["AAA", "AAB"]
    assert list(pairs["rejected_sequence"]) == ["AAD", "AAC"]
    assert (pairs["delta_margin"] > 0).all()


def test_delta_based_pairing_with_cross_component() -> None:
    base_df = pd.DataFrame(
        {
            "aa": ["AAA", "AAB", "AAC", "AAD", "AAE", "AAF"],
            "delta_M22_binding_enrichment_adj": [2.1, 1.5, 0.7, -0.2, -1.2, -2.0],
        }
    )

    pairs = build_dpo_pairs_from_clustered_dataframe(
        clustered_df=base_df,
        pairing_strategy="delta_based",
        delta_components=("cross",),
        cross_pairs_frac=0.5,
        min_score_margin=0.1,
        random_seed=42,
        source_view="base",
    )

    assert not pairs.empty
    assert "delta_component" in pairs.columns
    assert set(pairs["delta_component"].tolist()) == {"cross"}
    assert (pairs["delta_margin"] >= 0.1).all()


def test_train_val_test_split_sizes_sum_to_input() -> None:
    df = pd.DataFrame({"aa": [f"A{i}" for i in range(20)]})
    train_df, val_df, test_df = create_train_val_test_split(df, 0.6, 0.2, 0.2, seed=7)
    assert len(train_df) + len(val_df) + len(test_df) == len(df)

