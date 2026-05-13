import numpy as np
import pandas as pd
import pytest

from protein_design.dpo.dataset import (
    _downsample_pairs_to_train_controlled_split,
    build_dpo_pairs_from_clustered_dataframe,
)


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


def test_delta_based_count_mix_caps_each_component_by_requested_count() -> None:
    df = pd.DataFrame(
        {
            "aa": ["P1", "P2", "P3", "P4", "N1", "N2", "N3", "N4"],
            "delta_M22_binding_enrichment_adj": [4.0, 3.0, 2.0, 1.0, -1.0, -2.0, -3.0, -4.0],
        }
    )
    pairs = build_dpo_pairs_from_clustered_dataframe(
        df,
        pairing_strategy="delta_based",
        delta_components=["cross", "within_pos"],
        delta_mix_mode="count",
        delta_component_pair_counts={"cross": 2, "within_pos": 1},
        cross_pairs_frac=1.0,
        min_score_margin=0.1,
        random_seed=1,
    )
    counts = pairs["delta_component"].value_counts().to_dict()
    assert counts.get("cross", 0) == 2
    assert counts.get("within_pos", 0) == 1
    assert len(pairs) == 3


def test_delta_based_count_mix_uses_available_when_requested_count_is_too_large() -> None:
    df = pd.DataFrame(
        {
            "aa": ["P1", "P2", "P3", "P4", "N1", "N2", "N3", "N4"],
            "delta_M22_binding_enrichment_adj": [4.0, 3.0, 2.0, 1.0, -1.0, -2.0, -3.0, -4.0],
        }
    )
    pairs = build_dpo_pairs_from_clustered_dataframe(
        df,
        pairing_strategy="delta_based",
        delta_components=["cross", "within_pos"],
        delta_mix_mode="count",
        delta_component_pair_counts={"cross": 99, "within_pos": 99},
        cross_pairs_frac=1.0,
        min_score_margin=0.1,
        random_seed=1,
    )
    counts = pairs["delta_component"].value_counts().to_dict()
    assert counts.get("cross", 0) == 4
    assert counts.get("within_pos", 0) == 2
    assert len(pairs) == 6


def test_delta_based_fraction_mix_renormalizes_when_positive_fraction_component_is_not_selected() -> None:
    df = pd.DataFrame(
        {
            "aa": ["P1", "P2", "P3", "P4", "N1", "N2", "N3", "N4"],
            "delta_M22_binding_enrichment_adj": [4.0, 3.0, 2.0, 1.0, -1.0, -2.0, -3.0, -4.0],
        }
    )
    pairs = build_dpo_pairs_from_clustered_dataframe(
        df,
        pairing_strategy="delta_based",
        delta_components=["cross", "within_pos"],
        delta_mix_mode="fraction",
        delta_component_pair_fractions={
            "cross": 0.2,
            "within_pos": 0.0,
            "wt_anchors": 0.8,
        },
        cross_pairs_frac=1.0,
        min_score_margin=0.1,
        random_seed=1,
    )
    assert not pairs.empty
    assert set(pairs["delta_component"]) == {"cross"}


def test_train_controlled_split_sizes_use_train_as_anchor() -> None:
    train_df = pd.DataFrame({"x": range(25000)})
    val_df = pd.DataFrame({"x": range(20000)})
    test_df = pd.DataFrame({"x": range(20000)})
    out_train, out_val, out_test = _downsample_pairs_to_train_controlled_split(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_frac=0.8,
        val_frac=0.1,
        test_frac=0.1,
        rng=np.random.default_rng(1),
    )
    assert len(out_train) == 25000
    assert len(out_val) == 3125
    assert len(out_test) == 3125
