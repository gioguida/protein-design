from pathlib import Path

import pandas as pd

from protein_design.dpo.splitting import build_or_load_cluster_split_membership


def test_hamming_connected_components_and_cluster_split_membership(tmp_path: Path) -> None:
    base_df = pd.DataFrame(
        {
            "aa": ["AAA", "AAB", "ABB", "BBB", "CCC"],
            "num_mut": [2, 2, 2, 2, 2],
            "mut": ["A1A;A2A", "A1A;A2B", "A1A;B2B", "B1B;B2B", "C1C;C2C"],
            "M22_binding_enrichment_adj": [1.0, 0.5, -0.2, -0.1, 2.0],
            "delta_M22_binding_enrichment_adj": [0.1, 0.05, -0.03, -0.02, 0.2],
        }
    )

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    base_csv = processed_dir / "ED2_all.csv"
    base_df.to_csv(base_csv, index=False)

    membership = build_or_load_cluster_split_membership(
        base_df=base_df,
        base_csv_path=base_csv,
        processed_dir=processed_dir,
        train_frac=0.5,
        val_frac=0.0,
        test_frac=0.5,
        seed=123,
        force_rebuild=True,
        positive_threshold=0.0,
        stratify_bins=2,
        hamming_distance=1,
    )

    cluster_count = membership["cluster_id"].nunique()
    assert cluster_count == 2

    split_per_cluster = membership.groupby("cluster_id")["split"].nunique()
    assert (split_per_cluster == 1).all()

    assert set(membership["split"].unique()).issubset({"train", "val", "test"})


def test_cluster_split_membership_uses_all_num_mut_globally(tmp_path: Path) -> None:
    base_df = pd.DataFrame(
        {
            "aa": ["AAA", "AAB", "AAA", "AAC"],
            "num_mut": [2, 2, 3, 3],
            "mut": ["A1A;A2A", "A1A;A2B", "A1A;A2A;A3A", "A1A;A2A;A3C"],
            "M22_binding_enrichment_adj": [1.0, -0.1, 0.8, -0.2],
            "delta_M22_binding_enrichment_adj": [0.1, -0.01, 0.08, -0.02],
        }
    )

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    base_csv = processed_dir / "ED2_all.csv"
    base_df.to_csv(base_csv, index=False)

    membership = build_or_load_cluster_split_membership(
        base_df=base_df,
        base_csv_path=base_csv,
        processed_dir=processed_dir,
        train_frac=1.0,
        val_frac=0.0,
        test_frac=0.0,
        seed=123,
        force_rebuild=True,
        positive_threshold=0.0,
        stratify_bins=2,
        hamming_distance=1,
    )

    assert set(membership["num_mut"].astype(int)) == {2, 3}
    assert membership["cluster_id"].nunique() == 1
