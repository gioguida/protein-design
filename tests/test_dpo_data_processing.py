import pandas as pd

from protein_design.dpo.data_processing import (
    WT_M22_BINDING_ENRICHMENT,
    build_perplexity_eval_sets,
    get_distance2_data,
    organize_and_cluster,
)


def test_organize_and_cluster_preserves_delta_enrichment_column():
    df_d2 = pd.DataFrame(
        {
            "Unnamed: 0": [0, 1, 2],
            "aa": ["AAA", "AAB", "AAC"],
            "num_mut": [2, 2, 2],
            "mut": ["A1B;C2D", "A1B;E3F", "G4H;C2D"],
            "M22_binding_count_adj": [10.0, 8.0, 6.0],
            "M22_non_binding_count_adj": [5.0, 4.0, 3.0],
            "M22_binding_enrichment_adj": [1.0, 0.8, 0.6],
            "delta_M22_binding_enrichment_adj": [0.3, -0.1, 0.05],
        }
    )

    clustered = organize_and_cluster(df_d2, cluster_by=1)

    assert "delta_M22_binding_enrichment_adj" in clustered.columns
    assert "M22_binding_enrichment_adj" not in clustered.columns
    assert "M22_binding_enrichment" in clustered.columns


def test_get_distance2_data_supports_new_raw_schema_and_derives_delta():
    df_raw = pd.DataFrame(
        {
            "aa": ["AAA", "BBB", "CCC"],
            "num_mut": [2, 1, 2],
            "mut": ["A1B;C2D", "A1B", "E3F;G4H"],
            "count_ED2M22pos": [10.0, 4.0, 8.0],
            "count_ED2M22neg": [5.0, 2.0, 4.0],
            "M22_binding_enrichment_adj": [5.690013461, 4.690013461, 5.090013461],
        }
    )

    d2_df, n_bind, n_non_bind = get_distance2_data(df_raw)

    assert len(d2_df) == 3
    assert n_bind == 22.0
    assert n_non_bind == 11.0
    assert "M22_binding_count_adj" in d2_df.columns
    assert "M22_non_binding_count_adj" in d2_df.columns
    assert "delta_M22_binding_enrichment_adj" in d2_df.columns
    deltas = [round(float(v), 6) for v in d2_df["delta_M22_binding_enrichment_adj"].tolist()]
    assert deltas == [0.5, -0.5, -0.1]


class _DummyValConfig:
    class data:
        min_positive_delta = 0.2

        class val:
            n_val_pos = 10
            n_val_neg = 10


def test_build_perplexity_eval_sets_derives_delta_when_missing():
    df_val = pd.DataFrame(
        {
            "aa": ["AAA", "BBB", "CCC"],
            "num_mut": [2, 2, 2],
            "mut": ["A1B;C2D", "E3F;G4H", "I5J;K6L"],
            "M22_binding_enrichment_adj": [
                WT_M22_BINDING_ENRICHMENT + 1.0,
                WT_M22_BINDING_ENRICHMENT - 1.2,
                WT_M22_BINDING_ENRICHMENT + 0.1,
            ],
        }
    )

    val_pos, val_neg = build_perplexity_eval_sets(df_val=df_val, cfg=_DummyValConfig(), seed=42)

    assert not val_pos.empty
    assert not val_neg.empty
    assert (val_pos["delta_M22_binding_enrichment_adj"] > 0.2).all()
    assert (val_neg["delta_M22_binding_enrichment_adj"] < 0.0).all()


