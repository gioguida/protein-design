from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from protein_design.constants import C05_CDRH3
from protein_design.pssm_baseline import (
    STANDARD_AAS,
    build_output_rows,
    build_pssm_counts,
    counts_to_log_frequencies,
    load_train_dataframe,
    resolve_train_split_with_fallback,
    sample_cdrh3_sequences,
)


def _write_dms_config(tmp_path: Path, *, raw_path: Path) -> Path:
    config = {
        "split": {
            "enabled": True,
            "train_frac": 0.8,
            "val_frac": 0.1,
            "test_frac": 0.1,
            "seed": 42,
            "output_dir": str(tmp_path / "configured_splits"),
            "hamming_distance": 0,
            "stratify_bins": 2,
        },
        "datasets": {
            "toy_m22": {
                "path": str(raw_path),
                "sequence_col": "aa",
                "key_metric_col": "score",
            }
        },
    }
    config_path = tmp_path / "dms.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_resolve_train_split_with_fallback_uses_local_cached_split(tmp_path: Path) -> None:
    config_path = _write_dms_config(tmp_path, raw_path=tmp_path / "missing_raw.csv")
    local_split_dir = tmp_path / "local_splits" / "toy_m22"
    local_split_dir.mkdir(parents=True, exist_ok=True)
    train_csv = local_split_dir / "train.csv"
    pd.DataFrame({"aa": [C05_CDRH3], "score": [1.0]}).to_csv(train_csv, index=False)

    resolved, source = resolve_train_split_with_fallback("toy_m22", config_path, tmp_path / "local_splits")

    assert resolved == train_csv
    assert source == "local_fallback"


def test_load_train_dataframe_applies_threshold_filter(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "aa": [C05_CDRH3, "A" + C05_CDRH3[1:]],
            "score": [2.5, 0.5],
        }
    ).to_csv(raw, index=False)
    config_path = _write_dms_config(tmp_path, raw_path=raw)
    local_split_dir = tmp_path / "local_splits" / "toy_m22"
    local_split_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "aa": [C05_CDRH3, "A" + C05_CDRH3[1:]],
            "score": [2.5, 0.5],
        }
    ).to_csv(local_split_dir / "train.csv", index=False)

    df, sequence_col, metric_col, split_path, source = load_train_dataframe(
        "toy_m22",
        config_path,
        tmp_path / "local_splits",
        enrichment_threshold=1.0,
    )

    assert sequence_col == "aa"
    assert metric_col == "score"
    assert split_path.exists()
    assert source in {"configured", "local_fallback"}
    assert df["aa"].tolist() == [C05_CDRH3]


def test_pssm_sampling_is_deterministic_and_builds_expected_rows() -> None:
    sequences = [C05_CDRH3, C05_CDRH3, "A" + C05_CDRH3[1:]]
    counts = build_pssm_counts(sequences)
    assert counts.shape == (len(C05_CDRH3), len(STANDARD_AAS))
    assert counts[0].sum() == len(sequences)

    log_freq = counts_to_log_frequencies(counts, pseudocount=1.0)
    assert np.isfinite(log_freq).all()

    sampled_a = sample_cdrh3_sequences(log_freq, temperature=1.0, n_sequences=4, seed=7)
    sampled_b = sample_cdrh3_sequences(log_freq, temperature=1.0, n_sequences=4, seed=7)
    assert sampled_a == sampled_b
    assert all(len(seq) == len(C05_CDRH3) for seq in sampled_a)

    rows = build_output_rows(sampled_a)
    assert [row["chain_id"] for row in rows] == list(range(4))
    assert all(row["gibbs_step"] == 0 for row in rows)
    assert all(row["model_variant"] == "pssm" for row in rows)
