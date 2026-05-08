import time
from pathlib import Path

import pandas as pd
import yaml

from protein_design.dms_splitting import ensure_dataset_splits, resolve_dataset_split


def _write_config(
    tmp_path: Path,
    csv_path: Path,
    metric: str = "M22_binding_enrichment_adj",
    hamming_distance: int = 1,
) -> Path:
    cfg = {
        "split": {
            "enabled": True,
            "train_frac": 0.5,
            "val_frac": 0.25,
            "test_frac": 0.25,
            "seed": 7,
            "output_dir": str(tmp_path / "splits"),
            "hamming_distance": hamming_distance,
            "stratify_bins": 2,
        },
        "datasets": {
            "toy_m22": {
                "path": str(csv_path),
                "sequence_col": "aa",
                "key_metric_col": metric,
            }
        },
    }
    path = tmp_path / "dms.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_dms_splits_are_created_from_raw_and_reused(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "aa": ["AAAA", "AAAB", "BBBB", "BBBC", "CCCC", "DDDD", "EEEE", "FFFF"],
            "M22_binding_enrichment_adj": [5.0, -1.0, 4.0, -2.0, 0.0, 1.0, -3.0, 2.0],
        }
    ).to_csv(raw, index=False)
    cfg = _write_config(tmp_path, raw)

    paths = ensure_dataset_splits("toy_m22", cfg)
    assert set(paths) == {"train", "val", "test"}
    for path in paths.values():
        assert path.exists()

    split_aas = {
        split: set(pd.read_csv(path)["aa"].astype(str))
        for split, path in paths.items()
    }
    assert split_aas["train"].isdisjoint(split_aas["val"])
    assert split_aas["train"].isdisjoint(split_aas["test"])
    assert split_aas["val"].isdisjoint(split_aas["test"])
    assert "AAAB" in set().union(*split_aas.values())
    assert "BBBC" in set().union(*split_aas.values())

    mtimes = {split: path.stat().st_mtime for split, path in paths.items()}
    time.sleep(0.01)
    paths2 = ensure_dataset_splits("toy_m22", cfg)
    assert {split: path.stat().st_mtime for split, path in paths2.items()} == mtimes


def test_dms_split_cache_invalidates_when_metric_changes(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "aa": ["AAAA", "BBBB", "CCCC", "DDDD"],
            "M22_binding_enrichment_adj": [1.0, 2.0, 3.0, 4.0],
            "alt_metric": [4.0, 3.0, 2.0, 1.0],
        }
    ).to_csv(raw, index=False)

    cfg = _write_config(tmp_path, raw)
    first = resolve_dataset_split("toy_m22", "test", cfg)
    first_mtime = first.stat().st_mtime

    time.sleep(0.01)
    cfg = _write_config(tmp_path, raw, metric="alt_metric")
    second = resolve_dataset_split("toy_m22", "test", cfg)
    assert second.stat().st_mtime > first_mtime


def test_hamming_distance_zero_splits_by_individual_sequence(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "aa": ["AAAA", "AAAB", "AAAC", "AAAD", "AAAE", "AAAF", "AAAG", "AAAH"],
            "M22_binding_enrichment_adj": [8.0, 7.0, 6.0, 5.0, -1.0, -2.0, -3.0, -4.0],
        }
    ).to_csv(raw, index=False)
    cfg = _write_config(tmp_path, raw, hamming_distance=0)

    paths = ensure_dataset_splits("toy_m22", cfg)
    split_aas = {
        split: set(pd.read_csv(path)["aa"].astype(str))
        for split, path in paths.items()
    }

    assert sum(len(values) for values in split_aas.values()) == 8
    assert split_aas["train"].isdisjoint(split_aas["val"])
    assert split_aas["train"].isdisjoint(split_aas["test"])
    assert split_aas["val"].isdisjoint(split_aas["test"])
    assert len([split for split, values in split_aas.items() if values]) >= 2


def test_hamming_distance_zero_handles_duplicate_sequences_row_wise(tmp_path: Path) -> None:
    raw = tmp_path / "raw_dups.csv"
    pd.DataFrame(
        {
            "aa": ["AAAA"] * 12 + ["BBBB"] * 12,
            "M22_binding_enrichment_adj": list(range(12)) + list(range(-12, 0)),
        }
    ).to_csv(raw, index=False)
    cfg = _write_config(tmp_path, raw, hamming_distance=0)

    paths = ensure_dataset_splits("toy_m22", cfg, force=True)
    split_counts = {
        split: int((pd.read_csv(path)["aa"].astype(str) == "AAAA").sum())
        for split, path in paths.items()
    }

    # With hamming_distance=0, duplicate sequence rows should be split row-wise;
    # they must not all collapse into a single split.
    assert sum(1 for count in split_counts.values() if count > 0) >= 2
