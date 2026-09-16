import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from protein_design.cv_splitting import (
    assign_folds,
    cv_fold_key,
    ensure_cv_fold_splits,
    parse_cv_fold_key,
    pooled_oof_spearman,
)
from protein_design.dms_splitting import dataset_spec, resolve_dataset_split


def _write_cv_config(tmp_path: Path, csv_path: Path, val_frac: float = 0.2) -> Path:
    cfg = {
        "split": {
            "enabled": True,
            "train_frac": 0.8,
            "val_frac": 0.1,
            "test_frac": 0.1,
            "seed": 42,
            "output_dir": str(tmp_path / "base"),
            "hamming_distance": 0,
            "stratify_bins": 4,
        },
        "cv": {"n_folds": 5, "fold_seed": 0, "val_frac": val_frac, "output_dir": str(tmp_path / "cv")},
        "datasets": {
            "toy_m22": {
                "path": str(csv_path),
                "sequence_col": "aa",
                "key_metric_col": "M22_binding_enrichment_adj",
            }
        },
    }
    path = tmp_path / "cv.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _toy_raw(tmp_path: Path, n: int = 60) -> Path:
    rng = np.random.default_rng(0)
    raw = tmp_path / "raw.csv"
    pd.DataFrame(
        {
            "aa": [f"SEQ{i:04d}" for i in range(n)],
            "mut": [f"m{i}" for i in range(n)],
            "M22_binding_enrichment_adj": rng.normal(size=n),
        }
    ).to_csv(raw, index=False)
    return raw


def test_parse_and_build_cv_fold_key():
    assert parse_cv_fold_key("ed1_m22_cv5s0_f2").base_key == "ed1_m22"
    assert parse_cv_fold_key("ed1_m22_cv5s0_f2").n_folds == 5
    assert parse_cv_fold_key("ed1_m22_cv5s0_f2").fold_index == 2
    assert parse_cv_fold_key("ed1_m22") is None
    assert cv_fold_key("ed1_m22", 5, 0, 2) == "ed1_m22_cv5s0_f2"


def test_folds_tile_dataset_once_and_are_deterministic():
    metric = pd.Series(np.random.default_rng(1).normal(size=50))
    a = assign_folds(metric, n_folds=5, fold_seed=0, stratify_bins=4)
    b = assign_folds(metric, n_folds=5, fold_seed=0, stratify_bins=4)
    assert np.array_equal(a, b)  # deterministic
    assert set(np.unique(a)) == {0, 1, 2, 3, 4}
    # balanced (50 rows / 5 folds == 10 each here, but allow +/-1 for rounding)
    counts = np.bincount(a, minlength=5)
    assert counts.max() - counts.min() <= 1


def test_cv_fold_splits_no_leakage_and_cover(tmp_path: Path):
    raw = _toy_raw(tmp_path)
    cfg = _write_cv_config(tmp_path, raw)

    test_sets = []
    for i in range(5):
        key = cv_fold_key("toy_m22", 5, 0, i)
        paths = ensure_cv_fold_splits(key, cfg)
        train = set(pd.read_csv(paths["train"])["aa"])
        val = set(pd.read_csv(paths["val"])["aa"])
        test = set(pd.read_csv(paths["test"])["aa"])
        # No held-out leakage into train/val.
        assert test.isdisjoint(train)
        assert test.isdisjoint(val)
        assert train.isdisjoint(val)
        test_sets.append(test)
        meta = json.loads((paths["train"].parent / "split.meta.json").read_text())
        assert meta["fold_index"] == i and meta["n_folds"] == 5

    # The 5 test folds tile the dataset exactly once.
    pooled = [s for t in test_sets for s in t]
    assert len(pooled) == 60
    assert len(set(pooled)) == 60


def test_ensure_cv_splits_are_cached(tmp_path: Path):
    raw = _toy_raw(tmp_path)
    cfg = _write_cv_config(tmp_path, raw)
    key = cv_fold_key("toy_m22", 5, 0, 0)
    paths1 = ensure_cv_fold_splits(key, cfg)
    mtimes = {k: p.stat().st_mtime for k, p in paths1.items()}
    paths2 = ensure_cv_fold_splits(key, cfg)
    assert {k: p.stat().st_mtime for k, p in paths2.items()} == mtimes


def test_dms_splitting_delegates_cv_keys(tmp_path: Path):
    raw = _toy_raw(tmp_path)
    cfg = _write_cv_config(tmp_path, raw)
    key = cv_fold_key("toy_m22", 5, 0, 3)
    # resolve_dataset_split + dataset_spec must transparently handle fold keys.
    test_path = resolve_dataset_split(key, "test", cfg)
    assert test_path.exists()
    spec = dataset_spec(key, cfg)
    assert spec.key == key
    assert spec.key_metric_col == "M22_binding_enrichment_adj"


def test_pooled_oof_spearman_pools_not_averages():
    # Two folds; pooled Spearman over all 8 pairs, not the mean of per-fold rho.
    f0 = pd.DataFrame({"score": [1.0, 2.0, 3.0, 4.0], "M22_binding_enrichment_adj": [1.0, 2.0, 3.0, 4.0]})
    f1 = pd.DataFrame({"score": [1.0, 2.0, 3.0, 4.0], "M22_binding_enrichment_adj": [4.0, 3.0, 2.0, 1.0]})
    out = pooled_oof_spearman([f0, f1])
    assert out["n_pooled"] == 8
    assert out["n_folds"] == 2
    assert len(out["foldwise_spearman"]) == 2
    # f0 perfectly correlated (+1), f1 perfectly anti (-1); pooled is ~0.
    assert abs(out["pooled_spearman"]) < 0.5
