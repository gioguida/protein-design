"""Shared DMS dataset config and cached split utilities."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd
import yaml


SPLIT_NAMES = ("train", "val", "test")
DEFAULT_CONFIG_PATH = Path("conf/data/dms/default.yaml")


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    path: Path
    sequence_col: str
    key_metric_col: str
    split_source: Optional[str] = None


@dataclass(frozen=True)
class SplitConfig:
    enabled: bool
    train_frac: float
    val_frac: float
    test_frac: float
    seed: int
    output_dir: Path
    hamming_distance: int
    stratify_bins: int


@dataclass(frozen=True)
class DMSConfig:
    path: Path
    split: SplitConfig
    datasets: Mapping[str, DatasetSpec]


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _expand_path(value: str | Path, base_dir: Path) -> Path:
    raw = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(raw)
    if not path.is_absolute():
        path = base_dir / path
    return path


def _get(node: Any, key: str, default: Any = None) -> Any:
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def load_dms_config(config_path: str | Path | None = None) -> DMSConfig:
    root = project_root()
    path = _expand_path(config_path or DEFAULT_CONFIG_PATH, root)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    split_raw = raw.get("split", {}) or {}
    split = SplitConfig(
        enabled=bool(split_raw.get("enabled", True)),
        train_frac=float(split_raw.get("train_frac", 0.8)),
        val_frac=float(split_raw.get("val_frac", 0.1)),
        test_frac=float(split_raw.get("test_frac", 0.1)),
        seed=int(split_raw.get("seed", 42)),
        output_dir=_expand_path(split_raw.get("output_dir", "data/processed/dms_splits"), root),
        hamming_distance=int(split_raw.get("hamming_distance", 1)),
        stratify_bins=int(split_raw.get("stratify_bins", 10)),
    )
    if abs(split.train_frac + split.val_frac + split.test_frac - 1.0) >= 1e-6:
        raise ValueError("DMS split fractions must sum to 1.0.")
    if split.hamming_distance not in {0, 1}:
        raise ValueError("Only hamming_distance=0 or hamming_distance=1 is currently supported.")

    datasets: Dict[str, DatasetSpec] = {}
    for key, entry in (raw.get("datasets", {}) or {}).items():
        if not entry or "path" not in entry:
            raise ValueError(f"datasets.{key}.path is required in {path}")
        datasets[str(key)] = DatasetSpec(
            key=str(key),
            path=_expand_path(entry["path"], root),
            sequence_col=str(entry.get("sequence_col", "aa")),
            key_metric_col=str(entry.get("key_metric_col", "M22_binding_enrichment_adj")),
            split_source=None if entry.get("split_source") is None else str(entry.get("split_source")),
        )
    if not datasets:
        raise ValueError(f"No DMS datasets configured in {path}")
    return DMSConfig(path=path, split=split, datasets=datasets)


def dms_config_path_from_cfg(cfg: Any) -> Path:
    raw = _get(_get(cfg, "data", {}), "dms_config", None)
    return _expand_path(raw or DEFAULT_CONFIG_PATH, project_root())


def _split_paths(config: DMSConfig, dataset_key: str) -> Dict[str, Path]:
    out_dir = config.split.output_dir / dataset_key
    return {split: out_dir / f"{split}.csv" for split in SPLIT_NAMES}


def _meta_path(config: DMSConfig, dataset_key: str) -> Path:
    return config.split.output_dir / dataset_key / "split.meta.json"


def _cluster_ids_hamming_lte_one(sequences: list[str]) -> np.ndarray:
    n = len(sequences)
    uf = _UnionFind(n)
    by_len: Dict[int, list[int]] = {}
    for idx, seq in enumerate(sequences):
        by_len.setdefault(len(seq), []).append(idx)
    for indices in by_len.values():
        sig_to_first: Dict[str, int] = {}
        for idx in indices:
            seq = sequences[idx]
            if not seq:
                sig = "__empty__"
                prev = sig_to_first.get(sig)
                sig_to_first[sig] = idx if prev is None else prev
                if prev is not None:
                    uf.union(idx, prev)
                continue
            for pos in range(len(seq)):
                sig = seq[:pos] + "*" + seq[pos + 1 :]
                prev = sig_to_first.get(sig)
                sig_to_first[sig] = idx if prev is None else prev
                if prev is not None:
                    uf.union(idx, prev)
    root_to_id: Dict[int, int] = {}
    out = np.empty(n, dtype=np.int64)
    for idx in range(n):
        root = uf.find(idx)
        if root not in root_to_id:
            root_to_id[root] = len(root_to_id)
        out[idx] = root_to_id[root]
    return out


def _cluster_ids(sequences: list[str], hamming_distance: int) -> np.ndarray:
    if hamming_distance == 0:
        return np.arange(len(sequences), dtype=np.int64)
    if hamming_distance == 1:
        return _cluster_ids_hamming_lte_one(sequences)
    raise ValueError(f"Unsupported hamming_distance: {hamming_distance}")


def _target_counts(n: int, split: SplitConfig) -> Dict[str, int]:
    n_train = int(round(n * split.train_frac))
    n_val = int(round(n * split.val_frac))
    n_train = min(max(n_train, 0), n)
    n_val = min(max(n_val, 0), n - n_train)
    return {"train": n_train, "val": n_val, "test": n - n_train - n_val}


def _metric_strata(values: pd.Series, num_bins: int) -> pd.Series:
    if len(values) == 0:
        return pd.Series(dtype=np.int64, index=values.index)
    num_bins = max(1, min(int(num_bins), len(values)))
    if num_bins == 1:
        return pd.Series(np.zeros(len(values), dtype=np.int64), index=values.index)
    try:
        return pd.qcut(values, q=num_bins, labels=False, duplicates="drop").fillna(0).astype(int)
    except ValueError:
        return pd.Series(np.zeros(len(values), dtype=np.int64), index=values.index)


def _assign_rows_stratified(
    row_indices: pd.Index,
    metric_values: pd.Series,
    split: SplitConfig,
    rng: np.random.Generator,
) -> pd.Series:
    labels = pd.Series(index=row_indices, dtype=object)
    strata = _metric_strata(metric_values.loc[row_indices], split.stratify_bins)
    for stratum_id in sorted(strata.unique().tolist()):
        indices = strata.index[strata == stratum_id].to_numpy()
        indices = rng.permutation(indices)
        counts = _target_counts(len(indices), split)
        train_end = counts["train"]
        val_end = train_end + counts["val"]
        labels.loc[indices[:train_end]] = "train"
        labels.loc[indices[train_end:val_end]] = "val"
        labels.loc[indices[val_end:]] = "test"
    return labels


def _assign_small_cluster_splits(
    cluster_stats: pd.DataFrame,
    split: SplitConfig,
    rng: np.random.Generator,
    current_counts: Dict[str, int],
    target_counts: Dict[str, int],
) -> Dict[int, str]:
    if cluster_stats.empty:
        return {}
    num_bins = max(1, min(int(split.stratify_bins), len(cluster_stats)))
    strata = _metric_strata(cluster_stats["metric_mean"], num_bins)
    split_map: Dict[int, str] = {}
    for stratum_id in sorted(strata.unique().tolist()):
        stratum = cluster_stats.loc[strata == stratum_id].copy()
        stratum = stratum.iloc[rng.permutation(len(stratum))]
        for row in stratum.itertuples(index=False):
            cluster_size = int(row.cluster_size)
            deficits = {
                name: target_counts[name] - current_counts[name]
                for name in SPLIT_NAMES
            }
            fitting = [name for name in SPLIT_NAMES if deficits[name] >= cluster_size]
            if fitting:
                split_name = max(fitting, key=lambda name: deficits[name])
            else:
                split_name = max(SPLIT_NAMES, key=lambda name: deficits[name])
            split_map[int(row.cluster_id)] = split_name
            current_counts[split_name] += cluster_size
    return split_map


def _expected_meta(config: DMSConfig, dataset_key: str, spec: DatasetSpec, source_key: str) -> dict[str, Any]:
    stat = spec.path.stat()
    source_spec = config.datasets[source_key]
    source_stat = source_spec.path.stat()
    return {
        "version": 3,
        "method": "size_aware_hamming_components_with_stratified_oversized_component_fallback",
        "dataset_key": dataset_key,
        "split_source": source_key,
        "path": str(spec.path),
        "source_path": str(source_spec.path),
        "path_mtime": stat.st_mtime,
        "source_path_mtime": source_stat.st_mtime,
        "sequence_col": spec.sequence_col,
        "key_metric_col": spec.key_metric_col,
        "source_sequence_col": source_spec.sequence_col,
        "source_key_metric_col": source_spec.key_metric_col,
        "train_frac": config.split.train_frac,
        "val_frac": config.split.val_frac,
        "test_frac": config.split.test_frac,
        "seed": config.split.seed,
        "hamming_distance": config.split.hamming_distance,
        "stratify_bins": config.split.stratify_bins,
    }


def _metadata_matches(path: Path, expected: Mapping[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as fh:
            existing = json.load(fh)
    except Exception:
        return False
    return all(existing.get(k) == v for k, v in expected.items())


def _read_validated(spec: DatasetSpec) -> pd.DataFrame:
    if not spec.path.exists():
        raise FileNotFoundError(f"DMS dataset not found: {spec.path}")
    df = pd.read_csv(spec.path)
    missing = {spec.sequence_col, spec.key_metric_col}.difference(df.columns)
    if missing:
        raise ValueError(f"{spec.path} missing required columns: {sorted(missing)}")
    df = df.copy()
    df[spec.sequence_col] = df[spec.sequence_col].astype(str).str.strip()
    df = df[df[spec.sequence_col] != ""].reset_index(drop=True)
    df[spec.key_metric_col] = pd.to_numeric(df[spec.key_metric_col], errors="coerce")
    return df


def _build_source_membership(source_df: pd.DataFrame, spec: DatasetSpec, split: SplitConfig) -> pd.DataFrame:
    working = source_df.copy().reset_index(drop=True)
    working["_split_sequence"] = working[spec.sequence_col].astype(str)
    working["_split_metric"] = pd.to_numeric(working[spec.key_metric_col], errors="coerce")
    working["_split_metric_for_strata"] = working["_split_metric"].fillna(working["_split_metric"].median())
    if working["_split_metric_for_strata"].isna().all():
        working["_split_metric_for_strata"] = 0.0
    rng = np.random.default_rng(int(split.seed))
    working["cluster_id"] = _cluster_ids(
        working["_split_sequence"].tolist(),
        split.hamming_distance,
    )
    working["split"] = pd.Series(index=working.index, dtype=object)

    # For hamming_distance=0 each row is an independent unit, so perform a
    # direct row-wise stratified split to preserve the enrichment distribution.
    if int(split.hamming_distance) == 0:
        labels = _assign_rows_stratified(
            working.index,
            working["_split_metric_for_strata"],
            split,
            rng,
        )
        working["split"] = labels
        return working[[spec.sequence_col, "split", "cluster_id"]].rename(columns={spec.sequence_col: "split_key"})

    stats = (
        working.groupby("cluster_id", sort=True)["_split_metric_for_strata"]
        .agg(cluster_size="size", metric_mean="mean")
        .reset_index()
    )
    target_counts = _target_counts(len(working), split)
    current_counts = {name: 0 for name in SPLIT_NAMES}

    # ED2 contains very large Hamming-connected components. Keeping those
    # components indivisible can make the configured fractions impossible, so
    # only oversized components are split internally with the same metric
    # stratification used for the full split.
    oversized_threshold = max(1, min(target_counts["val"], target_counts["test"]))
    oversized_cluster_ids = set(
        stats.loc[stats["cluster_size"] > oversized_threshold, "cluster_id"].astype(int).tolist()
    )
    for cluster_id in sorted(oversized_cluster_ids):
        indices = working.index[working["cluster_id"] == cluster_id]
        labels = _assign_rows_stratified(indices, working["_split_metric_for_strata"], split, rng)
        working.loc[indices, "split"] = labels
        for split_name in SPLIT_NAMES:
            current_counts[split_name] += int((labels == split_name).sum())

    small_stats = stats[~stats["cluster_id"].astype(int).isin(oversized_cluster_ids)].copy()
    split_map = _assign_small_cluster_splits(
        small_stats,
        split,
        rng,
        current_counts=current_counts,
        target_counts=target_counts,
    )
    unset_mask = working["split"].isna()
    working.loc[unset_mask, "split"] = working.loc[unset_mask, "cluster_id"].map(split_map)
    if working["split"].isna().any():
        raise ValueError(f"Rows without split assignment in dataset {spec.key}")
    return working[[spec.sequence_col, "split", "cluster_id"]].rename(columns={spec.sequence_col: "split_key"})


def ensure_dataset_splits(
    dataset_key: str,
    config_path: str | Path | None = None,
    *,
    force: bool = False,
) -> Dict[str, Path]:
    config = load_dms_config(config_path)
    if dataset_key not in config.datasets:
        raise KeyError(f"Unknown DMS dataset key {dataset_key!r}. Available: {sorted(config.datasets)}")
    spec = config.datasets[dataset_key]
    source_key = spec.split_source or dataset_key
    if source_key not in config.datasets:
        raise KeyError(f"Dataset {dataset_key!r} references missing split_source {source_key!r}")
    source_spec = config.datasets[source_key]
    expected = _expected_meta(config, dataset_key, spec, source_key)
    paths = _split_paths(config, dataset_key)
    meta_path = _meta_path(config, dataset_key)
    if (
        not force
        and all(path.exists() for path in paths.values())
        and _metadata_matches(meta_path, expected)
    ):
        return paths

    df = _read_validated(spec)
    if source_key == dataset_key:
        membership = _build_source_membership(df, spec, config.split)
        split_values = membership["split"].astype(str).reset_index(drop=True)
        if len(split_values) != len(df):
            raise ValueError(
                f"Split membership length mismatch for dataset {dataset_key}: "
                f"{len(split_values)} memberships for {len(df)} rows."
            )
        df["_split"] = split_values
        key_col = spec.sequence_col
    else:
        source_paths = ensure_dataset_splits(source_key, config.path, force=force)
        membership_parts = []
        for split_name, path in source_paths.items():
            split_df = pd.read_csv(path, usecols=[source_spec.sequence_col])
            split_df = split_df.rename(columns={source_spec.sequence_col: "split_key"})
            split_df["split"] = split_name
            membership_parts.append(split_df)
        membership = pd.concat(membership_parts, ignore_index=True)
        key_col = spec.sequence_col
        split_lookup = dict(zip(membership["split_key"].astype(str), membership["split"].astype(str)))
        df["_split"] = df[key_col].astype(str).map(split_lookup)

    out_dir = config.split.output_dir / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, path in paths.items():
        split_df = df[df["_split"] == split_name].drop(columns=["_split"]).reset_index(drop=True)
        split_df.to_csv(path, index=False)

    counts = {split_name: int(pd.read_csv(path, usecols=[key_col]).shape[0]) for split_name, path in paths.items()}
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump({**expected, "counts": counts}, fh, indent=2, sort_keys=True)
    return paths


def resolve_dataset_split(
    dataset_key: str,
    split_name: str,
    config_path: str | Path | None = None,
    *,
    force: bool = False,
) -> Path:
    if split_name not in SPLIT_NAMES:
        raise ValueError(f"split_name must be one of {SPLIT_NAMES}, got {split_name!r}")
    return ensure_dataset_splits(dataset_key, config_path, force=force)[split_name]


def dataset_spec(dataset_key: str, config_path: str | Path | None = None) -> DatasetSpec:
    config = load_dms_config(config_path)
    return config.datasets[dataset_key]
