"""Cluster-based split utilities for DPO data preparation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

SPLIT_MEMBERSHIP_FILENAME = "ED2_cluster_split_membership.csv"
SPLIT_MEMBERSHIP_META_FILENAME = "ED2_cluster_split_membership.meta.json"

SPLIT_MEMBERSHIP_REQUIRED_COLUMNS = {
    "split_key",
    "split",
    "cluster_id",
    "cluster_size",
    "cluster_positive_fraction",
}


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
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


def split_membership_keys(df: pd.DataFrame) -> pd.Series:
    """Return stable row keys used to map split memberships across views."""
    if "Unnamed: 0" in df.columns:
        return df["Unnamed: 0"].astype(str)
    if {"aa", "mut"}.issubset(df.columns):
        return df["aa"].astype(str) + "||" + df["mut"].astype(str)
    if "aa" in df.columns:
        return df["aa"].astype(str)
    raise ValueError(
        "Cannot infer split membership keys. Expected one of: 'Unnamed: 0', ('aa' and 'mut'), or 'aa'."
    )


def _compute_cluster_ids_hamming_lte_one(sequences: Iterable[str]) -> np.ndarray:
    """Cluster sequences with connected components over edges where Hamming distance <= 1."""
    seq_list = [str(seq) for seq in sequences]
    n = len(seq_list)
    if n == 0:
        return np.array([], dtype=np.int64)

    uf = _UnionFind(n)
    by_length: Dict[int, list[int]] = {}
    for idx, seq in enumerate(seq_list):
        by_length.setdefault(len(seq), []).append(idx)

    for indices in by_length.values():
        if not indices:
            continue
        sig_to_first: Dict[str, int] = {}
        for idx in indices:
            seq = seq_list[idx]
            seq_len = len(seq)
            if seq_len == 0:
                sig = "__empty__"
                prev = sig_to_first.get(sig)
                if prev is None:
                    sig_to_first[sig] = idx
                else:
                    uf.union(idx, prev)
                continue
            for pos in range(seq_len):
                signature = seq[:pos] + "*" + seq[pos + 1 :]
                prev = sig_to_first.get(signature)
                if prev is None:
                    sig_to_first[signature] = idx
                else:
                    uf.union(idx, prev)

    root_to_cluster_id: Dict[int, int] = {}
    cluster_ids = np.empty(n, dtype=np.int64)
    next_id = 0
    for idx in range(n):
        root = uf.find(idx)
        if root not in root_to_cluster_id:
            root_to_cluster_id[root] = next_id
            next_id += 1
        cluster_ids[idx] = root_to_cluster_id[root]
    return cluster_ids


def _split_counts(total: int, train_frac: float, val_frac: float, test_frac: float) -> Tuple[int, int, int]:
    if abs(float(train_frac) + float(val_frac) + float(test_frac) - 1.0) >= 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")
    n_train = int(total * float(train_frac))
    n_val = int(total * float(val_frac))
    n_test = total - n_train - n_val
    return n_train, n_val, n_test


def _assign_cluster_splits_stratified(
    clusters_df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
    stratify_bins: int,
) -> Dict[int, str]:
    if clusters_df.empty:
        return {}

    num_bins = max(1, min(int(stratify_bins), len(clusters_df)))
    if num_bins == 1:
        strata = pd.Series(np.zeros(len(clusters_df), dtype=np.int64), index=clusters_df.index)
    else:
        try:
            strata = pd.qcut(
                clusters_df["cluster_positive_fraction"],
                q=num_bins,
                labels=False,
                duplicates="drop",
            )
            strata = strata.fillna(0).astype(int)
        except ValueError:
            strata = pd.Series(np.zeros(len(clusters_df), dtype=np.int64), index=clusters_df.index)

    rng = np.random.default_rng(int(seed))
    split_map: Dict[int, str] = {}
    for stratum_id in sorted(strata.unique().tolist()):
        mask = strata == stratum_id
        cluster_ids = clusters_df.loc[mask, "cluster_id"].astype(int).to_numpy()
        if len(cluster_ids) == 0:
            continue
        permuted = rng.permutation(cluster_ids)
        n_train, n_val, _ = _split_counts(len(permuted), train_frac, val_frac, test_frac)
        train_ids = permuted[:n_train]
        val_ids = permuted[n_train : n_train + n_val]
        test_ids = permuted[n_train + n_val :]
        for cid in train_ids:
            split_map[int(cid)] = "train"
        for cid in val_ids:
            split_map[int(cid)] = "val"
        for cid in test_ids:
            split_map[int(cid)] = "test"
    return split_map


def _metadata_matches(meta: Dict[str, object], expected: Dict[str, object]) -> bool:
    for key, value in expected.items():
        if meta.get(key) != value:
            return False
    return True


def build_or_load_cluster_split_membership(
    *,
    base_df: pd.DataFrame,
    base_csv_path: Path,
    processed_dir: Path,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
    force_rebuild: bool = False,
    positive_threshold: float = 0.0,
    stratify_bins: int = 10,
    hamming_distance: int = 1,
) -> pd.DataFrame:
    """Build or load cached cluster-based split memberships for the base ED2 dataset."""
    if int(hamming_distance) != 1:
        raise ValueError("Only hamming_distance=1 is currently supported.")

    processed_dir = Path(processed_dir)
    membership_path = processed_dir / SPLIT_MEMBERSHIP_FILENAME
    meta_path = processed_dir / SPLIT_MEMBERSHIP_META_FILENAME

    expected_meta = {
        "method": "hamming_connected_components_lte_1",
        "hamming_distance": int(hamming_distance),
        "train_frac": float(train_frac),
        "val_frac": float(val_frac),
        "test_frac": float(test_frac),
        "seed": int(seed),
        "positive_threshold": float(positive_threshold),
        "stratify_bins": int(stratify_bins),
    }

    should_rebuild = bool(force_rebuild)
    if not should_rebuild:
        if (not membership_path.exists()) or (not meta_path.exists()):
            should_rebuild = True
        elif membership_path.stat().st_mtime < base_csv_path.stat().st_mtime:
            should_rebuild = True
        else:
            try:
                with meta_path.open("r", encoding="utf-8") as fh:
                    existing_meta = json.load(fh)
                if not _metadata_matches(existing_meta, expected_meta):
                    should_rebuild = True
            except Exception:
                should_rebuild = True

    if not should_rebuild:
        cached = pd.read_csv(membership_path)
        missing = SPLIT_MEMBERSHIP_REQUIRED_COLUMNS.difference(cached.columns)
        if not missing:
            return cached
        should_rebuild = True

    if "aa" not in base_df.columns:
        raise ValueError("Base dataframe must contain column 'aa'.")
    if "M22_binding_enrichment_adj" not in base_df.columns:
        raise ValueError("Base dataframe must contain column 'M22_binding_enrichment_adj'.")

    clean_df = base_df.copy().reset_index(drop=True)
    clean_df["aa"] = clean_df["aa"].astype(str).str.strip()
    clean_df = clean_df[clean_df["aa"] != ""].reset_index(drop=True)
    if clean_df.empty:
        raise ValueError("Base dataframe has no valid sequences after cleaning.")

    enrichment = pd.to_numeric(clean_df["M22_binding_enrichment_adj"], errors="coerce")
    clean_df["M22_binding_enrichment_adj"] = enrichment.astype(float)
    clean_df["is_positive"] = (enrichment > float(positive_threshold)).fillna(False).astype(int)
    clean_df["split_key"] = split_membership_keys(clean_df).astype(str)

    cluster_ids = _compute_cluster_ids_hamming_lte_one(clean_df["aa"].tolist())
    clean_df["cluster_id"] = cluster_ids

    cluster_stats = (
        clean_df.groupby("cluster_id", sort=True)["is_positive"]
        .agg(cluster_size="size", cluster_positive_count="sum")
        .reset_index()
    )
    cluster_stats["cluster_positive_fraction"] = (
        cluster_stats["cluster_positive_count"] / cluster_stats["cluster_size"]
    )

    split_map = _assign_cluster_splits_stratified(
        clusters_df=cluster_stats,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=int(seed),
        stratify_bins=int(stratify_bins),
    )
    clean_df["split"] = clean_df["cluster_id"].map(split_map)
    if clean_df["split"].isna().any():
        raise ValueError("Found base rows without split assignment after cluster split.")

    membership_df = clean_df.merge(
        cluster_stats[["cluster_id", "cluster_size", "cluster_positive_fraction"]],
        on="cluster_id",
        how="left",
    )
    membership_df = membership_df[
        [
            "split_key",
            "aa",
            "mut",
            "num_mut",
            "M22_binding_enrichment_adj",
            "delta_M22_binding_enrichment_adj",
            "is_positive",
            "cluster_id",
            "cluster_size",
            "cluster_positive_fraction",
            "split",
        ]
    ].reset_index(drop=True)

    membership_df.to_csv(membership_path, index=False)
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(expected_meta, fh, sort_keys=True, indent=2)

    return membership_df


def summarize_split_membership(membership_df: pd.DataFrame) -> Dict[str, float]:
    """Return split and cluster summary statistics for logging."""
    if membership_df.empty:
        return {
            "num_clusters": 0.0,
            "cluster_size_min": 0.0,
            "cluster_size_median": 0.0,
            "cluster_size_max": 0.0,
            "num_sequences_train": 0.0,
            "num_sequences_val": 0.0,
            "num_sequences_test": 0.0,
            "num_positives_train": 0.0,
            "num_positives_val": 0.0,
            "num_positives_test": 0.0,
        }

    clusters = membership_df[["cluster_id", "cluster_size"]].drop_duplicates()
    out: Dict[str, float] = {
        "num_clusters": float(clusters["cluster_id"].nunique()),
        "cluster_size_min": float(clusters["cluster_size"].min()),
        "cluster_size_median": float(clusters["cluster_size"].median()),
        "cluster_size_max": float(clusters["cluster_size"].max()),
    }
    for split in ("train", "val", "test"):
        split_df = membership_df[membership_df["split"] == split]
        out[f"num_sequences_{split}"] = float(len(split_df))
        out[f"num_positives_{split}"] = float(split_df["is_positive"].sum())
    return out
