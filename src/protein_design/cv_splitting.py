"""K-fold cross-validation split materialization for the low-data DPO pipeline.

This is the fold-aware analogue of the ``<base_key>_splitseed<N>`` mechanism in
``protein_design.dms_splitting``. Where an external split seed re-draws ONE
train/val/test partition, k-fold CV partitions the *whole* small dataset into
``k`` folds so that every row is held out for test exactly once across the
``k`` runs. Pooling the held-out (prediction, ground-truth) pairs from all
folds and computing a SINGLE Spearman over the full dataset is the headline CV
metric -- much lower test-set variance than the single ~40-50 row held-out
split the small-dataset low-data curves used before (see
``report/meetings/14-07-26.ipynb``, "ED1 is tiny" caveat).

Dataset-key format (self-describing, mirrors ``_splitseed<N>``)::

    <base_key>_cv<K>s<SEED>_f<I>

    K    = n_folds
    SEED = fold_seed (deterministic fold assignment)
    I    = 0-based held-out fold index (0 <= I < K)

e.g. ``ed1_m22_cv5s0_f2`` is fold 2 of a 5-fold, seed-0 partition of
``ed1_m22``. The key is recorded verbatim in each run's ``resolved_config.yaml``
(``data.test.dataset_key``), so a run is fully self-documenting and the notebook
scanner can group folds by ``(base, K, SEED)`` and pool them.

Correctness guarantees (asserted in ``tests/test_cv_splitting.py``):

- **Deterministic, reproducible fold assignment** -- ``assign_folds`` depends
  only on ``(metric values, K, SEED, stratify_bins)`` and is independent of the
  fold index, so all ``K`` fold keys of one partition see the SAME assignment.
- **Each row is held out exactly once** -- the ``K`` test folds are a disjoint
  cover of the full (finite-metric) dataset.
- **No held-out leakage** -- the held-out fold is removed from the pool BEFORE
  the pool is split into train/val, so neither training nor checkpoint
  selection (which uses val) can see a test-fold row.

The CV knobs live in the ``cv:`` block of the DMS config YAML
(``conf/data/dms/cv.yaml``): ``val_frac`` (fraction of the non-test pool held
out for validation) and ``output_dir`` (isolated so canonical splits are never
touched). ``n_folds`` / ``fold_seed`` are carried by the key itself.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from .dms_splitting import (
    SPLIT_NAMES,
    DatasetSpec,
    SplitConfig,
    _assign_rows_stratified,
    _expand_path,
    _metadata_matches,
    _metric_strata,
    _read_validated,
    load_dms_config,
    project_root,
)

CV_FOLD_KEY_RE = re.compile(r"^(?P<base>.+)_cv(?P<k>\d+)s(?P<seed>\d+)_f(?P<i>\d+)$")

DEFAULT_CV_VAL_FRAC = 0.15
DEFAULT_CV_OUTPUT_DIR = "/cluster/project/infk/krause/${USER}/protein-design/data/dms_splits_cv"


@dataclass(frozen=True)
class CVFoldKey:
    """Parsed components of a ``<base>_cv<K>s<SEED>_f<I>`` dataset key."""

    full_key: str
    base_key: str
    n_folds: int
    fold_seed: int
    fold_index: int


@dataclass(frozen=True)
class CVConfig:
    val_frac: float
    output_dir: Path


def parse_cv_fold_key(dataset_key: str) -> Optional[CVFoldKey]:
    """Parse a CV fold key, or return ``None`` if it is not one.

    Validates ``0 <= fold_index < n_folds`` and ``n_folds >= 2`` so a malformed
    key fails loudly here rather than silently producing a degenerate split.
    """
    match = CV_FOLD_KEY_RE.match(str(dataset_key))
    if not match:
        return None
    n_folds = int(match.group("k"))
    fold_seed = int(match.group("seed"))
    fold_index = int(match.group("i"))
    if n_folds < 2:
        raise ValueError(f"CV fold key {dataset_key!r}: n_folds must be >= 2, got {n_folds}.")
    if not (0 <= fold_index < n_folds):
        raise ValueError(
            f"CV fold key {dataset_key!r}: fold index {fold_index} out of range for n_folds={n_folds}."
        )
    return CVFoldKey(
        full_key=str(dataset_key),
        base_key=match.group("base"),
        n_folds=n_folds,
        fold_seed=fold_seed,
        fold_index=fold_index,
    )


def is_cv_fold_key(dataset_key: str) -> bool:
    return parse_cv_fold_key(dataset_key) is not None


def cv_fold_key(base_key: str, n_folds: int, fold_seed: int, fold_index: int) -> str:
    """Construct a CV fold key (inverse of :func:`parse_cv_fold_key`)."""
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}.")
    if not (0 <= fold_index < n_folds):
        raise ValueError(f"fold index {fold_index} out of range for n_folds={n_folds}.")
    return f"{base_key}_cv{n_folds}s{fold_seed}_f{fold_index}"


def load_cv_config(config_path: str | Path | None = None) -> CVConfig:
    """Read the ``cv:`` block from a DMS config YAML (with safe defaults)."""
    root = project_root()
    path = _expand_path(config_path or "conf/data/dms/cv.yaml", root)
    raw: Mapping[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    cv_raw = (raw.get("cv") or {}) if isinstance(raw, Mapping) else {}
    val_frac = float(cv_raw.get("val_frac", DEFAULT_CV_VAL_FRAC))
    if not (0.0 < val_frac < 1.0):
        raise ValueError(f"cv.val_frac must be in (0, 1); got {val_frac}.")
    output_dir = _expand_path(cv_raw.get("output_dir", DEFAULT_CV_OUTPUT_DIR), root)
    return CVConfig(val_frac=val_frac, output_dir=output_dir)


def assign_folds(
    metric_values: pd.Series,
    n_folds: int,
    fold_seed: int,
    stratify_bins: int,
) -> np.ndarray:
    """Deterministic, metric-stratified fold assignment.

    Returns an int array of length ``len(metric_values)`` with values in
    ``[0, n_folds)``, positionally aligned to ``metric_values``. Within each
    metric stratum the rows are shuffled (seeded by ``fold_seed``) and dealt
    round-robin to the folds, so fold sizes stay balanced and each stratum is
    represented in every fold. Crucially it does NOT depend on any fold index,
    so all folds of one partition share the same assignment -> the test folds
    tile the dataset exactly once.
    """
    values = pd.Series(np.asarray(metric_values, dtype=float)).reset_index(drop=True)
    n = len(values)
    folds = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return folds
    strata = _metric_strata(values, stratify_bins)
    rng = np.random.default_rng(int(fold_seed))
    # Deal round-robin, but carry the offset across strata (continuous dealing)
    # so the per-stratum rounding remainders don't all pile onto folds 0,1,...
    # -> global fold sizes stay balanced within +/-1 while each stratum is still
    # spread across every fold.
    offset = 0
    for stratum_id in sorted(strata.unique().tolist()):
        positions = np.where(strata.values == stratum_id)[0]
        positions = rng.permutation(positions)
        folds[positions] = (np.arange(len(positions)) + offset) % int(n_folds)
        offset = (offset + len(positions)) % int(n_folds)
    if (folds < 0).any():
        raise RuntimeError("Internal error: some rows were left without a fold assignment.")
    return folds


def _expected_cv_meta(
    fold: CVFoldKey,
    spec: DatasetSpec,
    cv_cfg: CVConfig,
    split_cfg: SplitConfig,
) -> dict[str, Any]:
    stat = spec.path.stat()
    return {
        "version": 1,
        "method": "stratified_round_robin_kfold_with_stratified_pool_train_val",
        "dataset_key": fold.full_key,
        "base_key": fold.base_key,
        "n_folds": fold.n_folds,
        "fold_seed": fold.fold_seed,
        "fold_index": fold.fold_index,
        "path": str(spec.path),
        "path_mtime": stat.st_mtime,
        "sequence_col": spec.sequence_col,
        "key_metric_col": spec.key_metric_col,
        "val_frac": cv_cfg.val_frac,
        "stratify_bins": split_cfg.stratify_bins,
    }


def ensure_cv_fold_splits(
    dataset_key: str,
    config_path: str | Path | None = None,
    *,
    force: bool = False,
) -> Dict[str, Path]:
    """Materialize (and cache) train/val/test CSVs for one CV fold.

    ``test.csv`` is exactly the held-out fold; ``train.csv`` / ``val.csv`` are a
    stratified split of the remaining folds (the non-test pool). The layout,
    columns and ``split.meta.json`` marker match ``ensure_dataset_splits`` so
    the downstream DPO/LoRA-DPO training code consumes a fold key with no
    special-casing.
    """
    fold = parse_cv_fold_key(dataset_key)
    if fold is None:
        raise ValueError(f"{dataset_key!r} is not a CV fold key (expected <base>_cv<K>s<SEED>_f<I>).")

    config = load_dms_config(config_path)
    if fold.base_key not in config.datasets:
        raise KeyError(
            f"CV fold key {dataset_key!r}: base dataset {fold.base_key!r} not found in the DMS config. "
            f"Available: {sorted(config.datasets)}"
        )
    spec = config.datasets[fold.base_key]
    if spec.split_source is not None:
        raise ValueError(
            f"K-fold CV is only supported for datasets that are their own split source; "
            f"{fold.base_key!r} has split_source={spec.split_source!r}."
        )
    cv_cfg = load_cv_config(config_path)
    split_cfg = config.split

    out_dir = cv_cfg.output_dir / fold.full_key
    paths = {name: out_dir / f"{name}.csv" for name in SPLIT_NAMES}
    meta_path = out_dir / "split.meta.json"
    expected = _expected_cv_meta(fold, spec, cv_cfg, split_cfg)
    if (
        not force
        and all(path.exists() for path in paths.values())
        and _metadata_matches(meta_path, expected)
    ):
        return paths

    df = _read_validated(spec)
    metric = pd.to_numeric(df[spec.key_metric_col], errors="coerce")
    df = df.loc[metric.notna()].reset_index(drop=True)
    if len(df) < fold.n_folds:
        raise ValueError(
            f"Dataset {fold.base_key!r} has only {len(df)} usable rows -- fewer than n_folds={fold.n_folds}."
        )

    folds = assign_folds(df[spec.key_metric_col], fold.n_folds, fold.fold_seed, split_cfg.stratify_bins)
    test_mask = folds == fold.fold_index
    test_df = df.loc[test_mask].reset_index(drop=True)
    pool_df = df.loc[~test_mask].reset_index(drop=True)

    # Split the non-test pool into train/val ONLY (test_frac=0). Deterministic
    # seed derived from (fold_seed, fold_index) so each fold's val draw is
    # distinct yet reproducible. Reuses dms_splitting's stratified assigner;
    # any rows the rounding leaves labelled "test" here belong to the pool
    # (never the held-out fold) so we fold them back into train -- no leakage.
    pool_split = SplitConfig(
        enabled=True,
        train_frac=1.0 - cv_cfg.val_frac,
        val_frac=cv_cfg.val_frac,
        test_frac=0.0,
        seed=int(fold.fold_seed) * 1000 + int(fold.fold_index) + 1,
        output_dir=out_dir,
        hamming_distance=0,
        stratify_bins=split_cfg.stratify_bins,
    )
    pool_rng = np.random.default_rng(pool_split.seed)
    labels = _assign_rows_stratified(pool_df.index, pool_df[spec.key_metric_col], pool_split, pool_rng)
    val_df = pool_df.loc[labels == "val"].reset_index(drop=True)
    train_df = pool_df.loc[labels != "val"].reset_index(drop=True)

    # Belt-and-braces no-leakage assertion: the held-out test sequences must not
    # appear in train or val.
    test_seqs = set(test_df[spec.sequence_col].astype(str))
    for name, part in (("train", train_df), ("val", val_df)):
        overlap = test_seqs.intersection(part[spec.sequence_col].astype(str))
        if overlap:
            raise RuntimeError(
                f"CV leakage in {fold.full_key}: {len(overlap)} held-out sequence(s) also in {name}."
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
        part.to_csv(paths[name], index=False)
        counts[name] = int(len(part))
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump({**expected, "counts": counts, "n_total": int(len(df))}, fh, indent=2, sort_keys=True)
    return paths


def cv_fold_dataset_spec(dataset_key: str, config_path: str | Path | None = None) -> DatasetSpec:
    """Return the base dataset's :class:`DatasetSpec` re-keyed to the fold key.

    Lets the training code call ``dataset_spec(<fold_key>)`` transparently and
    recover the metric column / WT sequence of the underlying base dataset.
    """
    from dataclasses import replace

    fold = parse_cv_fold_key(dataset_key)
    if fold is None:
        raise ValueError(f"{dataset_key!r} is not a CV fold key.")
    config = load_dms_config(config_path)
    if fold.base_key not in config.datasets:
        raise KeyError(f"CV fold key {dataset_key!r}: base dataset {fold.base_key!r} not found.")
    return replace(config.datasets[fold.base_key], key=fold.full_key)


# --------------------------------------------------------------------------
# Pooled out-of-fold aggregation (the headline CV metric).
# --------------------------------------------------------------------------
def pooled_oof_spearman(
    fold_predictions: Sequence[pd.DataFrame],
    *,
    pred_col: str = "score",
    truth_col: str = "M22_binding_enrichment_adj",
) -> dict:
    """Pool held-out predictions across folds and compute ONE Spearman.

    ``fold_predictions`` is the list of per-fold ``test_predictions.csv`` frames
    (one per held-out fold). This concatenates every out-of-fold
    ``(prediction, ground_truth)`` pair and computes a single Spearman over the
    full pooled set -- explicitly NOT the mean of per-fold Spearmans. Returns a
    dict with the pooled rho/pval, the pooled example count, and the list of
    per-fold Spearmans (diagnostics only).
    """
    from scipy.stats import spearmanr

    frames = [f for f in fold_predictions if f is not None and len(f)]
    if not frames:
        return {"pooled_spearman": float("nan"), "pooled_pval": float("nan"),
                "n_pooled": 0, "n_folds": 0, "foldwise_spearman": []}

    foldwise = []
    for f in frames:
        if len(f) >= 3:
            r = spearmanr(f[pred_col].values, f[truth_col].values)
            foldwise.append(float(r.statistic))
        else:
            foldwise.append(float("nan"))

    pooled = pd.concat([f[[pred_col, truth_col]] for f in frames], ignore_index=True)
    if len(pooled) < 3:
        rho, pval = float("nan"), float("nan")
    else:
        res = spearmanr(pooled[pred_col].values, pooled[truth_col].values)
        rho, pval = float(res.statistic), float(res.pvalue)
    return {
        "pooled_spearman": rho,
        "pooled_pval": pval,
        "n_pooled": int(len(pooled)),
        "n_folds": len(frames),
        "foldwise_spearman": foldwise,
    }
