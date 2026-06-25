"""Low-data subsampling of the DPO *sequence* train split.

The DPO learning curve (the analogue of the embedding-probe learning curve) asks:
given only N measured variants, does an evotuned base produce a better DPO model
than vanilla ESM2? To answer it fairly we subsample the **train sequence split**
to a small size *before* preference pairs are derived, so the pairs reflect only
what the small measured set can support. Val/test pairs are left untouched, so the
held-out evaluation is constant across N.

Two schemes (decision: sequences-level, stratified with a random sanity band):

- ``stratified``  (default): bin the sequences by enrichment (qcut) and sample
  proportionally from each bin, so a small N still contains positives — otherwise
  some draws have no ``within_pos`` / ``cross`` pairs to build at all.
- ``random``: uniform sample without replacement (higher variance, the honest
  "whatever labels you happened to get" baseline).

Determinism is the key property: the same ``(n, scheme, seed)`` over the same
train split yields the same sequences regardless of the base model, so the only
thing that varies across the evo-vs-vanilla comparison is the model itself.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

LowDataScheme = Literal["stratified", "random"]

DEFAULT_METRIC_COL = "M22_binding_enrichment_adj"
DEFAULT_STRATIFY_BINS = 10


def _enrichment_strata(values: pd.Series, num_bins: int) -> pd.Series:
    """Quantile-bin the metric for stratification (mirrors dms_splitting)."""
    num_bins = max(1, min(int(num_bins), len(values)))
    if num_bins <= 1:
        return pd.Series(np.zeros(len(values), dtype=np.int64), index=values.index)
    try:
        return pd.qcut(values, q=num_bins, labels=False, duplicates="drop").fillna(0).astype(int)
    except ValueError:
        return pd.Series(np.zeros(len(values), dtype=np.int64), index=values.index)


def _stratified_indices(
    strata: pd.Series,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pick n positions, allocated across strata in proportion to their size.

    Largest-remainder allocation keeps the per-stratum counts summing to n; within
    each stratum the rows are drawn uniformly at random without replacement.
    """
    total = len(strata)
    stratum_ids = sorted(strata.unique().tolist())
    sizes = {sid: int((strata == sid).sum()) for sid in stratum_ids}

    raw = {sid: sizes[sid] * n / total for sid in stratum_ids}
    alloc = {sid: int(np.floor(raw[sid])) for sid in stratum_ids}
    remaining = n - sum(alloc.values())
    # Distribute the leftover by largest fractional remainder, capped by capacity.
    for sid in sorted(stratum_ids, key=lambda s: raw[s] - np.floor(raw[s]), reverse=True):
        if remaining <= 0:
            break
        if alloc[sid] < sizes[sid]:
            alloc[sid] += 1
            remaining -= 1

    picked: list[np.ndarray] = []
    for sid in stratum_ids:
        take = min(alloc[sid], sizes[sid])
        if take <= 0:
            continue
        pool = strata.index[strata == sid].to_numpy()
        chosen = rng.choice(pool, size=take, replace=False)
        picked.append(chosen)
    if not picked:
        return np.empty(0, dtype=strata.index.dtype)
    return np.concatenate(picked)


def subsample_train_sequences(
    train_df: pd.DataFrame,
    n: int,
    *,
    scheme: LowDataScheme = "stratified",
    metric_col: str = DEFAULT_METRIC_COL,
    stratify_bins: int = DEFAULT_STRATIFY_BINS,
    seed: int = 0,
) -> pd.DataFrame:
    """Return a low-data subset of ``n`` train sequences (deterministic).

    Rows with a non-finite metric are dropped first (they cannot be stratified and
    carry no usable label). If ``n`` is >= the number of usable rows the full
    (cleaned) frame is returned. Output preserves the input columns/schema so the
    delta-based pair builder can consume it unchanged.
    """
    if n is None or int(n) <= 0:
        raise ValueError(f"low-data n must be a positive integer, got {n!r}.")
    if metric_col not in train_df.columns:
        raise ValueError(f"train_df is missing the metric column {metric_col!r}.")

    metric = pd.to_numeric(train_df[metric_col], errors="coerce")
    usable = train_df.loc[metric.notna()].copy()
    usable[metric_col] = metric.loc[metric.notna()].astype(float)
    usable = usable.reset_index(drop=True)

    target = int(n)
    if target >= len(usable):
        return usable

    rng = np.random.default_rng(int(seed))
    if scheme == "random":
        idx = rng.choice(len(usable), size=target, replace=False)
    elif scheme == "stratified":
        strata = _enrichment_strata(usable[metric_col], stratify_bins)
        idx = _stratified_indices(strata, target, rng)
    else:
        raise ValueError(f"Unknown low-data scheme {scheme!r}. Expected 'stratified' or 'random'.")

    return usable.iloc[np.sort(np.asarray(idx))].reset_index(drop=True)
