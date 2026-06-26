from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from protein_design.analysis.registry import load_datasets_cfg


@dataclass(frozen=True)
class ReferenceSource:
    label: str
    path: Path
    seq_col: str
    split: str


def iter_reference_sources(repo_root: Path) -> list[ReferenceSource]:
    """Return the sequence sources used for novelty checks.

    The configured DMS datasets are the primary source of truth. The config
    points each dataset at a single split (the test split, used for scoring),
    but for novelty we want the *whole* dataset, so we expand every configured
    dataset to all the split CSVs that live alongside it (train/val/test). Each
    source records its ``split`` (the file stem) so callers can restrict the
    index to e.g. train-only when measuring training-set memorization.

    We also scan the legacy local raw/split directories to preserve the broader
    behavior of the existing novelty plot.
    """
    sources: list[ReferenceSource] = []
    seen: set[tuple[str, str]] = set()

    def _add(label: str, path: Path, seq_col: str, split: str) -> None:
        key = (str(path), seq_col)
        if key in seen:
            return
        seen.add(key)
        sources.append(ReferenceSource(label=label, path=path, seq_col=seq_col, split=split))

    cfg = load_datasets_cfg()
    for dataset_key, ds in (cfg.get("datasets", {}) or {}).items():
        configured = Path(str(ds["path"]))
        seq_col = str(ds.get("seq_col", "aa"))
        # Expand to every split CSV sitting next to the configured file so the
        # reference reflects the whole dataset, not just the configured split.
        siblings = sorted(configured.parent.glob("*.csv")) if configured.parent.exists() else []
        if not siblings:
            siblings = [configured]
        for csv_path in siblings:
            split = csv_path.stem  # 'train' / 'val' / 'test' / 'exp_enrichment' / ...
            _add(f"{dataset_key}:{split}", csv_path, seq_col, split)

    raw_dir = repo_root / "data" / "raw"
    for csv_path in sorted(raw_dir.glob("*.csv")):
        _add(f"raw:{csv_path.stem}", csv_path, "aa", csv_path.stem)

    splits_dir = repo_root / "data" / "dms_splits"
    for csv_path in sorted(splits_dir.rglob("*.csv")):
        rel = csv_path.relative_to(splits_dir).with_suffix("")
        _add(f"split:{rel.as_posix()}", csv_path, "aa", rel.name)

    return sources


def _read_unique_sequences(source: ReferenceSource, logger: logging.Logger | None = None) -> set[str]:
    logger = logger or logging.getLogger(__name__)
    if not source.path.exists():
        logger.warning("Skipping missing reference dataset %s (%s)", source.label, source.path)
        return set()
    try:
        series = pd.read_csv(source.path, usecols=[source.seq_col])[source.seq_col]
    except Exception as exc:
        logger.warning("Skipping %s (%s): %s", source.label, source.path, exc)
        return set()
    return {
        str(seq).strip()
        for seq in series.dropna().astype(str).tolist()
        if str(seq).strip()
    }


def build_reference_index(
    repo_root: Path,
    *,
    splits: set[str] | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, frozenset[str]]:
    """Map sequence -> dataset labels where it appears.

    ``splits`` optionally restricts the reference to sources whose split name is
    in the given set (e.g. ``{"train"}`` to measure training-set memorization).
    ``None`` (the default) includes every split — i.e. the whole dataset.
    """
    logger = logger or logging.getLogger(__name__)
    seq_to_labels: dict[str, set[str]] = defaultdict(set)

    for source in iter_reference_sources(repo_root):
        if splits is not None and source.split not in splits:
            continue
        seqs = _read_unique_sequences(source, logger=logger)
        for seq in seqs:
            seq_to_labels[seq].add(source.label)
        logger.info(
            "reference %-30s +%7d unique sequences",
            source.label,
            len(seqs),
        )

    return {seq: frozenset(sorted(labels)) for seq, labels in seq_to_labels.items()}


def annotate_sequence_membership(
    df: pd.DataFrame,
    *,
    seq_col: str,
    reference_index: dict[str, frozenset[str]],
) -> pd.DataFrame:
    """Add dataset-membership columns for the given sequence column."""
    annotated = df.copy()

    def _labels(seq: object) -> tuple[bool, int, str]:
        if pd.isna(seq):
            return False, 0, ""
        matches = reference_index.get(str(seq).strip(), frozenset())
        return bool(matches), len(matches), ";".join(matches)

    membership = annotated[seq_col].map(_labels)
    annotated["present_in_existing_dataset"] = membership.map(lambda item: item[0])
    annotated["n_matching_datasets"] = membership.map(lambda item: item[1])
    annotated["matching_datasets"] = membership.map(lambda item: item[2])
    return annotated


def annotate_generated_csv_in_place(
    csv_path: Path,
    *,
    seq_col: str = "cdrh3",
    reference_index: dict[str, frozenset[str]],
) -> None:
    df = pd.read_csv(csv_path)
    if seq_col not in df.columns:
        raise ValueError(f"{csv_path} is missing required column {seq_col!r}")
    annotated = annotate_sequence_membership(
        df,
        seq_col=seq_col,
        reference_index=reference_index,
    )
    annotated.to_csv(csv_path, index=False)
