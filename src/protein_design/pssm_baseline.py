"""Helpers for the CDR-H3 PSSM baseline sampler."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from protein_design.constants import C05_CDRH3, add_context
from protein_design.dms_splitting import dataset_spec, project_root, resolve_dataset_split

STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(STANDARD_AAS)}
CDRH3_LEN = len(C05_CDRH3)


def expand_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root() / path
    return path


def resolve_train_split_with_fallback(
    dataset_key: str,
    dms_config_path: str | Path,
    local_splits_dir: str | Path,
) -> tuple[Path, str]:
    """Resolve the train split path, falling back to local cached splits."""
    local_path = expand_repo_path(local_splits_dir) / dataset_key / "train.csv"
    try:
        resolved = resolve_dataset_split(dataset_key, "train", dms_config_path)
        if resolved.exists():
            return resolved, "configured"
    except FileNotFoundError:
        pass
    except OSError:
        pass
    if local_path.exists():
        return local_path, "local_fallback"
    raise FileNotFoundError(
        "Could not resolve the DMS train split from the configured location and no local fallback "
        f"was found at {local_path}"
    )


def load_train_dataframe(
    dataset_key: str,
    dms_config_path: str | Path,
    local_splits_dir: str | Path,
    enrichment_threshold: float | None = None,
) -> tuple[pd.DataFrame, str, str, Path, str]:
    spec = dataset_spec(dataset_key, dms_config_path)
    split_path, source = resolve_train_split_with_fallback(dataset_key, dms_config_path, local_splits_dir)
    df = pd.read_csv(split_path)
    missing = {spec.sequence_col, spec.key_metric_col}.difference(df.columns)
    if missing:
        raise ValueError(f"{split_path} missing required columns: {sorted(missing)}")
    df = df.copy()
    df[spec.sequence_col] = df[spec.sequence_col].astype(str).str.strip()
    df = df[df[spec.sequence_col] != ""].reset_index(drop=True)
    df[spec.key_metric_col] = pd.to_numeric(df[spec.key_metric_col], errors="coerce")
    if enrichment_threshold is not None:
        df = df[df[spec.key_metric_col] >= float(enrichment_threshold)].reset_index(drop=True)
    return df, spec.sequence_col, spec.key_metric_col, split_path, source


def build_pssm_counts(sequences: Iterable[str]) -> np.ndarray:
    counts = np.zeros((CDRH3_LEN, len(STANDARD_AAS)), dtype=np.float64)
    n_rows = 0
    for sequence in sequences:
        seq = str(sequence).strip()
        if len(seq) != CDRH3_LEN:
            raise ValueError(f"Expected length-{CDRH3_LEN} CDR-H3 sequence, got {seq!r}")
        for pos, aa in enumerate(seq):
            if aa not in AA_TO_INDEX:
                raise ValueError(f"Unsupported amino acid {aa!r} in sequence {seq!r}")
            counts[pos, AA_TO_INDEX[aa]] += 1.0
        n_rows += 1
    if n_rows == 0:
        raise ValueError("No valid train sequences available to build the PSSM.")
    return counts


def counts_to_log_frequencies(counts: np.ndarray, pseudocount: float = 1.0) -> np.ndarray:
    totals = counts.sum(axis=1, keepdims=True)
    return np.log(counts + pseudocount) - np.log(totals + counts.shape[1] * pseudocount)


def temperature_scaled_probabilities(log_frequencies: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    scaled = log_frequencies / float(temperature)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    probs = np.exp(scaled)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def sample_cdrh3_sequences(
    log_frequencies: np.ndarray,
    temperature: float,
    n_sequences: int,
    seed: int,
) -> list[str]:
    probs = temperature_scaled_probabilities(log_frequencies, temperature)
    rng = np.random.default_rng(seed)
    samples: list[str] = []
    aa_indices = np.arange(len(STANDARD_AAS))
    for _ in range(int(n_sequences)):
        residues = [STANDARD_AAS[int(rng.choice(aa_indices, p=probs[pos]))] for pos in range(CDRH3_LEN)]
        samples.append("".join(residues))
    return samples


def hamming_distance(a: str, b: str) -> int:
    if len(a) != len(b):
        raise ValueError(f"Cannot compute Hamming distance for lengths {len(a)} and {len(b)}")
    return sum(x != y for x, y in zip(a, b))


def build_output_rows(cdrh3_sequences: Iterable[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for chain_id, cdrh3 in enumerate(cdrh3_sequences):
        rows.append(
            {
                "chain_id": chain_id,
                "gibbs_step": 0,
                "sequence": add_context(cdrh3),
                "cdrh3": cdrh3,
                "n_mutations": hamming_distance(cdrh3, C05_CDRH3),
                "model_variant": "pssm",
            }
        )
    return rows
