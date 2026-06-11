"""Sequence entropy helpers for analysis scripts."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def position_entropy(
    seqs: Sequence[str],
    *,
    expected_length: int | None = None,
) -> np.ndarray:
    """Return per-position Shannon entropy in bits for a fixed-length sequence set."""
    if not seqs:
        length = int(expected_length or 0)
        return np.zeros(length, dtype=np.float32)

    if expected_length is None:
        expected_length = len(seqs[0])

    valid = [str(seq) for seq in seqs if len(str(seq)) == int(expected_length)]
    ent = np.zeros(int(expected_length), dtype=np.float32)
    if not valid:
        return ent

    n = len(valid)
    for idx in range(int(expected_length)):
        counts: dict[str, int] = {}
        for seq in valid:
            token = seq[idx]
            counts[token] = counts.get(token, 0) + 1
        probs = np.array([count / n for count in counts.values()], dtype=np.float32)
        probs = probs[probs > 0]
        ent[idx] = float(-(probs * np.log2(probs)).sum())
    return ent
