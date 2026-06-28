"""Helpers for the CDR-H3 random library-mutant baseline sampler.

A deliberately "dumb" generator, ported faithfully from the genetic-algorithm
seed-pool code in Lorenz Kaiser's ``uncertainty_protein`` repo
(``src/genetic_algorithm/alphabet.py``, ``build_position_alphabet`` +
``_random_mutant`` + ``build_seed_pool(mode="random_wt")``).

It is WT-centered and uniform over the observed residues at each position:

  * the per-position alphabet is the *set* of residues observed at that position
    in the train data (the WT residue is always included) -- NOT
    frequency-weighted (this is what makes it dumber than the PSSM baseline);
  * each sampled sequence picks ``k ~ Uniform[1, trust_radius]`` mutable
    positions and sets each to a uniformly-chosen observed residue != current,
    so ``trust_radius`` caps the number of edits from WT.

One intentional deviation from the Lorenz code: ``build_seed_pool`` dedups to
unique sequences, whereas here we emit exactly ``n_sequences`` rows allowing
duplicates, so the library matches the PSSM baseline's sample size (5000) for an
apples-to-apples comparison against the DPO 650M set.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from protein_design.constants import C05_CDRH3, add_context

# Reuse the shared pieces from the PSSM baseline so the two baselines resolve the
# same train split and share validation / helpers.
from protein_design.pssm_baseline import (
    AA_TO_INDEX,
    CDRH3_LEN,
    STANDARD_AAS,
    expand_repo_path,  # noqa: F401  (re-exported for symmetry with pssm_baseline)
    hamming_distance,
    load_train_dataframe,  # noqa: F401  (re-exported; used by the CLI/sweep)
    resolve_train_split_with_fallback,  # noqa: F401  (re-exported; used by the sweep)
)


def build_position_alphabet(
    sequences: Iterable[str], wt: str = C05_CDRH3
) -> list[list[str]]:
    """Per-position sorted set of residues observed across ``sequences``.

    The WT residue at each position is always included. Returns a list of length
    ``len(wt)``; element ``i`` is the sorted list of allowed residues at position
    ``i``. Sequences whose length differs from ``len(wt)`` are skipped (matching
    the Lorenz code), but every kept residue must be a standard amino acid.
    """
    if len(wt) != CDRH3_LEN:
        raise ValueError(f"Expected WT of length {CDRH3_LEN}, got {len(wt)}")
    observed: list[set[str]] = [set() for _ in range(CDRH3_LEN)]
    n_rows = 0
    for sequence in sequences:
        seq = str(sequence).strip()
        if len(seq) != CDRH3_LEN:
            continue
        for pos, aa in enumerate(seq):
            if aa not in AA_TO_INDEX:
                raise ValueError(f"Unsupported amino acid {aa!r} in sequence {seq!r}")
            observed[pos].add(aa)
        n_rows += 1
    if n_rows == 0:
        raise ValueError(
            f"No train sequences of length {CDRH3_LEN} available to build the "
            "position alphabet."
        )
    alphabet: list[list[str]] = []
    for pos in range(CDRH3_LEN):
        residues = observed[pos]
        residues.add(wt[pos])  # WT residue is always allowed
        alphabet.append(sorted(residues))
    return alphabet


def mutable_positions(position_alphabet: Sequence[Sequence[str]]) -> list[int]:
    """Positions with more than one observed residue (the ones we may mutate)."""
    return [i for i, residues in enumerate(position_alphabet) if len(residues) > 1]


def _random_mutant(
    wt: str,
    position_alphabet: Sequence[Sequence[str]],
    mutable: Sequence[int],
    trust_radius: int,
    rng: np.random.Generator,
) -> str:
    """One random sequence within ``trust_radius`` edits of WT, using only the
    per-position observed residues (faithful to Lorenz ``_random_mutant``)."""
    if not mutable:
        return wt
    k = rng.integers(1, min(trust_radius, len(mutable)) + 1)
    positions = rng.choice(mutable, size=int(k), replace=False)
    chars = list(wt)
    for i in positions:
        choices = [a for a in position_alphabet[i] if a != chars[i]]
        if choices:
            chars[i] = choices[rng.integers(len(choices))]
    return "".join(chars)


def sample_random_mutants(
    position_alphabet: Sequence[Sequence[str]],
    trust_radius: int,
    n_sequences: int,
    seed: int,
    wt: str = C05_CDRH3,
    allow_duplicates: bool = True,
) -> list[str]:
    """Sample ``n_sequences`` random WT-centered library mutants.

    With ``allow_duplicates=True`` (default, matches the PSSM baseline's sample
    size) exactly ``n_sequences`` rows are returned, duplicates included. With
    ``allow_duplicates=False`` the Lorenz dedup-to-unique behavior is used and
    fewer than ``n_sequences`` rows may be returned if the trust-radius
    neighborhood is small.
    """
    if trust_radius < 1:
        raise ValueError(f"trust_radius must be >= 1, got {trust_radius}")
    if len(wt) != CDRH3_LEN:
        raise ValueError(f"Expected WT of length {CDRH3_LEN}, got {len(wt)}")
    rng = np.random.default_rng(seed)
    mutable = mutable_positions(position_alphabet)

    if allow_duplicates:
        return [
            _random_mutant(wt, position_alphabet, mutable, trust_radius, rng)
            for _ in range(int(n_sequences))
        ]

    seen: set[str] = set()
    pool: list[str] = []
    attempts = 0
    max_attempts = int(n_sequences) * 50
    while len(pool) < int(n_sequences) and attempts < max_attempts:
        attempts += 1
        seq = _random_mutant(wt, position_alphabet, mutable, trust_radius, rng)
        if seq not in seen:
            seen.add(seq)
            pool.append(seq)
    return pool


def build_output_rows(cdrh3_sequences: Iterable[str]) -> list[dict[str, object]]:
    """Rows in the schema the generation notebooks/scorers expect, with
    ``model_variant="random"`` (mirrors ``pssm_baseline.build_output_rows``)."""
    rows: list[dict[str, object]] = []
    for chain_id, cdrh3 in enumerate(cdrh3_sequences):
        rows.append(
            {
                "chain_id": chain_id,
                "gibbs_step": 0,
                "sequence": add_context(cdrh3),
                "cdrh3": cdrh3,
                "n_mutations": hamming_distance(cdrh3, C05_CDRH3),
                "model_variant": "random",
            }
        )
    return rows
