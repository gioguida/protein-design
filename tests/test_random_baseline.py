import numpy as np
import pytest

from protein_design.constants import C05_CDRH3
from protein_design.random_baseline import (
    build_output_rows,
    build_position_alphabet,
    hamming_distance,
    mutable_positions,
    sample_random_mutants,
)


def _toy_train_sequences() -> list[str]:
    # WT plus a few single-position variants, so a handful of positions become
    # mutable and the rest stay fixed to WT.
    seqs = [C05_CDRH3]
    for pos, repl in [(0, "A"), (0, "C"), (5, "G"), (10, "W")]:
        chars = list(C05_CDRH3)
        chars[pos] = repl
        seqs.append("".join(chars))
    return seqs


def test_build_position_alphabet_includes_wt_and_observed() -> None:
    alphabet = build_position_alphabet(_toy_train_sequences(), wt=C05_CDRH3)
    assert len(alphabet) == len(C05_CDRH3)
    # WT residue always present at every position.
    for pos, residues in enumerate(alphabet):
        assert C05_CDRH3[pos] in residues
    # Position 0 saw WT + A + C; position 5 saw WT + G.
    assert set(alphabet[0]) == {C05_CDRH3[0], "A", "C"}
    assert set(alphabet[5]) == {C05_CDRH3[5], "G"}
    # An untouched position is fixed to WT only.
    assert alphabet[1] == [C05_CDRH3[1]]
    assert mutable_positions(alphabet) == [0, 5, 10]


def test_build_position_alphabet_skips_wrong_length_and_rejects_bad_aa() -> None:
    seqs = _toy_train_sequences() + ["TOOSHORT"]
    alphabet = build_position_alphabet(seqs, wt=C05_CDRH3)  # short seq skipped
    assert len(alphabet) == len(C05_CDRH3)
    with pytest.raises(ValueError):
        bad = list(C05_CDRH3)
        bad[0] = "1"
        build_position_alphabet(["".join(bad)], wt=C05_CDRH3)


def test_sampling_is_deterministic_and_within_trust_radius() -> None:
    alphabet = build_position_alphabet(_toy_train_sequences(), wt=C05_CDRH3)
    a = sample_random_mutants(alphabet, trust_radius=3, n_sequences=50, seed=7)
    b = sample_random_mutants(alphabet, trust_radius=3, n_sequences=50, seed=7)
    assert a == b  # deterministic by seed
    assert len(a) == 50  # duplicates kept -> exact sample size
    assert all(len(seq) == len(C05_CDRH3) for seq in a)
    # Every edit is within the trust radius and only at mutable positions.
    mutable = set(mutable_positions(alphabet))
    for seq in a:
        n_mut = hamming_distance(seq, C05_CDRH3)
        assert 0 <= n_mut <= 3
        diff = {i for i in range(len(seq)) if seq[i] != C05_CDRH3[i]}
        assert diff.issubset(mutable)


def test_unique_only_dedups() -> None:
    alphabet = build_position_alphabet(_toy_train_sequences(), wt=C05_CDRH3)
    pool = sample_random_mutants(
        alphabet, trust_radius=3, n_sequences=10_000, seed=1, allow_duplicates=False
    )
    assert len(pool) == len(set(pool))


def test_build_output_rows_schema_and_variant() -> None:
    alphabet = build_position_alphabet(_toy_train_sequences(), wt=C05_CDRH3)
    sampled = sample_random_mutants(alphabet, trust_radius=2, n_sequences=4, seed=0)
    rows = build_output_rows(sampled)
    assert [row["chain_id"] for row in rows] == list(range(4))
    assert all(row["gibbs_step"] == 0 for row in rows)
    assert all(row["model_variant"] == "random" for row in rows)
    assert all(set(row) == {"chain_id", "gibbs_step", "sequence", "cdrh3",
                            "n_mutations", "model_variant"} for row in rows)
    assert all(row["n_mutations"] == hamming_distance(row["cdrh3"], C05_CDRH3)
               for row in rows)


def test_trust_radius_must_be_positive() -> None:
    alphabet = build_position_alphabet(_toy_train_sequences(), wt=C05_CDRH3)
    with pytest.raises(ValueError):
        sample_random_mutants(alphabet, trust_radius=0, n_sequences=1, seed=0)
    # n_mutations grows with the trust radius (looser cap -> can edit more).
    rng_small = np.mean([hamming_distance(s, C05_CDRH3)
                         for s in sample_random_mutants(alphabet, 1, 200, seed=3)])
    rng_big = np.mean([hamming_distance(s, C05_CDRH3)
                       for s in sample_random_mutants(alphabet, 3, 200, seed=3)])
    assert rng_small <= rng_big
