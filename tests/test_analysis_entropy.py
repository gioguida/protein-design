import numpy as np

from protein_design.analysis.entropy import position_entropy


def test_position_entropy_returns_zero_for_empty_sequence_set() -> None:
    entropies = position_entropy([], expected_length=4)
    assert np.array_equal(entropies, np.zeros(4, dtype=np.float32))


def test_position_entropy_filters_wrong_length_sequences() -> None:
    entropies = position_entropy(["AAA", "ABA", "TOO_LONG"], expected_length=3)
    expected = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert np.allclose(entropies, expected)
