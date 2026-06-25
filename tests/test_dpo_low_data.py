import numpy as np
import pandas as pd
import pytest

from protein_design.dpo.low_data import subsample_train_sequences

METRIC = "M22_binding_enrichment_adj"


def _make_df(n: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    # Skewed enrichment: most negative, a minority of positives.
    enrichment = rng.normal(loc=-2.0, scale=2.0, size=n)
    return pd.DataFrame(
        {
            "aa": [f"SEQ{i:05d}" for i in range(n)],
            "mut": ["x"] * n,
            "num_mut": [1] * n,
            METRIC: enrichment,
            "delta_" + METRIC: enrichment - 5.19,
        }
    )


def test_size_and_subset_membership() -> None:
    df = _make_df()
    out = subsample_train_sequences(df, 100, scheme="random", seed=0)
    assert len(out) == 100
    assert set(out["aa"]).issubset(set(df["aa"]))


def test_deterministic_same_seed() -> None:
    df = _make_df()
    a = subsample_train_sequences(df, 150, scheme="stratified", seed=7)
    b = subsample_train_sequences(df, 150, scheme="stratified", seed=7)
    assert list(a["aa"]) == list(b["aa"])


def test_different_seed_differs() -> None:
    df = _make_df()
    a = subsample_train_sequences(df, 150, scheme="stratified", seed=0)
    b = subsample_train_sequences(df, 150, scheme="stratified", seed=1)
    assert list(a["aa"]) != list(b["aa"])


def test_n_geq_usable_returns_full() -> None:
    df = _make_df(50)
    out = subsample_train_sequences(df, 999, scheme="stratified", seed=0)
    assert len(out) == 50


def test_drops_non_finite_metric() -> None:
    df = _make_df(100)
    df.loc[:9, METRIC] = np.nan
    out = subsample_train_sequences(df, 999, scheme="random", seed=0)
    assert len(out) == 90
    assert out[METRIC].notna().all()


def test_stratified_preserves_positive_fraction_better_than_expected_at_small_n() -> None:
    # With a strongly negative-skewed metric a stratified small draw should keep a
    # non-trivial positive count (the whole point: enable within_pos/cross pairs).
    df = _make_df(2000)
    pos_frac_full = float((df[METRIC] > 0).mean())
    strat = subsample_train_sequences(df, 100, scheme="stratified", seed=0)
    strat_pos_frac = float((strat[METRIC] > 0).mean())
    # Stratified should track the global positive fraction within a tolerance.
    assert abs(strat_pos_frac - pos_frac_full) < 0.1


def test_invalid_scheme_and_n() -> None:
    df = _make_df(10)
    with pytest.raises(ValueError):
        subsample_train_sequences(df, 5, scheme="nope", seed=0)
    with pytest.raises(ValueError):
        subsample_train_sequences(df, 0, scheme="random", seed=0)
