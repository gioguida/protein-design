"""Dataset loading helpers for the delta-based DPO workflow."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, TypedDict, cast

import numpy as np
import pandas as pd

from protein_design.constants import WILD_TYPE
from protein_design.dms_splitting import (
    DEFAULT_CONFIG_PATH,
    dataset_spec,
    project_root,
    resolve_dataset_split,
)

from .data_processing import ensure_delta_m22_binding_enrichment
from .utils import _gap_pairs

RANDOM_SEED = 42
LOG = logging.getLogger(__name__)

PairingStrategy = Literal["delta_based"]
DeltaBasedComponent = Literal["within_pos", "within_neg", "wt_anchors", "cross"]
DeltaMixMode = Literal["count", "fraction"]
DELTA_BASED_COMPONENTS: Tuple[DeltaBasedComponent, ...] = (
    "within_pos",
    "within_neg",
    "wt_anchors",
    "cross",
)


class PairMember(TypedDict):
    aa: str
    score: float


PairTuple = Tuple[PairMember, PairMember]


class DeltaBasedParams(TypedDict):
    components: Tuple[DeltaBasedComponent, ...]
    mix_mode: DeltaMixMode
    component_pair_counts: Dict[DeltaBasedComponent, int]
    component_pair_fractions: Dict[DeltaBasedComponent, float]
    delta_col: str
    seq_col: str
    gap: float
    wt_pairs_frac: float
    cross_pairs_frac: float
    strong_pos_threshold: float
    strong_neg_threshold: float
    min_score_margin: float
    rng: np.random.Generator


PAIR_COLUMNS = [
    "source_view",
    "cluster_idx",
    "pair_rank_in_cluster",
    "pairing_strategy",
    "chosen_sequence",
    "rejected_sequence",
    "chosen_delta",
    "rejected_delta",
    "delta_margin",
]


def default_data_paths() -> Dict[str, Path]:
    root = project_root()
    return {
        "raw_m22": root / "data" / "raw" / "ED2_M22_binding_enrichment.csv",
        "processed_dir": root / "data" / "processed",
        "dms_config": root / DEFAULT_CONFIG_PATH,
    }


def validate_delta_based_components(
    components: Sequence[str],
) -> Tuple[DeltaBasedComponent, ...]:
    if not components:
        raise ValueError(
            "data.delta_based.components must include at least one component. "
            f"Valid components: {', '.join(DELTA_BASED_COMPONENTS)}"
        )
    normalized: List[str] = [str(component) for component in components]
    unknown = sorted({component for component in normalized if component not in DELTA_BASED_COMPONENTS})
    if unknown:
        raise ValueError(
            "Invalid data.delta_based.components value(s): "
            f"{', '.join(unknown)}. Valid components: {', '.join(DELTA_BASED_COMPONENTS)}"
        )
    return cast(Tuple[DeltaBasedComponent, ...], tuple(normalized))


def _reject_legacy_pairing(pairing_strategy: str) -> None:
    if str(pairing_strategy) != "delta_based":
        raise ValueError(
            "Only data.pairing_strategy='delta_based' is supported. "
            "Legacy mutation-view DPO strategies were removed because the DPO pipeline now trains on all num_mut values."
        )


def _next_random_state(rng: np.random.Generator) -> int:
    return int(rng.integers(0, np.iinfo(np.uint32).max, endpoint=True))


def _zero_component_counts() -> Dict[DeltaBasedComponent, int]:
    return {component: 0 for component in DELTA_BASED_COMPONENTS}


def _zero_component_fractions() -> Dict[DeltaBasedComponent, float]:
    return {component: 0.0 for component in DELTA_BASED_COMPONENTS}


def _coerce_component_counts(
    counts: Optional[Dict[str, int]],
) -> Dict[DeltaBasedComponent, int]:
    normalized = _zero_component_counts()
    if counts is None:
        return normalized
    for component in DELTA_BASED_COMPONENTS:
        if component in counts:
            normalized[component] = max(0, int(counts[component]))
    return normalized


def _coerce_component_fractions(
    fractions: Optional[Dict[str, float]],
) -> Dict[DeltaBasedComponent, float]:
    normalized = _zero_component_fractions()
    if fractions is None:
        return normalized
    for component in DELTA_BASED_COMPONENTS:
        if component in fractions:
            normalized[component] = max(0.0, float(fractions[component]))
    return normalized


def _normalize_mix_mode(mix_mode: str) -> DeltaMixMode:
    mode = str(mix_mode).strip().lower()
    if mode not in {"count", "fraction"}:
        raise ValueError(
            "data.delta_based.mix.mode must be 'count' or 'fraction'. "
            f"Got: {mix_mode!r}"
        )
    return cast(DeltaMixMode, mode)


def _sample_pairs_without_replacement(
    pairs: Sequence[PairTuple],
    sample_size: int,
    rng: np.random.Generator,
) -> List[PairTuple]:
    target = max(0, int(sample_size))
    if target <= 0:
        return []
    if target >= len(pairs):
        return list(pairs)
    indices = rng.choice(len(pairs), size=target, replace=False)
    return [pairs[int(idx)] for idx in indices]


def _count_targets_from_fractions(
    *,
    components: Tuple[DeltaBasedComponent, ...],
    available_counts: Dict[DeltaBasedComponent, int],
    component_pair_fractions: Dict[DeltaBasedComponent, float],
) -> Dict[DeltaBasedComponent, int]:
    selected_positive_fractions = {
        component: component_pair_fractions[component]
        for component in components
        if component_pair_fractions[component] > 0.0
    }
    if not selected_positive_fractions:
        return _zero_component_counts()

    selected_fraction_sum = float(sum(selected_positive_fractions.values()))
    normalized_fractions = {
        component: value / selected_fraction_sum
        for component, value in selected_positive_fractions.items()
    }

    feasible_totals = []
    for component, fraction in normalized_fractions.items():
        available = max(0, int(available_counts[component]))
        if fraction <= 0:
            continue
        feasible_totals.append(available / fraction)
    if not feasible_totals:
        return _zero_component_counts()
    target_total = max(0, int(np.floor(min(feasible_totals))))

    targets = _zero_component_counts()
    remainders: List[Tuple[float, DeltaBasedComponent]] = []
    assigned = 0
    for component, fraction in normalized_fractions.items():
        raw_target = fraction * float(target_total)
        base = min(int(np.floor(raw_target)), int(available_counts[component]))
        targets[component] = max(0, base)
        assigned += targets[component]
        remainders.append((raw_target - float(np.floor(raw_target)), component))

    remaining = max(0, target_total - assigned)
    if remaining > 0:
        remainders.sort(key=lambda item: item[0], reverse=True)
        for _, component in remainders:
            if remaining <= 0:
                break
            capacity = int(available_counts[component]) - targets[component]
            if capacity <= 0:
                continue
            add = min(capacity, remaining)
            targets[component] += add
            remaining -= add

    return targets


def _downsample_pairs_to_train_controlled_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frac_sum = float(train_frac) + float(val_frac) + float(test_frac)
    if abs(frac_sum - 1.0) >= 1e-6:
        raise ValueError(
            "Pair split fractions must sum to 1.0 when train-controlled split sizing is enabled. "
            f"Got train={train_frac}, val={val_frac}, test={test_frac} (sum={frac_sum})."
        )
    if float(train_frac) <= 0.0:
        raise ValueError(
            "train_frac must be > 0 when train-controlled split sizing is enabled. "
            f"Got: {train_frac}."
        )

    train_count = int(len(train_df))
    if train_count <= 0:
        return train_df, val_df.iloc[0:0].copy(), test_df.iloc[0:0].copy()

    target_total = float(train_count) / float(train_frac)
    target_val = max(0, int(np.floor(target_total * float(val_frac))))
    target_test = max(0, int(np.floor(target_total * float(test_frac))))

    def _downsample(df: pd.DataFrame, target: int) -> pd.DataFrame:
        if target >= len(df):
            return df
        if target <= 0:
            return df.iloc[0:0].copy()
        selected = rng.choice(len(df), size=target, replace=False)
        return df.iloc[np.sort(selected)].reset_index(drop=True)

    return (
        train_df.reset_index(drop=True),
        _downsample(val_df, target_val),
        _downsample(test_df, target_test),
    )


def _build_within_pos_pairs(
    cluster_df: pd.DataFrame,
    params: DeltaBasedParams,
    rng: np.random.Generator,
) -> List[PairTuple]:
    del rng
    delta_col = params["delta_col"]
    seq_col = params["seq_col"]
    positives = (
        cluster_df[cluster_df[delta_col].astype(float) > 0]
        .sort_values(delta_col, ascending=False)
        .reset_index(drop=True)
    )
    return _gap_pairs(positives, delta_col, seq_col, params["gap"])


def _build_within_neg_pairs(
    cluster_df: pd.DataFrame,
    params: DeltaBasedParams,
    rng: np.random.Generator,
) -> List[PairTuple]:
    delta_col = params["delta_col"]
    seq_col = params["seq_col"]
    positives = (
        cluster_df[cluster_df[delta_col].astype(float) > 0]
        .sort_values(delta_col, ascending=False)
        .reset_index(drop=True)
    )
    negatives = (
        cluster_df[cluster_df[delta_col].astype(float) < 0]
        .sort_values(delta_col, ascending=False)
        .reset_index(drop=True)
    )
    neg_sub_n = min(len(negatives), 2 * len(positives))
    if neg_sub_n <= 0:
        return []
    neg_sub = (
        negatives.sample(n=neg_sub_n, replace=False, random_state=_next_random_state(rng))
        .sort_values(delta_col, ascending=False)
        .reset_index(drop=True)
    )
    return _gap_pairs(neg_sub, delta_col, seq_col, params["gap"])


def _build_wt_anchor_pairs(cluster_df: pd.DataFrame, params: DeltaBasedParams) -> List[PairTuple]:
    delta_col = params["delta_col"]
    seq_col = params["seq_col"]
    rng = params["rng"]
    n = len(cluster_df)
    if n <= 1:
        return []
    positives = (
        cluster_df[cluster_df[delta_col].astype(float) > 0]
        .sort_values(delta_col, ascending=False)
        .reset_index(drop=True)
    )
    negatives = (
        cluster_df[cluster_df[delta_col].astype(float) < 0]
        .sort_values(delta_col, ascending=False)
        .reset_index(drop=True)
    )
    wt = {"aa": WILD_TYPE, "score": 0.0}
    all_pairs: List[PairTuple] = []
    num_wt = int(params["wt_pairs_frac"] * n)
    num_pos_wt = num_wt // 2
    num_neg_wt = num_wt - num_pos_wt
    strong_pos = positives[positives[delta_col].astype(float) > params["strong_pos_threshold"]]
    if len(strong_pos) > 0 and num_pos_wt > 0:
        sampled = strong_pos.sample(
            n=min(num_pos_wt, len(strong_pos)),
            replace=False,
            random_state=_next_random_state(rng),
        )
        for _, row in sampled.iterrows():
            all_pairs.append(({"aa": row[seq_col], "score": float(row[delta_col])}, wt))
    strong_neg = negatives[negatives[delta_col].astype(float) < params["strong_neg_threshold"]]
    if len(strong_neg) > 0 and num_neg_wt > 0:
        sampled = strong_neg.sample(
            n=min(num_neg_wt, len(strong_neg)),
            replace=False,
            random_state=_next_random_state(rng),
        )
        for _, row in sampled.iterrows():
            all_pairs.append((wt, {"aa": row[seq_col], "score": float(row[delta_col])}))
    return all_pairs


def _build_cross_pairs(
    cluster_df: pd.DataFrame,
    params: DeltaBasedParams,
    rng: np.random.Generator,
) -> List[PairTuple]:
    delta_col = params["delta_col"]
    seq_col = params["seq_col"]
    positives = (
        cluster_df[cluster_df[delta_col].astype(float) > 0]
        .sort_values(delta_col, ascending=False)
        .reset_index(drop=True)
    )
    negatives = (
        cluster_df[cluster_df[delta_col].astype(float) < 0]
        .sort_values(delta_col, ascending=False)
        .reset_index(drop=True)
    )
    num_cross = int(params["cross_pairs_frac"] * len(cluster_df))
    if len(positives) == 0 or len(negatives) == 0 or num_cross <= 0:
        return []
    cross_pos = positives.sample(
        n=min(num_cross, len(positives)),
        replace=len(positives) < num_cross,
        random_state=_next_random_state(rng),
    )
    cross_neg = negatives.sample(
        n=min(num_cross, len(negatives)),
        replace=False,
        random_state=_next_random_state(rng),
    )
    return [
        (
            {"aa": getattr(pos_row, seq_col), "score": float(getattr(pos_row, delta_col))},
            {"aa": getattr(neg_row, seq_col), "score": float(getattr(neg_row, delta_col))},
        )
        for pos_row, neg_row in zip(cross_pos.itertuples(), cross_neg.itertuples())
    ]


def _pair_delta_based(
    sequences_df: pd.DataFrame,
    params: DeltaBasedParams,
) -> List[Tuple[PairMember, PairMember, str]]:
    if len(sequences_df) <= 1:
        return []
    rng = params["rng"]
    builders = {
        "within_pos": lambda: _build_within_pos_pairs(sequences_df, params, rng),
        "within_neg": lambda: _build_within_neg_pairs(sequences_df, params, rng),
        "wt_anchors": lambda: _build_wt_anchor_pairs(sequences_df, params),
        "cross": lambda: _build_cross_pairs(sequences_df, params, rng),
    }
    per_component_pairs: Dict[DeltaBasedComponent, List[PairTuple]] = {
        component: [] for component in DELTA_BASED_COMPONENTS
    }
    for component in params["components"]:
        raw_pairs = builders[component]()
        per_component_pairs[component] = [
            (winner, loser)
            for winner, loser in raw_pairs
            if (winner["score"] - loser["score"]) >= float(params["min_score_margin"])
        ]

    available_counts: Dict[DeltaBasedComponent, int] = {
        component: len(per_component_pairs[component]) for component in DELTA_BASED_COMPONENTS
    }
    if params["mix_mode"] == "count":
        target_counts = _zero_component_counts()
        for component in params["components"]:
            target_counts[component] = min(
                int(params["component_pair_counts"][component]),
                int(available_counts[component]),
            )
    elif params["mix_mode"] == "fraction":
        target_counts = _count_targets_from_fractions(
            components=params["components"],
            available_counts=available_counts,
            component_pair_fractions=params["component_pair_fractions"],
        )
    else:  # pragma: no cover - guarded by _normalize_mix_mode
        raise ValueError(f"Unsupported mix mode: {params['mix_mode']!r}")

    selected_pairs: List[Tuple[PairMember, PairMember, str]] = []
    for component in params["components"]:
        chosen_pairs = _sample_pairs_without_replacement(
            per_component_pairs[component],
            target_counts[component],
            rng,
        )
        selected_pairs.extend((winner, loser, component) for winner, loser in chosen_pairs)
    return selected_pairs


def build_dpo_pairs_from_clustered_dataframe(
    clustered_df: pd.DataFrame,
    pairing_strategy: PairingStrategy = "delta_based",
    min_positive_delta: float = 0.0,
    min_delta_margin: float = 0.0,
    delta_components: Sequence[str] = DELTA_BASED_COMPONENTS,
    delta_mix_mode: str = "count",
    delta_component_pair_counts: Optional[Dict[str, int]] = None,
    delta_component_pair_fractions: Optional[Dict[str, float]] = None,
    gap: float = 0.5,
    wt_pairs_frac: float = 0.1,
    cross_pairs_frac: float = 0.1,
    strong_pos_threshold: float = 1.0,
    strong_neg_threshold: float = -5.0,
    min_score_margin: float = 0.1,
    rng: Optional[np.random.Generator] = None,
    random_seed: int = RANDOM_SEED,
    source_view: str = "base",
    delta_col: str = "delta_M22_binding_enrichment_adj",
    seq_col: str = "aa",
) -> pd.DataFrame:
    del min_positive_delta, min_delta_margin
    _reject_legacy_pairing(str(pairing_strategy))
    missing_cols = {seq_col, delta_col}.difference(clustered_df.columns)
    if missing_cols:
        raise ValueError(f"clustered_df is missing required columns: {', '.join(sorted(missing_cols))}")
    components = validate_delta_based_components(delta_components)
    mix_mode = _normalize_mix_mode(delta_mix_mode)
    component_pair_counts = _coerce_component_counts(delta_component_pair_counts)
    component_pair_fractions = _coerce_component_fractions(delta_component_pair_fractions)
    if delta_component_pair_counts is None and mix_mode == "count":
        unlimited_count = np.iinfo(np.int32).max
        for component in components:
            component_pair_counts[component] = unlimited_count

    params: DeltaBasedParams = {
        "components": components,
        "mix_mode": mix_mode,
        "component_pair_counts": component_pair_counts,
        "component_pair_fractions": component_pair_fractions,
        "delta_col": delta_col,
        "seq_col": seq_col,
        "gap": float(gap),
        "wt_pairs_frac": float(wt_pairs_frac),
        "cross_pairs_frac": float(cross_pairs_frac),
        "strong_pos_threshold": float(strong_pos_threshold),
        "strong_neg_threshold": float(strong_neg_threshold),
        "min_score_margin": float(min_score_margin),
        "rng": rng if rng is not None else np.random.default_rng(int(random_seed)),
    }
    pair_rows = []
    for pair_rank, (chosen, rejected, component) in enumerate(_pair_delta_based(clustered_df, params)):
        chosen_delta = float(chosen["score"])
        rejected_delta = float(rejected["score"])
        pair_rows.append(
            {
                "source_view": source_view,
                "cluster_idx": -1,
                "pair_rank_in_cluster": pair_rank,
                "pairing_strategy": "delta_based",
                "delta_component": component,
                "chosen_sequence": chosen["aa"],
                "rejected_sequence": rejected["aa"],
                "chosen_delta": chosen_delta,
                "rejected_delta": rejected_delta,
                "delta_margin": chosen_delta - rejected_delta,
            }
        )
    if not pair_rows:
        return pd.DataFrame(columns=PAIR_COLUMNS + ["delta_component"])
    return pd.DataFrame(pair_rows).reset_index(drop=True)


def _load_split_dataframe(
    *,
    split_name: str,
    dms_config_path: Path,
    dataset_key: str,
    force_rebuild: bool,
) -> pd.DataFrame:
    split_path = resolve_dataset_split(dataset_key, split_name, dms_config_path, force=force_rebuild)
    spec = dataset_spec(dataset_key, dms_config_path)
    df = pd.read_csv(split_path)
    if spec.key_metric_col != "M22_binding_enrichment_adj":
        df["M22_binding_enrichment_adj"] = pd.to_numeric(df[spec.key_metric_col], errors="coerce")
    return ensure_delta_m22_binding_enrichment(df)


def build_split_pair_dataframes_from_raw(
    pairing_strategy: PairingStrategy = "delta_based",
    include_views: Sequence[str] = (),
    raw_csv_path: Path = None,
    processed_dir: Path = None,
    force_rebuild: bool = False,
    min_positive_delta: float = 0.0,
    min_delta_margin: float = 0.0,
    delta_components: Sequence[str] = DELTA_BASED_COMPONENTS,
    delta_mix_mode: str = "count",
    delta_component_pair_counts: Optional[Dict[str, int]] = None,
    delta_component_pair_fractions: Optional[Dict[str, float]] = None,
    gap: float = 0.5,
    wt_pairs_frac: float = 0.1,
    cross_pairs_frac: float = 0.1,
    strong_pos_threshold: float = 1.0,
    strong_neg_threshold: float = -5.0,
    min_score_margin: float = 0.1,
    deduplicate_across_views: bool = True,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    split_hamming_distance: int = 1,
    split_stratify_bins: int = 10,
    enforce_train_controlled_split_sizes: bool = False,
    seed: int = RANDOM_SEED,
    dms_config_path: Path | str | None = None,
    dataset_key: str = "ed2_m22",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    del include_views, raw_csv_path, processed_dir, deduplicate_across_views
    del split_hamming_distance, split_stratify_bins
    _reject_legacy_pairing(str(pairing_strategy))
    dms_config_path = Path(dms_config_path or (project_root() / DEFAULT_CONFIG_PATH))
    split_frames = {
        split: _load_split_dataframe(
            split_name=split,
            dms_config_path=dms_config_path,
            dataset_key=dataset_key,
            force_rebuild=force_rebuild,
        )
        for split in ("train", "val", "test")
    }

    def build(split: str, offset: int) -> pd.DataFrame:
        return build_dpo_pairs_from_clustered_dataframe(
            clustered_df=split_frames[split],
            pairing_strategy="delta_based",
            min_positive_delta=min_positive_delta,
            min_delta_margin=min_delta_margin,
            delta_components=delta_components,
            delta_mix_mode=delta_mix_mode,
            delta_component_pair_counts=delta_component_pair_counts,
            delta_component_pair_fractions=delta_component_pair_fractions,
            gap=gap,
            wt_pairs_frac=wt_pairs_frac,
            cross_pairs_frac=cross_pairs_frac,
            strong_pos_threshold=strong_pos_threshold,
            strong_neg_threshold=strong_neg_threshold,
            min_score_margin=min_score_margin,
            rng=np.random.default_rng(int(seed) + offset),
            random_seed=int(seed) + offset,
            source_view=split,
        )

    train_df, val_df, test_df = build("train", 0), build("val", 1), build("test", 2)
    if bool(enforce_train_controlled_split_sizes):
        train_df, val_df, test_df = _downsample_pairs_to_train_controlled_split(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            train_frac=float(train_frac),
            val_frac=float(val_frac),
            test_frac=float(test_frac),
            rng=np.random.default_rng(int(seed) + 3),
        )
    return train_df, val_df, test_df


def load_dpo_pair_dataframe(**kwargs) -> pd.DataFrame:
    train_df, _, _ = build_split_pair_dataframes_from_raw(**kwargs)
    return train_df


def load_dpo_sequence_pairs(**kwargs) -> List[PairTuple]:
    pairs_df = load_dpo_pair_dataframe(**kwargs)
    return [
        (
            {"aa": str(chosen_aa), "score": float(chosen_delta)},
            {"aa": str(rejected_aa), "score": float(rejected_delta)},
        )
        for chosen_aa, rejected_aa, chosen_delta, rejected_delta in zip(
            pairs_df["chosen_sequence"],
            pairs_df["rejected_sequence"],
            pairs_df["chosen_delta"],
            pairs_df["rejected_delta"],
        )
    ]
