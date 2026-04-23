"""Dataset loading helpers for DPO workflow.

This module integrates preprocessing so callers can rely on raw data only.
If processed ED2 views are missing or stale, they are rebuilt automatically.
"""

import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, TypedDict, cast

import pandas as pd
import numpy as np


from protein_design.constants import WILD_TYPE

from .data_processing import build_processed_views
from .splitting import (
	build_or_load_cluster_split_membership,
	split_membership_keys,
	summarize_split_membership,
)
from .utils import _gap_pairs

RANDOM_SEED = 42
LOG = logging.getLogger(__name__)


PairingStrategy = Literal["positive_vs_tail", "positive_only_extremes", "both_structured", "delta_based"]
DeltaBasedComponent = Literal["within_pos", "within_neg", "wt_anchors", "cross"]
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


def _project_root() -> Path:
	"""Return the repository root path."""
	return Path(__file__).resolve().parents[3]


def default_data_paths() -> Dict[str, Path]:
	"""Return default raw and processed data paths."""
	root = _project_root()
	return {
		"raw_m22": root / "data" / "raw" / "ED2_M22_binding_enrichment.csv",
		"processed_dir": root / "data" / "processed",
	}


def _ensure_processed_data(
	raw_csv_path: Path = None,
	processed_dir: Path = None,
	force_rebuild: bool = False,
	verbose: bool = False,
) -> Dict[str, Path]:
	"""Ensure processed ED2 views exist and return their paths."""
	defaults = default_data_paths()
	raw_csv_path = defaults["raw_m22"] if raw_csv_path is None else Path(raw_csv_path)
	processed_dir = (
		defaults["processed_dir"] if processed_dir is None else Path(processed_dir)
	)

	return build_processed_views(
		raw_csv_path=raw_csv_path,
		processed_dir=processed_dir,
		force=force_rebuild,
		verbose=verbose,
	)


def load_distance2_dataframe(
	view: str = "mut1",
	raw_csv_path: Path = None,
	processed_dir: Path = None,
	force_rebuild: bool = False,
) -> pd.DataFrame:
	"""Load one processed dataframe view (base, mut1, or mut2)."""
	paths = _ensure_processed_data(
		raw_csv_path=raw_csv_path,
		processed_dir=processed_dir,
		force_rebuild=force_rebuild,
		verbose=False,
	)

	view_to_key = {
		"base": "ed2_all",
		"mut1": "d2_clustered_mut1",
		"mut2": "d2_clustered_mut2",
	}
	if view not in view_to_key:
		raise ValueError("view must be one of: base, mut1, mut2")

	return pd.read_csv(paths[view_to_key[view]])


def _pair_cluster_positive_vs_tail(
	cluster_df: pd.DataFrame,
	delta_col: str,
	min_positive_delta: float,
	min_delta_margin: float = 0.0,
) -> List[PairTuple]:
	"""Pair first with last, second with second-last, until chosen side is not positive."""
	cluster_sorted = cluster_df.sort_values(by=delta_col, ascending=False).reset_index(drop=True)
	deltas = cluster_sorted[delta_col].astype(float).to_numpy()
	n = len(cluster_sorted)
	left = 0
	right = n - 1
	pairs: List[PairTuple] = []

	while left < right and float(deltas[left]) > float(min_positive_delta):
		chosen = cluster_sorted.iloc[left]
		chosen_delta = float(chosen[delta_col])
		rejected = cluster_sorted.iloc[right]
		rejected_delta = float(rejected[delta_col])
		if chosen_delta - rejected_delta >= float(min_delta_margin):
			winner = {"aa": chosen["aa"], "score": chosen_delta}
			loser = {"aa": rejected["aa"], "score": rejected_delta}
			pairs.append((winner, loser))
		left += 1
		right -= 1

	return pairs


def _pair_cluster_positive_only_extremes(
	cluster_df: pd.DataFrame,
	delta_col: str,
	min_positive_delta: float,
	min_delta_margin: float = 0.0,
) -> List[PairTuple]:
	"""Pair i-th positive with (P-i)-th from tail in full cluster, where P is #positives."""
	cluster_sorted = cluster_df.sort_values(by=delta_col, ascending=False).reset_index(drop=True)
	deltas = cluster_sorted[delta_col].astype(float).to_numpy()
	positive_count = int((deltas > float(min_positive_delta)).sum())
	n = len(cluster_sorted)
	rejected_start = n - positive_count
	pairs: List[PairTuple] = []

	for i in range(positive_count):
		rejected_idx = rejected_start + i
		if rejected_idx >= n:
			break
		chosen = cluster_sorted.iloc[i]
		chosen_delta = float(chosen[delta_col])
		rejected = cluster_sorted.iloc[rejected_idx]
		rejected_delta = float(rejected[delta_col])
		if chosen_delta - rejected_delta >= float(min_delta_margin):
			winner = {"aa": chosen["aa"], "score": chosen_delta}
			loser = {"aa": rejected["aa"], "score": rejected_delta}
			pairs.append((winner, loser))

	return pairs


def _pair_cluster_both_structured_strategies(
	cluster_df: pd.DataFrame,
	delta_col: str,
	min_positive_delta: float,
	min_delta_margin: float = 0.0,
) -> List[PairTuple]:
	"""Combine both strucured stregies to get more pairs"""
	pairs_positive_vs_tail = _pair_cluster_positive_vs_tail(
		cluster_df=cluster_df,
		delta_col=delta_col,
		min_positive_delta=min_positive_delta,
		min_delta_margin=min_delta_margin,
	)

	pairs_positive_only_extremes = _pair_cluster_positive_only_extremes(
		cluster_df=cluster_df,
		delta_col=delta_col,
		min_positive_delta=min_positive_delta,
		min_delta_margin=min_delta_margin,
	)
	
	# Combine pairs and remove duplicates
	seen = set()
	combined_pairs = []
	for pair in pairs_positive_vs_tail + pairs_positive_only_extremes:
		chosen_seq = pair[0]["aa"]
		rejected_seq = pair[1]["aa"]
		if (chosen_seq, rejected_seq) not in seen:
			seen.add((chosen_seq, rejected_seq))
			combined_pairs.append(pair)

	return combined_pairs


def validate_delta_based_components(
	components: Sequence[str],
) -> Tuple[DeltaBasedComponent, ...]:
	"""Validate and normalize configured delta-based pair-source components."""
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


def _next_random_state(rng: np.random.Generator) -> int:
	"""Derive a deterministic integer seed for pandas sampling from a generator."""
	return int(rng.integers(0, np.iinfo(np.uint32).max, endpoint=True))


def _build_within_pos_pairs(
	cluster_df: pd.DataFrame,
	params: DeltaBasedParams,
	rng: np.random.Generator,
) -> List[PairTuple]:
	"""Build gap pairs among positive deltas."""
	del rng  # kept in signature for consistency across delta components
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
	"""Build gap pairs among negatives after subsampling."""
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


def _build_wt_anchor_pairs(
	cluster_df: pd.DataFrame,
	params: DeltaBasedParams,
) -> List[PairTuple]:
	"""Build WT-anchored pairs for strong positives and strong negatives."""
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
			winner = {"aa": row[seq_col], "score": float(row[delta_col])}
			all_pairs.append((winner, wt))

	strong_neg = negatives[negatives[delta_col].astype(float) < params["strong_neg_threshold"]]
	if len(strong_neg) > 0 and num_neg_wt > 0:
		sampled = strong_neg.sample(
			n=min(num_neg_wt, len(strong_neg)),
			replace=False,
			random_state=_next_random_state(rng),
		)
		for _, row in sampled.iterrows():
			loser = {"aa": row[seq_col], "score": float(row[delta_col])}
			all_pairs.append((wt, loser))

	return all_pairs


def _build_cross_pairs(
	cluster_df: pd.DataFrame,
	params: DeltaBasedParams,
	rng: np.random.Generator,
) -> List[PairTuple]:
	"""Build cross-class positive-vs-negative pairs."""
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

	all_pairs: List[PairTuple] = []
	for p_row, n_row in zip(cross_pos.itertuples(), cross_neg.itertuples()):
		winner = {"aa": getattr(p_row, seq_col), "score": float(getattr(p_row, delta_col))}
		loser = {"aa": getattr(n_row, seq_col), "score": float(getattr(n_row, delta_col))}
		all_pairs.append((winner, loser))

	return all_pairs


def _pair_delta_based(
	sequences_df: pd.DataFrame,
	params: DeltaBasedParams,
) -> List[Tuple[PairMember, PairMember, str]]:
	if len(sequences_df) <= 1:
		return []

	rng = params["rng"]
	component_builders = {
		"within_pos": lambda: _build_within_pos_pairs(sequences_df, params, rng),
		"within_neg": lambda: _build_within_neg_pairs(sequences_df, params, rng),
		"wt_anchors": lambda: _build_wt_anchor_pairs(sequences_df, params),
		"cross": lambda: _build_cross_pairs(sequences_df, params, rng),
	}

	all_pairs: List[Tuple[PairMember, PairMember, str]] = []
	for component in params["components"]:
		for winner, loser in component_builders[component]():
			all_pairs.append((winner, loser, component))

	return [
		(winner, loser, component)
		for winner, loser, component in all_pairs
		if (winner["score"] - loser["score"]) >= float(params["min_score_margin"])
	]

		
def build_dpo_pairs_from_clustered_dataframe(
	clustered_df: pd.DataFrame,
	pairing_strategy: PairingStrategy = "positive_vs_tail", 	# "positive_only_extremes", "both_structured" or "delta_based"
	min_positive_delta: float = 0.0,
	min_delta_margin: float = 0.0,
	delta_components: Sequence[str] = DELTA_BASED_COMPONENTS,
	gap: float = 0.5,
	wt_pairs_frac: float = 0.1,
	cross_pairs_frac: float = 0.1,
	strong_pos_threshold: float = 1.0,
	strong_neg_threshold: float = -5.0,
	min_score_margin: float = 0.1,
	rng: Optional[np.random.Generator] = None,
	random_seed: int = RANDOM_SEED,
	source_view: str = "",
) -> pd.DataFrame:
	"""Build DPO preference pairs from one clustered dataframe."""
	if pairing_strategy not in ("positive_vs_tail", "positive_only_extremes", "both_structured", "delta_based"):
		raise ValueError("pairing_strategy must be positive_vs_tail, positive_only_extremes, both_structured, or delta_based")

	seq_col = "aa"
	delta_col = "delta_M22_binding_enrichment_adj"
	cluster_col = "cluster_idx"
	required_cols = {seq_col, delta_col}
	if pairing_strategy != "delta_based":
		required_cols.add(cluster_col)
	missing_cols = required_cols.difference(clustered_df.columns)
	if missing_cols:
		missing = ", ".join(sorted(missing_cols))
		raise ValueError(f"clustered_df is missing required columns: {missing}")

	validated_components = (
		validate_delta_based_components(delta_components)
		if pairing_strategy == "delta_based"
		else DELTA_BASED_COMPONENTS
	)
	cluster_rng = rng if rng is not None else np.random.default_rng(int(random_seed))

	if pairing_strategy == "delta_based":
		delta_params: DeltaBasedParams = {
			"components": validated_components,
			"delta_col": delta_col,
			"seq_col": seq_col,
			"gap": float(gap),
			"wt_pairs_frac": float(wt_pairs_frac),
			"cross_pairs_frac": float(cross_pairs_frac),
			"strong_pos_threshold": float(strong_pos_threshold),
			"strong_neg_threshold": float(strong_neg_threshold),
			"min_score_margin": float(min_score_margin),
			"rng": cluster_rng,
		}
		global_pairs = _pair_delta_based(sequences_df=clustered_df, params=delta_params)
		pair_rows = []
		for pair_rank, (chosen, rejected, component) in enumerate(global_pairs):
			chosen_delta = float(chosen["score"])
			rejected_delta = float(rejected["score"])
			pair_rows.append(
				{
					"source_view": source_view,
					"cluster_idx": -1,
					"pair_rank_in_cluster": pair_rank,
					"pairing_strategy": pairing_strategy,
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

	pair_rows = []
	for cluster_value, cluster_df in clustered_df.groupby(cluster_col, sort=False):
		if pairing_strategy == "positive_vs_tail":
			cluster_pairs = _pair_cluster_positive_vs_tail(
				cluster_df=cluster_df,
				delta_col=delta_col,
				min_positive_delta=min_positive_delta,
				min_delta_margin=min_delta_margin,
			)
		elif pairing_strategy == "positive_only_extremes":
			cluster_pairs = _pair_cluster_positive_only_extremes(
				cluster_df=cluster_df,
				delta_col=delta_col,
				min_positive_delta=min_positive_delta,
				min_delta_margin=min_delta_margin,
			)
		elif pairing_strategy == "both_structured":
			cluster_pairs = _pair_cluster_both_structured_strategies(
				cluster_df=cluster_df,
				delta_col=delta_col,
				min_positive_delta=min_positive_delta,
				min_delta_margin=min_delta_margin,
			)

		for pair_rank, (chosen, rejected) in enumerate(cluster_pairs):
			chosen_delta = float(chosen["score"])
			rejected_delta = float(rejected["score"])
			pair_rows.append(
				{
					"source_view": source_view,
					"cluster_idx": cluster_value,
					"pair_rank_in_cluster": pair_rank,
					"pairing_strategy": pairing_strategy,
					"chosen_sequence": chosen["aa"],
					"rejected_sequence": rejected["aa"],
					"chosen_delta": chosen_delta,
					"rejected_delta": rejected_delta,
					"delta_margin": chosen_delta - rejected_delta,
				}
			)

	if not pair_rows:
		return pd.DataFrame(columns=PAIR_COLUMNS)

	pairs_df = pd.DataFrame(pair_rows)
	return pairs_df[PAIR_COLUMNS].reset_index(drop=True)


def load_dpo_pair_dataframe(
	pairing_strategy: PairingStrategy = "positive_vs_tail",
	include_views: Sequence[str] = ("mut1", "mut2"),
	raw_csv_path: Path = None,
	processed_dir: Path = None,
	force_rebuild: bool = False,
	min_positive_delta: float = 0.0,
	min_delta_margin: float = 0.0,
	delta_components: Sequence[str] = DELTA_BASED_COMPONENTS,
	gap: float = 0.5,
	wt_pairs_frac: float = 0.1,
	cross_pairs_frac: float = 0.1,
	strong_pos_threshold: float = 1.0,
	strong_neg_threshold: float = -5.0,
	min_score_margin: float = 0.1,
	seed: int = RANDOM_SEED,
	deduplicate_across_views: bool = True,
) -> pd.DataFrame:
	"""Load views and build a DPO pair dataframe.

	For delta_based, loads the base (unclustered) view regardless of include_views.
	For other strategies, loads the specified mut1/mut2 clustered views.
	"""

	validated_components = (
		validate_delta_based_components(delta_components)
		if pairing_strategy == "delta_based"
		else DELTA_BASED_COMPONENTS
	)
	delta_rng = np.random.default_rng(int(seed))

	if pairing_strategy == "delta_based":
		base_df = load_distance2_dataframe(
			view="base",
			raw_csv_path=raw_csv_path,
			processed_dir=processed_dir,
			force_rebuild=force_rebuild,
		)
		return build_dpo_pairs_from_clustered_dataframe(
			clustered_df=base_df,
			pairing_strategy=pairing_strategy,
			delta_components=validated_components,
			gap=gap,
			wt_pairs_frac=wt_pairs_frac,
			cross_pairs_frac=cross_pairs_frac,
			strong_pos_threshold=strong_pos_threshold,
			strong_neg_threshold=strong_neg_threshold,
			min_score_margin=min_score_margin,
			rng=delta_rng,
			random_seed=int(seed),
			source_view="base",
		)

	valid_views = {"mut1", "mut2"}
	if not include_views:
		raise ValueError("include_views must contain at least one of: mut1, mut2")
	for view in include_views:
		if view not in valid_views:
			raise ValueError("include_views must contain only: mut1, mut2")

	pairs_per_view = []
	for view in include_views:
		clustered_df = load_distance2_dataframe(
			view=view,
			raw_csv_path=raw_csv_path,
			processed_dir=processed_dir,
			force_rebuild=force_rebuild,
		)
		pairs_df = build_dpo_pairs_from_clustered_dataframe(
			clustered_df=clustered_df,
			pairing_strategy=pairing_strategy,
			min_positive_delta=min_positive_delta,
			min_delta_margin=min_delta_margin,
			delta_components=validated_components,
			gap=gap,
			wt_pairs_frac=wt_pairs_frac,
			cross_pairs_frac=cross_pairs_frac,
			strong_pos_threshold=strong_pos_threshold,
			strong_neg_threshold=strong_neg_threshold,
			min_score_margin=min_score_margin,
			rng=delta_rng,
			random_seed=int(seed),
			source_view=view,
		)
		pairs_per_view.append(pairs_df)

	if not pairs_per_view:
		return pd.DataFrame(columns=PAIR_COLUMNS)

	all_pairs = pd.concat(pairs_per_view, ignore_index=True)
	if deduplicate_across_views and not all_pairs.empty:
		all_pairs = all_pairs.drop_duplicates(
			subset=["chosen_sequence", "rejected_sequence"],
			keep="first",
		).reset_index(drop=True)

	return all_pairs


def load_dpo_sequence_pairs(
	pairing_strategy: PairingStrategy = "positive_vs_tail",
	include_views: Sequence[str] = ("mut1", "mut2"),
	raw_csv_path: Path = None,
	processed_dir: Path = None,
	force_rebuild: bool = False,
	min_positive_delta: float = 0.0,
	min_delta_margin: float = 0.0,
	delta_components: Sequence[str] = DELTA_BASED_COMPONENTS,
	gap: float = 0.5,
	wt_pairs_frac: float = 0.1,
	cross_pairs_frac: float = 0.1,
	strong_pos_threshold: float = 1.0,
	strong_neg_threshold: float = -5.0,
	min_score_margin: float = 0.1,
	seed: int = RANDOM_SEED,
	deduplicate_across_views: bool = True,
) -> List[PairTuple]:
	"""Return DPO preference pairs as ({aa, score}, {aa, score}) tuples."""
	pairs_df = load_dpo_pair_dataframe(
		pairing_strategy=pairing_strategy,
		include_views=include_views,
		raw_csv_path=raw_csv_path,
		processed_dir=processed_dir,
		force_rebuild=force_rebuild,
		min_positive_delta=min_positive_delta,
		min_delta_margin=min_delta_margin,
		delta_components=delta_components,
		gap=gap,
		wt_pairs_frac=wt_pairs_frac,
		cross_pairs_frac=cross_pairs_frac,
		strong_pos_threshold=strong_pos_threshold,
		strong_neg_threshold=strong_neg_threshold,
		min_score_margin=min_score_margin,
		seed=seed,
		deduplicate_across_views=deduplicate_across_views,
	)
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


def _build_pairs_for_split_views(
	clustered_views: Dict[str, pd.DataFrame],
	pairing_strategy: PairingStrategy,
	include_views: Sequence[str],
	min_positive_delta: float,
	min_delta_margin: float,
	delta_components: Sequence[str],
	gap: float,
	wt_pairs_frac: float,
	cross_pairs_frac: float,
	strong_pos_threshold: float,
	strong_neg_threshold: float,
	min_score_margin: float,
	seed: int,
	deduplicate_across_views: bool,
) -> pd.DataFrame:
	"""Build one pair dataframe from a dict of per-view clustered dataframes."""
	validated_components = (
		validate_delta_based_components(delta_components)
		if pairing_strategy == "delta_based"
		else DELTA_BASED_COMPONENTS
	)
	delta_rng = np.random.default_rng(int(seed))

	pairs_per_view: List[pd.DataFrame] = []
	for view in include_views:
		clustered_df = clustered_views[view]
		pairs_df = build_dpo_pairs_from_clustered_dataframe(
			clustered_df=clustered_df,
			pairing_strategy=pairing_strategy,
			min_positive_delta=min_positive_delta,
			min_delta_margin=min_delta_margin,
			delta_components=validated_components,
			gap=gap,
			wt_pairs_frac=wt_pairs_frac,
			cross_pairs_frac=cross_pairs_frac,
			strong_pos_threshold=strong_pos_threshold,
			strong_neg_threshold=strong_neg_threshold,
			min_score_margin=min_score_margin,
			rng=delta_rng,
			random_seed=int(seed),
			source_view=view,
		)
		pairs_per_view.append(pairs_df)

	if not pairs_per_view:
		return pd.DataFrame(columns=PAIR_COLUMNS)

	all_pairs = pd.concat(pairs_per_view, ignore_index=True)
	if deduplicate_across_views and not all_pairs.empty:
		all_pairs = all_pairs.drop_duplicates(
			subset=["chosen_sequence", "rejected_sequence"],
			keep="first",
		).reset_index(drop=True)

	return all_pairs


def build_split_pair_dataframes_from_raw(
	pairing_strategy: PairingStrategy = "positive_vs_tail",
	include_views: Sequence[str] = ("mut1", "mut2"),
	raw_csv_path: Path = None,
	processed_dir: Path = None,
	force_rebuild: bool = False,
	min_positive_delta: float = 0.0,
	min_delta_margin: float = 0.0,
	delta_components: Sequence[str] = DELTA_BASED_COMPONENTS,
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
	seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""Split base ED2 rows by cluster first, then build DPO pairs within each split."""
	if pairing_strategy == "delta_based":
		validate_delta_based_components(delta_components)

	defaults = default_data_paths()
	raw_csv_path = defaults["raw_m22"] if raw_csv_path is None else Path(raw_csv_path)
	processed_dir = defaults["processed_dir"] if processed_dir is None else Path(processed_dir)
	paths = _ensure_processed_data(
		raw_csv_path=raw_csv_path,
		processed_dir=processed_dir,
		force_rebuild=force_rebuild,
		verbose=False,
	)
	base_df = load_distance2_dataframe(
		view="base",
		raw_csv_path=raw_csv_path,
		processed_dir=processed_dir,
		force_rebuild=force_rebuild,
	)
	split_membership = build_or_load_cluster_split_membership(
		base_df=base_df,
		base_csv_path=Path(paths["ed2_all"]),
		processed_dir=processed_dir,
		train_frac=train_frac,
		val_frac=val_frac,
		test_frac=test_frac,
		seed=int(seed),
		force_rebuild=force_rebuild,
		positive_threshold=0.0,
		hamming_distance=int(split_hamming_distance),
		stratify_bins=int(split_stratify_bins),
	)
	summary = summarize_split_membership(split_membership)
	LOG.info(
		"Cluster split summary | clusters=%d | cluster_size(min/median/max)=%.0f/%.0f/%.0f | "
		"seq(train/val/test)=%.0f/%.0f/%.0f | pos(train/val/test)=%.0f/%.0f/%.0f",
		int(summary["num_clusters"]),
		summary["cluster_size_min"],
		summary["cluster_size_median"],
		summary["cluster_size_max"],
		summary["num_sequences_train"],
		summary["num_sequences_val"],
		summary["num_sequences_test"],
		summary["num_positives_train"],
		summary["num_positives_val"],
		summary["num_positives_test"],
	)

	train_keys = set(split_membership.loc[split_membership["split"] == "train", "split_key"].astype(str))
	val_keys = set(split_membership.loc[split_membership["split"] == "val", "split_key"].astype(str))
	test_keys = set(split_membership.loc[split_membership["split"] == "test", "split_key"].astype(str))

	if pairing_strategy == "delta_based":
		base_keys = split_membership_keys(base_df).astype(str)
		train_pairs = build_dpo_pairs_from_clustered_dataframe(
			clustered_df=base_df.loc[base_keys.isin(train_keys)].copy(),
			pairing_strategy=pairing_strategy,
			delta_components=delta_components,
			gap=gap,
			wt_pairs_frac=wt_pairs_frac,
			cross_pairs_frac=cross_pairs_frac,
			strong_pos_threshold=strong_pos_threshold,
			strong_neg_threshold=strong_neg_threshold,
			min_score_margin=min_score_margin,
			rng=np.random.default_rng(int(seed)),
			random_seed=int(seed),
			source_view="base",
		)
		val_pairs = build_dpo_pairs_from_clustered_dataframe(
			clustered_df=base_df.loc[base_keys.isin(val_keys)].copy(),
			pairing_strategy=pairing_strategy,
			delta_components=delta_components,
			gap=gap,
			wt_pairs_frac=wt_pairs_frac,
			cross_pairs_frac=cross_pairs_frac,
			strong_pos_threshold=strong_pos_threshold,
			strong_neg_threshold=strong_neg_threshold,
			min_score_margin=min_score_margin,
			rng=np.random.default_rng(int(seed) + 1),
			random_seed=int(seed) + 1,
			source_view="base",
		)
		test_pairs = build_dpo_pairs_from_clustered_dataframe(
			clustered_df=base_df.loc[base_keys.isin(test_keys)].copy(),
			pairing_strategy=pairing_strategy,
			delta_components=delta_components,
			gap=gap,
			wt_pairs_frac=wt_pairs_frac,
			cross_pairs_frac=cross_pairs_frac,
			strong_pos_threshold=strong_pos_threshold,
			strong_neg_threshold=strong_neg_threshold,
			min_score_margin=min_score_margin,
			rng=np.random.default_rng(int(seed) + 2),
			random_seed=int(seed) + 2,
			source_view="base",
		)
		return train_pairs, val_pairs, test_pairs

	valid_views = {"mut1", "mut2"}
	if not include_views:
		raise ValueError("include_views must contain at least one of: mut1, mut2")
	for view in include_views:
		if view not in valid_views:
			raise ValueError("include_views must contain only: mut1, mut2")

	clustered_sources = {
		view: load_distance2_dataframe(
			view=view,
			raw_csv_path=raw_csv_path,
			processed_dir=processed_dir,
			force_rebuild=force_rebuild,
		)
		for view in include_views
	}

	def filtered_views(keys: set) -> Dict[str, pd.DataFrame]:
		out: Dict[str, pd.DataFrame] = {}
		for view, df in clustered_sources.items():
			row_keys = split_membership_keys(df)
			out[view] = df[row_keys.isin(keys)].copy()
		return out

	train_pairs = _build_pairs_for_split_views(
		clustered_views=filtered_views(train_keys),
		pairing_strategy=pairing_strategy,
		include_views=include_views,
		min_positive_delta=min_positive_delta,
		min_delta_margin=min_delta_margin,
		delta_components=delta_components,
		gap=gap,
		wt_pairs_frac=wt_pairs_frac,
		cross_pairs_frac=cross_pairs_frac,
		strong_pos_threshold=strong_pos_threshold,
		strong_neg_threshold=strong_neg_threshold,
		min_score_margin=min_score_margin,
		seed=int(seed),
		deduplicate_across_views=deduplicate_across_views,
	)
	val_pairs = _build_pairs_for_split_views(
		clustered_views=filtered_views(val_keys),
		pairing_strategy=pairing_strategy,
		include_views=include_views,
		min_positive_delta=min_positive_delta,
		min_delta_margin=min_delta_margin,
		delta_components=delta_components,
		gap=gap,
		wt_pairs_frac=wt_pairs_frac,
		cross_pairs_frac=cross_pairs_frac,
		strong_pos_threshold=strong_pos_threshold,
		strong_neg_threshold=strong_neg_threshold,
		min_score_margin=min_score_margin,
		seed=int(seed) + 1,
		deduplicate_across_views=deduplicate_across_views,
	)
	test_pairs = _build_pairs_for_split_views(
		clustered_views=filtered_views(test_keys),
		pairing_strategy=pairing_strategy,
		include_views=include_views,
		min_positive_delta=min_positive_delta,
		min_delta_margin=min_delta_margin,
		delta_components=delta_components,
		gap=gap,
		wt_pairs_frac=wt_pairs_frac,
		cross_pairs_frac=cross_pairs_frac,
		strong_pos_threshold=strong_pos_threshold,
		strong_neg_threshold=strong_neg_threshold,
		min_score_margin=min_score_margin,
		seed=int(seed) + 2,
		deduplicate_across_views=deduplicate_across_views,
	)

	return train_pairs, val_pairs, test_pairs

