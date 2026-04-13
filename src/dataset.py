"""Dataset loading helpers for DPO workflow.

This module integrates preprocessing so callers can rely on raw data only.
If processed D2 files are missing or stale, they are rebuilt automatically.
"""

from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple, TypedDict

import pandas as pd
import numpy as np

if __package__:
	from .data_processing import build_processed_views
	from .utils import WILD_TYPE, _gap_pairs
else:  # pragma: no cover
	from data_processing import build_processed_views
	from utils import WILD_TYPE, _gap_pairs

RANDOM_SEED = 42


PairingStrategy = Literal["positive_vs_tail", "positive_only_extremes", "both_structured", "delta_based"]


class PairMember(TypedDict):
	aa: str
	score: float


PairTuple = Tuple[PairMember, PairMember]


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
	return Path(__file__).resolve().parents[1]


def default_data_paths() -> Dict[str, Path]:
	"""Return default raw and processed data paths."""
	root = _project_root()
	return {
		"raw_m22": root / "data" / "raw" / "M22_binding_enrichment.csv",
		"processed_dir": root / "data" / "processed",
	}


def _ensure_processed_data(
	raw_csv_path: Path = None,
	processed_dir: Path = None,
	force_rebuild: bool = False,
	verbose: bool = False,
) -> Dict[str, Path]:
	"""Ensure D2 processed views exist and return their paths."""
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
	"""Load one distance-2 dataframe view (base, mut1, or mut2)."""
	paths = _ensure_processed_data(
		raw_csv_path=raw_csv_path,
		processed_dir=processed_dir,
		force_rebuild=force_rebuild,
		verbose=False,
	)

	view_to_key = {
		"base": "d2",
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


def _pair_delta_based(
    sequences_df: pd.DataFrame,
    delta_col: str,
    seq_col: str,
    gap: float,
    wt_pairs_frac: float,
    cross_pairs_frac: float = 0.1,
    strong_pos_threshold: float = 1.0,
    strong_neg_threshold: float = -5.0,
    min_score_margin: float = 0.1,
) -> List[PairTuple]:

    n = len(sequences_df)
    if n <= 1:
        return []

    positives = (sequences_df[sequences_df[delta_col].astype(float) > 0]
                 .sort_values(delta_col, ascending=False).reset_index(drop=True))
    negatives = (sequences_df[sequences_df[delta_col].astype(float) < 0]
                 .sort_values(delta_col, ascending=False).reset_index(drop=True))

    wt = {"aa": WILD_TYPE, "score": 0.0}
    all_pairs: List[PairTuple] = []

    # 1 Within-positive pairs (fine-grained: good vs better) 
    all_pairs.extend(_gap_pairs(positives, delta_col, seq_col, gap))

    # 2 Within-negative pairs (subsampled to ~2x positives) 
    neg_sub_n = min(len(negatives), 2 * len(positives))
    neg_sub = (negatives.sample(n=neg_sub_n, replace=False)
               .sort_values(delta_col, ascending=False).reset_index(drop=True))
    all_pairs.extend(_gap_pairs(neg_sub, delta_col, seq_col, gap))

    # 3 WT-anchored pairs 
    num_wt = int(wt_pairs_frac * n)
    num_pos_wt = num_wt // 2
    num_neg_wt = num_wt - num_pos_wt

    strong_pos = positives[positives[delta_col].astype(float) > strong_pos_threshold]
    if len(strong_pos) > 0 and num_pos_wt > 0:
        sampled = strong_pos.sample(n=min(num_pos_wt, len(strong_pos)), replace=False)
        for _, row in sampled.iterrows():
            winner = {"aa": row[seq_col], "score": float(row[delta_col])}
            all_pairs.append((winner, wt))

    strong_neg = negatives[negatives[delta_col].astype(float) < strong_neg_threshold]
    if len(strong_neg) > 0 and num_neg_wt > 0:
        sampled = strong_neg.sample(n=min(num_neg_wt, len(strong_neg)), replace=False)
        for _, row in sampled.iterrows():
            loser = {"aa": row[seq_col], "score": float(row[delta_col])}
            all_pairs.append((wt, loser))

    # 4 Cross-class pairs (positive vs negative)
    num_cross = int(cross_pairs_frac * n)
    if len(positives) > 0 and len(negatives) > 0 and num_cross > 0:
        cross_pos = positives.sample(n=min(num_cross, len(positives)),
                                     replace=len(positives) < num_cross)
        cross_neg = negatives.sample(n=min(num_cross, len(negatives)), replace=False)
        for p_row, n_row in zip(cross_pos.itertuples(), cross_neg.itertuples()):
            winner = {"aa": getattr(p_row, seq_col), "score": float(getattr(p_row, delta_col))}
            loser  = {"aa": getattr(n_row, seq_col), "score": float(getattr(n_row, delta_col))}
            all_pairs.append((winner, loser))

    # Filter noisy pairs with tiny score differences 
    all_pairs = [(w, l) for w, l in all_pairs
                 if (w["score"] - l["score"]) >= min_score_margin]

    return all_pairs

		
def build_dpo_pairs_from_clustered_dataframe(
	clustered_df: pd.DataFrame,
	pairing_strategy: PairingStrategy = "positive_vs_tail", 	# "positive_only_extremes", "both_structured" or "delta_based"
	min_positive_delta: float = 0.0,
	min_delta_margin: float = 0.0,
	gap: float = 0.5,
	wt_pairs_frac: float = 0.1,
	cross_pairs_frac: float = 0.1,
	strong_pos_threshold: float = 1.0,
	strong_neg_threshold: float = -5.0,
	min_score_margin: float = 0.1,
	source_view: str = "",
) -> pd.DataFrame:
	"""Build DPO preference pairs from one clustered dataframe."""
	if pairing_strategy not in ("positive_vs_tail", "positive_only_extremes", "both_structured", "delta_based"):
		raise ValueError("pairing_strategy must be positive_vs_tail, positive_only_extremes, both_structured, or delta_based")

	seq_col = "aa"
	delta_col = "delta_M22_binding_enrichment_adj"
	cluster_col = "cluster_idx"
	required_cols = {seq_col, delta_col, cluster_col}
	missing_cols = required_cols.difference(clustered_df.columns)
	if missing_cols:
		missing = ", ".join(sorted(missing_cols))
		raise ValueError(f"clustered_df is missing required columns: {missing}")

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
		elif pairing_strategy == "delta_based":
			cluster_pairs = _pair_delta_based(
				sequences_df=cluster_df,
				delta_col=delta_col,
				seq_col=seq_col,
				gap=gap, 
				wt_pairs_frac=wt_pairs_frac, 
				cross_pairs_frac=cross_pairs_frac,
				strong_pos_threshold=strong_pos_threshold,
				strong_neg_threshold=strong_neg_threshold,
				min_score_margin=min_score_margin,
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
	gap: float = 0.5,
	wt_pairs_frac: float = 0.1,
	cross_pairs_frac: float = 0.1,
	strong_pos_threshold: float = 1.0,
	strong_neg_threshold: float = -5.0,
	min_score_margin: float = 0.1,
	deduplicate_across_views: bool = True,
) -> pd.DataFrame:
	"""Load clustered views and build a DPO pair dataframe."""

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
			gap=gap,
			wt_pairs_frac=wt_pairs_frac,
			cross_pairs_frac=cross_pairs_frac,
			strong_pos_threshold=strong_pos_threshold,
			strong_neg_threshold=strong_neg_threshold,
			min_score_margin=min_score_margin,
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
	gap: float = 0.5,
	wt_pairs_frac: float = 0.1,
	cross_pairs_frac: float = 0.1,
	strong_pos_threshold: float = 1.0,
	strong_neg_threshold: float = -5.0,
	min_score_margin: float = 0.1,
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
		gap=gap,
		wt_pairs_frac=wt_pairs_frac,
		cross_pairs_frac=cross_pairs_frac,
		strong_pos_threshold=strong_pos_threshold,
		strong_neg_threshold=strong_neg_threshold,
		min_score_margin=min_score_margin,
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


def _split_membership_keys(df: pd.DataFrame) -> pd.Series:
	"""Return stable row keys used to map base split membership into clustered views."""
	if "Unnamed: 0" in df.columns:
		return df["Unnamed: 0"].astype(str)

	if {"aa", "mut"}.issubset(df.columns):
		return df["aa"].astype(str) + "||" + df["mut"].astype(str)

	if "aa" in df.columns:
		return df["aa"].astype(str)

	raise ValueError(
		"Cannot infer split membership keys. Expected one of: 'Unnamed: 0', ('aa' and 'mut'), or 'aa'."
	)


def _build_pairs_for_split_views(
	clustered_views: Dict[str, pd.DataFrame],
	pairing_strategy: PairingStrategy,
	include_views: Sequence[str],
	min_positive_delta: float,
	min_delta_margin: float,
	gap: float,
	wt_pairs_frac: float,
	cross_pairs_frac: float,
	strong_pos_threshold: float,
	strong_neg_threshold: float,
	min_score_margin: float,
	deduplicate_across_views: bool,
) -> pd.DataFrame:
	"""Build one pair dataframe from a dict of per-view clustered dataframes."""
	pairs_per_view: List[pd.DataFrame] = []
	for view in include_views:
		clustered_df = clustered_views[view]
		pairs_df = build_dpo_pairs_from_clustered_dataframe(
			clustered_df=clustered_df,
			pairing_strategy=pairing_strategy,
			min_positive_delta=min_positive_delta,
			min_delta_margin=min_delta_margin,
			gap=gap,
			wt_pairs_frac=wt_pairs_frac,
			cross_pairs_frac=cross_pairs_frac,
			strong_pos_threshold=strong_pos_threshold,
			strong_neg_threshold=strong_neg_threshold,
			min_score_margin=min_score_margin,
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
	seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""Split base D2 rows first, then build DPO pairs inside each split independently."""
	valid_views = {"mut1", "mut2"}
	if not include_views:
		raise ValueError("include_views must contain at least one of: mut1, mut2")
	for view in include_views:
		if view not in valid_views:
			raise ValueError("include_views must contain only: mut1, mut2")

	base_df = load_distance2_dataframe(
		view="base",
		raw_csv_path=raw_csv_path,
		processed_dir=processed_dir,
		force_rebuild=force_rebuild,
	)
	train_base, val_base, test_base = create_train_val_test_split(
		base_df,
		train_frac=train_frac,
		val_frac=val_frac,
		test_frac=test_frac,
		seed=seed,
	)

	train_keys = set(_split_membership_keys(train_base).tolist())
	val_keys = set(_split_membership_keys(val_base).tolist())
	test_keys = set(_split_membership_keys(test_base).tolist())

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
			row_keys = _split_membership_keys(df)
			out[view] = df[row_keys.isin(keys)].copy()
		return out

	train_pairs = _build_pairs_for_split_views(
		clustered_views=filtered_views(train_keys),
		pairing_strategy=pairing_strategy,
		include_views=include_views,
		min_positive_delta=min_positive_delta,
		min_delta_margin=min_delta_margin,
		gap=gap,
		wt_pairs_frac=wt_pairs_frac,
		cross_pairs_frac=cross_pairs_frac,
		strong_pos_threshold=strong_pos_threshold,
		strong_neg_threshold=strong_neg_threshold,
		min_score_margin=min_score_margin,
		deduplicate_across_views=deduplicate_across_views,
	)
	val_pairs = _build_pairs_for_split_views(
		clustered_views=filtered_views(val_keys),
		pairing_strategy=pairing_strategy,
		include_views=include_views,
		min_positive_delta=min_positive_delta,
		min_delta_margin=min_delta_margin,
		gap=gap,
		wt_pairs_frac=wt_pairs_frac,
		cross_pairs_frac=cross_pairs_frac,
		strong_pos_threshold=strong_pos_threshold,
		strong_neg_threshold=strong_neg_threshold,
		min_score_margin=min_score_margin,
		deduplicate_across_views=deduplicate_across_views,
	)
	test_pairs = _build_pairs_for_split_views(
		clustered_views=filtered_views(test_keys),
		pairing_strategy=pairing_strategy,
		include_views=include_views,
		min_positive_delta=min_positive_delta,
		min_delta_margin=min_delta_margin,
		gap=gap,
		wt_pairs_frac=wt_pairs_frac,
		cross_pairs_frac=cross_pairs_frac,
		strong_pos_threshold=strong_pos_threshold,
		strong_neg_threshold=strong_neg_threshold,
		min_score_margin=min_score_margin,
		deduplicate_across_views=deduplicate_across_views,
	)

	return train_pairs, val_pairs, test_pairs


def create_train_val_test_split(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into train/val/test sets with fixed seed.

    Args:
        df: Input dataframe
        train_frac: Fraction for training set
        val_frac: Fraction for validation set
        test_frac: Fraction for test set
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n = len(df_shuffled)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train_df = df_shuffled.iloc[:train_end].reset_index(drop=True)
    val_df = df_shuffled.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df_shuffled.iloc[val_end:].reset_index(drop=True)

    return train_df, val_df, test_df
