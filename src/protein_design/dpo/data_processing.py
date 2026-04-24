import argparse
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd

from .splitting import (
    SPLIT_MEMBERSHIP_FILENAME,
    build_or_load_cluster_split_membership,
    split_membership_keys,
)

WT_M22_BINDING_ENRICHMENT = 5.190013461
DELTA_M22_BINDING_ENRICHMENT_COL = "delta_M22_binding_enrichment_adj"
M22_BINDING_ENRICHMENT_ADJ_COL = "M22_binding_enrichment_adj"

REQUIRED_RAW_COLUMNS = {
    "aa",
    "num_mut",
    "mut",
    "M22_binding_count_adj",
    "M22_non_binding_count_adj",
}

RAW_COLUMN_ALIASES = {
    "M22_binding_count_adj": ("count_ED2M22pos",),
    "M22_non_binding_count_adj": ("count_ED2M22neg",),
}

REQUIRED_ED5_COLUMNS = {
    "aa",
    "num_mut",
    "mut",
    "M22_binding_enrichment_adj",
}

ALLOWED_ED2_NUM_MUT = {2, 3, 4, 5}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _normalize_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize alternate raw-schema column names into canonical names."""
    normalized = df.copy()
    for canonical_name, aliases in RAW_COLUMN_ALIASES.items():
        if canonical_name in normalized.columns:
            continue
        for alias in aliases:
            if alias in normalized.columns:
                normalized[canonical_name] = normalized[alias]
                break
    return normalized


def ensure_delta_m22_binding_enrichment(
    df: pd.DataFrame,
    wt_binding_enrichment: float = WT_M22_BINDING_ENRICHMENT,
) -> pd.DataFrame:
    """Ensure delta enrichment exists, deriving it from adjusted enrichment if needed."""
    out = df.copy()

    if DELTA_M22_BINDING_ENRICHMENT_COL in out.columns:
        out[DELTA_M22_BINDING_ENRICHMENT_COL] = pd.to_numeric(
            out[DELTA_M22_BINDING_ENRICHMENT_COL],
            errors="coerce",
        ).astype(float)
        return out

    if M22_BINDING_ENRICHMENT_ADJ_COL not in out.columns:
        raise ValueError(
            "Raw data is missing required enrichment column(s): "
            f"expected '{DELTA_M22_BINDING_ENRICHMENT_COL}' or "
            f"'{M22_BINDING_ENRICHMENT_ADJ_COL}'."
        )

    enrichment = pd.to_numeric(out[M22_BINDING_ENRICHMENT_ADJ_COL], errors="coerce").astype(float)
    out[M22_BINDING_ENRICHMENT_ADJ_COL] = enrichment
    out[DELTA_M22_BINDING_ENRICHMENT_COL] = enrichment - float(wt_binding_enrichment)
    return out


def _read_raw_data(raw_csv_path: Union[str, Path]) -> pd.DataFrame:
    """Read raw M22 data and validate required columns."""
    raw_csv_path = Path(raw_csv_path)
    if not raw_csv_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_csv_path}")

    df = _normalize_raw_columns(pd.read_csv(raw_csv_path))
    missing_cols = REQUIRED_RAW_COLUMNS.difference(df.columns)
    if missing_cols:
        missing = ", ".join(sorted(missing_cols))
        raise ValueError(f"Raw data file is missing required columns: {missing}")
    return ensure_delta_m22_binding_enrichment(df)


def get_ed2_all_data(raw_input: Union[pd.DataFrame, str, Path]) -> Tuple[pd.DataFrame, float, float]:
    """Extract ED2 rows with num_mut in {2,3,4,5} and global binding/non-binding totals.

    The totals are computed on the full raw dataset (not only distance-2),
    matching your current enrichment normalization logic.
    """
    if isinstance(raw_input, pd.DataFrame):
        df = ensure_delta_m22_binding_enrichment(_normalize_raw_columns(raw_input))
    else:
        df = _read_raw_data(raw_input)

    df = df.copy()
    df["num_mut"] = pd.to_numeric(df["num_mut"], errors="coerce")
    df["M22_binding_count_adj"] = pd.to_numeric(df["M22_binding_count_adj"], errors="coerce")
    df["M22_non_binding_count_adj"] = pd.to_numeric(df["M22_non_binding_count_adj"], errors="coerce")
    df_filtered = df[df["num_mut"].isin(ALLOWED_ED2_NUM_MUT)].copy()
    n_bind = float(df["M22_binding_count_adj"].sum())
    n_non_bind = float(df["M22_non_binding_count_adj"].sum())
    return df_filtered, n_bind, n_non_bind


def get_distance2_data(raw_input: Union[pd.DataFrame, str, Path]) -> Tuple[pd.DataFrame, float, float]:
    """Backward-compatible alias returning ED2 rows with num_mut in {2,3,4,5}."""
    return get_ed2_all_data(raw_input)


def build_perplexity_eval_sets(
    df_val: pd.DataFrame,
    cfg: Any,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build fixed-size val_pos/val_neg subsets from the validation split."""
    df_val = ensure_delta_m22_binding_enrichment(df_val)

    min_positive_delta = float(cfg.data.min_positive_delta)
    n_val_pos = int(getattr(cfg.data.val, "n_val_pos", 200))
    n_val_neg = int(getattr(cfg.data.val, "n_val_neg", 400))

    delta_scores = pd.to_numeric(df_val[DELTA_M22_BINDING_ENRICHMENT_COL], errors="coerce")
    valid_rows = df_val.loc[delta_scores.notna()].copy()
    valid_rows[DELTA_M22_BINDING_ENRICHMENT_COL] = delta_scores.loc[delta_scores.notna()].astype(float)

    pos_pool = valid_rows[valid_rows[DELTA_M22_BINDING_ENRICHMENT_COL] > min_positive_delta]
    neg_pool = valid_rows[valid_rows[DELTA_M22_BINDING_ENRICHMENT_COL] < 0.0]

    if n_val_pos > 0 and len(pos_pool) > 0:
        val_pos = pos_pool.sample(
            n=min(n_val_pos, len(pos_pool)),
            replace=False,
            random_state=int(seed),
        ).reset_index(drop=True)
    else:
        val_pos = pos_pool.head(0).copy()

    if n_val_neg > 0 and len(neg_pool) > 0:
        val_neg = neg_pool.sample(
            n=min(n_val_neg, len(neg_pool)),
            replace=False,
            random_state=int(seed) + 1,
        ).reset_index(drop=True)
    else:
        val_neg = neg_pool.head(0).copy()

    return val_pos, val_neg


def clean_ed5_dataframe(raw_input: Union[pd.DataFrame, str, Path]) -> pd.DataFrame:
    """Clean ED5 raw data analogous to D2 preprocessing conventions."""
    if isinstance(raw_input, pd.DataFrame):
        df = raw_input.copy()
    else:
        raw_path = Path(raw_input)
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {raw_path}")
        df = pd.read_csv(raw_path)

    missing_cols = REQUIRED_ED5_COLUMNS.difference(df.columns)
    if missing_cols:
        missing = ", ".join(sorted(missing_cols))
        raise ValueError(f"ED5 raw file is missing required columns: {missing}")

    cleaned = df.replace(r"^\s*$", np.nan, regex=True).copy()
    cleaned["num_mut"] = pd.to_numeric(cleaned["num_mut"], errors="coerce")
    cleaned["M22_binding_enrichment_adj"] = pd.to_numeric(
        cleaned["M22_binding_enrichment_adj"],
        errors="coerce",
    )

    cleaned = cleaned.dropna(subset=["aa", "num_mut", "mut", "M22_binding_enrichment_adj"]).copy()
    cleaned["aa"] = cleaned["aa"].astype(str)
    cleaned["mut"] = cleaned["mut"].astype(str)
    cleaned = cleaned[cleaned["num_mut"] == 2].reset_index(drop=True)
    return cleaned


def build_clean_ed5_csv(
    raw_csv_path: Union[str, Path],
    processed_dir: Union[str, Path],
    force: bool = False,
    verbose: bool = True,
) -> Path:
    """Build D5.csv from raw ED5 data if missing or stale."""
    raw_csv_path = Path(raw_csv_path)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    output_path = processed_dir / "D5.csv"
    raw_mtime = raw_csv_path.stat().st_mtime
    output_is_fresh = output_path.exists() and output_path.stat().st_mtime >= raw_mtime
    should_rebuild = bool(force) or (not output_path.exists()) or (not output_is_fresh)

    if not should_rebuild:
        if verbose:
            print("D5.csv is up to date. Reusing existing file.")
        return output_path

    cleaned = clean_ed5_dataframe(raw_csv_path)
    cleaned.to_csv(output_path, index=False)

    if verbose:
        print(f"ED5 cleaned rows (num_mut==2 and non-null enrichment): {len(cleaned)}")
        print(f"Wrote: {output_path}")

    return output_path


def build_validation_perplexity_csvs(
    raw_csv_path: Union[str, Path],
    processed_dir: Union[str, Path],
    cfg: Any,
    seed: int,
    force: bool = False,
    verbose: bool = True,
) -> Dict[str, Path]:
    """Build validation eval CSVs from the cluster-based validation split of ED2."""
    raw_csv_path = Path(raw_csv_path)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "val_pos": processed_dir / "val_pos.csv",
        "val_neg": processed_dir / "val_neg.csv",
        "val_spearman": processed_dir / "val_spearman.csv",
    }

    processed_paths = build_processed_views(
        raw_csv_path=raw_csv_path,
        processed_dir=processed_dir,
        force=bool(getattr(cfg.data, "force_rebuild", False)),
        verbose=False,
    )
    base_df = pd.read_csv(processed_paths["ed2_all"])
    split_cfg = getattr(cfg.data, "split", None)
    split_membership = build_or_load_cluster_split_membership(
        base_df=base_df,
        base_csv_path=Path(processed_paths["ed2_all"]),
        processed_dir=processed_dir,
        train_frac=float(cfg.data.train_frac),
        val_frac=float(cfg.data.val_frac),
        test_frac=float(cfg.data.test_frac),
        seed=int(seed),
        force_rebuild=bool(force) or bool(getattr(cfg.data, "force_rebuild", False)),
        positive_threshold=0.0,
        stratify_bins=int(getattr(split_cfg, "stratify_bins", 10)),
        hamming_distance=int(getattr(split_cfg, "hamming_distance", 1)),
    )
    raw_mtime = raw_csv_path.stat().st_mtime
    split_mtime = (processed_dir / SPLIT_MEMBERSHIP_FILENAME).stat().st_mtime
    freshness_ref = max(raw_mtime, split_mtime)
    all_outputs_exist = all(path.exists() for path in output_paths.values())
    outputs_are_fresh = all(
        path.stat().st_mtime >= freshness_ref for path in output_paths.values() if path.exists()
    )
    should_rebuild = bool(force) or (not all_outputs_exist) or (not outputs_are_fresh)
    if not should_rebuild:
        if verbose:
            print("Validation perplexity CSVs are up to date. Reusing existing files.")
        return output_paths

    val_keys = set(
        split_membership.loc[split_membership["split"] == "val", "split_key"]
        .astype(str)
        .tolist()
    )
    base_keys = split_membership_keys(base_df).astype(str)
    df_val = base_df.loc[base_keys.isin(val_keys)].reset_index(drop=True)
    val_pos, val_neg = build_perplexity_eval_sets(df_val=df_val, cfg=cfg, seed=int(seed))
    # Spearman validation should cover the full validation split distribution
    # (no delta-score filtering, no num_mut restriction).
    val_spearman = df_val.copy()
    # keep only rows with num_mut = 2
    val_spearman = val_spearman[val_spearman["num_mut"] == 2].reset_index(drop=True)

    val_pos.to_csv(output_paths["val_pos"], index=False)
    val_neg.to_csv(output_paths["val_neg"], index=False)
    val_spearman.to_csv(output_paths["val_spearman"], index=False)

    if verbose:
        print(
            "Validation perplexity sets: "
            f"val_pos={len(val_pos)} val_neg={len(val_neg)}"
        )
        print(f"Wrote: {output_paths['val_pos']}")
        print(f"Wrote: {output_paths['val_neg']}")
        print(f"Wrote: {output_paths['val_spearman']}")

    return output_paths


def d2_stats(df_d2: pd.DataFrame) -> Dict[str, float]:
    """Compute summary stats for the distance-2 dataset."""
    df = df_d2.copy()
    stats: Dict[str, float] = {"total_entries": float(len(df))}

    muts = df["mut"].str.split(";", expand=True)
    if muts.shape[1] != 2:
        raise ValueError("Could not split mutations into exactly two parts.")

    df["mut1"] = muts[0]
    df["mut2"] = muts[1]

    clusters_mut1 = df.groupby("mut1").size()
    clusters_mut2 = df.groupby("mut2").size()

    pairs_mut1 = int(sum(n * (n - 1) // 2 for n in clusters_mut1))
    pairs_mut2 = int(sum(n * (n - 1) // 2 for n in clusters_mut2))

    stats.update(
        {
            "num_clusters_mut1": float(len(clusters_mut1)),
            "avg_cluster_size_mut1": float(clusters_mut1.mean()),
            "min_cluster_size_mut1": float(clusters_mut1.min()),
            "max_cluster_size_mut1": float(clusters_mut1.max()),
            "pairs_sharing_one_mutation": float(pairs_mut1 + pairs_mut2),
        }
    )
    return stats


def _compute_m22_binding_enrichment(
    bind_count_adj: pd.Series,
    non_bind_count_adj: pd.Series,
    n_bind: float,
    n_non_bind: float,
) -> pd.Series:
    """Compute log2 enrichment."""
    return (
        np.log2(bind_count_adj)
        - np.log2(non_bind_count_adj)
        - np.log2(n_bind)
        + np.log2(n_non_bind)
    )


def organize_and_cluster(
    df_d2: pd.DataFrame,
    cluster_by: int = 1,
    n_bind: float = None,
    n_non_bind: float = None,
) -> pd.DataFrame:
    """Cluster distance-2 data by first or second mutation and sort by enrichment."""
    if cluster_by not in (1, 2):
        raise ValueError("cluster_by must be 1 or 2")

    df = df_d2.copy()
    mut_idx = cluster_by - 1
    target_mut = f"mut{cluster_by}"

    mut_parts = df["mut"].str.split(";", expand=True)
    if mut_parts.shape[1] != 2:
        raise ValueError("Could not split mutations into exactly two parts.")
    df[target_mut] = mut_parts[mut_idx]

    unique_clusters = df[target_mut].unique()
    cluster_mapping = {mut: idx for idx, mut in enumerate(unique_clusters)}
    df["cluster_idx"] = df[target_mut].map(cluster_mapping)

    bind_count_adj = df["M22_binding_count_adj"]
    non_bind_count_adj = df["M22_non_binding_count_adj"]
    n_binding = n_bind if n_bind is not None else float(bind_count_adj.sum())
    n_non_binding = n_non_bind if n_non_bind is not None else float(non_bind_count_adj.sum())

    df["M22_binding_enrichment"] = _compute_m22_binding_enrichment(
        bind_count_adj=bind_count_adj,
        non_bind_count_adj=non_bind_count_adj,
        n_bind=n_binding,
        n_non_bind=n_non_binding,
    )

    df = df.sort_values(
        by=["cluster_idx", "M22_binding_enrichment"],
        ascending=[True, False],
    )

    if "M22_binding_enrichment_adj" in df.columns:
        df = df.drop(columns=["M22_binding_enrichment_adj"])

    ordered_cols = [
        "Unnamed: 0",
        "aa",
        "num_mut",
        "mut",
        "M22_binding_count_adj",
        "M22_non_binding_count_adj",
        "delta_M22_binding_enrichment_adj",
        "M22_binding_enrichment",
        "cluster_idx",
    ]
    final_cols = [col for col in ordered_cols if col in df.columns]
    return df[final_cols]


def build_processed_views(
    raw_csv_path: Union[str, Path],
    processed_dir: Union[str, Path],
    force: bool = False,
    verbose: bool = True,
) -> Dict[str, Path]:
    """Build ED2-all and distance-2 clustered views from raw data if missing or stale.

    Returns paths to:
    - ED2_all.csv
    - D2_clustered_mut1.csv
    - D2_clustered_mut2.csv
    """
    raw_csv_path = Path(raw_csv_path)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "ed2_all": processed_dir / "ED2_all.csv",
        "d2_clustered_mut1": processed_dir / "D2_clustered_mut1.csv",
        "d2_clustered_mut2": processed_dir / "D2_clustered_mut2.csv",
    }

    required_output_columns = {
        "ed2_all": {"aa", "num_mut", "mut", DELTA_M22_BINDING_ENRICHMENT_COL, "M22_binding_enrichment_adj"},
        "d2_clustered_mut1": {
            "aa",
            "num_mut",
            "mut",
            DELTA_M22_BINDING_ENRICHMENT_COL,
            "cluster_idx",
        },
        "d2_clustered_mut2": {
            "aa",
            "num_mut",
            "mut",
            DELTA_M22_BINDING_ENRICHMENT_COL,
            "cluster_idx",
        },
    }

    outputs_have_expected_schema = True
    for key, output_path in output_paths.items():
        if not output_path.exists():
            outputs_have_expected_schema = False
            break
        try:
            columns = set(pd.read_csv(output_path, nrows=0).columns)
        except Exception:
            outputs_have_expected_schema = False
            break
        if not required_output_columns[key].issubset(columns):
            outputs_have_expected_schema = False
            break

    raw_mtime = raw_csv_path.stat().st_mtime
    all_outputs_exist = all(path.exists() for path in output_paths.values())
    outputs_are_fresh = all(
        path.stat().st_mtime >= raw_mtime for path in output_paths.values() if path.exists()
    )

    should_rebuild = (
        force
        or (not all_outputs_exist)
        or (not outputs_are_fresh)
        or (not outputs_have_expected_schema)
    )
    if not should_rebuild:
        if verbose:
            print("Processed views are up to date. Reusing existing files.")
        return output_paths

    if verbose:
        print("Building processed views from raw M22 data...")

    raw_df = _read_raw_data(raw_csv_path)
    df_ed2_all, n_bind, n_non_bind = get_ed2_all_data(raw_df)
    df_d2 = df_ed2_all[df_ed2_all["num_mut"] == 2].copy()

    output_paths["ed2_all"].parent.mkdir(parents=True, exist_ok=True)
    df_ed2_all.to_csv(output_paths["ed2_all"], index=False)

    d2_clustered_mut1 = organize_and_cluster(
        df_d2,
        cluster_by=1,
        n_bind=n_bind,
        n_non_bind=n_non_bind,
    )
    d2_clustered_mut1.to_csv(output_paths["d2_clustered_mut1"], index=False)

    d2_clustered_mut2 = organize_and_cluster(
        df_d2,
        cluster_by=2,
        n_bind=n_bind,
        n_non_bind=n_non_bind,
    )
    d2_clustered_mut2.to_csv(output_paths["d2_clustered_mut2"], index=False)

    if verbose:
        stats = d2_stats(df_d2)
        print(
            "ED2 stats: "
            f"entries_all_mut={len(df_ed2_all)}, "
            f"entries_num_mut2={int(stats['total_entries'])}, "
            f"clusters(mut1)={int(stats['num_clusters_mut1'])}, "
            f"avg_cluster_size(mut1)={stats['avg_cluster_size_mut1']:.2f}"
        )
        print(f"Wrote: {output_paths['ed2_all']}")
        print(f"Wrote: {output_paths['d2_clustered_mut1']}")
        print(f"Wrote: {output_paths['d2_clustered_mut2']}")

    return output_paths


def load_processed_views(
    raw_csv_path: Union[str, Path],
    processed_dir: Union[str, Path],
    force: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Ensure processed views exist, then load them into memory."""
    paths = build_processed_views(raw_csv_path, processed_dir, force=force, verbose=False)
    return {
        "ed2_all": pd.read_csv(paths["ed2_all"]),
        "d2_clustered_mut1": pd.read_csv(paths["d2_clustered_mut1"]),
        "d2_clustered_mut2": pd.read_csv(paths["d2_clustered_mut2"]),
    }


if __name__ == "__main__":
    root = _project_root()
    parser = argparse.ArgumentParser(description="Build processed ED2 views from raw M22 data.")
    parser.add_argument(
        "--raw",
        type=Path,
        default=root / "data" / "raw" / "ED2_M22_binding_enrichment.csv",
        help="Path to raw M22 CSV.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=root / "data" / "processed",
        help="Directory to store processed CSVs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild outputs even if processed files are fresh.",
    )
    args = parser.parse_args()

    build_processed_views(
        raw_csv_path=args.raw,
        processed_dir=args.processed_dir,
        force=args.force,
        verbose=True,
    )

