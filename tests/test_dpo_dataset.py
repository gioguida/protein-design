"""
Test DPO dataset building with configuration matching conf/data/dpo/default.yaml
Run directly with: python tests/test_dpo_dataset.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from typing import List, Optional
import pandas as pd

from protein_design.dpo.dataset import build_split_pair_dataframes_from_raw

class DataConfig:
    """Config mimicking conf/data/dpo/default.yaml for easy parameter testing."""
    # File paths
    raw_csv: Optional[str] = None
    processed_dir: Optional[str] = None
    
    # Building strategy
    pairing_strategy: str = "delta_based"
    force_rebuild: bool = False
    include_views: List[str] = ["mut1", "mut2"]
    deduplicate_across_views: bool = True
    
    # Delta-based specific settings
    delta_components: List[str] = ["cross", "wt_anchors"]
    gap: float = 0.5
    wt_pairs_frac: float = 0.1
    cross_pairs_frac: float = 0.03
    strong_pos_threshold: float = 1.0
    strong_neg_threshold: float = -5.0
    min_score_margin: float = 0.1
    min_positive_delta: float = 3.0
    min_delta_margin: float = 5.0
    
    # Default splitting configurations
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1
    split_hamming_distance: int = 1
    split_stratify_bins: int = 10
    seed: int = 42


def print_stats(name: str, df: pd.DataFrame):
    if len(df) == 0:
        print(f"\n[{name} Set] - EMPTY")
        return
        
    print(f"\n{'='*40}")
    print(f"{name} Set Statistics")
    print(f"{'='*40}")
    print(f"Total pairs: {len(df):,}")
    
    if "delta_margin" in df.columns:
        margin = df["delta_margin"]
        print(f"\nDelta Margin Base Stats:")
        print(f"  Mean:   {margin.mean():.4f}")
        print(f"  Median: {margin.median():.4f}")
        print(f"  Min:    {margin.min():.4f}")
        print(f"  Max:    {margin.max():.4f}")
        
    if "pairing_strategy" in df.columns:
        print(f"\nPairing Strategies Distribution:")
        counts = df["pairing_strategy"].value_counts()
        for strategy, count in counts.items():
            print(f"  - {strategy}: {count:,}")
            
    if "delta_component" in df.columns:
        print(f"\nDelta Components Distribution:")
        counts = df["delta_component"].value_counts()
        for component, count in counts.items():
            print(f"  - {component}: {count:,}")
            
    if "source_view" in df.columns:
        print(f"\nSource Views Distribution:")
        counts = df["source_view"].value_counts()
        for view, count in counts.items():
            print(f"  - {view}: {count:,}")


def main():
    cfg = DataConfig()
    
    print("Building DPO datasets with configuration:")
    print(f"  Strategy:  {cfg.pairing_strategy}")
    print(f"  Views:     {cfg.include_views}")
    print(f"  Splits:    Train {cfg.train_frac} / Val {cfg.val_frac} / Test {cfg.test_frac}")
    print("...\n")

    train_pairs, val_pairs, test_pairs = build_split_pair_dataframes_from_raw(
        pairing_strategy=cfg.pairing_strategy,
        include_views=cfg.include_views,
        raw_csv_path=Path(cfg.raw_csv) if cfg.raw_csv else None,
        processed_dir=Path(cfg.processed_dir) if cfg.processed_dir else None,
        force_rebuild=cfg.force_rebuild,
        min_positive_delta=cfg.min_positive_delta,
        min_delta_margin=cfg.min_delta_margin,
        delta_components=cfg.delta_components,
        gap=cfg.gap,
        wt_pairs_frac=cfg.wt_pairs_frac,
        cross_pairs_frac=cfg.cross_pairs_frac,
        strong_pos_threshold=cfg.strong_pos_threshold,
        strong_neg_threshold=cfg.strong_neg_threshold,
        min_score_margin=cfg.min_score_margin,
        deduplicate_across_views=cfg.deduplicate_across_views,
        train_frac=cfg.train_frac,
        val_frac=cfg.val_frac,
        test_frac=cfg.test_frac,
        split_hamming_distance=cfg.split_hamming_distance,
        split_stratify_bins=cfg.split_stratify_bins,
        seed=cfg.seed,
    )

    print("\n" + "#" * 50)
    print(f"DATASET BUILDING COMPLETE")
    print(f"Total Combined Pairs: {len(train_pairs) + len(val_pairs) + len(test_pairs):,}")
    print("#" * 50)

    print_stats("Train", train_pairs)
    print_stats("Validation", val_pairs)
    print_stats("Test", test_pairs)


if __name__ == "__main__":
    main()
