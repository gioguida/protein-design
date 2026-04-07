import argparse
import sys
from pathlib import Path

class test_config:
    def __init__(self):
        self.pairing_strategy = "both_structured"  # "positive_vs_tail", "positive_only_extremes"
        self.preview_count = 0
        self.include_views = ("mut1", "mut2")
        self.force_rebuild = False
        self.min_positive_delta = 3.0
        self.min_delta_margin = 5.0
        self.deduplicate_across_views = True

def _add_repo_root_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_add_repo_root_to_path()

from src.dataset import default_data_paths, load_dpo_pair_dataframe


def main() -> None:
    args = test_config()
    paths = default_data_paths()
    pairs_df = load_dpo_pair_dataframe(
        pairing_strategy=args.pairing_strategy,
        include_views=args.include_views,
        raw_csv_path=paths["raw_m22"],
        processed_dir=paths["processed_dir"],
        force_rebuild=args.force_rebuild,
        min_positive_delta=args.min_positive_delta,
        min_delta_margin=args.min_delta_margin,
        deduplicate_across_views=args.deduplicate_across_views,
    )

    if pairs_df.empty:
        print("No pairs available after preprocessing.")
        return
    
    print(f"Pairing strategy: {args.pairing_strategy}")
    print(f"min_positive_delta: {args.min_positive_delta} | min_delta_margin: {args.min_delta_margin}")

    margins = pairs_df["delta_margin"].astype(float)
    print(
        f"Pair stats | n={len(pairs_df)} | "
        f"margin mean={margins.mean():.4f} median={margins.median():.4f} "
        f"min={margins.min():.4f} max={margins.max():.4f}"
    )

    by_view = pairs_df.groupby("source_view").size().sort_values(ascending=False)
    print("Pairs per view: " + ", ".join(f"{k}:{int(v)}" for k, v in by_view.items()))

    show_n = min(max(0, int(args.preview_count)), len(pairs_df))
    if show_n == 0:
        return

    top_examples = pairs_df.nlargest(show_n, "delta_margin")
    low_examples = pairs_df.nsmallest(show_n, "delta_margin")

    print("Top margin examples:")
    for _, row in top_examples.iterrows():
        print(
            f"  view={row['source_view']} cluster={row['cluster_idx']} "
            f"margin={float(row['delta_margin']):.4f} "
            f"chosen={row['chosen_sequence']} rejected={row['rejected_sequence']}"
        )

    print("Bottom margin examples:")
    for _, row in low_examples.iterrows():
        print(
            f"  view={row['source_view']} cluster={row['cluster_idx']} "
            f"margin={float(row['delta_margin']):.4f} "
            f"chosen={row['chosen_sequence']} rejected={row['rejected_sequence']}"
        )


if __name__ == "__main__":
    main()


