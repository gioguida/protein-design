import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def _add_repo_root_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_add_repo_root_to_path()

from src.dataset import default_data_paths

def main():
    paths = default_data_paths()
    # load D2 dataset
    D2_df = pd.read_csv(paths["processed_dir"] / "D2.csv")

    scores_d2 = D2_df["delta_M22_binding_enrichment_adj"].values
    positive_scores = scores_d2[scores_d2 > 0]
    negative_scores = scores_d2[scores_d2 < 0]

    print(f"D2 dataset: {len(scores_d2)} entries")
    print(f"Positive scores: {len(positive_scores)}, Range: [{positive_scores.min():.4f}, {positive_scores.max():.4f}], Mean: {positive_scores.mean():.4f})")
    print(f"Negative scores: {len(negative_scores)}, Range: [{negative_scores.min():.4f}, {negative_scores.max():.4f}], Mean: {negative_scores.mean():.4f})")

    plt.figure()
    plt.hist(scores_d2, bins=50, histtype="bar", rwidth=0.8)
    # color the bars based on positive/negative values
    for patch in plt.gca().patches:
        if patch.get_x() < 0:
            patch.set_facecolor("red")
        else:
            patch.set_facecolor("blue")
    plt.title("D2 delta M22 binding enrichment distribution")
    plt.xlabel("Delta M22 Binding Enrichment")
    plt.ylabel("Frequency")
    plt.show()



if __name__ == "__main__":
    main()