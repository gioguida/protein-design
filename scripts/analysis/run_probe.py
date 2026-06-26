"""Headless driver for the embedding linear-probe analysis.

The probe loads the cached `emb` artifacts (up to ~2 GB each, 343k x 1280 for the
big DMS splits) and fits Ridge + kNN over the full train split — too heavy for the
login node. Run it on a compute node instead; this script computes the comparison
table and both figures and writes them to disk, so the notebook only has to read
the results back:

    probe_scores.csv      report/figures/probe_scores.csv  (the evidence table)
    probe_spearman.pdf    report/figures/probe_spearman.pdf
    embedding_pca.pdf     report/figures/embedding_pca.pdf

The full-data probe saturates (≥200k labels wash out representation differences),
so `--mode curve` also computes a small-data *learning curve* — probe Spearman vs
train-set size — which is the regime that matters for a DPO starting base:
    learning_curve.csv    report/figures/learning_curve.csv
    learning_curve.pdf    report/figures/learning_curve.pdf

Usage (via SLURM):
    sbatch bash_scripts/probe.sbatch                       # full-data table + figures
    sbatch bash_scripts/probe.sbatch --mode curve          # small-data learning curve
    sbatch bash_scripts/probe.sbatch --mode both           # everything
    sbatch bash_scripts/probe.sbatch --models vanilla_650m,evo_c05_cdrmix --no-pca
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")  # headless: no display on compute nodes

from protein_design.analysis import figures as F  # noqa: E402

# Default comparison set (mirrors PROBE_MODELS in report/figures.ipynb): the full
# 650M family so the frozen-embedding spaces are comparable across models.
DEFAULT_MODELS = [
    "vanilla_650m",
    "evo_650m",
    "evo_c05_cdrmix",
    "evo_c05_cdrmix_spearman",
]


def _csv_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", type=_csv_list, default=DEFAULT_MODELS,
                   help="comma-separated model keys (default: 650M probe set)")
    p.add_argument("--datasets", default="all",
                   help="comma-separated dataset keys or 'all' (default)")
    p.add_argument("--ridge-alpha", type=float, default=10.0)
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-pca", action="store_true",
                   help="skip the (memory-heavy) PCA figure")
    p.add_argument("--mode", choices=["full", "curve", "both"], default="full",
                   help="full = saturated table+bars+PCA; curve = small-data "
                        "learning curve; both = everything (default: full)")
    p.add_argument("--n-repeats", type=int, default=5,
                   help="learning-curve: random train subsets averaged per size")
    p.add_argument("--out-name", default="learning_curve",
                   help="basename for the learning-curve outputs in report/figures/ "
                        "(default: learning_curve -> learning_curve.csv/.pdf). Use a "
                        "distinct name to avoid clobbering an existing comparison.")
    args = p.parse_args()

    datasets = "all" if args.datasets == "all" else _csv_list(args.datasets)
    probe_kwargs = dict(ridge_alpha=args.ridge_alpha, n_neighbors=args.n_neighbors,
                        cv_folds=args.cv_folds, seed=args.seed)

    print(f"Models:   {args.models}")
    print(f"Datasets: {datasets}")
    print(f"Mode:     {args.mode}")

    if args.mode in ("full", "both"):
        # 1. Comparison table (the actual evidence).
        df = F.probe_scores(args.models, datasets, **probe_kwargs)
        if df.empty:
            raise SystemExit("No probe rows produced — are the emb artifacts extracted? "
                             "Run preflight_emb / extract.sbatch --what emb first.")
        csv_path = F.FIGURES_DIR / "probe_scores.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
        print(df.round(3).to_string(index=False))

        # 2. Grouped-bar Spearman figure (reuse the table — don't recompute the probe).
        fig = F.plot_probe_spearman(args.models, datasets, df=df)
        F.save_fig(fig, "probe_spearman")

        # 3. PCA scatter (optional — concatenates all splits, the heaviest step).
        if not args.no_pca:
            fig = F.plot_embedding_pca(args.models, datasets)
            F.save_fig(fig, "embedding_pca")

    if args.mode in ("curve", "both"):
        # 4. Learning curve: probe quality vs amount of supervision (the regime
        #    that matters for a DPO starting base — full-data bars saturate).
        lc = F.probe_learning_curve(args.models, datasets, n_repeats=args.n_repeats,
                                    ridge_alpha=args.ridge_alpha, seed=args.seed)
        if lc.empty:
            raise SystemExit("No learning-curve rows — emb artifacts missing?")
        lc_path = F.FIGURES_DIR / f"{args.out_name}.csv"
        lc_path.parent.mkdir(parents=True, exist_ok=True)
        lc.to_csv(lc_path, index=False)
        print(f"Saved: {lc_path}")
        print(lc.round(3).to_string(index=False))
        fig = F.plot_learning_curve(args.models, datasets, df=lc)
        F.save_fig(fig, args.out_name)


if __name__ == "__main__":
    main()
