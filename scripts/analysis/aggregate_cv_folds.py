"""Aggregate per-fold held-out predictions into pooled out-of-fold CV metrics.

For each ``(model, base_dataset, N, low_data_seed)`` this scans ``$TRAIN_DIR``
for the K finished CV fold runs
(``lowdata_<model>_n<N>_s<seed>_cv<K>s<SEED>_f<I>_<ts>``), reads each fold run's
``test_predictions.csv``, concatenates the out-of-fold ``(prediction,
ground_truth)`` pairs, and computes a SINGLE pooled Spearman over the full
dataset -- the headline CV metric (explicitly NOT the mean of per-fold rho).

It writes one aggregated JSON per ``(model, dataset, N, low_data_seed)`` under
``$ANALYSIS_DIR/<model>/cv_pooled/<dataset>_cv<K>s<SEED>_n<N>_s<seed>.json``
containing the pooled Spearman, pooled example count, fold metadata, per-fold
Spearmans (diagnostics), and the contributing per-fold run paths.

Usage:
  uv run python scripts/analysis/aggregate_cv_folds.py \
      --dataset ed1_m22 --models vanilla_35m,evo_35m \
      --n 20,50,100,200,366 --seeds 0,1,2 --n-folds 5 --fold-seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd

from protein_design.cv_splitting import pooled_oof_spearman

_TS_RE = re.compile(r"_\d{8}_\d{6}$")


def _train_dir() -> Path:
    user = os.environ.get("USER", "unknown")
    scratch = os.environ.get("SCRATCH_DIR", f"/cluster/scratch/{user}/protein-design")
    return Path(os.environ.get("TRAIN_DIR", str(Path(scratch) / "train")))


def _analysis_dir() -> Path:
    return Path(os.environ.get("ANALYSIS_DIR", "/cluster/project/infk/krause/mdenegri/protein-design/analysis"))


def _latest_fold_run(train_dir: Path, base_name: str) -> Path | None:
    """Most recently finished run dir matching base_name (ignoring timestamp)."""
    pat = re.compile(r"^" + re.escape(base_name) + r"_\d{8}_\d{6}$")
    best, best_mt = None, -1.0
    for p in train_dir.glob(base_name + "_*"):
        if not pat.match(p.name):
            continue
        summ = p / "summary.json"
        preds = p / "test_predictions.csv"
        if not summ.exists() or not preds.exists():
            continue
        mt = summ.stat().st_mtime
        if mt > best_mt:
            best, best_mt = p, mt
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Base dataset key (e.g. ed1_m22)")
    parser.add_argument("--models", required=True, help="Comma-separated model keys")
    parser.add_argument("--n", required=True, help="Comma-separated N grid")
    parser.add_argument("--seeds", required=True, help="Comma-separated low_data seeds")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=0)
    parser.add_argument("--truth-col", default="M22_binding_enrichment_adj")
    args = parser.parse_args()

    train_dir = _train_dir()
    analysis_dir = _analysis_dir()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    ns = [int(x) for x in args.n.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    K, FS = args.n_folds, args.fold_seed

    n_written = 0
    for model in models:
        for n in ns:
            for seed in seeds:
                fold_dfs, fold_runs, missing = [], [], []
                for i in range(K):
                    base_name = f"lowdata_{model}_n{n}_s{seed}_cv{K}s{FS}_f{i}"
                    run_dir = _latest_fold_run(train_dir, base_name)
                    if run_dir is None:
                        missing.append(base_name)
                        continue
                    fold_dfs.append(pd.read_csv(run_dir / "test_predictions.csv"))
                    fold_runs.append(str(run_dir))
                if missing:
                    print(f"[skip] {model} n={n} s={seed}: {len(missing)}/{K} folds missing")
                    continue
                pooled = pooled_oof_spearman(fold_dfs, pred_col="score", truth_col=args.truth_col)
                out = {
                    "model": model,
                    "base_dataset": args.dataset,
                    "n_train": n,
                    "low_data_seed": seed,
                    "n_folds": K,
                    "fold_seed": FS,
                    "pooled_spearman": pooled["pooled_spearman"],
                    "pooled_pval": pooled["pooled_pval"],
                    "n_pooled": pooled["n_pooled"],
                    "foldwise_spearman": pooled["foldwise_spearman"],
                    "fold_run_dirs": fold_runs,
                }
                out_dir = analysis_dir / model / "cv_pooled"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{args.dataset}_cv{K}s{FS}_n{n}_s{seed}.json"
                out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
                print(f"[ok]   {out_path.name}: pooled rho={pooled['pooled_spearman']:.4f} "
                      f"(n={pooled['n_pooled']})")
                n_written += 1

    print(f"---- wrote {n_written} pooled-CV artifacts under {analysis_dir}")


if __name__ == "__main__":
    main()
