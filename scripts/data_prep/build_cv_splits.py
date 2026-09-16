"""Pre-materialize k-fold CV train/val/test CSVs before a CV sweep.

``protein_design.cv_splitting`` supports dataset keys of the form
``<base>_cv<K>s<SEED>_f<I>`` (fold I -> test; the remaining folds -> a
stratified train/val pool). Building them here once, sequentially, avoids many
concurrent SLURM jobs racing to write the same fold directory -- exactly the
precedent set by ``scripts/data_prep/build_external_splits.py`` for
``_splitseed<N>`` keys.

Usage:
  uv run python scripts/data_prep/build_cv_splits.py \
      --dms-config conf/data/dms/cv.yaml \
      --datasets ed1_m22,cetuximab_h --n-folds 5 --fold-seed 0

Add ``--verify`` to assert the K test folds tile the dataset exactly once
(each row held out exactly once, no leakage into train/val).
"""

from __future__ import annotations

import argparse

import pandas as pd

from protein_design.cv_splitting import cv_fold_key, ensure_cv_fold_splits, parse_cv_fold_key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", required=True, help="Comma-separated base dataset keys (e.g. ed1_m22,cetuximab_h)")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=0)
    parser.add_argument("--dms-config", default="conf/data/dms/cv.yaml", help="DMS config YAML with the cv: block")
    parser.add_argument("--force", action="store_true", help="Rebuild even if already present")
    parser.add_argument("--verify", action="store_true", help="Assert the folds tile the dataset exactly once")
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    for base in datasets:
        seq_col = None
        test_sequences: list[set] = []
        total_test_rows = 0
        for i in range(args.n_folds):
            key = cv_fold_key(base, args.n_folds, args.fold_seed, i)
            paths = ensure_cv_fold_splits(key, args.dms_config, force=args.force)
            counts = {name: len(pd.read_csv(p)) for name, p in paths.items()}
            print(f"{key}: {counts} -> {paths['train'].parent}")
            if args.verify:
                test_df = pd.read_csv(paths["test"])
                train_df = pd.read_csv(paths["train"])
                val_df = pd.read_csv(paths["val"])
                seq_col = "aa" if "aa" in test_df.columns else test_df.columns[0]
                t = set(test_df[seq_col].astype(str))
                if t & set(train_df[seq_col].astype(str)) or t & set(val_df[seq_col].astype(str)):
                    raise SystemExit(f"LEAKAGE: held-out fold {key} overlaps its own train/val.")
                test_sequences.append(t)
                total_test_rows += len(test_df)
        if args.verify and test_sequences:
            union = set().union(*test_sequences)
            pairwise_overlap = total_test_rows - len(union)
            status = "OK" if pairwise_overlap == 0 else f"OVERLAP={pairwise_overlap}"
            print(f"  [verify] {base}: {total_test_rows} pooled test rows across "
                  f"{args.n_folds} folds, {len(union)} unique -> each row held out once? {status}")
            if pairwise_overlap != 0:
                raise SystemExit(f"Fold test sets are not disjoint for {base}.")


if __name__ == "__main__":
    main()
