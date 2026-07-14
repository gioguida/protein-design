"""Pre-materialize external-split-seed train/val/test CSVs.

`protein_design.dms_splitting.ensure_dataset_splits` already supports keys of
the form `<dataset_key>_splitseed<N>`: same DatasetSpec as `<dataset_key>`, but
the FULL train/val/test split is re-instantiated with `split.seed=N`, isolated
in `<output_dir>/<dataset_key>_splitseed<N>/` (the canonical seed=42 split at
`<output_dir>/<dataset_key>/` is never touched). This is a second axis of
variation from `data.low_data.seed` (which subsamples an already-fixed train
split): it re-draws train+val+test membership itself.

Run this once, sequentially, before submitting a sweep grid over external
split seeds -- otherwise many concurrent SLURM jobs could race to build the
same split directory simultaneously.

Usage:
  uv run python scripts/data_prep/build_external_splits.py \
      --datasets ed1_m22,cetuximab_h --seeds 101,102,103,104,105
"""

from __future__ import annotations

import argparse

import pandas as pd

from protein_design.dms_splitting import ensure_dataset_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", required=True, help="Comma-separated base dataset keys (e.g. ed1_m22,cetuximab_h)")
    parser.add_argument("--seeds", required=True, help="Comma-separated split seeds (e.g. 101,102,103,104,105)")
    parser.add_argument("--force", action="store_true", help="Rebuild even if already present")
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    for dataset_key in datasets:
        for seed in seeds:
            ext_key = f"{dataset_key}_splitseed{seed}"
            paths = ensure_dataset_splits(ext_key, force=args.force)
            counts = {name: len(pd.read_csv(p)) for name, p in paths.items()}
            print(f"{ext_key}: {counts} -> {paths['train'].parent}")


if __name__ == "__main__":
    main()
