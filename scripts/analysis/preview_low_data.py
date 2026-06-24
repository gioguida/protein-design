#!/usr/bin/env python
"""Preview the low-data DPO train subsets before launching a sweep.

For each (n_train, scheme, seed) it subsamples the TRAIN sequence split exactly as
training will (same code path), derives the delta-based preference pairs, and
reports how many sequences/positives survive and how many pairs each component
yields. Use it to sanity-check that small N still produces usable pairs (a draw
with no positives builds no within_pos / cross pairs) before spending GPU on the
full evo-vs-vanilla learning curve.

Read-only: nothing is written unless --out is given.

Usage (login node, CPU is fine — no model is loaded):
    uv run python scripts/analysis/preview_low_data.py
    uv run python scripts/analysis/preview_low_data.py --n 50,100,200,500,1000 --seeds 0,1,2
    uv run python scripts/analysis/preview_low_data.py --scheme random --out report/figures/low_data_preview.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protein_design.dms_splitting import DEFAULT_CONFIG_PATH, project_root  # noqa: E402
from protein_design.dpo.dataset import (  # noqa: E402
    DELTA_BASED_COMPONENTS,
    _load_split_dataframe,
    build_dpo_pairs_from_clustered_dataframe,
)
from protein_design.dpo.low_data import (  # noqa: E402
    DEFAULT_METRIC_COL,
    subsample_train_sequences,
)

DEFAULT_N_GRID = [50, 100, 200, 500, 1000, 2000, 5000]


def _int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--n", type=_int_list, default=DEFAULT_N_GRID,
                   help="comma-separated train-sequence sizes")
    p.add_argument("--seeds", type=_int_list, default=[0, 1, 2],
                   help="comma-separated low-data seeds (repeats)")
    p.add_argument("--scheme", choices=["stratified", "random", "both"], default="both")
    p.add_argument("--dataset-key", default="ed2_m22")
    p.add_argument("--stratify-bins", type=int, default=10)
    p.add_argument("--out", type=Path, default=None,
                   help="optional CSV path for the composition table")
    args = p.parse_args()

    dms_config_path = project_root() / DEFAULT_CONFIG_PATH
    full_train = _load_split_dataframe(
        split_name="train",
        dms_config_path=dms_config_path,
        dataset_key=args.dataset_key,
        force_rebuild=False,
    )
    metric = pd.to_numeric(full_train[DEFAULT_METRIC_COL], errors="coerce")
    n_usable = int(metric.notna().sum())
    n_pos_full = int((metric > 0).sum())
    print(f"Full train split '{args.dataset_key}': {len(full_train)} rows "
          f"({n_usable} with finite metric, {n_pos_full} positive enrichment)")

    schemes = ["stratified", "random"] if args.scheme == "both" else [args.scheme]
    rows: list[dict] = []
    for scheme in schemes:
        for n in args.n:
            for seed in args.seeds:
                sub = subsample_train_sequences(
                    full_train, n, scheme=scheme,
                    stratify_bins=args.stratify_bins, seed=seed,
                )
                pairs = build_dpo_pairs_from_clustered_dataframe(
                    clustered_df=sub, source_view="train", random_seed=seed,
                )
                n_pos = int((sub[DEFAULT_METRIC_COL].astype(float) > 0).sum())
                comp_counts = (
                    pairs["delta_component"].value_counts().to_dict()
                    if not pairs.empty else {}
                )
                row = {
                    "scheme": scheme,
                    "n_train": n,
                    "seed": seed,
                    "n_seqs": len(sub),
                    "n_pos": n_pos,
                    "n_pairs": len(pairs),
                }
                for component in DELTA_BASED_COMPONENTS:
                    row[f"pairs_{component}"] = int(comp_counts.get(component, 0))
                rows.append(row)

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
        print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
