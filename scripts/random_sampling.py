"""Sample C05 CDR-H3 sequences from a train-split random library-mutant baseline.

WT-centered, uniform over the residues observed at each position (see
``protein_design.random_baseline``). The only knob is ``--trust-radius`` (the cap
on edits from WT), the analogue of the PSSM baseline's temperature.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protein_design.constants import C05_CDRH3
from protein_design.random_baseline import (
    build_output_rows,
    build_position_alphabet,
    load_train_dataframe,
    sample_random_mutants,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--dms-config", default="conf/data/dms/default.yaml")
    parser.add_argument("--local-splits-dir", default="data/dms_splits")
    parser.add_argument("--enrichment-threshold", type=float, default=None)
    parser.add_argument("--trust-radius", type=int, required=True)
    parser.add_argument("--n-sequences", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--unique-only",
        action="store_true",
        help="Dedup to unique sequences (Lorenz behavior). Default keeps "
        "duplicates so the library matches the PSSM baseline's sample size.",
    )
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[init] args: {vars(args)}")

    train_df, sequence_col, metric_col, split_path, split_source = load_train_dataframe(
        dataset_key=args.dataset_key,
        dms_config_path=args.dms_config,
        local_splits_dir=args.local_splits_dir,
        enrichment_threshold=args.enrichment_threshold,
    )
    print(f"[data] using train split ({split_source}): {split_path}")
    print(f"[data] rows available after cleanup/filtering: {len(train_df)}")
    if args.enrichment_threshold is not None:
        print(
            f"[data] enrichment threshold on {metric_col}: {args.enrichment_threshold} "
            f"(surviving rows: {len(train_df)})"
        )

    position_alphabet = build_position_alphabet(
        train_df[sequence_col].astype(str).tolist(), wt=C05_CDRH3
    )
    n_mutable = sum(1 for residues in position_alphabet if len(residues) > 1)
    print(
        f"[alphabet] {n_mutable}/{len(position_alphabet)} positions are mutable "
        f"(>1 observed residue)"
    )

    sampled_cdrh3 = sample_random_mutants(
        position_alphabet=position_alphabet,
        trust_radius=args.trust_radius,
        n_sequences=args.n_sequences,
        seed=args.seed,
        wt=C05_CDRH3,
        allow_duplicates=not args.unique_only,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df = pd.DataFrame(build_output_rows(sampled_cdrh3))[
        ["chain_id", "gibbs_step", "sequence", "cdrh3", "n_mutations", "model_variant"]
    ]
    output_df.to_csv(output_path, index=False)
    print(f"[done] wrote {len(output_df)} rows to {output_path}")

    meta = {
        "dataset_key": args.dataset_key,
        "dms_config": args.dms_config,
        "local_splits_dir": args.local_splits_dir,
        "train_split_path": str(split_path),
        "train_split_source": split_source,
        "sequence_col": sequence_col,
        "key_metric_col": metric_col,
        "enrichment_threshold": args.enrichment_threshold,
        "trust_radius": args.trust_radius,
        "n_sequences": args.n_sequences,
        "seed": args.seed,
        "unique_only": bool(args.unique_only),
        "n_mutable_positions": int(n_mutable),
        "train_rows_used": int(len(train_df)),
    }
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print(f"[done] wrote meta to {meta_path}")


if __name__ == "__main__":
    main()
