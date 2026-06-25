"""Sample C05 CDR-H3 sequences from a train-split PSSM baseline."""

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

from protein_design.pssm_baseline import (
    build_output_rows,
    build_pssm_counts,
    counts_to_log_frequencies,
    load_train_dataframe,
    sample_cdrh3_sequences,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--dms-config", default="conf/data/dms/default.yaml")
    parser.add_argument("--local-splits-dir", default="data/dms_splits")
    parser.add_argument("--enrichment-threshold", type=float, default=None)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--n-sequences", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
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

    counts = build_pssm_counts(train_df[sequence_col].astype(str).tolist())
    log_frequencies = counts_to_log_frequencies(counts, pseudocount=1.0)
    sampled_cdrh3 = sample_cdrh3_sequences(
        log_frequencies=log_frequencies,
        temperature=args.temperature,
        n_sequences=args.n_sequences,
        seed=args.seed,
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
        "temperature": args.temperature,
        "n_sequences": args.n_sequences,
        "seed": args.seed,
        "pssm_pseudocount": 1.0,
        "train_rows_used": int(len(train_df)),
    }
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print(f"[done] wrote meta to {meta_path}")


if __name__ == "__main__":
    main()
