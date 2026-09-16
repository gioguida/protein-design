#!/usr/bin/env python
"""Split a WT-similar-set (see build_wt_similar_set.py) into train/val/test.

Reuses split_for() with the exact same SplitConfig as the main OAS corpus
split (export_split_ids.py) rather than defining a new split scheme: every
selected sequence is itself a real OAS seq_id and already has a deterministic
split assignment, so reusing it keeps a sequence's train/val/test role
globally consistent across the project.

Usage:
  uv run scripts/data_prep/export_wt_similar_split_ids.py --wt-name c05 --top-n 5000
"""
import argparse
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from protein_design.evotuning.splits import Split, SplitConfig, split_for  # noqa: E402

SCHEMA = pa.schema([("seq_id", pa.string())])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    project = os.environ.get("PROJECT_DIR", ".")
    p.add_argument("--wt-name", required=True)
    p.add_argument("--top-n", type=int, default=5000)
    p.add_argument("--wt-dir", default=None, help="Default: $PROJECT_DIR/data/wt_similar/<wt-name>")
    p.add_argument("--salt", default="oas-v1")
    p.add_argument("--train-pct", type=int, default=90)
    p.add_argument("--val-pct", type=int, default=5)
    p.add_argument("--test-pct", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    project = os.environ["PROJECT_DIR"]
    wt_dir = args.wt_dir or os.path.join(project, "data", "wt_similar", args.wt_name)

    fasta_path = os.path.join(wt_dir, f"{args.wt_name}_wt_similar_top{args.top_n}.fasta")
    seq_ids: list[str] = []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                seq_ids.append(line[1:].strip().split()[0])
    print(f"[split] {len(seq_ids):,} seq_ids from {fasta_path}", flush=True)

    cfg = SplitConfig(salt=args.salt, train_pct=args.train_pct, val_pct=args.val_pct, test_pct=args.test_pct)
    buckets: dict[Split, list[str]] = {"train": [], "val": [], "test": []}
    for seq_id in seq_ids:
        buckets[split_for(seq_id, cfg)].append(seq_id)

    for split, ids in buckets.items():
        out_path = os.path.join(wt_dir, f"{args.wt_name}_wt_similar_top{args.top_n}_{split}_ids.parquet")
        table = pa.Table.from_pydict({"seq_id": ids}, schema=SCHEMA)
        pq.write_table(table, out_path, compression="snappy")
        print(f"[split] {split}: {len(ids):,} -> {out_path}", flush=True)

    total = sum(len(v) for v in buckets.values())
    print(f"[split] Done. {total:,} total (train={len(buckets['train']):,} "
          f"val={len(buckets['val']):,} test={len(buckets['test']):,})")


if __name__ == "__main__":
    main()
