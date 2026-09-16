#!/usr/bin/env python
"""Snapshot the train/val/test split as one seq_id-only Parquet file per split.

The training loader (protein_design.evotuning.data) assigns every sequence to
train/val/test on the fly, by hashing its seq_id against SplitConfig (see
protein_design.evotuning.splits.split_for) -- no split file is needed for
training itself. This script calls that exact same function once, so the
files it writes are guaranteed to match the live logic rather than
reimplementing it, and exist purely as a reference/audit artifact: a literal
record of split membership that survives any future change to the salt or
percentages, and lets other tooling (e.g. "sample only from val") work
without re-deriving the split.

Usage:
  uv run scripts/data_prep/export_split_ids.py \\
      --fasta $SCRATCH_DIR/oas_dedup_rep_seq.fasta \\
      --output-dir $SCRATCH_DIR \\
      [--salt oas-v1] [--train-pct 90] [--val-pct 5] [--test-pct 5]
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
BATCH_SIZE = 500_000


def iter_fasta_ids(fasta_path: str):
    """Yield seq_ids (FASTA headers) without parsing sequences."""
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                yield line[1:].strip().split()[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fasta", default=None, help="Default: $SCRATCH_DIR/oas_dedup_rep_seq.fasta")
    parser.add_argument("--output-dir", default=None, help="Default: $SCRATCH_DIR")
    parser.add_argument("--salt", default="oas-v1")
    parser.add_argument("--train-pct", type=int, default=90)
    parser.add_argument("--val-pct", type=int, default=5)
    parser.add_argument("--test-pct", type=int, default=5)
    args = parser.parse_args()

    load_dotenv()
    scratch_dir = os.environ.get("SCRATCH_DIR")
    if not scratch_dir:
        print("Error: SCRATCH_DIR env var not set", file=sys.stderr)
        sys.exit(1)

    fasta_path = args.fasta or os.path.join(scratch_dir, "oas_dedup_rep_seq.fasta")
    output_dir = args.output_dir or scratch_dir
    os.makedirs(output_dir, exist_ok=True)

    cfg = SplitConfig(
        salt=args.salt, train_pct=args.train_pct, val_pct=args.val_pct, test_pct=args.test_pct,
    )

    out_paths: dict[Split, str] = {
        s: os.path.join(output_dir, f"oas_split_{s}_ids.parquet") for s in ("train", "val", "test")
    }
    writers = {s: pq.ParquetWriter(p, SCHEMA, compression="snappy") for s, p in out_paths.items()}
    buffers: dict[Split, list[str]] = {"train": [], "val": [], "test": []}
    counts: dict[Split, int] = {"train": 0, "val": 0, "test": 0}

    def flush(split: Split) -> None:
        if not buffers[split]:
            return
        table = pa.Table.from_pydict({"seq_id": buffers[split]}, schema=SCHEMA)
        writers[split].write_table(table)
        buffers[split].clear()

    print(f"Scanning {fasta_path} (salt={cfg.salt!r}, {cfg.train_pct}/{cfg.val_pct}/{cfg.test_pct})...")
    try:
        for seq_id in iter_fasta_ids(fasta_path):
            split = split_for(seq_id, cfg)
            buffers[split].append(seq_id)
            counts[split] += 1
            if len(buffers[split]) >= BATCH_SIZE:
                flush(split)
        for split in ("train", "val", "test"):
            flush(split)
    finally:
        for w in writers.values():
            w.close()

    total = sum(counts.values())
    print(f"Done. {total:,} seq_ids -> train={counts['train']:,} val={counts['val']:,} test={counts['test']:,}")
    for split, path in out_paths.items():
        print(f"  {split}: {path}")


if __name__ == "__main__":
    main()
