#!/usr/bin/env python
"""Pre-compute train/val/test FASTA splits using the hash-based split assignment.

Reads a single FASTA and writes three output files:
  {stem}_train.fasta, {stem}_val.fasta, {stem}_test.fasta

The assignment is identical to what _load_fasta_by_split() does at runtime,
so pre-split files and the on-the-fly loader are interchangeable.

Usage:
  uv run scripts/data_prep/split_fasta.py \\
      --input $PROJECT_DIR/datasets/oas_dedup_rep_seq.fasta \\
      --output-dir $PROJECT_DIR/datasets \\
      [--salt oas-v1] [--train-pct 90] [--val-pct 5] [--test-pct 5]
"""

import argparse
import logging
import sys
from pathlib import Path

from Bio import SeqIO

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from protein_design.evotuning.splits import SplitConfig, split_for

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Input FASTA file")
    parser.add_argument("--output-dir", required=True, help="Directory for output split FASTAs")
    parser.add_argument("--salt", default="oas-v1")
    parser.add_argument("--train-pct", type=int, default=90)
    parser.add_argument("--val-pct", type=int, default=5)
    parser.add_argument("--test-pct", type=int, default=5)
    args = parser.parse_args()

    cfg = SplitConfig(
        salt=args.salt,
        train_pct=args.train_pct,
        val_pct=args.val_pct,
        test_pct=args.test_pct,
    )

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    out_paths = {
        "train": output_dir / f"{stem}_train.fasta",
        "val":   output_dir / f"{stem}_val.fasta",
        "test":  output_dir / f"{stem}_test.fasta",
    }

    for split, path in out_paths.items():
        if path.exists():
            log.warning("%s already exists — will overwrite", path)

    log.info("Input:  %s", input_path)
    log.info("Config: salt=%r train=%d%% val=%d%% test=%d%%", cfg.salt, cfg.train_pct, cfg.val_pct, cfg.test_pct)
    for split, path in out_paths.items():
        log.info("Output [%s]: %s", split, path)

    handles = {split: open(path, "w") for split, path in out_paths.items()}
    counts = {"train": 0, "val": 0, "test": 0}

    try:
        for i, record in enumerate(SeqIO.parse(str(input_path), "fasta")):
            split = split_for(record.id, cfg)
            SeqIO.write(record, handles[split], "fasta")
            counts[split] += 1
            if (i + 1) % 5_000_000 == 0:
                log.info("Processed %d sequences  train=%d val=%d test=%d", i + 1, counts["train"], counts["val"], counts["test"])
    finally:
        for h in handles.values():
            h.close()

    total = sum(counts.values())
    log.info("Done. Total=%d  train=%d (%.1f%%)  val=%d (%.1f%%)  test=%d (%.1f%%)",
             total,
             counts["train"], 100 * counts["train"] / total,
             counts["val"],   100 * counts["val"]   / total,
             counts["test"],  100 * counts["test"]  / total)


if __name__ == "__main__":
    main()
