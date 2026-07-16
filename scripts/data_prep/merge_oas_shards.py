#!/usr/bin/env python
"""Merge filter_oas.py shard outputs into the final oas_filtered.{fasta,csv.gz}.

Run after all shards from bash_scripts/utils/filter_oas_shard.sbatch have
completed (the corresponding .sbatch submits this with
`--dependency=afterok:<array_job_id>`).

- FASTA: plain concatenation (no headers to worry about).
- CSV: shards were written with --no-csv-header (data rows only), so this
  writes the header once, then concatenates the gzip-compressed shard bodies
  as additional gzip members -- gzip (and Python's gzip module, which pandas
  uses) transparently decompresses concatenated members as one stream, so
  this avoids decompressing and recompressing the whole corpus just to add
  a header.
- Summary: sums each shard's summary_{i}.json (written by filter_oas.py via
  --summary-json) instead of re-scanning the merged output.

Usage:
  uv run scripts/data_prep/merge_oas_shards.py --num-shards 32
"""

import argparse
import gzip
import io
import json
import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from filter_oas import FILE_META_COLS, SEQUENCE_COLS, print_summary  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()

    load_dotenv()
    scratch_dir = os.environ.get("SCRATCH_DIR")
    if not scratch_dir:
        print("Error: SCRATCH_DIR env var not set", file=sys.stderr)
        sys.exit(1)

    shards_dir = os.path.join(scratch_dir, "oas_shards")
    output_fasta = os.path.join(scratch_dir, "oas_filtered.fasta")
    output_csv = os.path.join(scratch_dir, "oas_filtered.csv.gz")

    K = args.num_shards
    shard_fastas = [os.path.join(shards_dir, f"oas_filtered_shard{i}.fasta") for i in range(K)]
    shard_csvs   = [os.path.join(shards_dir, f"oas_filtered_shard{i}.csv.gz") for i in range(K)]
    shard_jsons  = [os.path.join(shards_dir, f"summary_{i}.json") for i in range(K)]

    missing = [p for p in shard_fastas + shard_csvs + shard_jsons if not os.path.exists(p)]
    if missing:
        print(f"Error: {len(missing)} shard output(s) missing, e.g. {missing[0]}. "
              f"Did all {K} shard jobs finish successfully?", file=sys.stderr)
        sys.exit(1)

    # ── FASTA: plain concatenation ──────────────────────────────────────────
    print("Merging FASTA shards...")
    with open(output_fasta, "wb") as out:
        for p in shard_fastas:
            with open(p, "rb") as f:
                out.write(f.read())

    # ── CSV: header once, then concatenate gzip members ─────────────────────
    print("Merging CSV shards...")
    header_line = ",".join(["seq_id"] + FILE_META_COLS + SEQUENCE_COLS) + "\n"
    with open(output_csv, "wb") as out:
        header_buf = io.BytesIO()
        with gzip.GzipFile(fileobj=header_buf, mode="wb") as gz:
            gz.write(header_line.encode("utf-8"))
        out.write(header_buf.getvalue())
        for p in shard_csvs:
            with open(p, "rb") as f:
                out.write(f.read())

    # ── Summary: sum shard JSONs ─────────────────────────────────────────────
    total = {
        "total_passing": 0, "total_rejected": 0, "total_nterm_removed": 0,
        "total_cterm_removed": 0, "total_exact_dup": 0, "length_hist": {},
    }
    for p in shard_jsons:
        with open(p) as f:
            s = json.load(f)
        for k in ("total_passing", "total_rejected", "total_nterm_removed",
                   "total_cterm_removed", "total_exact_dup"):
            total[k] += s[k]
        for length, count in s["length_hist"].items():
            total["length_hist"][length] = total["length_hist"].get(length, 0) + count

    print_summary(total, output_fasta, output_csv)


if __name__ == "__main__":
    main()
