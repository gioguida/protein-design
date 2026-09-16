#!/usr/bin/env python
"""Subset the full OAS metadata table down to just the Linclust dedup survivors.

filter_oas.py's output (oas_filtered.parquet) carries metadata for every
sequence that passed cleaning, before deduplication. Only a fraction of those
survive run_linclust.sh's 95%-identity clustering as cluster representatives
(oas_dedup_rep_seq.fasta). This script reads the survivor seq_ids into memory
once (as a compact Arrow array, not a Python set -- see load_survivor_ids),
then hands the actual join to DuckDB: a semi-join between the metadata
Parquet dataset and the in-memory survivor list, streamed straight to the
output Parquet file. DuckDB's joins are out-of-core -- if the survivor hash
table doesn't fit in the configured memory_limit, it spills to disk instead
of exhausting whatever RAM the OS reports -- which is what a hand-rolled
pyarrow.acero version of this join kept failing on for reasons that weren't
diagnosable without direct access to the failing process.

The result is the file that gets promoted to
$PROJECT_DIR/data/oas/oas_dedup_meta.parquet alongside the dedup FASTA.

mmseqs easy-linclust preserves each representative's original FASTA header,
so seq_id stays a valid join key across the dedup step -- no reconciliation
needed.

Usage:
  uv run scripts/data_prep/extract_dedup_metadata.py \\
      --dedup-fasta $SCRATCH_DIR/oas_dedup_rep_seq.fasta \\
      --meta $SCRATCH_DIR/oas_filtered.parquet \\
      --output $SCRATCH_DIR/oas_dedup_meta.parquet
"""

import argparse
import os
import sys

import duckdb
import pyarrow as pa
from dotenv import load_dotenv

SURVIVOR_ID_BATCH_SIZE = 1_000_000


def load_survivor_ids(fasta_path: str, batch_size: int = SURVIVOR_ID_BATCH_SIZE) -> pa.Array:
    """Read seq_ids (FASTA headers) of the dedup representatives into one
    compact Arrow string array.

    Not a Python set: at ~200M survivors, a set[str] costs ~50-80 bytes of
    per-object overhead per string on top of the characters themselves
    (roughly 30GB total) and rehashes repeatedly as it grows -- that's what
    OOM-killed this step at both 64GB and 128GB before this rewrite, without
    ever reaching the main scan loop. An Arrow string array holds the same
    ~200M short strings as a few contiguous buffers, no per-element object
    overhead, at a few GB total.

    Uses large_string (64-bit offsets into the data buffer), not plain
    string (32-bit offsets, capped at 2GB of concatenated character data) --
    ~200M seq_ids comfortably exceeds that 2GB cap once concatenated.

    The Python-list intermediate is still needed to read lines one at a time,
    but it's batched (batch_size) and converted to a small Arrow array each
    round rather than collecting all ~200M headers into one Python list
    first, so peak memory during this read is bounded by batch_size, not the
    full survivor count.
    """
    arrays: list[pa.Array] = []
    batch: list[str] = []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                batch.append(line[1:].strip().split()[0])
                if len(batch) >= batch_size:
                    arrays.append(pa.array(batch, type=pa.large_string()))
                    batch = []
    if batch:
        arrays.append(pa.array(batch, type=pa.large_string()))
    return pa.concat_arrays(arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dedup-fasta", default=None, help="Default: $SCRATCH_DIR/oas_dedup_rep_seq.fasta")
    parser.add_argument("--meta", default=None, help="Default: $SCRATCH_DIR/oas_filtered.parquet")
    parser.add_argument("--output", default=None, help="Default: $SCRATCH_DIR/oas_dedup_meta.parquet")
    parser.add_argument(
        "--memory-limit-gb", type=float, default=None,
        help="DuckDB memory budget. Default: 80%% of mem-per-cpu * SLURM_CPUS_PER_TASK "
             "(or 32GB if not running under SLURM).",
    )
    args = parser.parse_args()

    load_dotenv()
    scratch_dir = os.environ.get("SCRATCH_DIR")
    if not scratch_dir:
        print("Error: SCRATCH_DIR env var not set", file=sys.stderr)
        sys.exit(1)

    dedup_fasta = args.dedup_fasta or os.path.join(scratch_dir, "oas_dedup_rep_seq.fasta")
    meta_path = args.meta or os.path.join(scratch_dir, "oas_filtered.parquet")
    output_path = args.output or os.path.join(scratch_dir, "oas_dedup_meta.parquet")

    print(f"Loading survivor seq_ids from {dedup_fasta} ...", flush=True)
    survivors = load_survivor_ids(dedup_fasta)
    print(f"  {len(survivors):,} survivor seq_ids ({survivors.nbytes / 1024**3:.2f} GB)", flush=True)

    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    if args.memory_limit_gb is not None:
        mem_limit_gb = args.memory_limit_gb
    else:
        mem_per_cpu_gb = float(os.environ.get("SLURM_MEM_PER_CPU", 0)) / 1024  # SLURM reports MB
        mem_limit_gb = mem_per_cpu_gb * n_cpus * 0.8 if mem_per_cpu_gb else 32.0

    # A directory (sharded filter_oas.py runs) vs. a single file (whole-corpus
    # runs) -- see merge_oas_shards.py. DuckDB's read_parquet needs an
    # explicit glob for the directory case.
    meta_glob = os.path.join(meta_path, "*.parquet") if os.path.isdir(meta_path) else meta_path

    # DuckDB does the actual join+filter+write: unlike the pyarrow.acero
    # attempt this replaces, DuckDB's joins are out-of-core -- if the
    # survivors hash table doesn't fit in memory_limit, it spills to disk
    # under temp_directory instead of relying on the OS to have enough RAM,
    # which is exactly the failure mode (unexplained OOMs) that motivated
    # switching to DuckDB for this step. `survivors` (an in-memory Arrow
    # array) is queryable directly via DuckDB's Arrow integration, no need
    # to write it to a file first.
    temp_dir = os.path.join(scratch_dir, "duckdb_tmp")
    os.makedirs(temp_dir, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads TO {n_cpus}")
    con.execute(f"SET memory_limit = '{mem_limit_gb:.1f}GB'")
    con.execute(f"SET temp_directory = '{temp_dir}'")
    con.execute("SET enable_progress_bar = true")
    print(f"DuckDB: threads={n_cpus} memory_limit={mem_limit_gb:.1f}GB temp_directory={temp_dir}", flush=True)

    survivors_table = pa.table({"seq_id": survivors})
    con.register("survivors", survivors_table)

    result = con.execute(
        f"""
        COPY (
            SELECT m.*
            FROM read_parquet('{meta_glob}') AS m
            SEMI JOIN survivors AS s ON m.seq_id = s.seq_id
        ) TO '{output_path}' (FORMAT PARQUET, COMPRESSION 'snappy')
        """
    ).fetchone()
    n_kept = result[0] if result else 0

    if n_kept != len(survivors):
        print(
            f"Warning: kept {n_kept:,} metadata rows but {len(survivors):,} "
            f"survivor seq_ids -- {len(survivors) - n_kept:,} representative(s) "
            f"had no matching row in {meta_path}.",
            file=sys.stderr,
        )

    print(f"Wrote {n_kept:,} rows to {output_path}")


if __name__ == "__main__":
    main()
