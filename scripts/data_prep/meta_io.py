"""Streaming reads over the OAS metadata Parquet table.

Backs every C05/analysis script that used to stream oas_filtered.csv.gz in
pandas chunks (pd.read_csv(path, usecols=..., chunksize=...)). Works
identically whether `path` is a single Parquet file (whole-corpus
filter_oas.py runs) or a directory of per-shard Parquet files (sharded runs
-- see filter_oas.py's module docstring for why those are never physically
merged), since pyarrow.dataset treats both as one logical table.
"""
import pyarrow.dataset as ds


def iter_meta_chunks(path: str, columns: list[str] | None = None, chunksize: int = 500_000):
    """Yield pandas DataFrames, chunk by chunk.

    Drop-in replacement for `pd.read_csv(path, usecols=columns, chunksize=chunksize)`
    iteration -- `chunksize` bounds memory the same way, though batch sizes
    follow the underlying row groups and won't be exactly `chunksize` rows.
    """
    dataset = ds.dataset(path)
    for batch in dataset.to_batches(columns=columns, batch_size=chunksize):
        yield batch.to_pandas()


def read_meta(path: str, columns: list[str] | None = None):
    """Read the whole table (optionally column-pruned) into one DataFrame."""
    return ds.dataset(path).to_table(columns=columns).to_pandas()
