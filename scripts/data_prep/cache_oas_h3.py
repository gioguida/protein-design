#!/usr/bin/env python3
"""One-pass stream of oas_filtered.parquet -> cached unique CDR-H3 table.

Builds a parquet of unique cdr3_aa with redundancy count and length, the
reusable artifact needed both for the ED-cloud diagnostic and for any future
C05/ED-similar corpus build. Heavy chains only, productive only.
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from meta_io import iter_meta_chunks  # noqa: E402

load_dotenv()

PROJECT = os.environ["PROJECT_DIR"]
SCRATCH = os.environ["SCRATCH_DIR"]
META = os.path.join(PROJECT, "data", "oas", "oas_filtered.parquet")
OUT = Path(SCRATCH) / "oas_h3_cache" / "oas_unique_h3.parquet"
OUT.parent.mkdir(parents=True, exist_ok=True)

counts: Counter[str] = Counter()
n = 0
for chunk in iter_meta_chunks(META, columns=["cdr3_aa", "chain"], chunksize=500_000):
    for h3, chain in zip(chunk["cdr3_aa"], chunk["chain"]):
        n += 1
        if n % 2_000_000 == 0:
            print(f"  ...{n:,} rows, {len(counts):,} unique H3", flush=True)
        h3 = (h3 or "").strip()
        if not h3 or "*" in h3 or "X" in h3:
            continue
        if (chain or "").strip().lower() not in ("", "heavy", "h"):
            continue
        counts[h3] += 1

print(f"Total rows={n:,}  unique H3={len(counts):,}", flush=True)
df = pd.DataFrame(
    {"cdr3_aa": list(counts.keys()), "count": list(counts.values())}
)
df["len"] = df["cdr3_aa"].str.len()
df = df.sort_values("count", ascending=False).reset_index(drop=True)
df.to_parquet(OUT)
print(f"Wrote {OUT}  ({len(df):,} rows)", flush=True)
