#!/usr/bin/env python3
"""One-pass stream of oas_filtered.csv.gz -> cached unique CDR-H3 table.

Builds a parquet of unique cdr3_aa with redundancy count and length, the
reusable artifact needed both for the ED-cloud diagnostic and for any future
C05/ED-similar corpus build. Heavy chains only, productive only.
"""
from __future__ import annotations

import csv
import gzip
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
csv.field_size_limit(sys.maxsize)

PROJECT = os.environ["PROJECT_DIR"]
SCRATCH = os.environ["SCRATCH_DIR"]
CSV = os.path.join(PROJECT, "data", "oas", "oas_filtered.csv.gz")
OUT = Path(SCRATCH) / "oas_h3_cache" / "oas_unique_h3.parquet"
OUT.parent.mkdir(parents=True, exist_ok=True)

counts: Counter[str] = Counter()
n = 0
with gzip.open(CSV, "rt") as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        n += 1
        if n % 2_000_000 == 0:
            print(f"  ...{n:,} rows, {len(counts):,} unique H3", flush=True)
        h3 = (row.get("cdr3_aa") or "").strip()
        if not h3 or "*" in h3 or "X" in h3:
            continue
        if (row.get("chain") or "").strip().lower() not in ("", "heavy", "h"):
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
