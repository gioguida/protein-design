#!/bin/bash
#SBATCH --job-name=recover_posid
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00
#SBATCH --output=bash_scripts/logs/recover_posid_%j.log

# One-off: recover the positional-identity threshold used to build
# c05_cdrh3_posid30.fasta by scanning OAS and taking min(posid) over its seq_ids.

set -euo pipefail
source bash_scripts/common_setup.sh

uv run python - <<'EOF'
import pandas as pd
from pathlib import Path

project = Path("/cluster/project/infk/krause/mdenegri/protein-design")
fasta = project / "datasets/c05/c05_cdrh3_posid30.fasta"
csv = project / "datasets/oas_filtered.csv.gz"
REF = "AKHMSMQQVVSAGWERADLVGDAFDV"

def posid(a, b):
    return sum(x == y for x, y in zip(a, b)) / max(len(a), len(b))

ids = set()
with open(fasta) as f:
    for line in f:
        if line.startswith(">"):
            ids.add(line[1:].strip().split()[0])
print(f"IDs in FASTA: {len(ids):,}", flush=True)

vals = []
rows = 0
for chunk in pd.read_csv(csv, usecols=["seq_id", "cdr3_aa"], chunksize=500_000):
    rows += len(chunk)
    sub = chunk[chunk["seq_id"].isin(ids)].dropna(subset=["cdr3_aa"])
    vals.extend(posid(str(s), REF) for s in sub["cdr3_aa"])
    if rows % 5_000_000 == 0:
        print(f"scanned {rows:,} rows, matched {len(vals):,}", flush=True)
print(f"Done. Scanned {rows:,} rows. Matched {len(vals):,}.", flush=True)

if vals:
    vals.sort()
    print(f"min posid     : {min(vals):.6f}")
    print(f"max posid     : {max(vals):.6f}")
    print(f"count <  0.20 : {sum(v <  0.20 for v in vals)}")
    print(f"count <  0.25 : {sum(v <  0.25 for v in vals)}")
    print(f"count <  0.30 : {sum(v <  0.30 for v in vals)}")
    print(f"5 smallest    : {[round(v, 4) for v in vals[:5]]}")
EOF
