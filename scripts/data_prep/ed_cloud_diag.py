#!/usr/bin/env python3
"""Diagnostic: characterize the ED2/ED5/ED811 CDR-H3 'cloud' around C05 and
its relationship to the natural OAS H3s already retrieved near C05.

No full-OAS scan needed: uses the ED train splits + the cdr_windows parquet
(natural H3s of the 6774 blosum25 seqs, i.e. the densest natural region near C05).

Questions answered:
  1. How far do ED variants spread from C05 (per library), in % identity?
  2. Are the natural near-C05 OAS H3s closer to C05, or to some ED variant?
     -> tests whether anchoring to the cloud would re-rank / reach differently.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner, substitution_matrices
from dotenv import load_dotenv

load_dotenv()
PROJECT = os.environ["PROJECT_DIR"]
SCRATCH = os.environ["SCRATCH_DIR"]

C05_H3 = "HMSMQQVVSAGWERADLVGDAFDV"          # 24-aa core
C05_H3_OAS = "AK" + C05_H3                    # 26-aa OAS format

# ---- BLOSUM62 global aligner matching the project's blosum pipeline ----
aligner = PairwiseAligner()
aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
aligner.open_gap_score = -11
aligner.extend_gap_score = -1
aligner.mode = "global"
_self = aligner.score(C05_H3_OAS, C05_H3_OAS)


def norm_blosum(a: str, b: str) -> float:
    return aligner.score(a, b) / _self


def ident24(a: str, b: str) -> float:
    """% identity for equal-length 24-aa cores (ED variants vs C05)."""
    return sum(x == y for x, y in zip(a, b)) / len(b)


# ---- 1. ED cloud geometry ----
print("=== 1. ED cloud geometry (identity to C05, 24-aa core) ===")
ed_variants: dict[str, np.ndarray] = {}
all_ed: set[str] = set()
for lib in ("ed2_m22", "ed5_m22", "ed811_m22"):
    p = f"{PROJECT}/data/dms_splits/{lib}/train.csv"
    aa = pd.read_csv(p, usecols=["aa"])["aa"].astype(str).str.strip()
    aa = aa[aa.str.len() == 24].unique()
    ed_variants[lib] = aa
    all_ed.update(aa.tolist())
    ids = np.array([ident24(s, C05_H3) for s in aa])
    print(f"  {lib:10s}  n_uniq={len(aa):7d}  identity-to-C05: "
          f"min={ids.min():.2f} mean={ids.mean():.2f} max={ids.max():.2f}")
all_ed_list = sorted(all_ed)
print(f"  union unique 24-aa ED variants: {len(all_ed_list):,}")

# ---- 2. Natural near-C05 OAS H3s: closer to C05 or to ED cloud? ----
print("\n=== 2. Natural near-C05 OAS H3s vs C05 vs ED cloud ===")
par = pd.read_parquet(f"{SCRATCH}/cdr_windows/c05_cdrh3_blosum25_flank3.parquet")
nat = par["cdr3_aa"].astype(str).str.strip()
nat = nat[nat.str.len() > 0].unique()
print(f"  unique natural H3s (near C05): {len(nat):,}")

# Subsample ED cloud for the pairwise pass (alignment is O(n*m)).
rng = np.random.default_rng(0)
ed_sample = rng.choice(all_ed_list, size=min(400, len(all_ed_list)), replace=False)

rows = []
for h3 in nat:
    n_c05 = norm_blosum(h3, C05_H3_OAS)
    # best normalized BLOSUM to any sampled ED variant (prepend AK to match frame)
    best_ed = max(norm_blosum(h3, "AK" + e) for e in ed_sample)
    rows.append((h3, n_c05, best_ed))
res = pd.DataFrame(rows, columns=["h3", "norm_c05", "norm_ed_best"])

closer_to_ed = (res["norm_ed_best"] > res["norm_c05"]).mean()
print(f"  norm-BLOSUM to C05    : mean={res['norm_c05'].mean():.3f} max={res['norm_c05'].max():.3f}")
print(f"  norm-BLOSUM to ED best: mean={res['norm_ed_best'].mean():.3f} max={res['norm_ed_best'].max():.3f}")
print(f"  fraction of natural H3s closer to an ED variant than to C05: {closer_to_ed:.1%}")
gain = (res["norm_ed_best"] - res["norm_c05"])
print(f"  mean similarity gain (ED_best - C05): {gain.mean():+.3f}  "
      f"(p90={np.percentile(gain,90):+.3f})")

out = Path(SCRATCH) / "oas_h3_cache" / "ed_cloud_diag_natural.parquet"
out.parent.mkdir(parents=True, exist_ok=True)
res.to_parquet(out)
print(f"\nWrote {out}")
