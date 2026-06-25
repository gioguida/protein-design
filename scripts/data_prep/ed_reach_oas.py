#!/usr/bin/env python3
"""Decisive test: does the ED2/5/811 H3 cloud reach natural OAS H3s that C05 misses?

Fast vectorized identity screen restricted to length-24 OAS H3s (the ED/C05 core
length). For C05 and a stratified sample of ED variants, compute the best (max)
identity to the OAS length-24 universe, and the union of OAS H3s reachable at
several identity thresholds. Substitution-only proxy (ignores indel neighbors).
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
PROJECT = os.environ["PROJECT_DIR"]
SCRATCH = os.environ["SCRATCH_DIR"]

C05 = "HMSMQQVVSAGWERADLVGDAFDV"  # 24-aa core
L = 24

# ---- load OAS length-24 unique H3s -> uint8 matrix ----
oas = pd.read_parquet(f"{SCRATCH}/oas_h3_cache/oas_unique_h3.parquet")
oas24 = oas[oas.len == L].reset_index(drop=True)
counts = oas24["count"].to_numpy()
M = np.frombuffer("".join(oas24["cdr3_aa"]).encode("ascii"), dtype=np.uint8).reshape(-1, L)
print(f"OAS length-{L} unique H3s: {M.shape[0]:,}  (covering {counts.sum():,} seqs)")

c05 = np.frombuffer(C05.encode("ascii"), dtype=np.uint8)


def max_identity_and_reach(query: np.ndarray, thresholds):
    """Return (max_identity, {thr: idx_array of OAS rows >= thr}) for one query."""
    ident = (M == query).sum(axis=1) / L
    return ident.max(), {t: np.where(ident >= t)[0] for t in thresholds}


THRS = [0.50, 0.60, 0.70]

# ---- C05 reference ----
c05_max, c05_reach = max_identity_and_reach(c05, THRS)
print(f"\nC05  best natural identity (len-24 OAS): {c05_max:.3f}")
for t in THRS:
    idx = c05_reach[t]
    print(f"  C05 reach >= {t:.2f} identity: {len(idx):>7,} unique H3  "
          f"({counts[idx].sum():>10,} seqs)")

# ---- ED cloud: stratified sample, best-natural-identity by library ----
print("\n=== ED variants: best natural identity (len-24 OAS) ===")
rng = np.random.default_rng(0)
ed_union_reach = {t: set() for t in THRS}
ed_best_by_lib = {}
SAMPLE = 1500  # per library
for lib in ("ed2_m22", "ed5_m22", "ed811_m22"):
    aa = pd.read_csv(f"{PROJECT}/data/dms_splits/{lib}/train.csv", usecols=["aa"])["aa"]
    aa = aa.astype(str).str.strip()
    aa = aa[aa.str.len() == L].unique()
    samp = rng.choice(aa, size=min(SAMPLE, len(aa)), replace=False)
    bests = []
    for s in samp:
        q = np.frombuffer(s.encode("ascii"), dtype=np.uint8)
        b, reach = max_identity_and_reach(q, THRS)
        bests.append(b)
        for t in THRS:
            ed_union_reach[t].update(reach[t].tolist())
    bests = np.array(bests)
    ed_best_by_lib[lib] = bests
    print(f"  {lib:10s} (n_samp={len(samp)}): best-nat-identity "
          f"min={bests.min():.2f} mean={bests.mean():.2f} max={bests.max():.2f}  "
          f"| frac with a natural >=0.6: {(bests>=0.6).mean():.1%}")

# ---- union reach of the ED sample vs C05 alone ----
print("\n=== Union natural OAS reach: ED sample vs C05 alone ===")
for t in THRS:
    ed_idx = np.array(sorted(ed_union_reach[t]), dtype=int)
    ed_seqs = counts[ed_idx].sum() if len(ed_idx) else 0
    c_idx = c05_reach[t]
    extra = len(set(ed_idx.tolist()) - set(c_idx.tolist()))
    print(f"  thr>={t:.2f}: C05={len(c_idx):>7,} H3  |  ED-sample={len(ed_idx):>7,} H3 "
          f"({ed_seqs:>10,} seqs)  |  NEW (not in C05 reach): {extra:>7,} H3")

print(f"\nNote: ED reach uses {SAMPLE}/library sample of ~1M variants -> union is a "
      "LOWER BOUND on the full cloud's reach.")
