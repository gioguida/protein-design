#!/usr/bin/env python3
"""Build a C05 CDR-H3 similarity corpus using BLOSUM62 + global pairwise alignment.

Pipeline:
  1. Stream oas_filtered.csv.gz, collect unique cdr3_aa strings and their seq_ids.
     Persist a pickled mapping unique_h3 -> [seq_id, ...] under $SCRATCH_DIR.
  2. For each unique H3, globally align to the C05 CDR-H3 using Biopython's
     PairwiseAligner with BLOSUM62 + BLAST-style affine gap penalties.
     Normalize: norm_score = align(x, ref) / align(ref, ref). Cache per-H3
     scores in a parquet under $SCRATCH_DIR so re-runs with different
     thresholds skip the alignment pass.
  3. Either --preview (histogram of normalized scores) or --threshold X
     (keep H3s with norm_score >= X, expand to seq_ids, stream
     oas_filtered.fasta into the final FASTA under $PROJECT_DIR/datasets/c05/).

Intermediates in $SCRATCH_DIR/c05_cdrh3_blosum/; final FASTA in
$PROJECT_DIR/datasets/c05/.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
from Bio.Align import PairwiseAligner, substitution_matrices
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from search_c05 import C05_CDRH3_OAS, ascii_bar  # noqa: E402
from _fasta_utils import stream_fasta_subset  # noqa: E402

load_dotenv()

SCORE_BINS = [
    (0.00, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 0.50),
    (0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90),
    (0.90, 1.00), (1.00, 1.0001),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    project = os.environ.get("PROJECT_DIR", ".")
    scratch = os.environ.get("SCRATCH_DIR", ".")

    p.add_argument("--csv", default=os.path.join(project, "datasets", "oas_filtered.csv.gz"))
    p.add_argument("--fasta", default=os.path.join(project, "datasets", "oas_filtered.fasta"))
    p.add_argument("--scratch-dir", default=scratch)
    p.add_argument("--output-dir", default=os.path.join(project, "datasets", "c05"))

    p.add_argument("--preview", action="store_true",
                   help="Print normalized-score histogram and exit.")
    p.add_argument("--threshold", type=float, default=None,
                   help="Normalized-score cutoff (0-1). Required unless --preview.")
    p.add_argument("--rebuild-index", action="store_true",
                   help="Force rebuild of the scratch H3 mapping.")
    p.add_argument("--rebuild-scores", action="store_true",
                   help="Force recomputation of the per-H3 alignment scores.")

    p.add_argument("--gap-open", type=float, default=-11.0, help="BLAST default: -11.")
    p.add_argument("--gap-extend", type=float, default=-1.0, help="BLAST default: -1.")
    p.add_argument("--chunksize", type=int, default=500_000)
    return p.parse_args()


def build_h3_mapping(csv_path: Path, mapping_pkl: Path, chunksize: int) -> None:
    print(f"[index] Building unique-H3 -> seq_ids mapping from {csv_path}", flush=True)
    mapping: dict[str, list[str]] = {}
    rows = 0
    reader = pd.read_csv(csv_path, usecols=["seq_id", "cdr3_aa"], chunksize=chunksize)
    for chunk in reader:
        rows += len(chunk)
        sub = chunk.dropna(subset=["cdr3_aa"])
        for seq_id, h3 in zip(sub["seq_id"], sub["cdr3_aa"].astype(str)):
            mapping.setdefault(h3, []).append(seq_id)
        if rows % 10_000_000 == 0:
            print(f"[index]   scanned {rows:,} rows, {len(mapping):,} unique H3", flush=True)
    print(f"[index] Done. {rows:,} rows, {len(mapping):,} unique H3.", flush=True)
    mapping_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(mapping_pkl, "wb") as f:
        pickle.dump(mapping, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_h3_mapping(mapping_pkl: Path) -> dict[str, list[str]]:
    with open(mapping_pkl, "rb") as f:
        return pickle.load(f)


def make_aligner(gap_open: float, gap_extend: float) -> PairwiseAligner:
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = gap_open
    aligner.extend_gap_score = gap_extend
    # No end-gap penalty discount; full global alignment.
    return aligner


def compute_scores(h3s: list[str], aligner: PairwiseAligner, ref: str) -> list[float]:
    self_score = float(aligner.score(ref, ref))
    print(f"[align] Self-score for reference ({len(ref)} aa): {self_score:.1f}", flush=True)
    out: list[float] = []
    for i, h3 in enumerate(h3s):
        try:
            s = float(aligner.score(h3, ref))
        except (KeyError, ValueError):
            # cdr3_aa may contain characters not in BLOSUM62 (e.g. '*' or 'X' stop/unk).
            s = float("nan")
        out.append(s / self_score)
        if (i + 1) % 100_000 == 0:
            print(f"[align]   aligned {i+1:,} / {len(h3s):,}", flush=True)
    return out


def histogram_scores(values: list[float]) -> None:
    total = sum(1 for v in values if v == v) or 1  # skip NaN in denominator
    print(f"{'bin':<10}  {'count':>10}  {'pct':>6}  histogram")
    for lo, hi in SCORE_BINS:
        if hi > 1.0:
            label, count = ">=1.00", sum(1 for v in values if v >= 1.0)
        else:
            label = f"{lo:.2f}-{hi:.2f}"
            count = sum(1 for v in values if v == v and lo <= v < hi)
        pct = 100.0 * count / total
        print(f"{label:<10}  {count:>10,}  {pct:>5.1f}%  {ascii_bar(pct)}")


def main() -> None:
    args = parse_args()
    if not args.preview and args.threshold is None:
        sys.exit("ERROR: pass --preview or --threshold X")

    work = Path(args.scratch_dir) / "c05_cdrh3_blosum"
    work.mkdir(parents=True, exist_ok=True)
    mapping_pkl = work / "h3_mapping.pkl"
    scores_parquet = work / "h3_scores.parquet"

    # Step 1: build or reuse the H3 mapping
    if args.rebuild_index or not mapping_pkl.exists():
        build_h3_mapping(Path(args.csv), mapping_pkl, args.chunksize)
    else:
        print(f"[index] Reusing cached {mapping_pkl}", flush=True)
    mapping = load_h3_mapping(mapping_pkl)
    ordered = list(mapping.keys())
    print(f"[index] {len(ordered):,} unique H3 sequences.", flush=True)

    # Step 2: compute or load normalized alignment scores
    if args.rebuild_scores or not scores_parquet.exists():
        aligner = make_aligner(args.gap_open, args.gap_extend)
        scores = compute_scores(ordered, aligner, C05_CDRH3_OAS)
        pd.DataFrame({"h3": ordered, "norm_score": scores}).to_parquet(scores_parquet, index=False)
        print(f"[align] Wrote {scores_parquet}", flush=True)
    else:
        print(f"[align] Reusing cached {scores_parquet}", flush=True)
    scored = pd.read_parquet(scores_parquet)

    valid = scored["norm_score"].dropna()
    print(f"[align] {len(valid):,} / {len(scored):,} H3 scored (non-NaN). "
          f"range [{valid.min():.3f}, {valid.max():.3f}]", flush=True)

    # Step 3: preview or build final FASTA
    if args.preview:
        histogram_scores(scored["norm_score"].tolist())
        print("\nCumulative seq counts at each threshold:")
        print(f"{'threshold':>10}  {'unique_h3':>10}  {'total_seqs':>12}")
        for t in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]:
            hit = scored[scored["norm_score"] >= t]
            total = sum(len(mapping[h]) for h in hit["h3"])
            print(f"{t:>10.2f}  {len(hit):>10,}  {total:>12,}")
        return

    thresh = float(args.threshold)
    hit = scored[scored["norm_score"] >= thresh]
    if hit.empty:
        sys.exit(f"ERROR: no H3 above norm_score {thresh}. Try a lower threshold.")
    wanted: set[str] = set()
    for h3 in hit["h3"]:
        wanted.update(mapping[h3])
    print(f"[expand] {len(hit):,} unique H3 -> {len(wanted):,} seq_ids", flush=True)

    pct = int(round(thresh * 100))
    out_path = Path(args.output_dir) / f"c05_cdrh3_blosum{pct}.fasta"
    written = stream_fasta_subset(Path(args.fasta), wanted, out_path)
    missing = len(wanted) - written
    print(f"[write] Wrote {written:,} sequences to {out_path}"
          + (f" ({missing:,} ids not found in FASTA)" if missing else ""))


if __name__ == "__main__":
    main()
