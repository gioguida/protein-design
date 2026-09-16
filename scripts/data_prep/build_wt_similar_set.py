#!/usr/bin/env python3
"""Build a general-purpose "WT-similar" evotuning corpus (see paper.py, WT-similar-set).

Ranks every OAS candidate's CDR-H3 against a given WT's CDR-H3 via BLOSUM62
global alignment (normalized by the WT's self-alignment score), takes the
top-N ranked sequences, and reports the achieved similarity distribution as
a diagnostic (rather than gating on a fixed threshold, which can silently
return too few -- or zero -- hits for a WT that is an outlier in OAS).

No germline (V/J gene) pre-filter: it would need the WT's own germline call,
which we have no direct way to compute (no ANARCI/IgBlast annotation
available for a sequence outside OAS), and CDR-H3-only scoring already
targets the hypervariable region directly.

Pipeline:
  1. Stream --meta, collect unique cdr3_aa strings and their seq_ids.
     Persist a pickled mapping unique_h3 -> [seq_id, ...] under --scratch-dir.
  2. For each unique H3, globally align to --wt-cdrh3 using Biopython's
     PairwiseAligner with BLOSUM62 + BLAST-style affine gap penalties.
     Normalize: norm_score = align(x, ref) / align(ref, ref). Cache per-H3
     scores in a parquet under --scratch-dir so re-runs with a different N
     skip the alignment pass.
  3. Rank unique H3 by score, expand to individual seq_ids (each seq_id
     inherits its H3's score) until --top-n sequences are collected. Report
     the min/median/max score among the selection.
  4. Stream --fasta to extract the selected seq_ids into the final FASTA.

Usage:
  uv run scripts/data_prep/build_wt_similar_set.py \\
      --wt-name c05 --wt-cdrh3 AKHMSMQQVVSAGWERADLVGDAFDV --top-n 5000
"""
from __future__ import annotations

import argparse
import os
import pickle
import statistics
import sys
from pathlib import Path

import pandas as pd
from Bio.Align import PairwiseAligner, substitution_matrices
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fasta_utils import stream_fasta_subset  # noqa: E402
from meta_io import iter_meta_chunks  # noqa: E402

load_dotenv()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    project = os.environ.get("PROJECT_DIR", ".")
    scratch = os.environ.get("SCRATCH_DIR", ".")

    p.add_argument("--wt-name", required=True, help="Short label for output files/caches, e.g. 'c05'.")
    p.add_argument("--wt-cdrh3", required=True, help="WT CDR-H3 in OAS format (with flanks, matches cdr3_aa convention).")
    p.add_argument("--top-n", type=int, default=5000)

    p.add_argument("--meta", default=os.path.join(project, "data", "oas", "oas_dedup_meta.parquet"))
    p.add_argument("--fasta", default=os.path.join(project, "data", "oas", "oas_dedup_rep_seq.fasta"))
    p.add_argument("--scratch-dir", default=scratch)
    p.add_argument("--output-dir", default=os.path.join(project, "data", "wt_similar"))

    p.add_argument("--rebuild-index", action="store_true", help="Force rebuild of the scratch H3 mapping.")
    p.add_argument("--rebuild-scores", action="store_true", help="Force recomputation of the per-H3 alignment scores.")
    p.add_argument("--gap-open", type=float, default=-11.0, help="BLAST default: -11.")
    p.add_argument("--gap-extend", type=float, default=-1.0, help="BLAST default: -1.")
    p.add_argument("--chunksize", type=int, default=500_000)
    return p.parse_args()


def build_h3_mapping(meta_path: Path, mapping_pkl: Path, chunksize: int) -> None:
    print(f"[index] Building unique-H3 -> seq_ids mapping from {meta_path}", flush=True)
    mapping: dict[str, list[str]] = {}
    rows = 0
    reader = iter_meta_chunks(str(meta_path), columns=["seq_id", "cdr3_aa"], chunksize=chunksize)
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
    return aligner


def compute_scores(h3s: list[str], aligner: PairwiseAligner, ref: str) -> list[float]:
    self_score = float(aligner.score(ref, ref))
    print(f"[align] Self-score for reference ({len(ref)} aa): {self_score:.1f}", flush=True)
    out: list[float] = []
    for i, h3 in enumerate(h3s):
        try:
            s = float(aligner.score(h3, ref))
        except (KeyError, ValueError):
            s = float("nan")
        out.append(s / self_score)
        if (i + 1) % 1_000_000 == 0:
            print(f"[align]   aligned {i+1:,} / {len(h3s):,}", flush=True)
    return out


def select_top_n(scored: pd.DataFrame, mapping: dict[str, list[str]], top_n: int) -> tuple[list[str], list[float]]:
    """Rank unique H3 by score desc, expand to seq_ids until top_n sequences collected."""
    ranked = scored.dropna(subset=["norm_score"]).sort_values("norm_score", ascending=False).reset_index(drop=True)
    seq_ids: list[str] = []
    seq_scores: list[float] = []
    for _, row in ranked.iterrows():
        h3, score = row["h3"], row["norm_score"]
        remaining = top_n - len(seq_ids)
        if remaining <= 0:
            break
        group = sorted(mapping[h3])  # deterministic order within a tied-score group
        take = group[:remaining]
        seq_ids.extend(take)
        seq_scores.extend([score] * len(take))
    return seq_ids, seq_scores


def main() -> None:
    args = parse_args()

    work = Path(args.scratch_dir) / "wt_similar" / args.wt_name
    work.mkdir(parents=True, exist_ok=True)
    mapping_pkl = work / "h3_mapping.pkl"
    scores_parquet = work / "h3_scores.parquet"

    if args.rebuild_index or not mapping_pkl.exists():
        build_h3_mapping(Path(args.meta), mapping_pkl, args.chunksize)
    else:
        print(f"[index] Reusing cached {mapping_pkl}", flush=True)
    mapping = load_h3_mapping(mapping_pkl)
    ordered = list(mapping.keys())
    print(f"[index] {len(ordered):,} unique H3 sequences.", flush=True)

    if args.rebuild_scores or not scores_parquet.exists():
        aligner = make_aligner(args.gap_open, args.gap_extend)
        scores = compute_scores(ordered, aligner, args.wt_cdrh3)
        pd.DataFrame({"h3": ordered, "norm_score": scores}).to_parquet(scores_parquet, index=False)
        print(f"[align] Wrote {scores_parquet}", flush=True)
    else:
        print(f"[align] Reusing cached {scores_parquet}", flush=True)
    scored = pd.read_parquet(scores_parquet)

    valid = scored["norm_score"].dropna()
    print(f"[align] {len(valid):,} / {len(scored):,} H3 scored (non-NaN). range [{valid.min():.3f}, {valid.max():.3f}]", flush=True)

    seq_ids, seq_scores = select_top_n(scored, mapping, args.top_n)
    print(f"[select] Selected {len(seq_ids):,} / {args.top_n:,} requested sequences.", flush=True)
    print(
        f"[select] score distribution: min={min(seq_scores):.3f} "
        f"median={statistics.median(seq_scores):.3f} max={max(seq_scores):.3f}",
        flush=True,
    )

    out_dir = Path(args.output_dir) / args.wt_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.wt_name}_wt_similar_top{args.top_n}.fasta"
    written = stream_fasta_subset(Path(args.fasta), set(seq_ids), out_path)
    missing = len(seq_ids) - written
    print(
        f"[write] Wrote {written:,} sequences to {out_path}"
        + (f" ({missing:,} ids not found in FASTA)" if missing else ""),
        flush=True,
    )

    scores_out = out_dir / f"{args.wt_name}_wt_similar_top{args.top_n}_scores.csv"
    pd.DataFrame({"seq_id": seq_ids, "norm_score": seq_scores}).to_csv(scores_out, index=False)
    print(f"[write] Wrote per-sequence scores to {scores_out}", flush=True)


if __name__ == "__main__":
    main()
