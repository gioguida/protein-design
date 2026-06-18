#!/usr/bin/env python3
"""Build per-sequence CDR-H3 ± flank windows for a VH FASTA corpus.

For every VH in ``--fasta`` we resolve its OAS ``cdr3_aa`` by streaming the
(huge, ~192M-row) ``oas_filtered.csv.gz`` in pandas chunks, filtered to just the
seq_ids present in the FASTA. We then locate ``cdr3_aa`` as a substring of the
VH and expand it by ``--flank`` residues on each side. The result is a small
parquet cache consumed by the ``cdr`` single-mask training variant.

Coordinates are residue indices (0-based) into the VH. The training collator
maps residue ``p`` to token ``p + 1`` (BOS offset).

Intermediates/caches live in ``$SCRATCH_DIR/cdr_windows/``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from dotenv import load_dotenv

# This repo keeps cluster paths in .env.local (loaded by common_setup.sh in
# sbatch); fall back to the default .env too.
load_dotenv(".env.local")
load_dotenv()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    project = os.environ.get("PROJECT_DIR", ".")
    scratch = os.environ.get("SCRATCH_DIR", ".")
    p.add_argument(
        "--fasta", required=True,
        help="VH FASTA whose headers are OAS seq_ids.",
    )
    p.add_argument(
        "--csv", default=os.path.join(project, "data", "oas", "oas_filtered.csv.gz"),
        help="OAS metadata CSV(.gz) with seq_id + cdr3_aa columns.",
    )
    p.add_argument("--flank", type=int, default=3, help="Framework residues per side.")
    p.add_argument("--chunksize", type=int, default=500_000)
    p.add_argument("--scratch-dir", default=scratch)
    p.add_argument("--rebuild", action="store_true", help="Recompute even if cached.")
    p.add_argument(
        "--validate-only", action="store_true",
        help="Build/load, print match-rate + 3 examples, do not require a full run.",
    )
    return p.parse_args()


def cache_path(scratch_dir: str, fasta: str, flank: int) -> Path:
    stem = Path(fasta).stem
    return Path(scratch_dir) / "cdr_windows" / f"{stem}_flank{flank}.parquet"


def load_fasta(fasta: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    for rec in SeqIO.parse(str(fasta), "fasta"):
        seqs[rec.id] = str(rec.seq)
    return seqs


def resolve_cdr3(
    csv_path: Path, wanted: set[str], chunksize: int
) -> dict[str, str]:
    """Stream the OAS CSV, returning seq_id -> cdr3_aa for wanted ids."""
    found: dict[str, str] = {}
    remaining = set(wanted)
    rows = 0
    reader = pd.read_csv(csv_path, usecols=["seq_id", "cdr3_aa"], chunksize=chunksize)
    for chunk in reader:
        rows += len(chunk)
        sub = chunk[chunk["seq_id"].isin(remaining)].dropna(subset=["cdr3_aa"])
        for sid, h3 in zip(sub["seq_id"].astype(str), sub["cdr3_aa"].astype(str)):
            if sid in remaining:
                found[sid] = h3
                remaining.discard(sid)
        print(
            f"[resolve] scanned {rows:,} rows; resolved {len(found):,}/{len(wanted):,}",
            flush=True,
        )
        if not remaining:
            print("[resolve] all seq_ids resolved; stopping early.", flush=True)
            break
    return found


def build_windows(
    seqs: dict[str, str], cdr3_map: dict[str, str], flank: int
) -> pd.DataFrame:
    records = []
    n_missing_cdr3 = 0
    n_substr_not_found = 0
    n_multi_match = 0
    for sid, vh in seqs.items():
        cdr3 = cdr3_map.get(sid)
        if cdr3 is None:
            n_missing_cdr3 += 1
            records.append(
                dict(seq_id=sid, cdr3_aa=None, vh_len=len(vh), cdr3_start=-1,
                     cdr3_end=-1, win_start=-1, win_end=-1, win_len=0, found=False)
            )
            continue
        start = vh.find(cdr3)
        if start < 0:
            n_substr_not_found += 1
            records.append(
                dict(seq_id=sid, cdr3_aa=cdr3, vh_len=len(vh), cdr3_start=-1,
                     cdr3_end=-1, win_start=-1, win_end=-1, win_len=0, found=False)
            )
            continue
        if vh.rfind(cdr3) != start:
            n_multi_match += 1  # ambiguous; keep first occurrence
        end = start + len(cdr3)
        win_start = max(0, start - flank)
        win_end = min(len(vh), end + flank)
        records.append(
            dict(seq_id=sid, cdr3_aa=cdr3, vh_len=len(vh), cdr3_start=start,
                 cdr3_end=end, win_start=win_start, win_end=win_end,
                 win_len=win_end - win_start, found=True)
        )
    df = pd.DataFrame.from_records(records)
    df.attrs["n_missing_cdr3"] = n_missing_cdr3
    df.attrs["n_substr_not_found"] = n_substr_not_found
    df.attrs["n_multi_match"] = n_multi_match
    return df


def report(df: pd.DataFrame, seqs: dict[str, str], flank: int) -> None:
    n = len(df)
    found = int(df["found"].sum())
    print(f"\n[report] sequences: {n:,}")
    print(f"[report] resolved windows: {found:,} ({100.0*found/max(n,1):.2f}%)")
    print(f"[report] missing cdr3_aa:  {df.attrs.get('n_missing_cdr3', 0):,}")
    print(f"[report] substr not found: {df.attrs.get('n_substr_not_found', 0):,}")
    print(f"[report] multi-match (rfind != find): {df.attrs.get('n_multi_match', 0):,}")
    ok = df[df["found"]]
    if len(ok):
        print(
            f"[report] win_len range [{int(ok['win_len'].min())}, "
            f"{int(ok['win_len'].max())}], mean {ok['win_len'].mean():.1f}"
        )
    print("\n[report] 3 examples (| marks the ±flank window):")
    for _, r in ok.head(3).iterrows():
        vh = seqs[r["seq_id"]]
        ws, we = int(r["win_start"]), int(r["win_end"])
        marked = vh[:ws] + "|" + vh[ws:we] + "|" + vh[we:]
        print(f"  {r['seq_id']}  cdr3={r['cdr3_aa']}")
        print(f"    {marked}")


def main() -> None:
    args = parse_args()
    out = cache_path(args.scratch_dir, args.fasta, args.flank)

    if out.exists() and not args.rebuild:
        print(f"[cache] reusing {out}", flush=True)
        df = pd.read_parquet(out)
        # attrs are not persisted in parquet; recompute summary counts.
        df.attrs["n_missing_cdr3"] = int((~df["found"] & df["cdr3_aa"].isna()).sum())
        df.attrs["n_substr_not_found"] = int((~df["found"] & df["cdr3_aa"].notna()).sum())
        df.attrs["n_multi_match"] = 0
        seqs = load_fasta(Path(args.fasta))
        report(df, seqs, args.flank)
        return

    seqs = load_fasta(Path(args.fasta))
    print(f"[fasta] loaded {len(seqs):,} VH sequences from {args.fasta}", flush=True)

    cdr3_map = resolve_cdr3(Path(args.csv), set(seqs.keys()), args.chunksize)
    print(f"[resolve] resolved cdr3_aa for {len(cdr3_map):,} seq_ids", flush=True)

    df = build_windows(seqs, cdr3_map, args.flank)
    report(df, seqs, args.flank)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\n[write] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
