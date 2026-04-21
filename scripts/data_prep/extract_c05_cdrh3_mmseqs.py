#!/usr/bin/env python3
"""Build a C05 CDR-H3 similarity corpus using MMseqs2 (short-sequence search).

Pipeline:
  1. Stream oas_filtered.csv.gz and build a deduped CDR-H3 FASTA under $SCRATCH_DIR
     (one record per unique cdr3_aa string; many OAS seq_ids share the same H3).
     Also persist a pickled mapping unique_h3 -> [seq_id, ...] for the final lookup.
  2. Run MMseqs2 easy-search with the C05 CDR-H3 as query, using short-sequence
     tuning (-k 5, --comp-bias-corr 0).
  3. Either --preview (print a pident histogram and exit) or --threshold X
     (keep hits with pident >= X, expand to the full seq_id set, stream
     oas_filtered.fasta to write the final FASTA under $PROJECT_DIR/datasets/c05/).

All MMseqs2 intermediates (query/target FASTAs, tmp/, results.tsv, mapping.pkl)
live under $SCRATCH_DIR/c05_cdrh3_mmseqs/. Only the final FASTA goes to the
project datasets directory.
"""
from __future__ import annotations

import argparse
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from search_c05 import C05_CDRH3_OAS, ascii_bar  # noqa: E402
from _fasta_utils import stream_fasta_subset  # noqa: E402

load_dotenv()

MMSEQS_FORMAT = "query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits"
MMSEQS_COLS = MMSEQS_FORMAT.split(",")

PIDENT_BINS = [
    (0.20, 0.30), (0.30, 0.40), (0.40, 0.50), (0.50, 0.60),
    (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.00),
    (1.00, 1.0001),
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
                   help="Run search + print pident histogram, do not write a final FASTA.")
    p.add_argument("--threshold", type=float, default=None,
                   help="pident cutoff (0-1) for the final corpus. Required unless --preview.")
    p.add_argument("--rebuild-index", action="store_true",
                   help="Force rebuild of the scratch H3 target FASTA + mapping cache.")

    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--sensitivity", type=float, default=7.5)
    p.add_argument("--kmer", type=int, default=5, help="MMseqs2 k-mer size for prefilter (default 5 for short seqs).")
    p.add_argument("--evalue", type=float, default=100.0,
                   help="Loose evalue (short seqs need relaxed cutoff).")
    p.add_argument("--min-seq-id", type=float, default=0.2)
    p.add_argument("--max-seqs", type=int, default=200_000)
    p.add_argument("--chunksize", type=int, default=500_000)
    return p.parse_args()


def build_h3_index(csv_path: Path, target_fasta: Path, mapping_pkl: Path, chunksize: int) -> None:
    """Stream OAS CSV, build deduped H3 FASTA + {h3: [seq_ids]} mapping."""
    print(f"[index] Building deduped H3 FASTA from {csv_path}", flush=True)
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

    target_fasta.parent.mkdir(parents=True, exist_ok=True)
    with open(target_fasta, "w") as f:
        for i, h3 in enumerate(mapping.keys()):
            f.write(f">h3_{i}\n{h3}\n")
    # Save index in the SAME order so results.tsv "target" (h3_<i>) maps back cleanly.
    ordered = list(mapping.keys())
    with open(mapping_pkl, "wb") as f:
        pickle.dump({"ordered": ordered, "mapping": mapping}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[index] Wrote {target_fasta} ({len(ordered):,} records) and {mapping_pkl}", flush=True)


def load_index(mapping_pkl: Path) -> tuple[list[str], dict[str, list[str]]]:
    with open(mapping_pkl, "rb") as f:
        obj = pickle.load(f)
    return obj["ordered"], obj["mapping"]


def write_query_fasta(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f">C05_cdrh3\n{C05_CDRH3_OAS}\n")


def run_mmseqs(query: Path, target: Path, results: Path, tmp: Path, args: argparse.Namespace) -> None:
    tmp.mkdir(parents=True, exist_ok=True)
    cmd = [
        "mmseqs", "easy-search",
        str(query), str(target), str(results), str(tmp),
        "-s", str(args.sensitivity),
        "--search-type", "1",
        "-k", str(args.kmer),
        "--comp-bias-corr", "0",
        "--min-seq-id", str(args.min_seq_id),
        "-e", str(args.evalue),
        "--max-seqs", str(args.max_seqs),
        "--threads", str(args.threads),
        "--format-output", MMSEQS_FORMAT,
    ]
    print(f"[mmseqs] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def load_results(results: Path) -> pd.DataFrame:
    if not results.exists() or results.stat().st_size == 0:
        return pd.DataFrame(columns=MMSEQS_COLS)
    df = pd.read_csv(results, sep="\t", header=None, names=MMSEQS_COLS)
    df["pident"] = df["pident"] / 100.0  # MMseqs2 reports percent; convert to fraction
    return df.sort_values("pident", ascending=False).reset_index(drop=True)


def histogram_pident(values: list[float]) -> None:
    total = len(values) if values else 1
    print(f"{'bin':<10}  {'count':>10}  {'pct':>6}  histogram")
    for lo, hi in PIDENT_BINS:
        if hi > 1.0:
            label, count = "100%", sum(1 for v in values if v >= 1.0)
        else:
            label = f"{int(lo*100)}-{int(hi*100)}%"
            count = sum(1 for v in values if lo <= v < hi)
        pct = 100.0 * count / total
        print(f"{label:<10}  {count:>10,}  {pct:>5.1f}%  {ascii_bar(pct)}")


def main() -> None:
    args = parse_args()
    if not args.preview and args.threshold is None:
        sys.exit("ERROR: pass --preview or --threshold X")
    if shutil.which("mmseqs") is None:
        sys.exit("ERROR: 'mmseqs' not on PATH. Load the MMseqs2 module in common_setup.sh.")

    work = Path(args.scratch_dir) / "c05_cdrh3_mmseqs"
    work.mkdir(parents=True, exist_ok=True)
    query_fa = work / "query_h3.fasta"
    target_fa = work / "target_h3.fasta"
    mapping_pkl = work / "h3_mapping.pkl"
    results_tsv = work / "results.tsv"
    tmp_dir = work / "mmseqs_tmp"

    # Step 1: build or reuse the H3 index
    if args.rebuild_index or not target_fa.exists() or not mapping_pkl.exists():
        build_h3_index(Path(args.csv), target_fa, mapping_pkl, args.chunksize)
    else:
        print(f"[index] Reusing cached {target_fa} and {mapping_pkl}", flush=True)

    # Step 2: query + MMseqs2 search
    write_query_fasta(query_fa)
    run_mmseqs(query_fa, target_fa, results_tsv, tmp_dir, args)

    df = load_results(results_tsv)
    print(f"[search] {len(df):,} hits. pident range [{df['pident'].min():.3f}, {df['pident'].max():.3f}]"
          if len(df) else "[search] No hits.", flush=True)

    # Step 3: preview or build final FASTA
    if args.preview:
        histogram_pident(df["pident"].tolist())
        thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
        ordered, mapping = load_index(mapping_pkl)
        print("\nCumulative seq counts at each pident threshold:")
        print(f"{'threshold':>10}  {'unique_h3':>10}  {'total_seqs':>12}")
        for t in thresholds:
            hit_mask = df["pident"] >= t
            unique_h3 = hit_mask.sum()
            if unique_h3 == 0:
                print(f"{t:>10.2f}  {0:>10}  {0:>12}")
                continue
            targets = df.loc[hit_mask, "target"].tolist()
            total = sum(len(mapping[ordered[int(t_name.removeprefix('h3_'))]]) for t_name in targets)
            print(f"{t:>10.2f}  {unique_h3:>10,}  {total:>12,}")
        return

    # Build mode
    thresh = float(args.threshold)
    hit = df[df["pident"] >= thresh].reset_index(drop=True)
    if hit.empty:
        sys.exit(f"ERROR: no hits above pident {thresh}. Try a lower threshold.")
    ordered, mapping = load_index(mapping_pkl)
    wanted: set[str] = set()
    for target_name in hit["target"]:
        idx = int(target_name.removeprefix("h3_"))
        wanted.update(mapping[ordered[idx]])
    print(f"[expand] {len(hit):,} unique H3 hits -> {len(wanted):,} seq_ids", flush=True)

    pct = int(round(thresh * 100))
    out_path = Path(args.output_dir) / f"c05_cdrh3_mmseqs{pct}.fasta"
    written = stream_fasta_subset(Path(args.fasta), wanted, out_path)
    missing = len(wanted) - written
    print(f"[write] Wrote {written:,} sequences to {out_path}"
          + (f" ({missing:,} ids not found in FASTA)" if missing else ""))


if __name__ == "__main__":
    main()
