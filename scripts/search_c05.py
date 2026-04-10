#!/usr/bin/env python3
"""Search the filtered OAS dataset for sequences similar to wildtype C05.

Runs two complementary analyses:
1. Exact substring scan for the C05 CDRH3 — ground truth for "is the wildtype present?"
2. MMseqs2 easy-search of the full C05 heavy chain against the filtered FASTA —
   surfaces germline/framework neighbors and characterizes the similarity distribution.

Writes a human-readable report to stdout and to ``$SCRATCH_DIR/c05_search/search_report.txt``.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from Bio import SeqIO

C05_HEAVY = (
    "EVQLQESGGGLVQPGESLRLSCVGSGSSFGESTLSYYAVSWVRQAPGKGLEWLSIINAGGGDIDYADSVEG"
    "RFTISRDNSKETLYLQMTNLRVEDTGVYYCAKHMSMQQVVSAGWERADLVGDAFDVWGQGTMVTVSS"
)
C05_CDRH3 = "HMSMQQVVSAGWERADLVGDAFDV"

MMSEQS_FORMAT = (
    "query,target,pident,alnlen,mismatch,gapopen,"
    "qstart,qend,tstart,tend,evalue,bits,qcov,tcov,tseq"
)
MMSEQS_COLS = MMSEQS_FORMAT.split(",")

IDENTITY_BINS = [
    (0.30, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 0.70),
    (0.70, 0.80), (0.80, 0.90), (0.90, 0.95), (0.95, 1.00),
    (1.00, 1.0001),  # 100% bin
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--fasta",
        default="/cluster/project/infk/krause/mdenegri/protein-design/oas_filtered.fasta",
        help="Target FASTA to search.",
    )
    p.add_argument(
        "--scratch-dir",
        default=os.environ.get("SCRATCH_DIR"),
        help="Scratch directory; work dir is <scratch>/c05_search/. Defaults to $SCRATCH_DIR.",
    )
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--min-seq-id", type=float, default=0.3)
    p.add_argument("--evalue", type=float, default=1e-5)
    p.add_argument("--max-seqs", type=int, default=5000)
    return p.parse_args()


def exact_cdrh3_scan(fasta: Path) -> list[str]:
    """Single pass over the target FASTA collecting IDs containing the C05 CDRH3 substring."""
    print(f"[scan] Scanning {fasta} for exact CDRH3 substring '{C05_CDRH3}'...", flush=True)
    matches: list[str] = []
    n = 0
    for rec in SeqIO.parse(str(fasta), "fasta"):
        n += 1
        if C05_CDRH3 in str(rec.seq):
            matches.append(rec.id)
        if n % 1_000_000 == 0:
            print(f"[scan]   processed {n:,} sequences, {len(matches)} matches so far", flush=True)
    print(f"[scan] Done. Total sequences scanned: {n:,}. Exact CDRH3 matches: {len(matches)}", flush=True)
    return matches


def write_query_fasta(path: Path) -> None:
    path.write_text(f">C05_heavy\n{C05_HEAVY}\n")


def run_mmseqs(query: Path, target: Path, results: Path, tmp: Path, args: argparse.Namespace) -> None:
    cmd = [
        "mmseqs", "easy-search",
        str(query), str(target), str(results), str(tmp),
        "-s", "7.5",
        "--search-type", "1",
        "--max-seqs", str(args.max_seqs),
        "-e", str(args.evalue),
        "--min-seq-id", str(args.min_seq_id),
        "--threads", str(args.threads),
        "--format-output", MMSEQS_FORMAT,
    ]
    print(f"[mmseqs] Running: {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[mmseqs] FAILED with exit code {e.returncode}", file=sys.stderr)
        raise


def load_results(results: Path) -> pd.DataFrame:
    if not results.exists() or results.stat().st_size == 0:
        return pd.DataFrame(columns=MMSEQS_COLS)
    df = pd.read_csv(results, sep="\t", header=None, names=MMSEQS_COLS)
    # pident is reported as fraction (0-1) by MMseqs2; keep as fraction internally
    return df.sort_values("pident", ascending=False).reset_index(drop=True)


def histogram(values: list[float]) -> list[tuple[str, int, float]]:
    """Bin values into the standard identity bins. Returns (label, count, pct)."""
    out = []
    total = len(values) if values else 1
    for lo, hi in IDENTITY_BINS:
        if hi > 1.0:
            label = "100%"
            count = sum(1 for v in values if v >= 1.0)
        else:
            label = f"{int(lo*100)}-{int(hi*100)}%"
            count = sum(1 for v in values if lo <= v < hi)
        out.append((label, count, 100.0 * count / total))
    return out


def ascii_bar(pct: float, width: int = 40) -> str:
    n = int(round(pct / 100.0 * width))
    return "#" * n + "." * (width - n)


def build_report(
    cdrh3_matches: list[str],
    df: pd.DataFrame,
    target_fasta: Path,
    args: argparse.Namespace,
) -> str:
    lines: list[str] = []
    w = lines.append

    w("=" * 78)
    w("C05 SIMILARITY SEARCH REPORT")
    w("=" * 78)
    w(f"Target FASTA : {target_fasta}")
    w(f"Query        : C05 heavy chain ({len(C05_HEAVY)} aa)")
    w(f"C05 CDRH3    : {C05_CDRH3} ({len(C05_CDRH3)} aa)")
    w(f"MMseqs2      : -s 7.5  -e {args.evalue}  --min-seq-id {args.min_seq_id}  --max-seqs {args.max_seqs}")
    w("")

    # ── Section 1: Exact CDRH3 match ────────────────────────────────────────
    w("-" * 78)
    w("[1] EXACT WILDTYPE CDRH3 MATCH")
    w("-" * 78)
    if cdrh3_matches:
        w(f"Exact CDRH3 match found: YES  ({len(cdrh3_matches)} sequence(s))")
        w("Matching sequence IDs:")
        for sid in cdrh3_matches[:50]:
            w(f"  {sid}")
        if len(cdrh3_matches) > 50:
            w(f"  ... ({len(cdrh3_matches) - 50} more)")
    else:
        w("Exact CDRH3 match found: NO")
        w("The wildtype C05 CDRH3 substring is not present in the dataset.")
    w("")

    # ── Section 2: Top 20 hits by full-chain identity ───────────────────────
    w("-" * 78)
    w("[2] TOP 20 MOST SIMILAR ANTIBODIES (full heavy chain)")
    w("-" * 78)
    if df.empty:
        w("No MMseqs2 hits above thresholds.")
        w("Try lowering --min-seq-id or raising --evalue.")
    else:
        w(f"{'rank':>4}  {'seq_id':<40}  {'pident':>7}  {'alnlen':>6}  {'evalue':>10}  {'bits':>6}")
        for i, row in df.head(20).iterrows():
            w(
                f"{i+1:>4}  {row['target']:<40}  "
                f"{row['pident']*100:>6.2f}%  {int(row['alnlen']):>6}  "
                f"{row['evalue']:>10.2e}  {row['bits']:>6.1f}"
            )
    w("")

    # ── Section 3: Identity distribution ────────────────────────────────────
    w("-" * 78)
    w("[3] FULL-CHAIN IDENTITY DISTRIBUTION")
    w("-" * 78)
    if df.empty:
        w("(no hits)")
    else:
        bins = histogram(df["pident"].tolist())
        w(f"{'bin':<10}  {'count':>8}  {'pct':>6}  histogram")
        for label, count, pct in bins:
            w(f"{label:<10}  {count:>8}  {pct:>5.1f}%  {ascii_bar(pct)}")
    w("")

    # ── Section 4: Summary ──────────────────────────────────────────────────
    w("-" * 78)
    w("[4] SUMMARY")
    w("-" * 78)
    w(f"Exact wildtype CDRH3 present : {'YES' if cdrh3_matches else 'NO'}")
    w(f"Total MMseqs2 hits           : {len(df)}")
    if not df.empty:
        w(f"Highest full-chain identity  : {df['pident'].max()*100:.2f}%")
        w(f"Mean full-chain identity     : {df['pident'].mean()*100:.2f}%")
        if not cdrh3_matches:
            best = df.iloc[0]
            w("")
            w("Wildtype CDRH3 not present — closest full-chain match:")
            w(f"  seq_id : {best['target']}")
            w(f"  pident : {best['pident']*100:.2f}%")
            w(f"  alnlen : {int(best['alnlen'])}")
            w(f"  evalue : {best['evalue']:.2e}")
            w(f"  tseq   : {best['tseq']}")
    w("=" * 78)

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    if not args.scratch_dir:
        sys.exit("ERROR: --scratch-dir not provided and $SCRATCH_DIR is not set.")
    if shutil.which("mmseqs") is None:
        sys.exit("ERROR: 'mmseqs' not found on PATH. Load the MMseqs2 module first.")

    target_fasta = Path(args.fasta)
    if not target_fasta.exists():
        sys.exit(f"ERROR: target FASTA not found: {target_fasta}")

    work = Path(args.scratch_dir) / "c05_search"
    work.mkdir(parents=True, exist_ok=True)
    query_fa = work / "query.fasta"
    results_tsv = work / "results.tsv"
    tmp_dir = work / "mmseqs_search_tmp"
    report_path = work / "search_report.txt"

    # Step 1: query FASTA
    write_query_fasta(query_fa)

    # Step 2: exact CDRH3 substring scan
    cdrh3_matches = exact_cdrh3_scan(target_fasta)

    # Step 3: MMseqs2 easy-search
    run_mmseqs(query_fa, target_fasta, results_tsv, tmp_dir, args)

    # Step 4: parse
    df = load_results(results_tsv)
    print(f"[parse] Loaded {len(df)} hits from {results_tsv}", flush=True)

    # Step 5: report
    report = build_report(cdrh3_matches, df, target_fasta, args)
    report_path.write_text(report)
    print(report)
    print(f"[done] Report written to {report_path}")


if __name__ == "__main__":
    main()
