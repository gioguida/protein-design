#!/usr/bin/env python3
"""Extract C05 training-corpus variants used for ablation finetuning runs.

Modes:
  --preview-cdrh3
      Scan oas_filtered.csv.gz (OAS with cdr3_aa), print a CDR-H3 identity
      histogram vs. the C05 reference CDR-H3. No files written.

  --variant fullseq_60 [--pident-threshold 0.60]
      Filter the MMseqs2 results.tsv (from search_c05.py) by full-chain
      pident >= threshold and write c05_fullseq_<PCT>.fasta from the tseq
      column. No FASTA re-lookup needed.

  --variant cdrh3_sim --cdrh3-threshold X
      Scan oas_filtered.csv.gz for rows with CDR-H3 identity >= X vs. the
      C05 reference, then stream oas_filtered.fasta to extract their full
      sequences into c05_cdrh3_sim.fasta.
"""
from __future__ import annotations

import argparse
import gzip
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Reuse C05 reference sequences and the CDR-H3 identity metric from search_c05.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from search_c05 import (  # noqa: E402
    C05_CDRH3_OAS,
    CDRH3_BINS,
    MMSEQS_COLS,
    ascii_bar,
    cdrh3_identity,
    histogram,
)

load_dotenv()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    project = os.environ.get("PROJECT_DIR", ".")
    scratch = os.environ.get("SCRATCH_DIR", ".")

    p.add_argument("--preview-cdrh3", action="store_true",
                   help="Print CDR-H3 identity histogram over the full OAS CSV and exit.")
    p.add_argument("--variant", choices=["fullseq_60", "cdrh3_sim"],
                   help="Which variant FASTA to produce.")

    p.add_argument("--results-tsv", default=os.path.join(scratch, "c05_search", "results.tsv"),
                   help="MMseqs2 results TSV (for fullseq_60).")
    p.add_argument("--csv", default=os.path.join(project, "datasets", "oas_filtered.csv.gz"),
                   help="OAS filtered metadata CSV with seq_id + cdr3_aa (gzipped).")
    p.add_argument("--fasta", default=os.path.join(project, "datasets", "oas_filtered.fasta"),
                   help="OAS filtered FASTA (for cdrh3_sim lookup).")
    p.add_argument("--output-dir", default=os.path.join(project, "datasets", "c05"),
                   help="Directory for output FASTA files.")

    p.add_argument("--pident-threshold", type=float, default=0.60,
                   help="Full-chain pident cutoff for fullseq_60 variant.")
    p.add_argument("--cdrh3-threshold", type=float, default=None,
                   help="CDR-H3 identity cutoff for cdrh3_sim variant (required for that mode).")
    p.add_argument("--chunksize", type=int, default=200_000)
    return p.parse_args()


def load_cdrh3_ids(csv_path: Path, threshold: float, chunksize: int) -> tuple[set[str], int]:
    """Stream the OAS CSV, compute CDR-H3 identity, return seq_ids >= threshold plus row count."""
    keep: set[str] = set()
    total = 0
    print(f"[cdrh3] Scanning {csv_path} for cdrh3_id >= {threshold:.3f}...", flush=True)
    reader = pd.read_csv(csv_path, usecols=["seq_id", "cdr3_aa"], chunksize=chunksize)
    for chunk in reader:
        total += len(chunk)
        sub = chunk.dropna(subset=["cdr3_aa"])
        ids = sub["cdr3_aa"].astype(str).map(lambda s: cdrh3_identity(s, C05_CDRH3_OAS))
        mask = ids >= threshold
        keep.update(sub.loc[mask, "seq_id"].tolist())
        if total % 1_000_000 == 0:
            print(f"[cdrh3]   scanned {total:,} rows, kept {len(keep):,} so far", flush=True)
    print(f"[cdrh3] Done. Scanned {total:,} rows. Kept {len(keep):,}.", flush=True)
    return keep, total


def preview_cdrh3_histogram(csv_path: Path, chunksize: int) -> None:
    print(f"[preview] Computing CDR-H3 identity distribution over {csv_path}", flush=True)
    print(f"[preview] Reference (OAS format): {C05_CDRH3_OAS} ({len(C05_CDRH3_OAS)} aa)", flush=True)
    all_ids: list[float] = []
    total = 0
    reader = pd.read_csv(csv_path, usecols=["cdr3_aa"], chunksize=chunksize)
    for chunk in reader:
        total += len(chunk)
        sub = chunk.dropna(subset=["cdr3_aa"])
        all_ids.extend(cdrh3_identity(str(s), C05_CDRH3_OAS) for s in sub["cdr3_aa"])
        if total % 1_000_000 == 0:
            print(f"[preview]   processed {total:,} rows", flush=True)
    print(f"[preview] Processed {total:,} rows total.", flush=True)

    bins = histogram(all_ids, CDRH3_BINS)
    print()
    print(f"{'bin':<10}  {'count':>10}  {'pct':>6}  histogram")
    for label, count, pct in bins:
        print(f"{label:<10}  {count:>10,}  {pct:>5.1f}%  {ascii_bar(pct)}")
    print()
    thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]
    print("Cumulative counts (>= threshold):")
    for t in thresholds:
        n = sum(1 for v in all_ids if v >= t)
        pct = 100.0 * n / max(total, 1)
        print(f"  >= {t:.2f}: {n:>10,}  ({pct:5.2f}%)")
    if all_ids:
        print(f"\nMean CDR-H3 identity : {sum(all_ids) / len(all_ids) * 100:.3f}%")
        print(f"Max  CDR-H3 identity : {max(all_ids) * 100:.2f}%")


def write_fullseq_variant(
    results_tsv: Path,
    output_path: Path,
    pident_threshold: float,
) -> int:
    if not results_tsv.exists():
        sys.exit(f"ERROR: results.tsv not found at {results_tsv}")
    df = pd.read_csv(results_tsv, sep="\t", header=None, names=MMSEQS_COLS)
    df = df.drop_duplicates(subset="target").reset_index(drop=True)
    # MMseqs2 pident is a percentage; normalize to fraction.
    df["pident_frac"] = df["pident"] / 100.0
    keep = df[df["pident_frac"] >= pident_threshold].reset_index(drop=True)
    print(f"[fullseq] {len(df):,} unique hits -> {len(keep):,} at pident >= {pident_threshold:.2f}", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for _, row in keep.iterrows():
            f.write(f">{row['target']}\n{row['tseq']}\n")
    print(f"[fullseq] Wrote {len(keep):,} sequences to {output_path}")
    return len(keep)


def stream_fasta_subset(fasta_path: Path, wanted_ids: set[str], output_path: Path) -> int:
    """One-pass stream of a FASTA, writing records whose header is in wanted_ids."""
    opener = gzip.open if str(fasta_path).endswith(".gz") else open
    written = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with opener(fasta_path, "rt") as fin, open(output_path, "w") as fout:
        keep = False
        for line in fin:
            if line.startswith(">"):
                header = line[1:].strip().split()[0]
                keep = header in wanted_ids
                if keep:
                    fout.write(f">{header}\n")
                    written += 1
            elif keep:
                fout.write(line)
    return written


def write_cdrh3_variant(
    csv_path: Path,
    fasta_path: Path,
    output_path: Path,
    threshold: float,
    chunksize: int,
) -> int:
    wanted, _ = load_cdrh3_ids(csv_path, threshold, chunksize)
    if not wanted:
        sys.exit(f"ERROR: no sequences matched cdrh3_id >= {threshold}. Lower the threshold.")
    print(f"[cdrh3] Extracting {len(wanted):,} sequences from {fasta_path}...", flush=True)
    written = stream_fasta_subset(fasta_path, wanted, output_path)
    missing = len(wanted) - written
    print(f"[cdrh3] Wrote {written:,} sequences to {output_path}"
          + (f" ({missing:,} ids not found in FASTA)" if missing else ""))
    return written


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    fasta_path = Path(args.fasta)
    results_tsv = Path(args.results_tsv)
    out_dir = Path(args.output_dir)

    if args.preview_cdrh3:
        if not csv_path.exists():
            sys.exit(f"ERROR: CSV not found: {csv_path}")
        preview_cdrh3_histogram(csv_path, args.chunksize)
        return

    if args.variant == "fullseq_60":
        pct = int(round(args.pident_threshold * 100))
        out = out_dir / f"c05_fullseq_{pct}.fasta"
        write_fullseq_variant(results_tsv, out, args.pident_threshold)
        return

    if args.variant == "cdrh3_sim":
        if args.cdrh3_threshold is None:
            sys.exit("ERROR: --cdrh3-threshold is required for --variant cdrh3_sim")
        if not csv_path.exists():
            sys.exit(f"ERROR: CSV not found: {csv_path}")
        if not fasta_path.exists():
            sys.exit(f"ERROR: FASTA not found: {fasta_path}")
        out = out_dir / "c05_cdrh3_sim.fasta"
        write_cdrh3_variant(csv_path, fasta_path, out, args.cdrh3_threshold, args.chunksize)
        return

    sys.exit("ERROR: pass --preview-cdrh3 or --variant {fullseq_60,cdrh3_sim}")


if __name__ == "__main__":
    main()
