#!/usr/bin/env python3
"""Extract C05 TTT training corpora from MMseqs2 search results.

Reads the results.tsv produced by search_c05.py and writes:
  1. c05_vh_pid30.fasta — all C05-similar OAS sequences (MSA-TTT corpus)
  2. c05_ref.fasta      — the C05 VH reference sequence alone (single-seq TTT)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

C05_HEAVY = (
    "EVQLQESGGGLVQPGESLRLSCVGSGSSFGESTLSYYAVSWVRQAPGKGLEWLSIINAGGGDIDYADSVEG"
    "RFTISRDNSKETLYLQMTNLRVEDTGVYYCAKHMSMQQVVSAGWERADLVGDAFDVWGQGTMVTVSS"
)

MMSEQS_COLS = [
    "query", "target", "pident", "alnlen", "mismatch", "gapopen",
    "qstart", "qend", "tstart", "tend", "evalue", "bits", "qcov", "tcov", "tseq",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    scratch = os.environ.get("SCRATCH_DIR", ".")
    project = os.environ.get("PROJECT_DIR", ".")
    p.add_argument(
        "--results-tsv",
        default=os.path.join(scratch, "c05_search", "results.tsv"),
        help="MMseqs2 results TSV from search_c05.py.",
    )
    p.add_argument(
        "--output-dir",
        default=os.path.join(project, "data", "c05"),
        help="Directory for output FASTA files.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_path = Path(args.results_tsv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load search results
    df = pd.read_csv(results_path, sep="\t", header=None, names=MMSEQS_COLS)
    print(f"Loaded {len(df)} hits from {results_path}")

    # Deduplicate by target ID (MMseqs2 may return duplicates)
    df = df.drop_duplicates(subset="target").reset_index(drop=True)
    print(f"After dedup: {len(df)} unique sequences")

    # Write MSA-TTT corpus (full C05-similar set at pid >= 0.30)
    msa_path = output_dir / "c05_vh_pid30.fasta"
    with open(msa_path, "w") as f:
        for _, row in df.iterrows():
            f.write(f">{row['target']}\n{row['tseq']}\n")
    print(f"Wrote {len(df)} sequences to {msa_path}")

    # Write single-sequence TTT corpus
    single_path = output_dir / "c05_ref.fasta"
    with open(single_path, "w") as f:
        f.write(f">C05_heavy\n{C05_HEAVY}\n")
    print(f"Wrote C05 VH to {single_path}")


if __name__ == "__main__":
    main()
