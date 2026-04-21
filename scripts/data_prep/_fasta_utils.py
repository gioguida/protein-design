"""Shared FASTA helpers for data-prep scripts."""
from __future__ import annotations

import gzip
from pathlib import Path


def stream_fasta_subset(fasta_path: Path, wanted_ids: set[str], output_path: Path) -> int:
    """One-pass stream of a FASTA, writing records whose header ID is in wanted_ids.

    Header parsing matches the convention elsewhere in this repo: ID is the first
    whitespace-delimited token after '>'. Transparently handles gzipped input.
    Returns the number of records written.
    """
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
