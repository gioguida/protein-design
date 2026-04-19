#!/usr/bin/env python
"""Filter OAS CSV files and write passing sequences to a FASTA file and metadata CSV.

Reads all .csv.gz files from $SCRATCH_DIR/oas_raw/, applies quality filters,
and writes:
  - oas_filtered.fasta  : one entry per passing sequence, header = {run}_{row_index}
  - oas_filtered.csv.gz : flat table with file-level + per-sequence metadata

Filters applied in order:
  1. Quality: productive==T, ANARCI_status not flagged, seq length ≥ 50, CDR3 present
  2. N-terminal fragmentation: first IMGT position in ANARCI_numbering must be ≤ 16
  3. C-terminal fragmentation: at least one IMGT position ≥ 121 must be present
"""

import gzip
import glob
import json
import os
import re
import sys

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm


# "Shorter" is intentionally absent — N/C-terminal truncation is handled explicitly
# by _apply_fragmentation() using IMGT position thresholds.
ANARCI_REJECT = ["Missing Conserved Cysteine", "insert"]

MIN_SEQ_LEN = 50

FRAG_MAX_NTERM_POS = 16   # first IMGT position must be ≤ this
FRAG_MIN_CTERM_POS = 121  # at least one IMGT position ≥ this must be present

# Matches IMGT positions 1–16 (N-terminal coverage check)
# Equivalent to: min(positions) <= FRAG_MAX_NTERM_POS
_NTERM_PASS_RE = re.compile(r"'(?:[1-9]|1[0-6])\s*':\s*'[A-Z-]'")

# Matches IMGT positions 121–199 (covers full FWH4 range and beyond)
_CTERM_RE = re.compile(r"'(?:12[1-9]|1[3-9]\d)\s*':\s*'[A-Z-]'")

# Standard amino acids (X = unknown, kept as-is)
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWYX")

# Per-sequence AIRR columns to keep in the CSV
SEQUENCE_COLS = [
    "sequence_alignment_aa",
    "v_call", "d_call", "j_call",
    "v_identity", "d_identity", "j_identity",
    "cdr1_aa", "cdr2_aa", "cdr3_aa",
    "junction_aa",
    "fwr1_aa", "fwr2_aa", "fwr3_aa", "fwr4_aa",
    "productive",
    "ANARCI_status",
    "Redundancy",
]

def clean_seq(s: str) -> str:
    """Strip alignment gaps and replace non-standard characters with X."""
    s = str(s).replace("-", "")
    return "".join(c if c in STANDARD_AA else "X" for c in s)


def _parse_file_meta(csv_path: str) -> dict:
    """Read the OAS JSON metadata from row 0 of a gzipped CSV."""
    with gzip.open(csv_path, "rt") as fh:
        raw = fh.readline().strip().strip('"').replace('""', '"')
    meta = json.loads(raw)
    return {
        "run":         str(meta["Run"]) if meta.get("Run") is not None else None,
        "species":     meta.get("Species"),
        "age":         str(meta["Age"]) if meta.get("Age") is not None else None,
        "b_source":    meta.get("BSource"),
        "b_type":      meta.get("BType"),
        "vaccine":     meta.get("Vaccine"),
        "disease":     meta.get("Disease"),
        "subject":     meta.get("Subject"),
        "longitudinal": meta.get("Longitudinal"),
        "isotype":     meta.get("Isotype"),
        "chain":       meta.get("Chain"),
        "source_url":  meta.get("Link"),
    }


def _apply_fragmentation(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Return (filtered_df, n_nterm_removed, n_cterm_removed)."""
    if "ANARCI_numbering" not in df.columns:
        return df, 0, 0

    # ── N-terminal ──────────────────────────────────────────────────────────
    # Pass if any IMGT position ≤ FRAG_MAX_NTERM_POS (16) is present.
    # Logically equivalent to min(positions) <= 16, but uses str.contains
    # instead of str.extractall to avoid building a massive MultiIndex Series.
    nterm_pass = df["ANARCI_numbering"].str.contains(_NTERM_PASS_RE, na=False)
    n_nterm = int((~nterm_pass).sum())
    df = df.loc[nterm_pass]

    if df.empty:
        return df, n_nterm, 0

    # ── C-terminal ──────────────────────────────────────────────────────────
    cterm_pass = df["ANARCI_numbering"].str.contains(_CTERM_RE, na=False)
    n_cterm = int((~cterm_pass).sum())
    df = df.loc[cterm_pass]

    return df, n_nterm, n_cterm


def passes_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Return (passing_df, n_nterm_removed, n_cterm_removed)."""
    mask = pd.Series(True, index=df.index)

    if "productive" in df.columns:
        mask &= df["productive"] == "T"

    if "ANARCI_status" in df.columns:
        pattern = "|".join(ANARCI_REJECT)
        mask &= ~df["ANARCI_status"].str.contains(pattern, case=False, na=False)

    if "sequence_alignment_aa" in df.columns:
        mask &= df["sequence_alignment_aa"].notna()
        mask &= df["sequence_alignment_aa"].str.len() >= MIN_SEQ_LEN

    if "cdr3_aa" in df.columns:
        mask &= df["cdr3_aa"].notna()
        mask &= df["cdr3_aa"].str.len() > 0

    df = df.loc[mask]
    if df.empty:
        return df, 0, 0

    return _apply_fragmentation(df)


def main() -> None:
    load_dotenv()
    scratch_dir = os.environ.get("SCRATCH_DIR")
    if not scratch_dir:
        print("Error: SCRATCH_DIR env var not set", file=sys.stderr)
        sys.exit(1)

    input_dir = os.path.join(scratch_dir, "oas_raw")
    output_fasta = os.path.join(scratch_dir, "oas_filtered.fasta")
    output_csv   = os.path.join(scratch_dir, "oas_filtered.csv.gz")

    csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv.gz")))
    if not csv_files:
        print(f"No .csv.gz files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV files in {input_dir}")

    total_passing       = 0
    total_rejected      = 0
    total_nterm_removed = 0
    total_cterm_removed = 0

    csv_header_written = False

    with open(output_fasta, "w") as fasta, gzip.open(output_csv, "wt") as csv_fh:
        for csv_path in tqdm(csv_files, desc="Processing files"):
            fname = os.path.basename(csv_path)

            try:
                file_meta = _parse_file_meta(csv_path)
            except Exception as e:
                print(f"  [ERROR] {fname}: failed to parse metadata: {e}", file=sys.stderr)
                continue

            # Use filename stem (not Run) as the seq_id prefix: ~5% of OAS files
            # share a Run with another file (unit splits like SRRxxx_1, _2, _3),
            # so {run}_{i} would collide. {file_stem}_{i} is globally unique.
            file_stem = fname.removesuffix(".csv.gz")

            try:
                # header=1 because row 0 is OAS JSON metadata
                df = pd.read_csv(csv_path, header=1, compression="gzip", low_memory=False)
            except Exception as e:
                print(f"  [ERROR] {fname}: {e}", file=sys.stderr)
                continue

            n_total = len(df)
            filtered, n_nterm, n_cterm = passes_filters(df)
            n_pass   = len(filtered)
            n_reject = n_total - n_pass

            total_passing       += n_pass
            total_rejected      += n_reject
            total_nterm_removed += n_nterm
            total_cterm_removed += n_cterm

            if n_pass == 0:
                print(
                    f"  {fname}: {n_pass:,} pass / {n_reject:,} reject (of {n_total:,})"
                    f"  [N-term frag: {n_nterm:,}, C-term frag: {n_cterm:,}]"
                )
                continue

            filtered = filtered.reset_index(drop=True)
            seq_ids = [f"{file_stem}_{i}" for i in range(len(filtered))]

            # ── FASTA ────────────────────────────────────────────────────────
            for seq_id, seq in zip(seq_ids, filtered["sequence_alignment_aa"]):
                fasta.write(f">{seq_id}\n{clean_seq(seq)}\n")

            # ── CSV batch ────────────────────────────────────────────────────
            batch = {"seq_id": seq_ids}
            for k, v in file_meta.items():
                batch[k] = [v] * n_pass
            for col in SEQUENCE_COLS:
                batch[col] = filtered[col].tolist() if col in filtered.columns else [None] * n_pass

            batch_df = pd.DataFrame(batch)
            batch_df.to_csv(csv_fh, header=not csv_header_written, index=False)
            csv_header_written = True

            print(
                f"  {fname}: {n_pass:,} pass / {n_reject:,} reject (of {n_total:,})"
                f"  [N-term frag: {n_nterm:,}, C-term frag: {n_cterm:,}]"
            )

    print(f"\nSummary:")
    print(f"  Total passing:               {total_passing:,}")
    print(f"  Total rejected:              {total_rejected:,}")
    print(f"  -- N-terminal fragmentation: {total_nterm_removed:,}")
    print(f"  -- C-terminal fragmentation: {total_cterm_removed:,}")
    print(f"  FASTA: {output_fasta}")
    print(f"  CSV:   {output_csv}")


if __name__ == "__main__":
    main()
