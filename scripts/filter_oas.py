#!/usr/bin/env python
"""Filter OAS CSV files and write passing sequences to a FASTA file.

Reads all .csv.gz files from $SCRATCH_DIR/oas_raw/, applies quality filters,
and writes a single FASTA file with sequential IDs.
"""

import glob
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm


# Substrings in ANARCI_status that indicate bad sequences
ANARCI_REJECT = ["Shorter", "Missing Conserved Cysteine", "insert"]

MIN_SEQ_LEN = 50


def passes_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply quality filters to an OAS dataframe and return passing rows."""
    mask = pd.Series(True, index=df.index)

    # productive == "T"
    if "productive" in df.columns:
        mask &= df["productive"] == "T"

    # ANARCI_status does not contain reject substrings
    if "ANARCI_status" in df.columns:
        pattern = "|".join(ANARCI_REJECT)
        mask &= ~df["ANARCI_status"].str.contains(pattern, case=False, na=False)

    # sequence_alignment_aa is non-null and long enough
    if "sequence_alignment_aa" in df.columns:
        mask &= df["sequence_alignment_aa"].notna()
        mask &= df["sequence_alignment_aa"].str.len() >= MIN_SEQ_LEN

    # cdr3_aa is non-null and non-empty
    if "cdr3_aa" in df.columns:
        mask &= df["cdr3_aa"].notna()
        mask &= df["cdr3_aa"].str.len() > 0

    return df.loc[mask]


def main() -> None:
    load_dotenv()
    scratch_dir = os.environ.get("SCRATCH_DIR")
    if not scratch_dir:
        print("Error: SCRATCH_DIR env var not set", file=sys.stderr)
        sys.exit(1)

    input_dir = os.path.join(scratch_dir, "oas_raw")
    output_path = os.path.join(scratch_dir, "oas_filtered.fasta")

    csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv.gz")))
    if not csv_files:
        print(f"No .csv.gz files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV files in {input_dir}")

    seq_id = 0
    total_passing = 0
    total_rejected = 0

    with open(output_path, "w") as fasta:
        for csv_path in tqdm(csv_files, desc="Processing files"):
            fname = os.path.basename(csv_path)
            try:
                # header=1 because row 0 is OAS JSON metadata
                df = pd.read_csv(csv_path, header=1, compression="gzip", low_memory=False)
            except Exception as e:
                print(f"  [ERROR] {fname}: {e}")
                continue

            n_total = len(df)
            filtered = passes_filters(df)
            n_pass = len(filtered)
            n_reject = n_total - n_pass

            total_passing += n_pass
            total_rejected += n_reject

            # Write passing sequences to FASTA
            for seq in filtered["sequence_alignment_aa"]:
                clean_seq = str(seq).replace("-", "")  # strip alignment gaps
                fasta.write(f">seq_{seq_id}\n{clean_seq}\n")
                seq_id += 1

            print(f"  {fname}: {n_pass:,} pass / {n_reject:,} reject (of {n_total:,})")

    print(f"\nSummary:")
    print(f"  Total passing:  {total_passing:,}")
    print(f"  Total rejected: {total_rejected:,}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
