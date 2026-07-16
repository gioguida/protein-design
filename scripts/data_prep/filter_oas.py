#!/usr/bin/env python
"""Filter OAS CSV files and write passing sequences to a FASTA file and metadata CSV.

Two modes:
  - Whole-corpus (default): reads every .csv.gz file in $SCRATCH_DIR/oas_raw/.
  - Shard mode (--input-list): reads only the files listed in a manifest, for
    parallel SLURM-array processing of the corpus (needed because file sizes
    are extremely skewed -- a handful of files are 1000x the median size, so
    a single-process run both takes very long and holds an exact-dup hash set
    that grows without bound across the whole corpus). See
    scripts/data_prep/make_oas_shards.py to build manifests,
    bash_scripts/utils/filter_oas_shard.sbatch to run them as an array, and
    scripts/data_prep/merge_oas_shards.py to combine shard outputs.

Writes:
  - <output-fasta>  : one entry per passing sequence, header = {file_stem}_{row_index}
  - <output-csv>    : flat table with file-level + per-sequence metadata
  - <summary-json>  : optional; machine-readable summary counts + length
                       histogram, so merge_oas_shards.py can aggregate them
                       without re-scanning the (potentially huge) outputs

Filters applied in order:
  1. Quality: productive==T, ANARCI_status not flagged, seq length ≥ 50, CDR3 present
  2. N-terminal fragmentation: first IMGT position in ANARCI_numbering must be ≤ 17
  3. C-terminal fragmentation: at least one IMGT position ≥ 121 must be present
  4. Exact-duplicate removal: by hash of the cleaned sequence, first occurrence
     wins, scoped to all files one process sees (the whole corpus in
     whole-corpus mode, one shard in shard mode -- cross-shard exact dups are
     still caught by the later 95%-identity Linclust step, which is a superset
     of exact-match clustering).
"""

import argparse
import gzip
import glob
import hashlib
import json
import os
import re
import sys

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm


ANARCI_REJECT = ["Missing Conserved Cysteine", "insert"]

MIN_SEQ_LEN = 50

# IMGT V-domain numbering runs 1..128. "Missing >16 N-term residues" means
# rejecting only when the first present position is >= 18, i.e. passing
# requires first present position <= 17 (missing exactly 16 residues passes).
FRAG_MAX_NTERM_POS = 17   # first IMGT position must be ≤ this
FRAG_MIN_CTERM_POS = 121  # at least one IMGT position ≥ this must be present

# Matches IMGT positions 1–17 (N-terminal coverage check)
# Equivalent to: min(positions) <= FRAG_MAX_NTERM_POS
_NTERM_PASS_RE = re.compile(r"'(?:[1-9]|1[0-7])\s*':\s*'[A-Z-]'")

# Matches IMGT positions 121–199 (covers full FWH4 range and beyond)
_CTERM_RE = re.compile(r"'(?:12[1-9]|1[3-9]\d)\s*':\s*'[A-Z-]'")

# Standard amino acids (X = unknown, kept as-is)
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWYX")
_NON_STANDARD_RE = re.compile(f"[^{''.join(sorted(STANDARD_AA))}]")

# Digest size (bytes) for the exact-dup hash set. 16 bytes (128 bit) keeps
# collision probability negligible at corpus scale (~10^-19 at 3x10^8
# sequences by the birthday bound) while using far less memory than storing
# full sequence strings.
DEDUP_DIGEST_SIZE = 16

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

# Raw OAS CSVs carry ~97 AIRR columns (nucleotide alignments, CIGAR strings,
# per-segment start/end indices, ...) that filter_oas.py never reads. Loading
# all of them via pandas (each cell a separate Python object) is the dominant
# memory cost for the largest files (up to ~2.2GB compressed, ~670k rows) --
# large enough to OOM a 32GB job on a single file. ANARCI_numbering is needed
# for fragmentation checks (_apply_fragmentation) but dropped from the output.
NEEDED_COLS = sorted(set(SEQUENCE_COLS + ["ANARCI_numbering", "cdr3_aa"]))


def clean_seq_series(s: pd.Series) -> pd.Series:
    """Vectorized: strip alignment gaps, replace non-standard chars with X."""
    s = s.str.replace("-", "", regex=False)
    return s.str.replace(_NON_STANDARD_RE, "X", regex=True)


def dedup_exact(cleaned_seqs: list[str], seen: set[bytes]) -> list[bool]:
    """Exact-duplicate filter: keep only the first occurrence of each cleaned
    sequence among the files this process sees (mutates `seen` in place)."""
    keep = []
    for s in cleaned_seqs:
        digest = hashlib.blake2b(s.encode("ascii"), digest_size=DEDUP_DIGEST_SIZE).digest()
        if digest in seen:
            keep.append(False)
        else:
            seen.add(digest)
            keep.append(True)
    return keep


# Keys of the dict returned by _parse_file_meta, in order -- exported so
# merge_oas_shards.py can reconstruct the CSV header without duplicating
# this list by hand.
FILE_META_COLS = [
    "run", "species", "age", "b_source", "b_type", "vaccine", "disease",
    "subject", "longitudinal", "isotype", "chain", "source_url",
]


def _parse_file_meta(csv_path: str) -> dict:
    """Read the OAS JSON metadata from row 0 of a gzipped CSV."""
    with gzip.open(csv_path, "rt") as fh:
        raw = fh.readline().strip().strip('"').replace('""', '"')
    meta = json.loads(raw)
    return dict(zip(FILE_META_COLS, [
        str(meta["Run"]) if meta.get("Run") is not None else None,
        meta.get("Species"),
        str(meta["Age"]) if meta.get("Age") is not None else None,
        meta.get("BSource"),
        meta.get("BType"),
        meta.get("Vaccine"),
        meta.get("Disease"),
        meta.get("Subject"),
        meta.get("Longitudinal"),
        meta.get("Isotype"),
        meta.get("Chain"),
        meta.get("Link"),
    ]))


def _apply_fragmentation(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Return (filtered_df, n_nterm_removed, n_cterm_removed)."""
    if "ANARCI_numbering" not in df.columns:
        return df, 0, 0

    # ── N-terminal ──────────────────────────────────────────────────────────
    # Pass if any IMGT position ≤ FRAG_MAX_NTERM_POS (17) is present.
    # Logically equivalent to min(positions) <= 17, but uses str.contains
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


# Rows per pd.read_csv chunk. Bounds peak memory to a small, roughly constant
# multiple of this regardless of file size -- needed because ANARCI_numbering
# (required for fragmentation checks) is ~13x the length of the sequence
# itself, and a handful of raw files have 600k-1M+ rows: loading one such file
# whole, then filtering it through several sequential DataFrame copies
# (passes_filters -> _apply_fragmentation's two .loc[] calls -> dedup), was
# enough to OOM a 48GB job on a single file.
CHUNK_SIZE = 100_000


def _process_chunk(
    chunk: pd.DataFrame, file_stem: str, row_offset: int, fasta, csv_fh,
    file_meta: dict, seen_hashes: set[bytes], length_hist: dict[int, int],
    csv_header_written: bool,
) -> tuple[int, int, int, int, int, int, int, bool]:
    """Filter, dedup, and write one chunk. Returns
    (n_total, n_pass, n_nterm, n_cterm, n_dup, n_final, new_row_offset, csv_header_written).
    """
    n_total = len(chunk)
    filtered, n_nterm, n_cterm = passes_filters(chunk)
    n_pass = len(filtered)

    if n_pass == 0:
        return n_total, n_pass, n_nterm, n_cterm, 0, 0, row_offset, csv_header_written

    filtered = filtered.reset_index(drop=True)
    cleaned_seqs = clean_seq_series(filtered["sequence_alignment_aa"]).tolist()

    # Exact-dup removal (scope = all chunks/files this process sees; see module docstring).
    keep_mask = dedup_exact(cleaned_seqs, seen_hashes)
    n_dup = keep_mask.count(False)
    if n_dup:
        filtered = filtered.loc[keep_mask].reset_index(drop=True)
        cleaned_seqs = [s for s, k in zip(cleaned_seqs, keep_mask) if k]

    n_final = len(filtered)
    if n_final == 0:
        return n_total, n_pass, n_nterm, n_cterm, n_dup, 0, row_offset, csv_header_written

    seq_ids = [f"{file_stem}_{row_offset + i}" for i in range(n_final)]

    # ── FASTA ────────────────────────────────────────────────────────────
    for seq_id, seq in zip(seq_ids, cleaned_seqs):
        fasta.write(f">{seq_id}\n{seq}\n")
        length_hist[len(seq)] = length_hist.get(len(seq), 0) + 1

    # ── CSV batch ────────────────────────────────────────────────────────
    batch = {"seq_id": seq_ids}
    for k, v in file_meta.items():
        batch[k] = [v] * n_final
    for col in SEQUENCE_COLS:
        batch[col] = filtered[col].tolist() if col in filtered.columns else [None] * n_final

    batch_df = pd.DataFrame(batch)
    batch_df.to_csv(csv_fh, header=not csv_header_written, index=False)

    return n_total, n_pass, n_nterm, n_cterm, n_dup, n_final, row_offset + n_final, True


def process_files(csv_files: list[str], output_fasta: str, output_csv: str, write_csv_header: bool) -> dict:
    """Core filtering loop. Returns a JSON-serializable summary dict."""
    total_passing       = 0
    total_rejected      = 0
    total_nterm_removed = 0
    total_cterm_removed = 0
    total_exact_dup     = 0

    csv_header_written = not write_csv_header
    seen_hashes: set[bytes] = set()
    # Length -> count, over final (post-dedup) sequences. Lengths are bounded
    # (roughly 50-300 aa), so this histogram stays tiny regardless of corpus size.
    length_hist: dict[int, int] = {}

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
                # header=1 because row 0 is OAS JSON metadata. Peek at the header
                # first (nrows=0, cheap) so usecols only requests columns this
                # file actually has -- avoids loading the ~80 unused AIRR
                # columns (nucleotide alignments, CIGAR strings, etc.) that
                # are a big chunk of the memory cost for the largest files.
                header_cols = pd.read_csv(csv_path, header=1, compression="gzip", nrows=0).columns
                usecols = [c for c in NEEDED_COLS if c in header_cols]
                chunk_iter = pd.read_csv(csv_path, header=1, compression="gzip",
                                          usecols=usecols, chunksize=CHUNK_SIZE)
            except Exception as e:
                print(f"  [ERROR] {fname}: {e}", file=sys.stderr)
                continue

            n_total_f = n_pass_f = n_nterm_f = n_cterm_f = n_dup_f = n_final_f = 0
            row_offset = 0
            try:
                for chunk in chunk_iter:
                    (n_total_c, n_pass_c, n_nterm_c, n_cterm_c, n_dup_c, n_final_c,
                     row_offset, csv_header_written) = _process_chunk(
                        chunk, file_stem, row_offset, fasta, csv_fh, file_meta,
                        seen_hashes, length_hist, csv_header_written,
                    )
                    n_total_f  += n_total_c
                    n_pass_f   += n_pass_c
                    n_nterm_f  += n_nterm_c
                    n_cterm_f  += n_cterm_c
                    n_dup_f    += n_dup_c
                    n_final_f  += n_final_c
            except Exception as e:
                print(f"  [ERROR] {fname}: failed mid-file: {e}", file=sys.stderr)
                continue

            total_passing       += n_pass_f
            total_rejected      += n_total_f - n_pass_f
            total_nterm_removed += n_nterm_f
            total_cterm_removed += n_cterm_f
            total_exact_dup     += n_dup_f

            print(
                f"  {fname}: {n_pass_f:,} pass / {n_total_f - n_pass_f:,} reject (of {n_total_f:,})"
                f"  [N-term frag: {n_nterm_f:,}, C-term frag: {n_cterm_f:,}, exact dup: {n_dup_f:,}]"
            )

    return {
        "total_passing": total_passing,
        "total_rejected": total_rejected,
        "total_nterm_removed": total_nterm_removed,
        "total_cterm_removed": total_cterm_removed,
        "total_exact_dup": total_exact_dup,
        "length_hist": {str(k): v for k, v in length_hist.items()},
    }


def print_summary(summary: dict, output_fasta: str, output_csv: str) -> None:
    total_passing = summary["total_passing"]
    total_exact_dup = summary["total_exact_dup"]

    print(f"\nSummary:")
    print(f"  Total passing quality+frag filters: {total_passing:,}")
    print(f"  -- N-terminal fragmentation removed: {summary['total_nterm_removed']:,}")
    print(f"  -- C-terminal fragmentation removed: {summary['total_cterm_removed']:,}")
    print(f"  Total rejected (quality+frag):       {summary['total_rejected']:,}")
    print(f"  Total exact duplicates removed:      {total_exact_dup:,}")
    print(f"  Total written (final):               {total_passing - total_exact_dup:,}")

    length_hist = {int(k): v for k, v in summary["length_hist"].items()}
    if length_hist:
        lengths = sorted(length_hist)
        n_final = sum(length_hist.values())
        cum, pct = 0, {}
        targets = [1, 5, 25, 50, 75, 95, 99]
        for L in lengths:
            cum += length_hist[L]
            for t in list(targets):
                if cum >= t / 100 * n_final:
                    pct[t] = L
                    targets.remove(t)
        mean_len = sum(L * c for L, c in length_hist.items()) / n_final
        print(f"  Length distribution (final, n={n_final:,}):")
        print(f"    min={lengths[0]} max={lengths[-1]} mean={mean_len:.1f}")
        print(f"    percentiles: {pct}")

    print(f"  FASTA: {output_fasta}")
    print(f"  CSV:   {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input-list", default=None,
        help="Manifest file listing filenames (one per line, relative to $SCRATCH_DIR/oas_raw) "
             "to process. Default: process every .csv.gz file in $SCRATCH_DIR/oas_raw.",
    )
    parser.add_argument("--output-fasta", default=None, help="Default: $SCRATCH_DIR/oas_filtered.fasta")
    parser.add_argument("--output-csv", default=None, help="Default: $SCRATCH_DIR/oas_filtered.csv.gz")
    parser.add_argument(
        "--summary-json", default=None,
        help="Optional path to dump the summary counts + length histogram as JSON "
             "(used by merge_oas_shards.py to aggregate shard results).",
    )
    parser.add_argument(
        "--no-csv-header", action="store_true",
        help="Omit the CSV header row (shard mode; merge_oas_shards.py writes it once).",
    )
    args = parser.parse_args()

    load_dotenv()
    scratch_dir = os.environ.get("SCRATCH_DIR")
    if not scratch_dir:
        print("Error: SCRATCH_DIR env var not set", file=sys.stderr)
        sys.exit(1)

    input_dir = os.path.join(scratch_dir, "oas_raw")

    if args.input_list:
        with open(args.input_list) as f:
            names = [line.strip() for line in f if line.strip()]
        csv_files = [os.path.join(input_dir, n) for n in names]
        missing = [f for f in csv_files if not os.path.exists(f)]
        if missing:
            print(f"Error: {len(missing)} file(s) from {args.input_list} not found, "
                  f"e.g. {missing[0]}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(csv_files)} CSV files (from {args.input_list})")
    else:
        csv_files = sorted(glob.glob(os.path.join(input_dir, "*.csv.gz")))
        if not csv_files:
            print(f"No .csv.gz files found in {input_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(csv_files)} CSV files in {input_dir}")

    output_fasta = args.output_fasta or os.path.join(scratch_dir, "oas_filtered.fasta")
    output_csv = args.output_csv or os.path.join(scratch_dir, "oas_filtered.csv.gz")

    summary = process_files(csv_files, output_fasta, output_csv, write_csv_header=not args.no_csv_header)

    if args.summary_json:
        with open(args.summary_json, "w") as f:
            json.dump(summary, f)

    print_summary(summary, output_fasta, output_csv)


if __name__ == "__main__":
    main()
