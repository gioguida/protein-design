#!/usr/bin/env python3
"""Pack a VH corpus into the memory-mapped form the trainer reads.

Holding an OAS-scale corpus in memory as Python objects, or as fixed-width
numpy arrays, costs tens of GB for the training split alone, and a per-sequence
CDR-H3 lookup keyed by sequence id costs about as much again. Packing the
corpus once removes both costs: sequences live in a flat byte file that the
operating system pages in on demand, and everything the masking policies need
(the CDR-H3 boundaries, the flanked window, the split assignment) becomes a
small array indexed by position.

Packing is also where the corpus is *fixed*. Sequences whose CDR-H3 cannot be
located in the VH are dropped here, once, so every masking policy afterwards
trains on exactly the same set of sequences in exactly the same order. Dropping
them later, inside whichever policy happens to need a window, would leave the
whole-chain policy training on a larger corpus than the CDR policies, which
changes the epoch length and breaks the correspondence the two-phase training
schedule relies on.

Non-standard residues are not stored separately. They survive filtering as the
letter X, so a position is identifiable as non-standard directly from the
packed bytes and is excluded from masking at that point.

Output layout, one directory per (corpus, flank):

    meta.json      corpus provenance, counts, drop reasons, split settings
    seqs.u8        every sequence's residues, concatenated, ASCII
    offsets.i64    n+1 offsets into seqs.u8
    ids.u8         every sequence id, concatenated, ASCII
    id_offsets.i64 n+1 offsets into ids.u8
    cdr3.i32       (n, 2) CDR-H3 start/end, residue coordinates
    win.i32        (n, 2) CDR-H3 expanded by `flank` on each side, clipped
    split.i8       (n,) 0 = train, 1 = val, 2 = test

Build at the widest flank any policy needs. A policy wanting a narrower region
reads cdr3.i32 instead of win.i32, so one pack serves all of them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from protein_design.evotuning.splits import SplitConfig, split_for  # noqa: E402

from filter_oas import clean_seq_series  # noqa: E402
from meta_io import iter_meta_chunks  # noqa: E402

load_dotenv(".env.local")
load_dotenv()

SPLIT_CODE = {"train": 0, "val": 1, "test": 2}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    project = os.environ.get("PROJECT_DIR", ".")
    scratch = os.environ.get("SCRATCH_DIR", ".")
    p.add_argument(
        "--meta", default=os.path.join(project, "data", "oas", "oas_dedup_meta.parquet"),
        help="Metadata table carrying seq_id, sequence_alignment_aa and cdr3_aa.",
    )
    p.add_argument(
        "--fasta", default=None,
        help=(
            "Optional. Restrict the pack to the sequence ids in this FASTA, and "
            "take its sequences as authoritative. Only for corpora small enough "
            "for their id set to fit in memory (e.g. a WT-similar set). Omit to "
            "pack every row of --meta."
        ),
    )
    p.add_argument("--name", default=None, help="Pack name. Default: derived from the input.")
    p.add_argument("--flank", type=int, default=5, help="Framework residues per side.")
    p.add_argument(
        "--max-residues", type=int, default=254,
        help="Residues that fit alongside the BOS/EOS tokens at the model's context length.",
    )
    p.add_argument("--out-dir", default=os.path.join(scratch, "packed"))
    p.add_argument("--chunksize", type=int, default=500_000)
    p.add_argument("--salt", default="oas-v1")
    p.add_argument("--train-pct", type=int, default=90)
    p.add_argument("--val-pct", type=int, default=5)
    p.add_argument("--test-pct", type=int, default=5)
    p.add_argument("--rebuild", action="store_true")
    return p.parse_args()


def load_fasta_ids_and_seqs(fasta_path: str) -> dict[str, str]:
    """Read a FASTA into {seq_id: sequence}. Small corpora only."""
    seqs: dict[str, str] = {}
    sid = None
    parts: list[str] = []
    with open(fasta_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if sid is not None:
                    seqs[sid] = "".join(parts)
                sid = line[1:].split()[0]
                parts = []
            else:
                parts.append(line)
    if sid is not None:
        seqs[sid] = "".join(parts)
    return seqs


class PackWriter:
    """Append-only writer for the packed corpus files."""

    def __init__(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        self.out = out
        self.f_seqs = open(out / "seqs.u8", "wb")
        self.f_ids = open(out / "ids.u8", "wb")
        self.seq_offsets = [0]
        self.id_offsets = [0]
        self.cdr3: list[tuple[int, int]] = []
        self.win: list[tuple[int, int]] = []
        self.split: list[int] = []

    def add(
        self, seq_id: str, seq: str, cdr3_start: int, cdr3_end: int,
        win_start: int, win_end: int, split_code: int,
    ) -> None:
        sb = seq.encode("ascii")
        ib = seq_id.encode("ascii")
        self.f_seqs.write(sb)
        self.f_ids.write(ib)
        self.seq_offsets.append(self.seq_offsets[-1] + len(sb))
        self.id_offsets.append(self.id_offsets[-1] + len(ib))
        self.cdr3.append((cdr3_start, cdr3_end))
        self.win.append((win_start, win_end))
        self.split.append(split_code)

    def close(self, meta: dict) -> None:
        self.f_seqs.close()
        self.f_ids.close()
        np.asarray(self.seq_offsets, dtype=np.int64).tofile(self.out / "offsets.i64")
        np.asarray(self.id_offsets, dtype=np.int64).tofile(self.out / "id_offsets.i64")
        np.asarray(self.cdr3, dtype=np.int32).reshape(-1, 2).tofile(self.out / "cdr3.i32")
        np.asarray(self.win, dtype=np.int32).reshape(-1, 2).tofile(self.out / "win.i32")
        np.asarray(self.split, dtype=np.int8).tofile(self.out / "split.i8")
        (self.out / "meta.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    args = parse_args()
    split_cfg = SplitConfig(
        salt=args.salt, train_pct=args.train_pct,
        val_pct=args.val_pct, test_pct=args.test_pct,
    )

    source = args.fasta or args.meta
    name = args.name or Path(source).stem
    out = Path(args.out_dir) / f"{name}_flank{args.flank}"
    if (out / "meta.json").exists() and not args.rebuild:
        print(f"[pack] reusing {out}")
        print((out / "meta.json").read_text())
        return

    wanted: dict[str, str] | None = None
    if args.fasta:
        wanted = load_fasta_ids_and_seqs(args.fasta)
        print(f"[pack] restricting to {len(wanted):,} ids from {args.fasta}", flush=True)

    writer = PackWriter(out)
    counts = {"train": 0, "val": 0, "test": 0}
    n_rows = n_kept = 0
    n_no_cdr3 = n_not_found = n_multi = n_too_long = n_not_wanted = 0

    columns = ["seq_id", "sequence_alignment_aa", "cdr3_aa"]
    for chunk in iter_meta_chunks(args.meta, columns=columns, chunksize=args.chunksize):
        n_rows += len(chunk)
        chunk = chunk.dropna(subset=["sequence_alignment_aa"])
        # Reproduce the sequence exactly as filtering wrote it, and clean the
        # annotated CDR-H3 the same way. Cleaning only the VH is what makes a
        # CDR-H3 containing a non-standard residue stop matching as a
        # substring: the VH gets an X, the annotation keeps the original letter.
        seqs = clean_seq_series(chunk["sequence_alignment_aa"]).tolist()
        h3s = clean_seq_series(chunk["cdr3_aa"].fillna("")).tolist()
        ids = chunk["seq_id"].astype(str).tolist()

        for sid, seq, h3 in zip(ids, seqs, h3s):
            if wanted is not None:
                if sid not in wanted:
                    n_not_wanted += 1
                    continue
                seq = wanted[sid]
            if not h3:
                n_no_cdr3 += 1
                continue
            start = seq.find(h3)
            if start < 0:
                n_not_found += 1
                continue
            if seq.rfind(h3) != start:
                n_multi += 1  # ambiguous, keep the first occurrence
            end = start + len(h3)
            if end > args.max_residues:
                # The CDR-H3 would be cut off by the model's context length,
                # leaving nothing for the CDR policies to mask.
                n_too_long += 1
                continue
            split = split_for(sid, split_cfg)
            counts[split] += 1
            writer.add(
                sid, seq, start, end,
                max(0, start - args.flank),
                min(len(seq), end + args.flank),
                SPLIT_CODE[split],
            )
            n_kept += 1

        if n_rows % (10 * args.chunksize) < args.chunksize:
            print(f"[pack] scanned {n_rows:,} rows, kept {n_kept:,}", flush=True)

    dropped = n_no_cdr3 + n_not_found + n_too_long
    considered = n_kept + dropped
    meta = {
        "name": name,
        "meta_source": str(args.meta),
        "fasta_source": str(args.fasta) if args.fasta else None,
        "flank": args.flank,
        "max_residues": args.max_residues,
        "n_sequences": n_kept,
        "counts": counts,
        "split": {
            "salt": split_cfg.salt, "train_pct": split_cfg.train_pct,
            "val_pct": split_cfg.val_pct, "test_pct": split_cfg.test_pct,
        },
        "rows_scanned": n_rows,
        "dropped": {
            "missing_cdr3": n_no_cdr3,
            "cdr3_not_in_vh": n_not_found,
            "cdr3_beyond_context": n_too_long,
            "total": dropped,
            "rate": (dropped / considered) if considered else 0.0,
        },
        "ambiguous_cdr3_match": n_multi,
        "not_in_fasta": n_not_wanted,
    }
    writer.close(meta)
    print(json.dumps(meta, indent=2))
    print(f"[pack] wrote {out}")


if __name__ == "__main__":
    main()
