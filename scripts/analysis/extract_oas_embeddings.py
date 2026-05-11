"""Extract whole-VH and CDRH3 mean-pooled embeddings for an OAS background sample.

Dataset-agnostic counterpart to ``extract_embeddings.py``. The OAS reservoir sample is
deterministic for a fixed ``(seed=42, max_oas, oas_fasta)`` triple, so the embeddings here are a
function only of ``(checkpoint, max_oas, oas_fasta, oas_meta)`` — not of any DMS dataset. Splitting
this stage out lets the bash pipeline cache OAS embeddings once per checkpoint and reuse them
across DMS dataset runs.

Reads
-----
- OAS FASTA (`>seq_id` headers): background antibody sequences.
- OAS metadata CSV.gz: V-/J-call, V-identity, CDR3 segment etc.
- ESM2 model weights (vanilla from HF hub or a checkpoint dir).

Writes one .npz file with the following schema:

    whole_seq_embs           (N, 480) float32
    cdrh3_embs               (N, 480) float32     NaN row where CDR3 not locatable
    sequences                (N,)     <U          full VH string
    seq_ids                  (N,)     <U          OAS FASTA seq_id
    v_family                 (N,)     <U          IGHV{1..7}
    v_call                   (N,)     <U          gene-level e.g. "IGHV3-23"
    j_call                   (N,)     <U          gene-level e.g. "IGHJ4"
    v_identity               (N,)     float32     percent identity to germline; NaN if missing
    cdr3_aa                  (N,)     <U          empty string if missing
    cdr3_len                 (N,)     int32       -1 if missing
    model_variant            (1,)     <U
    max_oas                  (1,)     int32       cache header
    oas_fasta                (1,)     <U          cache header
    oas_meta                 (1,)     <U          cache header
    schema_version           (1,)     <U
"""

from __future__ import annotations

import argparse
import gzip
import logging
import random
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

# Reuse ESM2 loading + batched inference from the DMS-side script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_embeddings import (  # noqa: E402
    EMB_DIM,
    ESM2_MODEL_ID,
    SEED,
    extract_batch,
    load_esm_encoder,
)

SCHEMA_VERSION = "v2"

DEFAULT_OAS_FASTA = "/cluster/project/infk/krause/mdenegri/protein-design/data/oas_dedup_rep_seq.fasta"
DEFAULT_OAS_META = "/cluster/project/infk/krause/mdenegri/protein-design/data/oas_filtered.csv.gz"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("extract_oas_embeddings")


# --------------------------------------------------------------------------- I/O


def reservoir_sample_fasta(path: Path, k: int) -> List[Tuple[str, str]]:
    """Single-pass reservoir sample of (seq_id, sequence) pairs from a FASTA.

    Deterministic for a given (seed, k, file content): identical across runs.
    """
    rng = random.Random(SEED)
    opener = gzip.open if str(path).endswith(".gz") else open
    reservoir: List[Tuple[str, str]] = []
    with opener(path, "rt") as fh:
        for i, rec in enumerate(SeqIO.parse(fh, "fasta")):
            item = (rec.id, str(rec.seq))
            if i < k:
                reservoir.append(item)
            else:
                j = rng.randint(0, i)
                if j < k:
                    reservoir[j] = item
    return reservoir


def _v_family_from_call(v_call) -> str:
    """'IGHV3-21*01,IGHV3-21*02' -> 'IGHV3'. Empty string for NaN/missing."""
    if not isinstance(v_call, str) or not v_call:
        return ""
    first = v_call.split(",", 1)[0].strip()
    return first.split("-", 1)[0] if "-" in first else first


def _gene_from_call(call) -> str:
    """'IGHV3-23*01,IGHV3-23*02' -> 'IGHV3-23'; 'IGHJ4*02' -> 'IGHJ4'."""
    if not isinstance(call, str) or not call:
        return ""
    first = call.split(",", 1)[0].strip()
    return first.split("*", 1)[0]


def _v_identity_to_float(value) -> float:
    """Coerce v_identity to float; NaN for missing/non-numeric."""
    try:
        if value is None:
            return float("nan")
        f = float(value)
        return f if not np.isnan(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


# Stored as a small named-tuple-like dict for one OAS row's metadata
class _OasMeta:
    __slots__ = ("cdr3_aa", "cdr3_start", "v_family", "v_call", "j_call", "v_identity")

    def __init__(self, cdr3_aa: str, cdr3_start: int, v_family: str,
                 v_call: str, j_call: str, v_identity: float):
        self.cdr3_aa = cdr3_aa
        self.cdr3_start = cdr3_start
        self.v_family = v_family
        self.v_call = v_call
        self.j_call = j_call
        self.v_identity = v_identity


def build_oas_metadata_lookup(meta_path: Path, wanted_ids: Iterable[str]) -> dict[str, _OasMeta]:
    """Stream OAS metadata, return {seq_id: _OasMeta} restricted to wanted_ids.

    cdr3_start is the 0-based index of CDR-H3 in the full VH, computed from segment lengths.
    """
    wanted = set(wanted_ids)
    out: dict[str, _OasMeta] = {}
    cols = ["seq_id", "cdr3_aa",
            "fwr1_aa", "cdr1_aa", "fwr2_aa", "cdr2_aa", "fwr3_aa",
            "v_call", "j_call", "v_identity"]
    for chunk in pd.read_csv(meta_path, usecols=cols, chunksize=500_000):
        hit = chunk[chunk["seq_id"].isin(wanted)]
        for row in hit.itertuples(index=False):
            cdr = row.cdr3_aa if isinstance(row.cdr3_aa, str) else ""
            start = sum(
                len(s) if isinstance(s, str) else 0
                for s in (row.fwr1_aa, row.cdr1_aa, row.fwr2_aa, row.cdr2_aa, row.fwr3_aa)
            )
            out[str(row.seq_id)] = _OasMeta(
                cdr3_aa=cdr,
                cdr3_start=start,
                v_family=_v_family_from_call(row.v_call),
                v_call=_gene_from_call(row.v_call),
                j_call=_gene_from_call(row.j_call),
                v_identity=_v_identity_to_float(row.v_identity),
            )
    return out


# --------------------------------------------------------------------- assemble + infer


def build_oas_table(
    oas_fasta: Path, oas_meta_path: Path, max_oas: int
) -> Tuple[pd.DataFrame, int]:
    """Build a DataFrame of OAS rows + per-row CDRH3 token positions.

    Returns (df, n_unmatched). df has columns:
      seq_id, sequence, cdrh3_token_positions (List[int] | None),
      v_family, v_call, j_call, v_identity, cdr3_aa, cdr3_len.
    """
    log.info("Reservoir-sampling %d OAS sequences from %s", max_oas, oas_fasta)
    oas = reservoir_sample_fasta(oas_fasta, max_oas)
    log.info("Looking up metadata for %d OAS seq_ids in %s", len(oas), oas_meta_path)
    lookup = build_oas_metadata_lookup(oas_meta_path, [sid for sid, _ in oas])

    rows: list[dict] = []
    n_unmatched = 0
    for sid, seq in oas:
        meta = lookup.get(sid)
        if meta is None:
            n_unmatched += 1
            rows.append({
                "seq_id": sid, "sequence": seq, "cdrh3_token_positions": None,
                "v_family": "", "v_call": "", "j_call": "",
                "v_identity": float("nan"), "cdr3_aa": "", "cdr3_len": -1,
            })
            continue
        cdr, start = meta.cdr3_aa, meta.cdr3_start
        positions: Optional[List[int]] = None
        if cdr and 0 <= start and start + len(cdr) <= len(seq) and seq[start:start + len(cdr)] == cdr:
            # +1 because token 0 is BOS
            positions = list(range(start + 1, start + 1 + len(cdr)))
        else:
            n_unmatched += 1
        rows.append({
            "seq_id": sid,
            "sequence": seq,
            "cdrh3_token_positions": positions,
            "v_family": meta.v_family,
            "v_call": meta.v_call,
            "j_call": meta.j_call,
            "v_identity": meta.v_identity,
            "cdr3_aa": cdr,
            "cdr3_len": len(cdr) if cdr else -1,
        })
    if n_unmatched:
        log.warning("%d / %d OAS sequences have no locatable CDRH3 (NaN cdrh3_emb)",
                    n_unmatched, len(oas))
    return pd.DataFrame(rows), n_unmatched


def run_inference(
    df: pd.DataFrame,
    model: EsmModel,
    tokenizer,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Batch through ESM and return (whole_seq_embs, cdrh3_embs)."""
    n = len(df)
    whole_out = np.full((n, EMB_DIM), np.nan, dtype=np.float32)
    cdr_out = np.full((n, EMB_DIM), np.nan, dtype=np.float32)
    seqs = df["sequence"].tolist()
    positions = df["cdrh3_token_positions"].tolist()
    for start in tqdm(range(0, n, batch_size), desc="embed oas"):
        sl = slice(start, start + batch_size)
        w, c = extract_batch(model, tokenizer, seqs[sl], positions[sl], device)
        whole_out[sl] = w.astype(np.float32)
        cdr_out[sl] = c.astype(np.float32)
    return whole_out, cdr_out


# --------------------------------------------------------------------- skip helper


def existing_is_current(path: Path, max_oas: int, oas_fasta: str, oas_meta: str) -> bool:
    """True iff `path` exists with matching schema_version AND matching cache headers."""
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            if "schema_version" not in z.files:
                return False
            if str(z["schema_version"][0]) != SCHEMA_VERSION:
                return False
            if int(z["max_oas"][0]) != int(max_oas):
                log.warning("Cache %s has max_oas=%d but caller asked for %d", path,
                            int(z["max_oas"][0]), max_oas)
                return False
            if str(z["oas_fasta"][0]) != oas_fasta or str(z["oas_meta"][0]) != oas_meta:
                log.warning("Cache %s headers (oas_fasta/oas_meta) mismatch caller; rebuilding",
                            path)
                return False
            return True
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not inspect cache %s (%s); will rebuild", path, exc)
        return False


# --------------------------------------------------------------------- main


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-variant", required=True)
    p.add_argument("--checkpoint-path", default=None,
                   help="HF-format checkpoint dir or .pt path; omit for vanilla")
    p.add_argument("--output-path", required=True, help="Destination .npz path")
    p.add_argument("--oas-fasta", default=DEFAULT_OAS_FASTA)
    p.add_argument("--oas-meta", default=DEFAULT_OAS_META)
    p.add_argument("--max-oas", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--skip-if-current", action="store_true",
                   help="Exit 0 without recomputing if --output-path is already present with a "
                        "matching schema_version and cache headers.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    out_path = Path(args.output_path)
    if args.skip_if_current and existing_is_current(
        out_path, args.max_oas, args.oas_fasta, args.oas_meta
    ):
        log.info("Skip: %s exists with matching schema_version=%s and headers",
                 out_path, SCHEMA_VERSION)
        return 0
    if args.skip_if_current and out_path.exists():
        log.warning("Existing %s has stale schema/headers — rebuilding", out_path)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    log.info("Loading tokenizer from %s", ESM2_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_ID)
    model = load_esm_encoder(args.checkpoint_path).eval().to(device)
    for p_ in model.parameters():
        p_.requires_grad = False
    if device.type == "cuda":
        model = model.half()

    df, _ = build_oas_table(Path(args.oas_fasta), Path(args.oas_meta), args.max_oas)
    log.info("OAS table: %d rows", len(df))

    whole, cdr = run_inference(df, model, tokenizer, device, args.batch_size)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        whole_seq_embs=whole,
        cdrh3_embs=cdr,
        sequences=np.array(df["sequence"].tolist()),
        seq_ids=np.array(df["seq_id"].tolist()),
        v_family=np.array(df["v_family"].tolist()),
        v_call=np.array(df["v_call"].tolist()),
        j_call=np.array(df["j_call"].tolist()),
        v_identity=df["v_identity"].to_numpy(dtype=np.float32),
        cdr3_aa=np.array(df["cdr3_aa"].tolist()),
        cdr3_len=df["cdr3_len"].to_numpy(dtype=np.int32),
        model_variant=np.array([args.model_variant]),
        max_oas=np.array([args.max_oas], dtype=np.int32),
        oas_fasta=np.array([args.oas_fasta]),
        oas_meta=np.array([args.oas_meta]),
        schema_version=np.array([SCHEMA_VERSION]),
    )
    log.info("Wrote %s  (%d rows, embedding dim %d)", out_path, len(df), EMB_DIM)
    return 0


if __name__ == "__main__":
    sys.exit(main())
