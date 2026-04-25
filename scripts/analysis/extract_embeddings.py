"""Extract whole-VH and CDRH3 mean-pooled embeddings from one ESM2-35M variant.

Reads
-----
- DMS scoring CSVs (M22, SI06): C05 CDRH3 variants + binding enrichments.
- OAS FASTA + metadata (`seq_id`, `cdr3_aa`): background antibody sequences.
- Optional Gibbs CSV (columns include `chain_id`, `gibbs_step`, `sequence`).
- ESM2 model weights (vanilla from HF hub or a checkpoint dir).

Writes one .npz file with the schema below — this is the integration contract consumed by
`compute_projections.py`; keep keys/dtypes stable.

    whole_seq_embs           (N, 480) float32
    cdrh3_embs               (N, 480) float32  NaN row where CDR3 not locatable
    sequences                (N,)     <U       full VH string
    source_labels            (N,)     <U       wt | dms | oas | gibbs
    M22_binding_enrichment   (N,)     float32  NaN where unavailable
    SI06_binding_enrichment  (N,)     float32  NaN where unavailable
    cdrh3_identity_to_wt     (N,)     float32  NaN for OAS
    gibbs_step               (N,)     int32    -1 for non-gibbs
    chain_id                 (N,)     int32    -1 for non-gibbs
    model_variant            (1,)     <U
"""

from __future__ import annotations

import argparse
import gzip
import logging
import random
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel

from protein_design.constants import (
    C05_CDRH3,
    C05_CDRH3_END,
    C05_CDRH3_START,
    C05_VH,
    add_context,
)

ESM2_MODEL_ID = "facebook/esm2_t12_35M_UR50D"
SEED = 42
EMB_DIM = 480

# Default input paths (CLAUDE.md). Overridable via CLI.
DEFAULT_DMS_M22 = "/cluster/project/infk/krause/mdenegri/protein-design/datasets/scoring/D2_M22.csv"
DEFAULT_DMS_SI06 = "/cluster/project/infk/krause/mdenegri/protein-design/datasets/scoring/D2_SI06.csv"
DEFAULT_OAS_FASTA = "/cluster/project/infk/krause/mdenegri/protein-design/datasets/oas_dedup_rep_seq.fasta"
DEFAULT_OAS_META = "/cluster/project/infk/krause/mdenegri/protein-design/datasets/oas_filtered.csv.gz"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("extract_embeddings")


# --------------------------------------------------------------------------- I/O


def load_dms(m22_path: Path, si06_path: Path, max_n: int) -> pd.DataFrame:
    """Outer-merge M22 + SI06 on `aa` (24-aa CDRH3), sample up to max_n rows."""
    m22 = pd.read_csv(m22_path)[["aa", "M22_binding_enrichment_adj"]]
    si06 = pd.read_csv(si06_path)[["aa", "SI06_binding_enrichment_adj"]]
    merged = m22.merge(si06, on="aa", how="outer")
    if len(merged) > max_n:
        merged = merged.sample(n=max_n, random_state=SEED).reset_index(drop=True)
    return merged


def reservoir_sample_fasta(path: Path, k: int) -> List[Tuple[str, str]]:
    """Single-pass reservoir sample of (seq_id, sequence) pairs from a FASTA."""
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


def build_cdr3_lookup(meta_path: Path, wanted_ids: Iterable[str]) -> dict[str, tuple[str, int]]:
    """Stream OAS metadata, return {seq_id: (cdr3_aa, cdr3_start)} restricted to wanted_ids.

    cdr3_start is the 0-based index of the CDR-H3 in the full VH sequence, computed
    exactly from segment lengths (fwr1+cdr1+fwr2+cdr2+fwr3) — no substring search.
    """
    wanted = set(wanted_ids)
    out: dict[str, tuple[str, int]] = {}
    cols = ["seq_id", "cdr3_aa", "fwr1_aa", "cdr1_aa", "fwr2_aa", "cdr2_aa", "fwr3_aa"]
    for chunk in pd.read_csv(meta_path, usecols=cols, chunksize=500_000):
        hit = chunk[chunk["seq_id"].isin(wanted)]
        for row in hit.itertuples(index=False):
            cdr = row.cdr3_aa if isinstance(row.cdr3_aa, str) else ""
            start = sum(
                len(s) if isinstance(s, str) else 0
                for s in (row.fwr1_aa, row.cdr1_aa, row.fwr2_aa, row.cdr2_aa, row.fwr3_aa)
            )
            out[str(row.seq_id)] = (cdr, start)
    return out


def load_gibbs(path: Path, max_n: int) -> Optional[pd.DataFrame]:
    """Load Gibbs sequences as DataFrame with columns sequence/chain_id/gibbs_step.

    Accepts CSV (from scripts/gibbs_sampling.py) or FASTA (chain_id=0, step=row index).
    Returns None if path does not exist.
    """
    if not path.exists():
        log.warning("Gibbs path %s not found — skipping gibbs source", path)
        return None
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        cols = {"sequence"}
        if not cols.issubset(df.columns):
            raise ValueError(f"Gibbs CSV missing required column 'sequence': {path}")
        if "chain_id" not in df.columns:
            df["chain_id"] = 0
        if "gibbs_step" not in df.columns:
            df["gibbs_step"] = np.arange(len(df))
        df = df[["sequence", "chain_id", "gibbs_step"]]
    else:
        records = list(SeqIO.parse(str(path), "fasta"))
        df = pd.DataFrame({
            "sequence": [str(r.seq) for r in records],
            "chain_id": 0,
            "gibbs_step": np.arange(len(records)),
        })
    if len(df) > max_n:
        df = df.sample(n=max_n, random_state=SEED).reset_index(drop=True)
    return df


# --------------------------------------------------------------------- reference


def cdrh3_identity_to_wt(full_vh: str) -> float:
    """Fraction of CDRH3 positions matching C05 WT. Assumes 138-aa fixed VH."""
    cdr = full_vh[C05_CDRH3_START:C05_CDRH3_END]
    if len(cdr) != len(C05_CDRH3):
        return float("nan")
    return sum(a == b for a, b in zip(cdr, C05_CDRH3)) / len(C05_CDRH3)


def fixed_vh_cdrh3_token_positions() -> List[int]:
    """Token indices for CDRH3 in a 138-aa VH (BOS at 0, residue i → token i+1)."""
    return list(range(C05_CDRH3_START + 1, C05_CDRH3_END + 1))


def build_reference_set(
    dms_m22: Path,
    dms_si06: Path,
    oas_fasta: Path,
    oas_meta: Path,
    gibbs_path: Optional[Path],
    max_dms: int,
    max_oas: int,
    max_gibbs: int,
) -> pd.DataFrame:
    """Assemble the per-row table all subsequent steps operate on.

    Columns: sequence, source, M22_enrich, SI06_enrich, cdrh3_token_positions (list[int]
    or None), cdrh3_identity_to_wt (float, NaN for OAS), gibbs_step, chain_id.
    """
    rows: list[dict] = []

    # 1) WT
    rows.append({
        "sequence": C05_VH,
        "source": "wt",
        "M22_enrich": np.nan,
        "SI06_enrich": np.nan,
        "cdrh3_token_positions": fixed_vh_cdrh3_token_positions(),
        "cdrh3_identity_to_wt": 1.0,
        "gibbs_step": -1,
        "chain_id": -1,
    })

    # 2) DMS
    log.info("Loading DMS …")
    dms = load_dms(dms_m22, dms_si06, max_dms)
    fixed_pos = fixed_vh_cdrh3_token_positions()
    for _, r in dms.iterrows():
        full = add_context(r["aa"])
        rows.append({
            "sequence": full,
            "source": "dms",
            "M22_enrich": float(r["M22_binding_enrichment_adj"]) if pd.notna(r["M22_binding_enrichment_adj"]) else np.nan,
            "SI06_enrich": float(r["SI06_binding_enrichment_adj"]) if pd.notna(r["SI06_binding_enrichment_adj"]) else np.nan,
            "cdrh3_token_positions": fixed_pos,
            "cdrh3_identity_to_wt": cdrh3_identity_to_wt(full),
            "gibbs_step": -1,
            "chain_id": -1,
        })

    # 3) OAS
    log.info("Reservoir-sampling %d OAS sequences from %s", max_oas, oas_fasta)
    oas = reservoir_sample_fasta(oas_fasta, max_oas)
    log.info("Looking up CDR3 metadata for %d OAS seq_ids", len(oas))
    cdr3_lookup = build_cdr3_lookup(oas_meta, [sid for sid, _ in oas])
    n_unmatched = 0
    for sid, seq in oas:
        entry = cdr3_lookup.get(sid)
        if entry and entry[0]:
            cdr, start = entry
            # +1 because token 0 is BOS; verify the annotated region matches the sequence
            if start >= 0 and start + len(cdr) <= len(seq) and seq[start:start + len(cdr)] == cdr:
                positions = list(range(start + 1, start + 1 + len(cdr)))
            else:
                positions = None
                n_unmatched += 1
        else:
            positions = None
            n_unmatched += 1
        rows.append({
            "sequence": seq,
            "source": "oas",
            "M22_enrich": np.nan,
            "SI06_enrich": np.nan,
            "cdrh3_token_positions": positions,
            "cdrh3_identity_to_wt": np.nan,
            "gibbs_step": -1,
            "chain_id": -1,
        })
    if n_unmatched:
        log.warning("%d / %d OAS sequences have no locatable CDRH3 (NaN cdrh3_emb)", n_unmatched, len(oas))

    # 4) Gibbs
    if gibbs_path is not None:
        gibbs_df = load_gibbs(gibbs_path, max_gibbs)
        if gibbs_df is not None:
            log.info("Loaded %d Gibbs rows", len(gibbs_df))
            for _, r in gibbs_df.iterrows():
                full = str(r["sequence"])
                rows.append({
                    "sequence": full,
                    "source": "gibbs",
                    "M22_enrich": np.nan,
                    "SI06_enrich": np.nan,
                    "cdrh3_token_positions": fixed_pos,
                    "cdrh3_identity_to_wt": cdrh3_identity_to_wt(full),
                    "gibbs_step": int(r["gibbs_step"]),
                    "chain_id": int(r["chain_id"]),
                })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------- model load


def load_esm_encoder(checkpoint_path: Optional[str]) -> EsmModel:
    """Return an `EsmModel` (encoder only) from one of three checkpoint shapes.

    1. None → vanilla ESM2-35M from the HF hub.
    2. HF-format directory containing `model.safetensors` or `pytorch_model.bin`.
    3. Directory containing `best.pt`/`final.pt`, or a direct `.pt` path — torch state
       dict produced by this repo's training pipeline. Keys are prefixed `model.esm.*`
       (the `ESM2Model` wrapper class wraps `EsmForMaskedLM`); we strip that prefix and
       drop the `lm_head.*` keys, then load into a vanilla-initialized `EsmModel`.
    """
    if checkpoint_path is None:
        log.info("Loading vanilla ESM2 from %s", ESM2_MODEL_ID)
        return EsmModel.from_pretrained(ESM2_MODEL_ID)

    p = Path(checkpoint_path)
    if p.is_dir() and ((p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists()):
        log.info("Loading HF-format checkpoint from %s", p)
        return EsmModel.from_pretrained(str(p))

    pt_path = p if p.is_file() else next(
        (p / name for name in ("best.pt", "final.pt") if (p / name).exists()), None
    )
    if pt_path is None:
        raise FileNotFoundError(
            f"No HF weights, best.pt, or final.pt found at {checkpoint_path}"
        )

    log.info("Loading state-dict checkpoint %s into a vanilla encoder", pt_path)
    raw = torch.load(pt_path, map_location="cpu", weights_only=False)
    state = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw

    encoder_state: dict = {}
    for k, v in state.items():
        if k.startswith("model.esm."):
            encoder_state[k[len("model.esm."):]] = v
        elif k.startswith("esm."):
            encoder_state[k[len("esm."):]] = v
        # silently drop lm_head and other non-encoder keys

    model = EsmModel.from_pretrained(ESM2_MODEL_ID)
    missing, unexpected = model.load_state_dict(encoder_state, strict=False)
    non_pooler_missing = [m for m in missing if not m.startswith("pooler.")]
    if non_pooler_missing:
        raise RuntimeError(
            f"Checkpoint missing required encoder keys: {non_pooler_missing[:5]} (+{len(non_pooler_missing)-5} more)"
        )
    if unexpected:
        log.warning("Ignored %d unexpected keys (e.g. %s)", len(unexpected), unexpected[:3])
    log.info("Loaded encoder weights (pooler kept at vanilla init — unused for last_hidden_state).")
    return model


# --------------------------------------------------------------------- inference


@torch.no_grad()
def extract_batch(
    model: EsmModel,
    tokenizer,
    sequences: Sequence[str],
    cdrh3_positions: Sequence[Optional[List[int]]],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    spaced = [" ".join(list(s)) for s in sequences]
    enc = tokenizer(spaced, return_tensors="pt", padding=True, add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    out = model(input_ids=input_ids, attention_mask=attn)
    h = out.last_hidden_state.float()  # [B, L, D]

    whole, cdr = [], []
    nan_row = torch.full((h.shape[-1],), float("nan"), device=device)
    for i, seq in enumerate(sequences):
        L = len(seq)
        whole.append(h[i, 1 : 1 + L].mean(dim=0))
        positions = cdrh3_positions[i]
        if positions:
            idx = torch.tensor(positions, device=device, dtype=torch.long)
            cdr.append(h[i].index_select(0, idx).mean(dim=0))
        else:
            cdr.append(nan_row)
    return torch.stack(whole).cpu().numpy(), torch.stack(cdr).cpu().numpy()


def run_inference(
    df: pd.DataFrame,
    model: EsmModel,
    tokenizer,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Group by source (similar lengths → less padding waste), batch through ESM."""
    n = len(df)
    whole_out = np.full((n, EMB_DIM), np.nan, dtype=np.float32)
    cdr_out = np.full((n, EMB_DIM), np.nan, dtype=np.float32)
    df = df.reset_index(drop=True)
    for source, group in df.groupby("source", sort=False):
        idxs = group.index.to_numpy()
        seqs = group["sequence"].tolist()
        positions = group["cdrh3_token_positions"].tolist()
        for start in tqdm(range(0, len(idxs), batch_size), desc=f"embed {source}"):
            sl = slice(start, start + batch_size)
            w, c = extract_batch(model, tokenizer, seqs[sl], positions[sl], device)
            whole_out[idxs[sl]] = w.astype(np.float32)
            cdr_out[idxs[sl]] = c.astype(np.float32)
    return whole_out, cdr_out


# ------------------------------------------------------------------------- main


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-variant", required=True, help="Label string (vanilla, evotuned, …)")
    p.add_argument("--checkpoint-path", default=None, help="HF-format checkpoint dir; omit for vanilla")
    p.add_argument("--gibbs-path", default=None, help="CSV or FASTA of Gibbs-sampled sequences")
    p.add_argument("--output-path", required=True, help="Destination .npz path")
    p.add_argument("--dms-m22", default=DEFAULT_DMS_M22)
    p.add_argument("--dms-si06", default=DEFAULT_DMS_SI06)
    p.add_argument("--oas-fasta", default=DEFAULT_OAS_FASTA)
    p.add_argument("--oas-meta", default=DEFAULT_OAS_META)
    p.add_argument("--max-dms", type=int, default=500)
    p.add_argument("--max-oas", type=int, default=2000)
    p.add_argument("--max-gibbs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def main() -> int:
    args = parse_args()
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

    df = build_reference_set(
        Path(args.dms_m22),
        Path(args.dms_si06),
        Path(args.oas_fasta),
        Path(args.oas_meta),
        Path(args.gibbs_path) if args.gibbs_path else None,
        args.max_dms,
        args.max_oas,
        args.max_gibbs,
    )
    log.info("Reference set: %d rows  (%s)", len(df), dict(df["source"].value_counts()))

    whole, cdr = run_inference(df, model, tokenizer, device, args.batch_size)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        whole_seq_embs=whole,
        cdrh3_embs=cdr,
        sequences=np.array(df["sequence"].tolist()),
        source_labels=np.array(df["source"].tolist()),
        M22_binding_enrichment=df["M22_enrich"].to_numpy(dtype=np.float32),
        SI06_binding_enrichment=df["SI06_enrich"].to_numpy(dtype=np.float32),
        cdrh3_identity_to_wt=df["cdrh3_identity_to_wt"].to_numpy(dtype=np.float32),
        gibbs_step=df["gibbs_step"].to_numpy(dtype=np.int32),
        chain_id=df["chain_id"].to_numpy(dtype=np.int32),
        model_variant=np.array([args.model_variant]),
    )
    log.info("Wrote %s  (%d rows, embedding dim %d)", out_path, len(df), EMB_DIM)
    return 0


if __name__ == "__main__":
    sys.exit(main())
