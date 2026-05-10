"""Extract whole-VH and CDRH3 mean-pooled embeddings from one ESM2-35M variant.

DMS-side stage of the embedding pipeline. OAS embeddings live in a separate, dataset-agnostic
artifact written by ``extract_oas_embeddings.py`` — they are not regenerated when the DMS dataset
changes.

Reads
-----
- DMS scoring CSVs (M22, SI06): C05 CDRH3 variants + binding enrichments.
- Optional Gibbs CSV (columns include `chain_id`, `gibbs_step`, `sequence`).
- ESM2 model weights (vanilla from HF hub or a checkpoint dir).

Writes one .npz file with the schema below.

    whole_seq_embs           (N, 480) float32
    cdrh3_embs               (N, 480) float32
    sequences                (N,)     <U       full VH string
    source_labels            (N,)     <U       wt | dms | gibbs
    M22_binding_enrichment   (N,)     float32  NaN where unavailable
    SI06_binding_enrichment  (N,)     float32  NaN where unavailable
    cdrh3_identity_to_wt     (N,)     float32
    gibbs_step               (N,)     int32    -1 for non-gibbs
    chain_id                 (N,)     int32    -1 for non-gibbs
    gibbs_config             (N,)     <U
    model_variant            (1,)     <U
    dms_dataset              (1,)     <U       which DMS source (ed2, ed5, …)
    schema_version           (1,)     <U       "v2" — bump when the schema changes
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

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
SCHEMA_VERSION = "v2"  # bump when the on-disk schema changes

# Default input paths (CLAUDE.md). Overridable via CLI.
_DMS_BASE = "/cluster/project/infk/krause/mdenegri/protein-design/datasets/scoring"
DMS_DATASETS: dict[str, dict[str, str]] = {
    "ed2": {
        "m22": f"{_DMS_BASE}/D2_M22.csv",
        "si06": f"{_DMS_BASE}/D2_SI06.csv",
    },
    "ed5": {
        "m22": f"{_DMS_BASE}/ED5_M22_binding_enrichment.csv",
        "si06": f"{_DMS_BASE}/ED5_SI06_binding_enrichment.csv",
    },
    "ed811": {
        "m22": f"{_DMS_BASE}/ED811_M22_enrichment_full.csv",
    },
}
DEFAULT_DMS_DATASET = "ed2"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("extract_embeddings")


# --------------------------------------------------------------------------- I/O


def load_dms(
    m22_path: Optional[Path], si06_path: Optional[Path], max_n: int
) -> pd.DataFrame:
    """Outer-merge M22 + SI06 on `aa` (24-aa CDRH3), sample up to max_n rows.

    Either path may be ``None`` (assay missing for this dataset). The
    corresponding ``*_binding_enrichment_adj`` column is then filled with NaN.
    If both are ``None`` the result is an empty DataFrame.
    """
    frames: List[pd.DataFrame] = []
    if m22_path is not None:
        frames.append(pd.read_csv(m22_path)[["aa", "M22_binding_enrichment_adj"]])
    if si06_path is not None:
        frames.append(pd.read_csv(si06_path)[["aa", "SI06_binding_enrichment_adj"]])
    if not frames:
        return pd.DataFrame(columns=["aa", "M22_binding_enrichment_adj",
                                     "SI06_binding_enrichment_adj"])
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on="aa", how="outer")
    for col in ("M22_binding_enrichment_adj", "SI06_binding_enrichment_adj"):
        if col not in merged.columns:
            merged[col] = np.nan
    if len(merged) > max_n:
        merged = merged.sample(n=max_n, random_state=SEED).reset_index(drop=True)
    return merged


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
    dms_m22: Optional[Path],
    dms_si06: Optional[Path],
    gibbs_paths: List[Tuple[str, Path]],
    max_dms: int,
    max_gibbs: int,
) -> pd.DataFrame:
    """Assemble the per-row table (WT + DMS + Gibbs) all subsequent steps operate on.

    Columns: sequence, source, M22_enrich, SI06_enrich, cdrh3_token_positions (fixed for
    all rows since the VH framework is constant), cdrh3_identity_to_wt, gibbs_step,
    chain_id, gibbs_config.
    """
    rows: list[dict] = []
    fixed_pos = fixed_vh_cdrh3_token_positions()

    # 1) WT
    rows.append({
        "sequence": C05_VH,
        "source": "wt",
        "M22_enrich": np.nan,
        "SI06_enrich": np.nan,
        "cdrh3_token_positions": fixed_pos,
        "cdrh3_identity_to_wt": 1.0,
        "gibbs_step": -1,
        "chain_id": -1,
        "gibbs_config": "",
    })

    # 2) DMS
    log.info("Loading DMS …")
    dms = load_dms(dms_m22, dms_si06, max_dms)
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
            "gibbs_config": "",
        })

    # 3) Gibbs (one or more named configs)
    for config_name, gibbs_path in gibbs_paths:
        gibbs_df = load_gibbs(gibbs_path, max_gibbs)
        if gibbs_df is None:
            continue
        log.info("Loaded %d Gibbs rows from config=%s path=%s",
                 len(gibbs_df), config_name, gibbs_path)
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
                "gibbs_config": config_name,
            })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------- model load


def _extract_state_dict(raw) -> dict:
    """Handle evotuning ('model_state_dict' wrapper, keys prefixed 'model.')
    and DPO ('policy_state_dict' wrapper, HF-format keys) checkpoint shapes."""
    if isinstance(raw, dict):
        for key in ("policy_state_dict", "model_state_dict"):
            if key in raw and isinstance(raw[key], dict):
                return raw[key]
    return raw


def load_esm_encoder(checkpoint_path: Optional[str]) -> EsmModel:
    """Return an `EsmModel` (encoder only) from one of four checkpoint shapes.

    1. None → vanilla ESM2-35M from the HF hub.
    2. HF-format directory containing `model.safetensors` or `pytorch_model.bin`.
    3. Evotuning .pt: ``{"model_state_dict": {...}}`` with keys prefixed
       ``model.esm.*`` / ``model.lm_head.*`` (the ``ESM2Model`` wrapper).
    4. DPO .pt: ``{"policy_state_dict": {...}}`` with HF-format keys
       ``esm.*`` / ``lm_head.*`` (no wrapper prefix).

    In all .pt cases we strip the ``model.`` prefix and drop ``lm_head.*``,
    then load into a vanilla-initialized ``EsmModel``.
    """
    if checkpoint_path is None:
        log.info("Loading vanilla ESM2 from %s", ESM2_MODEL_ID)
        return EsmModel.from_pretrained(ESM2_MODEL_ID)

    checkpoint_path = os.path.expandvars(os.path.expanduser(checkpoint_path))
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
    state = _extract_state_dict(raw)

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
    p.add_argument("--gibbs-path", action="append", default=[],
                   help="CSV or FASTA of Gibbs-sampled sequences. Repeatable. "
                        "Format: NAME=PATH to tag rows with a config name "
                        "(e.g. 'distribution=outputs/gibbs/distribution/vanilla.csv'); "
                        "bare PATH is treated as NAME='default'.")
    p.add_argument("--output-path", required=True, help="Destination .npz path")
    p.add_argument("--dms-dataset", default=DEFAULT_DMS_DATASET, choices=sorted(DMS_DATASETS),
                   help="Named DMS dataset whose CSVs should be loaded "
                        "(resolves --dms-m22/--dms-si06 unless those are passed explicitly).")
    p.add_argument("--dms-m22", default=None,
                   help="Override M22 CSV path (defaults to the --dms-dataset entry).")
    p.add_argument("--dms-si06", default=None,
                   help="Override SI06 CSV path (defaults to the --dms-dataset entry).")
    p.add_argument("--max-dms", type=int, default=500)
    p.add_argument("--max-gibbs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--skip-if-current", action="store_true",
                   help="If --output-path already exists with the current SCHEMA_VERSION, "
                        "exit 0 without recomputing. Old/missing schema triggers a rebuild.")
    return p.parse_args()


def _existing_is_current(path: Path) -> bool:
    """True iff `path` exists and contains a matching schema_version field."""
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as z:
            if "schema_version" not in z.files:
                return False
            return str(z["schema_version"][0]) == SCHEMA_VERSION
    except Exception as exc:  # noqa: BLE001 — opening unknown legacy formats
        log.warning("Could not inspect %s (%s); will rebuild", path, exc)
        return False


def main() -> int:
    args = parse_args()

    out_path = Path(args.output_path)
    if args.skip_if_current and _existing_is_current(out_path):
        log.info("Skip: %s exists with schema_version=%s", out_path, SCHEMA_VERSION)
        return 0
    if args.skip_if_current and out_path.exists():
        log.warning("Existing %s has stale schema — rebuilding", out_path)

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

    gibbs_paths: List[Tuple[str, Path]] = []
    for spec in args.gibbs_path:
        if "=" in spec:
            name, path_str = spec.split("=", 1)
        else:
            name, path_str = "default", spec
        gibbs_paths.append((name, Path(path_str)))

    dataset_paths = DMS_DATASETS[args.dms_dataset]
    dms_m22 = args.dms_m22 if args.dms_m22 is not None else dataset_paths.get("m22")
    dms_si06 = args.dms_si06 if args.dms_si06 is not None else dataset_paths.get("si06")
    if dms_m22 is None:
        log.warning("DMS dataset %s has no M22 assay — M22 column will be NaN", args.dms_dataset)
    if dms_si06 is None:
        log.warning("DMS dataset %s has no SI06 assay — SI06 column will be NaN", args.dms_dataset)
    log.info("DMS dataset: %s (M22=%s, SI06=%s)", args.dms_dataset, dms_m22, dms_si06)

    df = build_reference_set(
        Path(dms_m22) if dms_m22 else None,
        Path(dms_si06) if dms_si06 else None,
        gibbs_paths,
        args.max_dms,
        args.max_gibbs,
    )
    log.info("Reference set: %d rows  (%s)", len(df), dict(df["source"].value_counts()))

    whole, cdr = run_inference(df, model, tokenizer, device, args.batch_size)

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
        gibbs_config=np.array(df["gibbs_config"].tolist()),
        model_variant=np.array([args.model_variant]),
        dms_dataset=np.array([args.dms_dataset]),
        schema_version=np.array([SCHEMA_VERSION]),
    )
    log.info("Wrote %s  (%d rows, embedding dim %d)", out_path, len(df), EMB_DIM)
    return 0


if __name__ == "__main__":
    sys.exit(main())
