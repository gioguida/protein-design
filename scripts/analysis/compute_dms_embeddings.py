"""
Extract CDR-H3 mean-pooled embeddings for one model on one DMS dataset and cache
them in the per-model analysis tree, for the linear-probe / PCA figures.

Artifact location (see protein_design.analysis.registry):
    $ANALYSIS_DIR/<model>/emb/<dataset>.npz   (+ <dataset>.npz.meta.json)

The .npz holds, for each available split of the dataset, three arrays:
    <split>_emb  (n, D) float32   CDR-H3 mean-pooled last-hidden-state
    <split>_y    (n,)   float32   experimental enrichment (the probe target)
    <split>_seq  (n,)   <U        the 24-aa CDR-H3 string
plus  splits (<U), dim (int), seq_col (<U).

Splits come from the dataset's directory: train.csv / test.csv siblings of the
registry path (the probe fits on train, evaluates on test). A single-file
dataset (no train/test siblings, e.g. `exp`) is stored under the split "all" and
the probe falls back to cross-validation.

The artifact is reused unless its provenance (checkpoint / base_model / dataset
path) changed or --force is passed.

Usage:
    uv run python scripts/analysis/compute_dms_embeddings.py --model evo_650m --dataset ed2_m22
    uv run python scripts/analysis/compute_dms_embeddings.py --model vanilla_650m --dataset all

Prefer running via: sbatch bash_scripts/extract.sbatch --what emb --model … --dataset …
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForMaskedLM

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from protein_design.constants import C05_CDRH3_START, C05_CDRH3_END, add_context  # noqa: E402
from protein_design.analysis import registry  # noqa: E402

ESM2_35M_ID = "facebook/esm2_t12_35M_UR50D"
ESM2_650M_ID = "facebook/esm2_t33_650M_UR50D"

# CDR-H3 token positions in the fixed 138-aa VH (+1 for the CLS/BOS token).
_CDR_START = C05_CDRH3_START + 1
_CDR_END = C05_CDRH3_END + 1

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── model loading (mirrors compute_pll.py: registry base_model is authoritative) ──

def _extract_state_dict(raw: dict) -> dict:
    for key in ("policy_state_dict", "model_state_dict"):
        if key in raw and isinstance(raw[key], dict):
            return raw[key]
    return raw


def _load_pt_into_mlm(pt_path: Path, base_model: str) -> EsmForMaskedLM:
    log.info("Loading state-dict checkpoint %s (base=%s)", pt_path, base_model)
    raw = torch.load(pt_path, map_location="cpu", weights_only=False)
    state = _extract_state_dict(raw)
    new_state = {k[len("model."):] if k.startswith("model.") else k: v
                 for k, v in state.items()}
    model = EsmForMaskedLM.from_pretrained(base_model)
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    non_optional = [m for m in missing
                    if not m.startswith("esm.contact_head.") and "position_ids" not in m]
    if non_optional:
        raise RuntimeError(f"Checkpoint missing required keys: {non_optional[:5]}")
    if unexpected:
        log.warning("Ignored %d unexpected keys (e.g. %s)", len(unexpected), unexpected[:3])
    return model


def load_esm_encoder(checkpoint: str, base_model: str):
    """Return the EsmModel encoder for `checkpoint` (empty -> vanilla base_model)."""
    if not checkpoint:
        log.info("No checkpoint given — loading vanilla %s", base_model)
        return EsmForMaskedLM.from_pretrained(base_model).esm
    p = Path(checkpoint)
    if p.is_file() and p.suffix == ".pt":
        return _load_pt_into_mlm(p, base_model).esm
    if p.is_dir():
        if (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists():
            return EsmForMaskedLM.from_pretrained(str(p)).esm
        for name in ("best.pt", "final.pt"):
            if (p / name).exists():
                return _load_pt_into_mlm(p / name, base_model).esm
        raise FileNotFoundError(f"No HF weights, best.pt, or final.pt found at {checkpoint}")
    return EsmForMaskedLM.from_pretrained(checkpoint).esm


# ── inference ────────────────────────────────────────────────────────────────

@torch.no_grad()
def cdrh3_embeddings(
    encoder,
    tokenizer,
    cdrh3_strings: list[str],
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    """Mean-pool the encoder's last hidden state over the 24 CDR-H3 token positions.

    Every sequence shares the fixed 138-aa VH (constant framework), so the CDR-H3
    token slice is the same for all rows — only the H3 residues vary, which is
    exactly the signal the probe should read.
    """
    full_vhs = [add_context(s) for s in cdrh3_strings]
    spaced = [" ".join(list(s)) for s in full_vhs]
    out: list[np.ndarray] = []
    for start in tqdm(range(0, len(spaced), batch_size), desc="embed"):
        batch = spaced[start:start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=True)
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            h = encoder(input_ids=input_ids, attention_mask=attn).last_hidden_state
        cdr = h[:, _CDR_START:_CDR_END, :].float().mean(dim=1)  # (b, D)
        out.append(cdr.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


# ── split discovery ──────────────────────────────────────────────────────────

def _dataset_splits(ds_path: str) -> dict[str, Path]:
    """Map split name -> CSV path for one dataset.

    Uses train.csv/test.csv siblings of the registry path when present (the probe
    needs both); otherwise the registry file itself is the single split "all".
    """
    p = Path(ds_path)
    parent = p.parent
    found = {name: parent / f"{name}.csv"
             for name in ("train", "test") if (parent / f"{name}.csv").exists()}
    return found if found else {"all": p}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="Model key from conf/analysis/models.yaml (also the artifact "
                        "folder). Unknown keys require --checkpoint/--base-model.")
    p.add_argument("--checkpoint", default=None,
                   help="Override the registry checkpoint (HF dir, .pt, or HF id). "
                        "Empty string forces vanilla --base-model.")
    p.add_argument("--base-model", default=None,
                   help=f"Override the registry base_model, e.g. {ESM2_35M_ID} or {ESM2_650M_ID}.")
    p.add_argument("--dataset", required=True,
                   help="Dataset key from conf/analysis/dms_datasets.yaml, or 'all'.")
    p.add_argument("--batch-size", type=int, default=128,
                   help="Sequences per forward pass (128 fits a 650M model in fp16 on a 20G GPU).")
    p.add_argument("--force", action="store_true",
                   help="Recompute even if a fresh artifact already exists.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spec = registry.resolve_model(args.model, checkpoint=args.checkpoint,
                                  base_model=args.base_model)
    checkpoint = spec["checkpoint"] or ""
    base_model = spec["base_model"]
    cfg = registry.load_datasets_cfg()

    pending = []
    for key in registry.dataset_keys(args.dataset):
        out_npz = registry.artifact_path(args.model, "emb", f"{key}.npz")
        expected = {"checkpoint": checkpoint, "base_model": base_model,
                    "dataset_path": cfg["datasets"][key]["path"]}
        if not registry.needs_recompute(out_npz, expected, force=args.force):
            log.info("[skip] %s is up to date — pass --force to recompute", out_npz)
            continue
        pending.append((key, out_npz, expected))

    if not pending:
        log.info("Nothing to do.")
        return

    device = torch.device(args.device)
    log.info("Loading ESM2 model %r for embeddings: %s (base=%s)",
             args.model, checkpoint or "vanilla", base_model)
    encoder = load_esm_encoder(checkpoint, base_model).eval().to(device)
    for prm in encoder.parameters():
        prm.requires_grad = False
    if device.type == "cuda":
        encoder = encoder.half()
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    for key, out_npz, expected in pending:
        ds = cfg["datasets"][key]
        seq_col, enrich_col = ds["seq_col"], ds["enrichment_col"]
        arrays: dict[str, np.ndarray] = {}
        split_names: list[str] = []
        for split, path in _dataset_splits(ds["path"]).items():
            df = pd.read_csv(path)
            if seq_col not in df.columns or enrich_col not in df.columns:
                raise ValueError(f"{path} missing {seq_col!r}/{enrich_col!r}; got {list(df.columns)}")
            df = (df[[seq_col, enrich_col]]
                  .dropna(subset=[seq_col, enrich_col])
                  .drop_duplicates(subset=[seq_col], keep="first"))
            seqs = df[seq_col].astype(str).tolist()
            y = df[enrich_col].to_numpy(np.float32)
            log.info("[%s/%s] embedding %d sequences from %s", key, split, len(seqs), path)
            emb = cdrh3_embeddings(encoder, tokenizer, seqs, device, args.batch_size)
            arrays[f"{split}_emb"] = emb
            arrays[f"{split}_y"] = y
            arrays[f"{split}_seq"] = np.array(seqs)
            split_names.append(split)

        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_npz, splits=np.array(split_names),
                 dim=np.array([next(iter(arrays.values())).shape[1]]),
                 seq_col=np.array([seq_col]), **arrays)
        registry.write_meta(out_npz, splits=split_names, seq_col=seq_col, **expected)
        log.info("[%s] wrote %s  (splits=%s)", key, out_npz, split_names)


if __name__ == "__main__":
    main()
