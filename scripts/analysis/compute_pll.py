"""
Compute CDR-H3 pseudo-log-likelihood (PLL) for one model on one DMS dataset and
cache the result in the per-model analysis tree.

Artifact location (see protein_design.analysis.registry):
    $ANALYSIS_DIR/<model>/pll/<dataset>.csv   (+ <dataset>.csv.meta.json)

Columns written: <seq_col>, pll
The artifact is reused unless its provenance (checkpoint / base_model / dataset
path) changed or --force is passed.

The model is identified by a key in conf/analysis/models.yaml, which supplies
its checkpoint and base_model. For an ad-hoc model not in the registry, pass
--checkpoint / --base-model explicitly.

Usage:
    uv run python scripts/analysis/compute_pll.py --model evo_35m --dataset ed2_m22
    uv run python scripts/analysis/compute_pll.py --model evo_35m --dataset all

Prefer running via: sbatch bash_scripts/extract.sbatch --what pll --model … --dataset …
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

_CDR_START = C05_CDRH3_START + 1   # +1 for CLS token
_CDR_END = C05_CDRH3_END + 1

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


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


def load_esm_for_mlm(checkpoint: str, base_model: str) -> EsmForMaskedLM:
    if not checkpoint:
        log.info("No checkpoint given — loading vanilla %s", base_model)
        return EsmForMaskedLM.from_pretrained(base_model)
    p = Path(checkpoint)
    if p.is_file() and p.suffix == ".pt":
        return _load_pt_into_mlm(p, base_model)
    if p.is_dir():
        if (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists():
            return EsmForMaskedLM.from_pretrained(str(p))
        for name in ("best.pt", "final.pt"):
            if (p / name).exists():
                return _load_pt_into_mlm(p / name, base_model)
        raise FileNotFoundError(f"No HF weights, best.pt, or final.pt found at {checkpoint}")
    return EsmForMaskedLM.from_pretrained(checkpoint)


@torch.no_grad()
def compute_pll(
    model: EsmForMaskedLM,
    tokenizer,
    cdrh3_strings: list[str],
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Sum of one-at-a-time masked log-probs over the 24 CDR-H3 positions.

    All sequences share the same length (full VH = 138 + special tokens), so we
    flatten every (sequence, masked-position) job into one stream and run it in
    large batches. This keeps the GPU saturated — one big forward per batch
    instead of 24 length-`batch_size` passes per sequence-batch — for the same
    total FLOPs. `batch_size` here counts (sequence, position) pairs.
    """
    cdr_positions = list(range(_CDR_START, _CDR_END))
    L = len(cdr_positions)
    n = len(cdrh3_strings)
    mask_id = tokenizer.mask_token_id

    full_vhs = [add_context(s) for s in cdrh3_strings]
    spaced = [" ".join(list(s)) for s in full_vhs]
    enc = tokenizer(spaced, return_tensors="pt", padding=True, add_special_tokens=True)
    base_tokens = enc["input_ids"]      # (n, T) on CPU
    base_attn = enc["attention_mask"]   # (n, T) on CPU

    # One row per (sequence, masked-position): row k -> seq k // L, pos k % L.
    seq_idx = torch.arange(n).repeat_interleave(L)        # (n*L,)
    pos_idx = torch.tensor(cdr_positions).repeat(n)       # (n*L,)
    total = n * L
    out = torch.empty(total, dtype=torch.float32)

    for start in tqdm(range(0, total, batch_size), desc="PLL"):
        sl = slice(start, start + batch_size)
        s = seq_idx[sl]
        p = pos_idx[sl].to(device)
        toks = base_tokens[s].to(device)    # fancy-index copies; safe to mask
        attn = base_attn[s].to(device)
        rows = torch.arange(toks.shape[0], device=device)
        true_ids = toks[rows, p].clone()
        toks[rows, p] = mask_id
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(input_ids=toks, attention_mask=attn).logits
        sel = logits[rows, p, :].float()    # (b, V) — slice before softmax
        log_probs = torch.log_softmax(sel, dim=-1)
        out[sl] = log_probs[rows, true_ids].cpu()

    return out.view(n, L).sum(dim=1).numpy()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True,
                   help="Model key from conf/analysis/models.yaml (e.g. evo_35m). "
                        "Also the artifact folder name. Unknown keys require "
                        "--checkpoint/--base-model.")
    p.add_argument("--checkpoint", default=None,
                   help="Override the registry checkpoint (HF dir, .pt, or HF id). "
                        "Empty string forces vanilla --base-model.")
    p.add_argument("--base-model", default=None,
                   help=f"Override the registry base_model (HF id for architecture + "
                        f"tokenizer), e.g. {ESM2_35M_ID} or {ESM2_650M_ID}.")
    p.add_argument("--dataset", required=True,
                   help="Dataset key from conf/analysis/dms_datasets.yaml, or 'all'.")
    p.add_argument("--batch-size", type=int, default=256,
                   help="(sequence, masked-position) pairs per forward pass. "
                        "256 fits a 650M model in fp16 on a 20G GPU with margin; "
                        "the 35M models can go much higher (e.g. 2048).")
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
        out_csv = registry.artifact_path(args.model, "pll", f"{key}.csv")
        expected = {"checkpoint": checkpoint, "base_model": base_model,
                    "dataset_path": cfg["datasets"][key]["path"]}
        if not registry.needs_recompute(out_csv, expected, force=args.force):
            log.info("[skip] %s is up to date — pass --force to recompute", out_csv)
            continue
        pending.append((key, out_csv, expected))

    if not pending:
        log.info("Nothing to do.")
        return

    device = torch.device(args.device)
    log.info("Loading ESM2 model %r for PLL: %s (base=%s)",
             args.model, checkpoint or "vanilla", base_model)
    model = load_esm_for_mlm(checkpoint, base_model).eval().to(device)
    if device.type == "cuda":
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    for key, out_csv, expected in pending:
        ds = cfg["datasets"][key]
        log.info("[%s] reading %s", key, ds["path"])
        df = pd.read_csv(ds["path"])
        seq_col = ds["seq_col"]
        if seq_col not in df.columns:
            raise ValueError(f"{ds['path']} missing column {seq_col!r}; "
                             f"got: {list(df.columns)}")
        seqs = df[seq_col].astype(str).drop_duplicates().tolist()
        log.info("[%s] computing PLL on %d unique sequences", key, len(seqs))
        pll = compute_pll(model, tokenizer, seqs, device, args.batch_size)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({seq_col: seqs, "pll": pll}).to_csv(out_csv, index=False)
        registry.write_meta(out_csv, n=len(seqs), seq_col=seq_col, **expected)
        log.info("[%s] wrote %s  (n=%d)", key, out_csv, len(seqs))


if __name__ == "__main__":
    main()
