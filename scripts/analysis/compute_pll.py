"""
Compute CDR-H3 pseudo-log-likelihood (PLL) for one model on one DMS dataset and
cache the result. The plot script (plot_dms_correlations.py) reads this cache.

Cache location:
    $cache_root/pll/<model_name>/<dataset>.csv     (cache_root from conf yaml)

Columns written: <seq_col>, pll
If the cache file already exists it is reused unless --force is passed.

Usage:
    uv run python scripts/analysis/compute_pll.py \
        --model-name evodpo_4ep_step1376 \
        --checkpoint /cluster/.../checkpoints/evodpo_4ep/step_1376.pt \
        --dataset ed2_m22

Pass --dataset all to compute every dataset listed in the registry.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForMaskedLM

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from protein_design.constants import C05_CDRH3_START, C05_CDRH3_END, add_context  # noqa: E402

ESM2_35M_ID = "facebook/esm2_t12_35M_UR50D"
CONFIG_PATH = REPO_ROOT / "conf" / "analysis" / "dms_datasets.yaml"

_CDR_START = C05_CDRH3_START + 1   # +1 for CLS token
_CDR_END = C05_CDRH3_END + 1

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _extract_state_dict(raw: dict) -> dict:
    for key in ("policy_state_dict", "model_state_dict"):
        if key in raw and isinstance(raw[key], dict):
            return raw[key]
    return raw


def _load_pt_into_mlm(pt_path: Path) -> EsmForMaskedLM:
    log.info("Loading state-dict checkpoint %s", pt_path)
    raw = torch.load(pt_path, map_location="cpu", weights_only=False)
    state = _extract_state_dict(raw)
    new_state = {k[len("model."):] if k.startswith("model.") else k: v
                 for k, v in state.items()}
    model = EsmForMaskedLM.from_pretrained(ESM2_35M_ID)
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    non_optional = [m for m in missing
                    if not m.startswith("esm.contact_head.") and "position_ids" not in m]
    if non_optional:
        raise RuntimeError(f"Checkpoint missing required keys: {non_optional[:5]}")
    if unexpected:
        log.warning("Ignored %d unexpected keys (e.g. %s)", len(unexpected), unexpected[:3])
    return model


def load_esm_for_mlm(checkpoint: str) -> EsmForMaskedLM:
    if not checkpoint:
        log.info("No checkpoint given — loading vanilla %s", ESM2_35M_ID)
        return EsmForMaskedLM.from_pretrained(ESM2_35M_ID)
    p = Path(checkpoint)
    if p.is_file() and p.suffix == ".pt":
        return _load_pt_into_mlm(p)
    if p.is_dir():
        if (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists():
            return EsmForMaskedLM.from_pretrained(str(p))
        for name in ("best.pt", "final.pt"):
            if (p / name).exists():
                return _load_pt_into_mlm(p / name)
        raise FileNotFoundError(f"No HF weights, best.pt, or final.pt found at {checkpoint}")
    return EsmForMaskedLM.from_pretrained(checkpoint)


@torch.no_grad()
def compute_pll(
    model: EsmForMaskedLM,
    tokenizer,
    cdrh3_strings: list[str],
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    cdr_token_positions = list(range(_CDR_START, _CDR_END))
    n = len(cdrh3_strings)
    out = np.zeros((n, len(cdr_token_positions)), dtype=np.float32)
    mask_id = tokenizer.mask_token_id
    full_vhs = [add_context(s) for s in cdrh3_strings]

    for start in tqdm(range(0, n, batch_size), desc="PLL"):
        batch = full_vhs[start:start + batch_size]
        spaced = [" ".join(list(s)) for s in batch]
        enc = tokenizer(spaced, return_tensors="pt", padding=True, add_special_tokens=True)
        tokens = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        B = tokens.shape[0]

        for p_idx, token_pos in enumerate(cdr_token_positions):
            masked = tokens.clone()
            masked[:, token_pos] = mask_id
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(input_ids=masked, attention_mask=attn).logits
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            true_ids = tokens[:, token_pos]
            picks = log_probs[torch.arange(B, device=device), token_pos, true_ids]
            out[start:start + B, p_idx] = picks.cpu().numpy()

    return out.sum(axis=1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", required=True,
                   help="Folder name used in the cache path. Pick something human-readable "
                        "(e.g. evodpo_4ep_step1376).")
    p.add_argument("--checkpoint", default="",
                   help="ESM2-35M checkpoint (HF dir, .pt, or HF model ID). "
                        "Omit to use vanilla ESM2-35M.")
    p.add_argument("--dataset", required=True,
                   help="Dataset key from conf/analysis/dms_datasets.yaml, or 'all'.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--force", action="store_true",
                   help="Recompute even if the cache file already exists.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with CONFIG_PATH.open() as f:
        cfg = yaml.safe_load(f)
    cache_root = Path(cfg["paths"]["cache_root"]) / "pll" / args.model_name

    if args.dataset == "all":
        dataset_keys = list(cfg["datasets"].keys())
    else:
        if args.dataset not in cfg["datasets"]:
            raise SystemExit(f"Unknown dataset {args.dataset!r}. "
                             f"Known: {list(cfg['datasets'])}")
        dataset_keys = [args.dataset]

    pending = []
    for key in dataset_keys:
        out_csv = cache_root / f"{key}.csv"
        if out_csv.exists() and not args.force:
            log.info("[skip] %s exists — pass --force to recompute", out_csv)
            continue
        pending.append((key, out_csv))

    if not pending:
        log.info("Nothing to do.")
        return

    device = torch.device(args.device)
    log.info("Loading ESM2 model for PLL: %s", args.checkpoint or "vanilla")
    model = load_esm_for_mlm(args.checkpoint).eval().to(device)
    if device.type == "cuda":
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(ESM2_35M_ID)

    for key, out_csv in pending:
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
        log.info("[%s] wrote %s  (n=%d)", key, out_csv, len(seqs))


if __name__ == "__main__":
    main()
