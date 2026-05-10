"""Joint PLL PCA — masked per-position log P(true aa) on DMS, joint PCA across model variants.

Two analyses on the same forward passes:

* **C1** — Sum per-position log-probs to a sequence-level score, build N×M
  matrix (sequences × models), centre per model, fit PCA(2). PC1 = consensus
  sequence quality across models. PC2 = axis of model disagreement.
  A biplot with model loadings as arrows shows where each model "pulls."

* **C2** — Skip the position aggregation: build N×(M·P) matrix where P=24 CDR
  positions. Centre per (model, position). Fit PCA(4). The loadings
  (M × P heatmap per PC) reveal which CDR-H3 positions are most affected by
  each fine-tuning step — the scientifically richest output.

Compute cost: ~24 forward passes per sequence per model (one per masked
position). For N=500 DMS, M=3 models: ~36k forward passes total. Trivially
fast on GPU.

Caveat: PLL has different semantic content across model variants (vanilla
scores "is this a plausible protein?"; DPO-tuned models score "is this a
likely-to-bind antibody?"). Per-column centring is essential before PCA;
even so, the PC1 "consensus" axis aggregates over heterogeneous quantities.
Treat the result as descriptive, not as a unified quality score.

Inputs
------
``--variant LABEL=CHECKPOINT`` repeated once per model variant. CHECKPOINT
is one of: HuggingFace model ID, an HF-format directory, an HF directory
containing ``best.pt`` / ``final.pt``, or a direct ``.pt`` path.

DMS data path defaults match ``extract_embeddings.py`` and the C05 datasets
memory.

Writes
------
Single .npz at ``--output-path`` containing per-position raw scores, the
DMS metadata columns, and both C1 and C2 PCA outputs. Companion plots are
produced by ``plot_pll_pca.py``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForMaskedLM

from protein_design.constants import (
    C05_CDRH3_END,
    C05_CDRH3_START,
    add_context,
)

ESM2_MODEL_ID = "facebook/esm2_t12_35M_UR50D"
SEED = 42
DEFAULT_DMS_M22 = "/cluster/project/infk/krause/mdenegri/protein-design/datasets/scoring/D2_M22.csv"
DEFAULT_DMS_SI06 = "/cluster/project/infk/krause/mdenegri/protein-design/datasets/scoring/D2_SI06.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("compute_pll_pca")


# --------------------------------------------------------------------------- I/O


def load_dms(
    m22_path: Path,
    si06_path: Path | None,
    max_n: int,
    m22_col: str = "M22_binding_enrichment_adj",
    si06_col: str = "SI06_binding_enrichment_adj",
) -> pd.DataFrame:
    """Outer-merge M22 + SI06 on `aa` (24-aa CDRH3), sample up to max_n rows."""
    m22 = pd.read_csv(m22_path)[["aa", m22_col]].rename(columns={m22_col: "M22_binding_enrichment_adj"})
    if si06_path is None:
        merged = m22
        merged["SI06_binding_enrichment_adj"] = np.nan
    else:
        si06 = pd.read_csv(si06_path)[["aa", si06_col]].rename(columns={si06_col: "SI06_binding_enrichment_adj"})
        merged = m22.merge(si06, on="aa", how="outer")
    if len(merged) > max_n:
        merged = merged.sample(n=max_n, random_state=SEED).reset_index(drop=True)
    return merged.reset_index(drop=True)


# --------------------------------------------------------------------- model load


def _extract_state_dict(raw) -> dict:
    """Handle three .pt shapes: evotuning ('model_state_dict' wrapper, keys
    prefixed 'model.'), DPO ('policy_state_dict' wrapper, HF-format keys), or
    bare state dicts."""
    if isinstance(raw, dict):
        for key in ("policy_state_dict", "model_state_dict"):
            if key in raw and isinstance(raw[key], dict):
                return raw[key]
    return raw


def _load_pt_into_mlm(pt_path: Path) -> EsmForMaskedLM:
    """Load a .pt state dict produced by this repo's training pipeline."""
    log.info("Loading state-dict checkpoint %s", pt_path)
    raw = torch.load(pt_path, map_location="cpu", weights_only=False)
    state = _extract_state_dict(raw)
    new_state: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        new_state[k[len("model."):] if k.startswith("model.") else k] = v
    model = EsmForMaskedLM.from_pretrained(ESM2_MODEL_ID)
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    non_optional_missing = [m for m in missing
                            if not m.startswith("esm.contact_head.")
                            and "position_ids" not in m]
    if non_optional_missing:
        raise RuntimeError(
            f"Checkpoint missing required keys: {non_optional_missing[:5]} "
            f"(+{max(0, len(non_optional_missing)-5)} more)"
        )
    if unexpected:
        log.warning("Ignored %d unexpected keys (e.g. %s)", len(unexpected), unexpected[:3])
    return model


def load_esm_for_mlm(checkpoint: str) -> EsmForMaskedLM:
    """Resolve a checkpoint string to an ``EsmForMaskedLM``.

    Accepts: HF model ID, HF-format dir, dir with ``best.pt``/``final.pt``,
    or a direct ``.pt`` file. Empty/None falls back to vanilla ESM2-35M.
    """
    if not checkpoint:
        log.info("Loading vanilla ESM2 from %s", ESM2_MODEL_ID)
        return EsmForMaskedLM.from_pretrained(ESM2_MODEL_ID)

    p = Path(checkpoint)
    if p.is_file() and p.suffix == ".pt":
        return _load_pt_into_mlm(p)
    if p.is_dir():
        if (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists():
            log.info("Loading HF-format checkpoint from %s", p)
            return EsmForMaskedLM.from_pretrained(str(p))
        pt_path = next((p / n for n in ("best.pt", "final.pt") if (p / n).exists()), None)
        if pt_path is not None:
            return _load_pt_into_mlm(pt_path)
        raise FileNotFoundError(
            f"No HF weights, best.pt, or final.pt found at {checkpoint}"
        )
    # Treat as HF model ID (e.g. 'facebook/esm2_...').
    log.info("Loading HF model ID %s", checkpoint)
    return EsmForMaskedLM.from_pretrained(checkpoint)


# --------------------------------------------------------------------- inference


@torch.no_grad()
def per_position_log_probs(
    model: EsmForMaskedLM,
    tokenizer,
    full_vh_sequences: List[str],
    device: torch.device,
    cdr_token_positions: List[int],
    batch_size: int = 32,
) -> np.ndarray:
    """Mask each CDR-H3 token in turn, return log P(true aa | context).

    Returns an array of shape ``(len(sequences), len(cdr_token_positions))``
    of dtype float32. Each entry is the log-probability assigned to the true
    amino acid at that position when only that position is masked.
    """
    n = len(full_vh_sequences)
    P = len(cdr_token_positions)
    out = np.zeros((n, P), dtype=np.float32)
    mask_id = tokenizer.mask_token_id

    n_batches = (n + batch_size - 1) // batch_size
    for start in tqdm(range(0, n, batch_size), total=n_batches, desc="PLL"):
        batch = full_vh_sequences[start:start + batch_size]
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
            row_idx = torch.arange(B, device=device)
            picks = log_probs[row_idx, token_pos, true_ids]
            out[start:start + B, p_idx] = picks.cpu().numpy().astype(np.float32)

    return out


# --------------------------------------------------------------------------- PCA


def fit_c1(per_pos_pll: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sum-over-positions → N×M, centre per model, fit PCA(min(2, M))."""
    seq_pll = per_pos_pll.sum(axis=2)  # (N, M)
    centered = seq_pll - seq_pll.mean(axis=0, keepdims=True)
    n_components = min(2, seq_pll.shape[1])
    pca = PCA(n_components=n_components, random_state=SEED)
    coords = pca.fit_transform(centered).astype(np.float32)
    loadings = pca.components_.astype(np.float32)  # (n_components, M)
    return coords, loadings, pca.explained_variance_ratio_.astype(np.float32)


def fit_c2(per_pos_pll: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reshape to N × (M·P), centre per (model, position), fit PCA(min(4, ...))."""
    N, M, P = per_pos_pll.shape
    flat = per_pos_pll.reshape(N, M * P)
    centered = flat - flat.mean(axis=0, keepdims=True)
    n_components = min(4, M * P, N - 1)
    pca = PCA(n_components=n_components, random_state=SEED)
    coords = pca.fit_transform(centered).astype(np.float32)
    loadings = pca.components_.reshape(n_components, M, P).astype(np.float32)
    return coords, loadings, pca.explained_variance_ratio_.astype(np.float32)


# ------------------------------------------------------------------------- main


def parse_variant_spec(spec: str) -> Tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"--variant must be of the form LABEL=CHECKPOINT, got: {spec!r}"
        )
    label, ckpt = spec.split("=", 1)
    return label.strip(), ckpt.strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--variant",
        action="append",
        required=True,
        type=parse_variant_spec,
        help="LABEL=CHECKPOINT; repeat once per variant. CHECKPOINT may be empty "
             "(vanilla), an HF model ID, an HF dir, or a .pt file.",
    )
    p.add_argument("--dms-m22", default=DEFAULT_DMS_M22)
    p.add_argument("--dms-si06", default=None)
    p.add_argument("--dms-m22-col", default="M22_binding_enrichment_adj")
    p.add_argument("--dms-si06-col", default="SI06_binding_enrichment_adj")
    p.add_argument("--max-dms", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--output-path", type=Path, required=True,
                   help="Destination .npz path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    log.info("Loading DMS …")
    dms = load_dms(
        Path(args.dms_m22),
        Path(args.dms_si06) if args.dms_si06 else None,
        args.max_dms,
        m22_col=args.dms_m22_col,
        si06_col=args.dms_si06_col,
    )
    log.info("DMS: %d sequences", len(dms))

    cdr_strings = dms["aa"].astype(str).tolist()
    full_vhs = [add_context(s) for s in cdr_strings]
    cdr_token_positions = list(range(C05_CDRH3_START + 1, C05_CDRH3_END + 1))
    P = len(cdr_token_positions)
    N = len(full_vhs)
    M = len(args.variant)

    tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_ID)

    per_position_pll = np.zeros((N, M, P), dtype=np.float32)
    variant_labels: List[str] = []
    for m_idx, (label, checkpoint) in enumerate(args.variant):
        if label in variant_labels:
            raise ValueError(f"Duplicate variant label: {label!r}")
        variant_labels.append(label)
        log.info("=== %s (checkpoint=%r) ===", label, checkpoint)
        model = load_esm_for_mlm(checkpoint).eval().to(device)
        for p_ in model.parameters():
            p_.requires_grad = False
        if device.type == "cuda":
            model = model.half()

        scores = per_position_log_probs(
            model, tokenizer, full_vhs, device, cdr_token_positions,
            batch_size=args.batch_size,
        )
        log.info("[%s] mean log P = %.3f, min = %.3f, max = %.3f",
                 label, float(scores.mean()), float(scores.min()), float(scores.max()))
        per_position_pll[:, m_idx, :] = scores

        # Free GPU memory before loading the next variant.
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    log.info("Fitting C1 (sequence-level joint PCA) …")
    c1_coords, c1_loadings, c1_ev = fit_c1(per_position_pll)
    log.info("C1: PC1=%.1f%% PC2=%.1f%%", 100 * c1_ev[0],
             100 * c1_ev[1] if len(c1_ev) > 1 else float("nan"))

    log.info("Fitting C2 (per-position joint PCA) …")
    c2_coords, c2_loadings, c2_ev = fit_c2(per_position_pll)
    log.info("C2: PC1=%.1f%% PC2=%.1f%% PC3=%.1f%% PC4=%.1f%%",
             100 * c2_ev[0],
             100 * c2_ev[1] if len(c2_ev) > 1 else float("nan"),
             100 * c2_ev[2] if len(c2_ev) > 2 else float("nan"),
             100 * c2_ev[3] if len(c2_ev) > 3 else float("nan"))

    np.savez(
        args.output_path,
        per_position_pll=per_position_pll,
        model_variants=np.array(variant_labels),
        cdr_position_index=np.arange(P, dtype=np.int32),
        dms_cdrh3=np.array(cdr_strings),
        dms_M22_enrich=dms["M22_binding_enrichment_adj"].to_numpy(dtype=np.float32),
        dms_SI06_enrich=dms["SI06_binding_enrichment_adj"].to_numpy(dtype=np.float32),
        c1_coords=c1_coords,
        c1_loadings=c1_loadings,
        c1_explained_variance=c1_ev,
        c2_coords=c2_coords,
        c2_loadings=c2_loadings,
        c2_explained_variance=c2_ev,
    )
    log.info("Wrote %s", args.output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
