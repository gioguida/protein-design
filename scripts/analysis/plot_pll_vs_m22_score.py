"""
Correlation between CDR-H3 PLL and M22 flu-scorer enrichment on sampled sequences.

Loads a Gibbs or beam-search CSV, computes CDR-H3 PLL under the user's ESM2
checkpoint, then scores every sequence with the M22 flu binding oracle.  Produces
a blue-hexbin scatter with a red OLS regression line and Spearman ρ annotation,
matching the style used in W&B training plots.

The M22 scorer (ESM2-8M backbone + MLP head) is loaded directly from the .pt
checkpoint without the ``esme`` library.  The esm_state_dict keys are remapped
from esme format to transformers format at load time.

Input CSV must have a ``cdrh3`` column (24-aa CDR-H3 strings).

Outputs
-------
<output-dir>/pll_vs_m22_score.png
<output-dir>/pll_vs_m22_scores.csv    (cdrh3, pll, m22_score)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForMaskedLM, EsmModel

# protein-design package (two dirs up from scripts/analysis/)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from protein_design.constants import C05_CDRH3_START, C05_CDRH3_END, add_context
from protein_design.wandb_plots import set_publication_style

ESM2_35M_ID = "facebook/esm2_t12_35M_UR50D"
ESM2_8M_ID = "facebook/esm2_t6_8M_UR50D"
DEFAULT_SCORER_CKPT = "/cluster/project/infk/krause/ssussex/flu/models/M22_best.pt"

# CDR-H3 slice in token space: CLS(1) + left_context(103) = 104, end = 104 + 24 = 128
_CDR_START = C05_CDRH3_START + 1   # +1 for CLS token
_CDR_END = C05_CDRH3_END + 1

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ESM2 model loading (PLL)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# PLL computation (CDR-H3 only, masked per-position)
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_pll(
    model: EsmForMaskedLM,
    tokenizer,
    cdrh3_strings: list[str],
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Return per-sequence CDR-H3 PLL (sum of masked per-position log-probs)."""
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


# ---------------------------------------------------------------------------
# M22 flu scorer (no esme / flash_attn dependency)
# ---------------------------------------------------------------------------


def _remap_esme_to_transformers(esme_sd: dict) -> dict:
    """Remap esme ESM2 state dict keys to transformers EsmModel keys.

    Both use the same 33-token alphabet (L=4, A=5, ...) and the same
    weight shapes for all attention and FFN components, so the remapping
    is purely a key rename.
    """
    mapping = {
        "self_attn.norm.weight":  "attention.LayerNorm.weight",
        "self_attn.norm.bias":    "attention.LayerNorm.bias",
        "self_attn.q.weight":     "attention.self.query.weight",
        "self_attn.q.bias":       "attention.self.query.bias",
        "self_attn.k.weight":     "attention.self.key.weight",
        "self_attn.k.bias":       "attention.self.key.bias",
        "self_attn.v.weight":     "attention.self.value.weight",
        "self_attn.v.bias":       "attention.self.value.bias",
        "self_attn.out.weight":   "attention.output.dense.weight",
        "self_attn.out.bias":     "attention.output.dense.bias",
        "final.0.weight":         "LayerNorm.weight",
        "final.0.bias":           "LayerNorm.bias",
        "final.1.weight":         "intermediate.dense.weight",
        "final.1.bias":           "intermediate.dense.bias",
        "final.3.weight":         "output.dense.weight",
        "final.3.bias":           "output.dense.bias",
    }
    new_sd = {}
    for k, v in esme_sd.items():
        if k == "embed_tokens.weight":
            new_sd["embeddings.word_embeddings.weight"] = v
        elif k.startswith("layers."):
            parts = k.split(".")
            layer_idx = parts[1]
            rest = ".".join(parts[2:])
            if rest in mapping:
                new_sd[f"encoder.layer.{layer_idx}.{mapping[rest]}"] = v
        elif k in ("emb_layer_norm_after.weight", "emb_layer_norm_after.bias"):
            new_sd[f"encoder.{k}"] = v
        # lm_head keys are skipped — we use EsmModel, not EsmForMaskedLM
    return new_sd


class M22Scorer:
    """M22 flu binding scorer using transformers ESM2-8M backbone.

    Loads the checkpoint's esm_state_dict (esme format) and head_state_dict
    directly into a transformers EsmModel + a small MLP — no esme or flash_attn
    needed.  Implements the same forward pass as flu/src/model.py with
    aggregate='mean_pooling' and use_context=True.
    """

    def __init__(self, ckpt_path: str, device: torch.device) -> None:
        log.info("Loading M22 scorer from %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        remapped = _remap_esme_to_transformers(ckpt["esm_state_dict"])
        self.esm = EsmModel.from_pretrained(ESM2_8M_ID)
        missing, _ = self.esm.load_state_dict(remapped, strict=False)
        real_missing = [m for m in missing
                        if "rotary_embeddings" not in m
                        and "contact_head" not in m
                        and "position_ids" not in m
                        and "pooler" not in m]
        if real_missing:
            raise RuntimeError(f"Scorer checkpoint missing keys: {real_missing[:5]}")

        self.head = nn.Sequential(
            nn.Linear(320, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.head.load_state_dict(ckpt["head_state_dict"])

        self.target_mean: float = ckpt["target_mean"]
        self.target_std: float = ckpt["target_std"]
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(ESM2_8M_ID)

        self.esm = self.esm.to(device).eval()
        self.head = self.head.to(device).eval()
        log.info("M22 scorer ready (target_mean=%.4f  target_std=%.4f)",
                 self.target_mean, self.target_std)

    @torch.no_grad()
    def score_batch(self, cdrh3_strings: list[str]) -> np.ndarray:
        full_vhs = [add_context(s) for s in cdrh3_strings]
        spaced = [" ".join(list(s)) for s in full_vhs]
        enc = self.tokenizer(spaced, return_tensors="pt", padding=True, add_special_tokens=True)
        tokens = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)

        out = self.esm(input_ids=tokens, attention_mask=attn)
        # mean pool the 24 CDR-H3 token positions
        cdr_hidden = out.last_hidden_state[:, _CDR_START:_CDR_END, :].float()
        embeddings = cdr_hidden.mean(dim=1)  # [batch, 320]

        logits = self.head(embeddings).squeeze(-1)  # [batch]
        scores = logits * self.target_std + self.target_mean
        return scores.cpu().numpy()

    def score_all(self, cdrh3_strings: list[str], batch_size: int = 32) -> np.ndarray:
        results = []
        for i in tqdm(range(0, len(cdrh3_strings), batch_size), desc="M22 score"):
            results.append(self.score_batch(cdrh3_strings[i:i + batch_size]))
        return np.concatenate(results)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot_pll_vs_score(
    pll: np.ndarray,
    score: np.ndarray,
    model_variant: str,
    out_path: Path,
) -> None:
    set_publication_style()

    mask = np.isfinite(pll) & np.isfinite(score)
    pll_c, score_c = pll[mask], score[mask]
    n = int(mask.sum())
    rho, pval = spearmanr(pll_c, score_c)

    fig, ax = plt.subplots(figsize=(5, 4.5), constrained_layout=True)

    if n > 500:
        ax.hexbin(pll_c, score_c, gridsize=40, cmap="Blues", mincnt=1)
    else:
        ax.scatter(pll_c, score_c, s=8, alpha=0.35, color="#648FFF", linewidths=0)

    # red OLS regression line
    coeffs = np.polyfit(pll_c, score_c, 1)
    x_line = np.linspace(pll_c.min(), pll_c.max(), 200)
    ax.plot(x_line, np.polyval(coeffs, x_line),
            color="#DC267F", linewidth=1.5, label="OLS fit")

    rho_txt = rf"$\rho$ = {rho:.3f}  ($p$ = {pval:.1e}, $n$ = {n})"
    ax.text(0.03, 0.97, rho_txt, ha="left", va="top",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"))

    ax.set_xlabel("CDR-H3 PLL")
    ax.set_ylabel("M22 log-enrichment score")
    ax.set_title(f"PLL vs M22 scorer — {model_variant}")
    ax.legend(fontsize=9)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequences-csv", type=Path, required=True,
                   help="Gibbs or beam CSV with a 'cdrh3' column.")
    p.add_argument("--checkpoint-path", default="",
                   help="ESM2-35M checkpoint for PLL (HF dir, .pt, or HF model ID). "
                        "Omit to use vanilla ESM2-35M.")
    p.add_argument("--model-variant", required=True,
                   help="Label used in the plot title (e.g. 'evotuned+dpo').")
    p.add_argument("--scorer-ckpt", default=DEFAULT_SCORER_CKPT,
                   help="Path to M22 scorer .pt checkpoint.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-sequences", type=int, default=None,
                   help="Cap on unique CDR-H3 sequences (useful for quick checks).")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    log.info("Reading sequences from %s", args.sequences_csv)
    df = pd.read_csv(args.sequences_csv)
    if "cdrh3" not in df.columns:
        raise ValueError(f"CSV must contain a 'cdrh3' column; got: {list(df.columns)}")
    seqs = df["cdrh3"].dropna().astype(str).unique().tolist()
    if args.max_sequences is not None:
        seqs = seqs[: args.max_sequences]
    log.info("Scoring %d unique CDR-H3 sequences", len(seqs))

    # ── PLL (user's ESM2-35M checkpoint) ─────────────────────────────────────
    log.info("Loading ESM2 model for PLL: %s", args.checkpoint_path or "vanilla")
    pll_model = load_esm_for_mlm(args.checkpoint_path).eval().to(device)
    if device.type == "cuda":
        pll_model = pll_model.half()
    pll_tokenizer = AutoTokenizer.from_pretrained(ESM2_35M_ID)

    pll = compute_pll(pll_model, pll_tokenizer, seqs, device, args.batch_size)

    del pll_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── M22 score (ESM2-8M oracle) ────────────────────────────────────────────
    scorer = M22Scorer(args.scorer_ckpt, device)
    m22_scores = scorer.score_all(seqs, args.batch_size)

    # ── save CSV ──────────────────────────────────────────────────────────────
    scores_df = pd.DataFrame({"cdrh3": seqs, "pll": pll, "m22_score": m22_scores})
    csv_path = args.output_dir / "pll_vs_m22_scores.csv"
    scores_df.to_csv(csv_path, index=False)
    log.info("Wrote %s", csv_path)

    valid = np.isfinite(pll) & np.isfinite(m22_scores)
    rho, pval = spearmanr(pll[valid], m22_scores[valid])
    log.info("Spearman rho=%.3f  p=%.2e  n=%d", rho, pval, int(valid.sum()))

    # ── plot ──────────────────────────────────────────────────────────────────
    plot_pll_vs_score(
        pll, m22_scores, args.model_variant,
        args.output_dir / "pll_vs_m22_score.png",
    )


if __name__ == "__main__":
    main()
