#!/usr/bin/env python3
"""Spearman-correlation heatmap: models × DMS scoring regions (ED2-M22).

Rows = models
Columns = [Left-1, Left-3, Left-5, All-CDR, Right-1, Right-3, Right-5]
  • Left-k: sequences where at least one mutation falls in the first k CDR-H3 positions
  • Right-k: sequences where at least one mutation falls in the last k CDR-H3 positions
  • All-CDR: all sequences in the split
Color encodes Spearman rho magnitude (diverging, centered at 0).

Per-model scores are cached to disk on first run so subsequent invocations
(e.g. with a different split or enrichment column) skip the forward passes.
  • Checkpoint-based models: cache written to the checkpoint's parent directory.
  • HuggingFace / vanilla models: cache written to
    data/model_scores/<sanitized-label>/ inside the project root.

Usage
-----
uv run scripts/analysis/plot_spearman_heatmap.py \\
    --model "ESM2 35M|facebook/esm2_t12_35M_UR50D" \\
    --model "Evotuned|/path/to/checkpoint.pt" \\
    --dms-dir data/processed/dms_splits/ed2_m22 \\
    --output-dir reports/spearman_heatmap

Model spec: LABEL|CHECKPOINT  or  LABEL|SIZE|CHECKPOINT  (SIZE is ignored).
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from transformers import EsmForMaskedLM

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protein_design.config import build_model_config
from protein_design.constants import C05_CDRH3
from protein_design.eval import (
    _mutation_positions_per_row,
    evaluate_spearman,
    score_double_mutants,
    score_sequences_cdr_pll,
)
from protein_design.model import ESM2Model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("plot_spearman_heatmap")

BASE_MODEL_CONF_PATH = "conf/model/esm2_35m.yaml"
ENRICHMENT_COL = "M22_binding_enrichment_adj"
FLANK_KS = (1, 3, 5)
SCORE_COL = "pll"

# Column order and display labels
COLUMN_KEYS = [
    "spearman_avg_left1",
    "spearman_avg_left3",
    "spearman_avg_left5",
    "spearman_avg",
    "spearman_avg_right1",
    "spearman_avg_right3",
    "spearman_avg_right5",
]
COLUMN_LABELS = ["Left 1", "Left 3", "Left 5", "All CDR", "Right 1", "Right 3", "Right 5"]
N_KEYS = [
    "n_left_1",
    "n_left_3",
    "n_left_5",
    None,   # All CDR — n comes from the full dataframe
    "n_right_1",
    "n_right_3",
    "n_right_5",
]


# ---------------------------------------------------------------------------
# Model loading (mirrors compare_models.py)
# ---------------------------------------------------------------------------


def _extract_state_dict(raw: Any) -> dict[str, torch.Tensor]:
    if isinstance(raw, dict):
        for key in ("policy_state_dict", "model_state_dict"):
            if key in raw and isinstance(raw[key], dict):
                return raw[key]
    if isinstance(raw, dict):
        return raw
    raise TypeError(f"Unsupported checkpoint payload type: {type(raw)}")


def _load_pt_into_esm(model: ESM2Model, pt_path: Path) -> None:
    raw = torch.load(pt_path, map_location="cpu", weights_only=False)
    state = _extract_state_dict(raw)
    new_state: dict[str, torch.Tensor] = {
        k[len("model."):] if k.startswith("model.") else k: v
        for k, v in state.items()
    }
    missing, unexpected = model.model.load_state_dict(new_state, strict=False)
    non_optional = [
        m for m in missing
        if not m.startswith("esm.contact_head.") and "position_ids" not in m
    ]
    if non_optional:
        raise RuntimeError(f"Checkpoint missing required keys: {non_optional[:5]}")
    if unexpected:
        log.warning("Ignored %d unexpected keys from %s", len(unexpected), pt_path)


def _load_model(device: torch.device, checkpoint: str) -> ESM2Model:
    cfg = OmegaConf.load(REPO_ROOT / BASE_MODEL_CONF_PATH)
    model = ESM2Model(build_model_config(cfg, device=str(device)))

    ckpt = Path(checkpoint)
    if ckpt.is_file() and ckpt.suffix == ".pt":
        _load_pt_into_esm(model, ckpt)
    elif ckpt.is_dir():
        if (ckpt / "model.safetensors").exists() or (ckpt / "pytorch_model.bin").exists():
            model.model = EsmForMaskedLM.from_pretrained(str(ckpt))
        else:
            pt_path = next(
                (ckpt / n for n in ("best.pt", "final.pt") if (ckpt / n).exists()), None
            )
            if pt_path is None:
                raise FileNotFoundError(f"No HF weights, best.pt, or final.pt found at {ckpt}")
            _load_pt_into_esm(model, pt_path)
    elif ckpt.is_absolute():
        raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt}")
    else:
        # HuggingFace model ID
        model.model = EsmForMaskedLM.from_pretrained(checkpoint)

    model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# Score caching
# ---------------------------------------------------------------------------


def _cache_dir(checkpoint: str, label: str) -> Path:
    """Derive the directory where score CSVs are stored for this model.

    Local checkpoints → parent directory of the .pt file (or the dir itself).
    HuggingFace model IDs → data/model_scores/<sanitized-label>/ in project root.
    """
    p = Path(checkpoint)
    if p.is_file():
        return p.parent
    if p.is_dir():
        return p
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", label).strip("_")
    return REPO_ROOT / "data" / "model_scores" / safe


def _scores_csv_path(cache_dir: Path, dms_dir: Path, split: str) -> Path:
    return cache_dir / f"{dms_dir.name}_{split}_scores.csv"


def _load_cached_scores(cache_path: Path, df: pd.DataFrame) -> np.ndarray | None:
    if not cache_path.exists():
        return None
    cached = pd.read_csv(cache_path)
    if SCORE_COL not in cached.columns:
        log.info("Cache %s has no %r column — will recompute", cache_path, SCORE_COL)
        return None
    if len(cached) != len(df):
        log.warning(
            "Cache row count mismatch (%d vs %d) in %s — will recompute",
            len(cached), len(df), cache_path,
        )
        return None
    if "aa" in cached.columns and not (cached["aa"].values == df["aa"].values).all():
        log.warning("Cache sequence mismatch in %s — will recompute", cache_path)
        return None
    log.info("Loaded cached scores from %s", cache_path)
    return cached[SCORE_COL].to_numpy(dtype=np.float32)


def _save_cached_scores(
    cache_path: Path, df: pd.DataFrame, scores: np.ndarray, enrichment_col: str
) -> None:
    keep = [c for c in ["aa", "mut", "num_mut", enrichment_col] if c in df.columns]
    out = df[keep].copy()
    out[SCORE_COL] = scores
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False)
    log.info("Saved scores → %s", cache_path)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _load_dms(dms_dir: Path, split: str, enrichment_col: str) -> pd.DataFrame:
    csv = dms_dir / f"{split}.csv"
    if not csv.exists():
        raise FileNotFoundError(f"DMS split file not found: {csv}")
    df = pd.read_csv(csv)
    df["aa"] = df["aa"].astype(str).str.strip()
    df = df[(df["aa"] != "") & pd.to_numeric(df[enrichment_col], errors="coerce").notna()].copy()
    df[enrichment_col] = pd.to_numeric(df[enrichment_col], errors="coerce")
    df = df.reset_index(drop=True)
    log.info("Loaded %d DMS sequences from %s", len(df), csv)
    return df


def _compute_raw_scores(
    model: ESM2Model,
    df: pd.DataFrame,
    scoring_mode: str,
    batch_size: int,
) -> np.ndarray:
    if scoring_mode == "cdr_pll":
        sequences = df["aa"].astype(str).str.strip().tolist()
        return score_sequences_cdr_pll(scorer=model, sequences=sequences, batch_size=batch_size)
    return score_double_mutants(
        model=None, tokenizer=None, wt=C05_CDRH3, df=df, device=None,
        strategy="average", batch_size=batch_size, scorer=model,
    )


def _get_scores(
    checkpoint: str,
    label: str,
    df: pd.DataFrame,
    enrichment_col: str,
    scoring_mode: str,
    batch_size: int,
    device: torch.device,
    dms_dir: Path,
    split: str,
) -> np.ndarray:
    """Return pll scores for every row in df, loading from cache when available."""
    cache_path = _scores_csv_path(_cache_dir(checkpoint, label), dms_dir, split)

    scores = _load_cached_scores(cache_path, df)
    if scores is not None:
        return scores

    log.info("Loading model %r ...", label)
    model = _load_model(device, checkpoint)
    if device.type == "cuda":
        model.model = model.model.half()
    try:
        log.info("Scoring %d sequences (mode=%s) ...", len(df), scoring_mode)
        scores = _compute_raw_scores(model, df, scoring_mode, batch_size)
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _save_cached_scores(cache_path, df, scores, enrichment_col)
    return scores


def _compute_spearmans(
    scores: np.ndarray,
    df: pd.DataFrame,
    enrichment_col: str,
    n_all: int,
) -> dict[str, float | int]:
    enrichment = np.asarray(df[enrichment_col].values, dtype=float)
    row: dict[str, float | int] = {"n_all": n_all}

    rho, _ = evaluate_spearman(scores, enrichment)
    row["spearman_avg"] = float(rho)

    if "mut" in df.columns:
        cdr_len = len(C05_CDRH3)
        row_muts = _mutation_positions_per_row(df, C05_CDRH3)
        for k in FLANK_KS:
            left_w = set(range(k))
            right_w = set(range(cdr_len - k, cdr_len))
            for side, window in (("left", left_w), ("right", right_w)):
                mask = np.array([bool(m & window) for m in row_muts])
                row[f"n_{side}_{k}"] = int(mask.sum())
                rho_k, _ = evaluate_spearman(scores[mask], enrichment[mask])
                row[f"spearman_avg_{side}{k}"] = float(rho_k)
    else:
        for k in FLANK_KS:
            for side in ("left", "right"):
                row[f"n_{side}_{k}"] = 0
                row[f"spearman_avg_{side}{k}"] = float("nan")

    return row


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_model_spec(spec: str) -> tuple[str, str]:
    """Accept LABEL|CHECKPOINT or LABEL|SIZE|CHECKPOINT (SIZE is ignored)."""
    parts = [p.strip() for p in spec.split("|")]
    if len(parts) == 2:
        return parts[0], parts[1]
    if len(parts) == 3:
        return parts[0], parts[2]
    raise argparse.ArgumentTypeError(
        f"Invalid --model spec {spec!r}. Expected LABEL|CHECKPOINT or LABEL|SIZE|CHECKPOINT"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", required=True, type=_parse_model_spec,
                   metavar="LABEL|CHECKPOINT")
    p.add_argument("--dms-dir", type=Path,
                   default=REPO_ROOT / "data/processed/dms_splits/ed2_m22",
                   help="Directory containing {train,val,test}.csv for the DMS split")
    p.add_argument("--split-name", choices=("train", "val", "test"), default="test")
    p.add_argument("--enrichment-col", default=ENRICHMENT_COL)
    p.add_argument("--scoring-mode", choices=("mutation_path", "cdr_pll"),
                   default="mutation_path",
                   help="mutation_path uses double-mutant path avg; cdr_pll uses full CDR PLL")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--filename", default="spearman_heatmap_ed2_m22.png")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_heatmap(
    model_labels: list[str],
    rho_matrix: np.ndarray,
    n_matrix: np.ndarray,
    col_labels: list[str],
    out_path: Path,
    scoring_mode: str,
    split_name: str,
) -> None:
    n_models, n_cols = rho_matrix.shape

    col_w = 1.6
    row_h = 0.9
    fig_w = col_w * n_cols + 4.0   # extra room for y-axis labels
    fig_h = row_h * n_models + 2.5  # extra room for x-axis labels + title
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmax = max(0.05, float(np.nanmax(np.abs(rho_matrix))))
    im = ax.imshow(rho_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    for i in range(n_models):
        for j in range(n_cols):
            rho = rho_matrix[i, j]
            if np.isnan(rho):
                cell_text, text_color = "n/a", "grey"
            else:
                cell_text = f"{rho:+.3f}"
                text_color = "white" if abs(rho) > 0.55 * vmax else "black"
            ax.text(j, i, cell_text, ha="center", va="center",
                    fontsize=10, color=text_color, fontweight="bold")

    # n is the same for all models per column — embed it in the column header
    n_per_col = n_matrix[0]  # first row is representative
    tick_labels = [f"{lbl}\n(n={n:,})" for lbl, n in zip(col_labels, n_per_col)]
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(model_labels, fontsize=10)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    ax.axvline(2.5, color="white", linewidth=2.0)
    ax.axvline(3.5, color="white", linewidth=2.0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Spearman ρ", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    scoring_label = "double-mutant path avg" if scoring_mode == "mutation_path" else "CDR PLL"
    ax.set_title(
        f"Spearman ρ vs ED2-M22 binding enrichment  |  {split_name} split  |  scoring: {scoring_label}",
        fontsize=11, pad=14,
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved heatmap → %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_dms(args.dms_dir, args.split_name, args.enrichment_col)
    n_all = len(df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    model_labels: list[str] = []
    rows: list[dict] = []

    for label, checkpoint in args.model:
        scores = _get_scores(
            checkpoint=checkpoint,
            label=label,
            df=df,
            enrichment_col=args.enrichment_col,
            scoring_mode=args.scoring_mode,
            batch_size=args.batch_size,
            device=device,
            dms_dir=args.dms_dir,
            split=args.split_name,
        )
        row = _compute_spearmans(scores, df, args.enrichment_col, n_all)
        model_labels.append(label)
        rows.append(row)

    # Build matrices (models × columns)
    n_models = len(model_labels)
    n_cols = len(COLUMN_KEYS)
    rho_matrix = np.full((n_models, n_cols), np.nan)
    n_matrix = np.zeros((n_models, n_cols), dtype=int)

    for i, row in enumerate(rows):
        for j, (col_key, n_key) in enumerate(zip(COLUMN_KEYS, N_KEYS)):
            rho_matrix[i, j] = row.get(col_key, np.nan)
            n_matrix[i, j] = row["n_all"] if n_key is None else row.get(n_key, 0)

    # Save summary CSV alongside the plot
    summary = [{"model": lbl, **r} for lbl, r in zip(model_labels, rows)]
    pd.DataFrame(summary).to_csv(args.output_dir / "spearman_scores_ed2_m22.csv", index=False)

    _plot_heatmap(
        model_labels=model_labels,
        rho_matrix=rho_matrix,
        n_matrix=n_matrix,
        col_labels=COLUMN_LABELS,
        out_path=args.output_dir / args.filename,
        scoring_mode=args.scoring_mode,
        split_name=args.split_name,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
