"""Gibbs sampling diagnostics across model variants and named configs.

Each ``--gibbs`` entry tags a Gibbs CSV with a (variant, config) pair. When
multiple configs are passed for a variant, comparison plots use a
config × variant grid; per-config plots split into separate files.

Inputs
------
``--gibbs LABEL=CHECKPOINT=GIBBS_CSV[=CONFIG]`` repeated once per (variant,
config). CONFIG defaults to ``default``. CHECKPOINT matches the forms
accepted by ``compute_pll_pca.py``.

Writes (in ``--output-dir``)
----------------------------
- ``gibbs_pll_dist_{M22,SI06}.png`` — DMS+Gibbs violins, rows = config,
  cols = variant; DMS strip coloured by enrichment, WT PLL marked.
- ``gibbs_pll_trajectory.png`` — sequence PLL across Gibbs steps; rows =
  config, cols = variant; one line per chain, WT/DMS bands.
- ``gibbs_pairwise_hamming.png`` — pairwise CDR-H3 Hamming histograms;
  rows = config, cols = variant.
- ``gibbs_edit_distance.png`` — n_mutations from WT; rows = config, cols
  = variant.
- ``gibbs_sequence_logo[_{config}].png`` — per-position AA frequency stacks
  per variant. One file per config when ≥ 2 configs.
- ``gibbs_position_mutation_freq[_{config}].png`` — per-position mutation
  rate heatmap. One file per config when ≥ 2 configs.
- ``gibbs_summary.csv`` — per (variant, config, slice) row of summary stats.

Each plot above is also emitted with an ``_early`` suffix, restricted to
Gibbs samples within ``--early-max-ed`` mutations of the C05 WT (default 10)
and capped at ``--early-max-chains`` chains (default 10). Use ``--skip-early``
to disable. The early slice is a separate row in ``gibbs_summary.csv``
(``slice=early`` vs ``slice=full``).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForMaskedLM

from protein_design.constants import (
    C05_CDRH3,
    C05_CDRH3_END,
    C05_CDRH3_START,
    add_context,
)

ESM2_MODEL_ID = "facebook/esm2_t12_35M_UR50D"
SEED = 42
_DMS_BASE = "/cluster/project/infk/krause/mdenegri/protein-design/datasets/scoring"
DMS_DATASETS: Dict[str, Dict[str, str]] = {
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
log = logging.getLogger("gibbs_diagnostics")


# --------------------------------------------------------------------- model load
# Same prefix-stripping logic as compute_pll_pca.py — kept inline here so the
# script is standalone.


def _extract_state_dict(raw) -> dict:
    """Handle evotuning ('model_state_dict' wrapper) and DPO ('policy_state_dict'
    wrapper) checkpoint shapes alongside bare state dicts."""
    if isinstance(raw, dict):
        for key in ("policy_state_dict", "model_state_dict"):
            if key in raw and isinstance(raw[key], dict):
                return raw[key]
    return raw


def _load_pt_into_mlm(pt_path: Path) -> EsmForMaskedLM:
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
        raise RuntimeError(f"Checkpoint missing required keys: {non_optional_missing[:5]}")
    if unexpected:
        log.warning("Ignored %d unexpected keys (e.g. %s)", len(unexpected), unexpected[:3])
    return model


def load_esm_for_mlm(checkpoint: str) -> EsmForMaskedLM:
    if not checkpoint:
        return EsmForMaskedLM.from_pretrained(ESM2_MODEL_ID)
    p = Path(checkpoint)
    if p.is_file() and p.suffix == ".pt":
        return _load_pt_into_mlm(p)
    if p.is_dir():
        if (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists():
            return EsmForMaskedLM.from_pretrained(str(p))
        pt_path = next((p / n for n in ("best.pt", "final.pt") if (p / n).exists()), None)
        if pt_path is not None:
            return _load_pt_into_mlm(pt_path)
        raise FileNotFoundError(f"No HF weights, best.pt, or final.pt found at {checkpoint}")
    return EsmForMaskedLM.from_pretrained(checkpoint)


# --------------------------------------------------------------------- inference


@torch.no_grad()
def per_position_cdr_log_probs(
    model: EsmForMaskedLM,
    tokenizer,
    cdrh3_strings: List[str],
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Return (N, 24) log P(true aa | masked context) over each CDR-H3 position."""
    cdr_token_positions = list(range(C05_CDRH3_START + 1, C05_CDRH3_END + 1))
    P = len(cdr_token_positions)
    n = len(cdrh3_strings)
    out = np.zeros((n, P), dtype=np.float32)
    mask_id = tokenizer.mask_token_id

    full_vhs = [add_context(s) for s in cdrh3_strings]
    n_batches = (n + batch_size - 1) // batch_size
    for start in tqdm(range(0, n, batch_size), total=n_batches, desc="PLL"):
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
            out[start:start + B, p_idx] = picks.cpu().numpy().astype(np.float32)
    return out


def sequence_pll(per_pos_log_probs: np.ndarray) -> np.ndarray:
    """Sum per-position log-probs to a sequence-level CDR PLL."""
    return per_pos_log_probs.sum(axis=1)


# ----------------------------------------------------------------- I/O helpers


def load_dms(
    m22_path: "Path | None", si06_path: "Path | None", max_n: int
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Return aligned (cdrh3 list, M22 enrichment array, SI06 enrichment array).

    Either path may be None (assay missing for this dataset). The corresponding
    enrichment array is then returned all-NaN with the same length as the
    surviving CDR-H3 list. If both are None, all three returns are empty.
    """
    frames: List[pd.DataFrame] = []
    if m22_path is not None:
        frames.append(pd.read_csv(m22_path)[["aa", "M22_binding_enrichment_adj"]])
    if si06_path is not None:
        frames.append(pd.read_csv(si06_path)[["aa", "SI06_binding_enrichment_adj"]])
    if not frames:
        empty = np.zeros(0, dtype=np.float32)
        return [], empty, empty
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on="aa", how="outer")
    for col in ("M22_binding_enrichment_adj", "SI06_binding_enrichment_adj"):
        if col not in merged.columns:
            merged[col] = np.nan
    if len(merged) > max_n:
        merged = merged.sample(n=max_n, random_state=SEED).reset_index(drop=True)
    return (
        merged["aa"].astype(str).tolist(),
        merged["M22_binding_enrichment_adj"].to_numpy(dtype=np.float32),
        merged["SI06_binding_enrichment_adj"].to_numpy(dtype=np.float32),
    )


def load_gibbs_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"chain_id", "gibbs_step", "cdrh3", "n_mutations"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Gibbs CSV missing columns {missing}: {path}")
    df = df[df["cdrh3"].astype(str).str.len() == len(C05_CDRH3)].copy()
    return df


# ------------------------------------------------------------------------ plots


FITNESS_ROWS = [
    ("dms_m22", "M22 binding enrichment"),
    ("dms_si06", "SI06 binding enrichment"),
]

# Standard chemistry-style AA palette for sequence logos.
AA_COLORS: Dict[str, str] = {
    "A": "#33a02c", "V": "#33a02c", "L": "#33a02c", "I": "#33a02c", "M": "#33a02c",
    "F": "#6a3d9a", "W": "#6a3d9a", "Y": "#6a3d9a",
    "K": "#1f78b4", "R": "#1f78b4", "H": "#1f78b4",
    "D": "#e31a1c", "E": "#e31a1c",
    "S": "#ff7f00", "T": "#ff7f00", "N": "#ff7f00", "Q": "#ff7f00",
    "G": "#b15928",
    "P": "#fb9a99",
    "C": "#ffff99",
}


def plot_pll_violin_grid(
    per_variant: Dict[Tuple[str, str], Dict[str, np.ndarray]],
    labels: List[str],
    configs: List[str],
    fkey: str,
    flabel: str,
    out_path: Path,
) -> None:
    """Rows = configs, cols = variants. Each cell: DMS+Gibbs violins, DMS
    strip colored by enrichment, WT line. One figure per readout.
    """
    n_cols = len(labels)
    n_rows = len(configs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols + 1.0, 4.0 * n_rows),
                             squeeze=False)

    rng = np.random.default_rng(SEED)

    for row, cfg in enumerate(configs):
        for col, v in enumerate(labels):
            ax = axes[row, col]
            key = (v, cfg)
            if key not in per_variant:
                ax.axis("off")
                continue
            d = per_variant[key]
            dms_pll = d["dms_pll"]
            gibbs_pll = d["gibbs_pll"]
            wt_pll = float(d["wt_pll"])
            fitness = d[fkey]

            datasets = [dms_pll]
            positions = [1.0]
            colors = ["tab:blue"]
            if len(gibbs_pll):
                datasets.append(gibbs_pll)
                positions.append(2.0)
                colors.append("tab:green")
            parts = ax.violinplot(datasets, positions=positions,
                                  widths=0.8, showmeans=False, showmedians=True,
                                  showextrema=False)
            for pc, color in zip(parts["bodies"], colors):
                pc.set_facecolor(color)
                pc.set_edgecolor("black")
                pc.set_alpha(0.45)

            valid = ~np.isnan(fitness)
            jitter = rng.uniform(-0.18, 0.18, size=int(valid.sum()))
            sc = ax.scatter(
                np.full(int(valid.sum()), 1.0) + jitter,
                dms_pll[valid],
                c=fitness[valid], cmap="viridis", s=12, alpha=0.85,
                edgecolors="black", linewidths=0.2, zorder=4,
            )
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=flabel)

            ax.axhline(wt_pll, color="red", linestyle="--", linewidth=1.2, zorder=5)
            ax.text(0.02, wt_pll, "WT", color="red", fontsize=9, fontweight="bold",
                    transform=ax.get_yaxis_transform(), ha="left", va="bottom")

            ax.set_xticks([1.0, 2.0])
            ax.set_xticklabels(["DMS", "Gibbs"], fontsize=9)
            if row == 0:
                ax.set_title(v, fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{cfg}\n\nCDR-H3 sequence PLL", fontsize=10)
            else:
                ax.set_ylabel("CDR-H3 sequence PLL", fontsize=9)

    fig.suptitle(
        f"{flabel} — DMS vs Gibbs PLL  (rows = config, cols = variant)\n"
        "DMS strip colored by enrichment; WT PLL marked",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def plot_pll_trajectory(
    per_variant: Dict[Tuple[str, str], Dict[str, np.ndarray]],
    labels: List[str],
    configs: List[str],
    out_path: Path,
) -> None:
    """Rows = configs, cols = variants. One line per chain. WT PLL dashed;
    DMS [5, 95] percentile band shaded grey.
    """
    n_cols = len(labels)
    n_rows = len(configs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.4 * n_rows),
                             squeeze=False)

    chain_cmap = plt.get_cmap("tab10")
    legend_seeded = False

    for row, cfg in enumerate(configs):
        for col, v in enumerate(labels):
            ax = axes[row, col]
            key = (v, cfg)
            if key not in per_variant:
                ax.axis("off")
                continue
            d = per_variant[key]
            gibbs_pll = d["gibbs_pll"]
            steps = d["gibbs_step"]
            chains = d["chain_id"]
            wt_pll = float(d["wt_pll"])
            dms_pll = d["dms_pll"]

            lo, hi = float(np.percentile(dms_pll, 5)), float(np.percentile(dms_pll, 95))
            ax.axhspan(lo, hi, color="lightgrey", alpha=0.4, zorder=1,
                       label="DMS [5, 95]% PLL" if not legend_seeded else None)

            unique_chains = sorted(set(int(c_) for c_ in chains))
            for i, ch in enumerate(unique_chains):
                mask = chains == ch
                order = np.argsort(steps[mask])
                x = steps[mask][order]
                y = gibbs_pll[mask][order]
                ax.plot(x, y, "-", color=chain_cmap(i % 10), linewidth=1.0,
                        alpha=0.85, zorder=3,
                        label=f"chain {ch}" if not legend_seeded else None)

            ax.axhline(wt_pll, color="red", linestyle="--", linewidth=1.2, zorder=4,
                       label="WT PLL" if not legend_seeded else None)
            legend_seeded = True

            if row == 0:
                ax.set_title(v, fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{cfg}\nCDR-H3 sequence PLL", fontsize=9)
            ax.set_xlabel("Gibbs step")

    handles, leg_labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, leg_labels, loc="upper right", fontsize=8, ncol=1,
                   bbox_to_anchor=(0.998, 0.998))
    fig.suptitle("Gibbs PLL trajectory — rows = config, cols = variant", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def plot_sequence_logo(
    per_variant: Dict[str, Dict[str, np.ndarray]],
    out_path: Path,
    min_freq: float = 0.02,
) -> None:
    """One row per variant; each column = a CDR position with stacked AA
    rectangles sized by frequency. Tall single letters → collapse onto one
    consensus residue.
    """
    variants = list(per_variant.keys())
    P = len(C05_CDRH3)
    n_vars = len(variants)
    fig, axes = plt.subplots(n_vars, 1, figsize=(0.45 * P + 4.0, 1.7 * n_vars + 1.4),
                             squeeze=False, sharex=True)

    for row, v in enumerate(variants):
        ax = axes[row, 0]
        chars: np.ndarray = per_variant[v]["cdrh3_chars"]  # (N, P)
        for p_idx in range(P):
            counts = Counter(chars[:, p_idx].tolist())
            total = sum(counts.values())
            if total == 0:
                continue
            sorted_aas = sorted(counts.items(), key=lambda kv: -kv[1])
            y = 0.0
            for aa, cnt in sorted_aas:
                freq = cnt / total
                if freq < min_freq:
                    break
                color = AA_COLORS.get(aa, "lightgrey")
                ax.add_patch(
                    plt.Rectangle((p_idx - 0.42, y), 0.84, freq,
                                  facecolor=color, edgecolor="white",
                                  linewidth=0.4, alpha=0.85)
                )
                if freq > 0.06:
                    ax.text(p_idx, y + freq / 2, aa, ha="center", va="center",
                            fontsize=8, fontweight="bold", color="black")
                y += freq

        ax.set_xlim(-0.6, P - 0.4)
        ax.set_ylim(0.0, 1.02)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_ylabel(v, rotation=0, ha="right", va="center", fontsize=10,
                      fontweight="bold")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    axes[-1, 0].set_xticks(range(P))
    axes[-1, 0].set_xticklabels([f"{i + 1}\n{a}" for i, a in enumerate(C05_CDRH3)],
                                fontsize=8)
    axes[-1, 0].set_xlabel("CDR-H3 position (1-indexed; WT residue beneath)")
    fig.suptitle(
        "Sequence logo — per-position AA frequency in Gibbs samples\n"
        "(tall single letters indicate collapse onto one consensus residue)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def plot_pairwise_hamming(
    per_variant: Dict[Tuple[str, str], Dict[str, np.ndarray]],
    labels: List[str],
    configs: List[str],
    out_path: Path,
    max_n: int = 1000,
) -> None:
    """Rows = configs, cols = variants. Histogram of pairwise CDR-H3 Hamming
    distances. Cap N to ``max_n`` per cell to keep O(N²) tractable.
    """
    n_cols = len(labels)
    n_rows = len(configs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.2 * n_rows),
                             squeeze=False, sharex=True, sharey=True)

    P = len(C05_CDRH3)
    rng = np.random.default_rng(SEED)

    for row, cfg in enumerate(configs):
        for col, v in enumerate(labels):
            ax = axes[row, col]
            key = (v, cfg)
            if key not in per_variant:
                ax.axis("off")
                continue
            chars: np.ndarray = per_variant[key]["cdrh3_chars"]
            n = chars.shape[0]
            if n < 2:
                ax.text(0.5, 0.5, "n<2", transform=ax.transAxes, ha="center", va="center")
                if row == 0:
                    ax.set_title(v, fontsize=10)
                continue
            if n > max_n:
                keep = rng.choice(n, size=max_n, replace=False)
                chars = chars[keep]
                n = max_n

            eq = (chars[:, None, :] == chars[None, :, :])
            dist = P - eq.sum(axis=2)
            iu = np.triu_indices(n, k=1)
            d_flat = dist[iu]

            ax.hist(d_flat, bins=range(0, P + 2), align="left",
                    color="tab:purple", edgecolor="white", linewidth=0.4)
            if row == 0:
                ax.set_title(f"{v}", fontsize=10)
            if row == n_rows - 1:
                ax.set_xlabel("pairwise Hamming")
            if col == 0:
                ax.set_ylabel(f"{cfg}\npair count", fontsize=9)
            ax.set_xlim(-0.5, P + 0.5)

    fig.suptitle("Pairwise Hamming among Gibbs CDR-H3s — collapse → mass at 0",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def plot_edit_distance(
    per_variant: Dict[Tuple[str, str], Dict[str, np.ndarray]],
    labels: List[str],
    configs: List[str],
    out_path: Path,
) -> None:
    """Rows = configs, cols = variants. Histogram of n_mutations vs C05 WT."""
    n_cols = len(labels)
    n_rows = len(configs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.2 * n_rows),
                             squeeze=False)
    for row, cfg in enumerate(configs):
        for col, v in enumerate(labels):
            ax = axes[row, col]
            key = (v, cfg)
            if key not in per_variant:
                ax.axis("off")
                continue
            nm = per_variant[key]["n_mutations"]
            if not len(nm):
                ax.axis("off")
                continue
            ax.hist(nm, bins=range(0, int(nm.max()) + 2), color="tab:green",
                    edgecolor="white", linewidth=0.4, align="left")
            if row == 0:
                ax.set_title(v, fontsize=11)
            if row == n_rows - 1:
                ax.set_xlabel("edit distance from C05 WT")
            if col == 0:
                ax.set_ylabel(f"{cfg}\ncount", fontsize=9)
            ax.set_xlim(left=-0.5)
    fig.suptitle("Gibbs samples — edit distance from C05 WT CDR-H3", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def plot_position_mutation_freq(
    per_variant: Dict[str, Dict[str, np.ndarray]], out_path: Path
) -> None:
    variants = list(per_variant.keys())
    P = len(C05_CDRH3)
    matrix = np.zeros((len(variants), P), dtype=np.float32)
    for i, v in enumerate(variants):
        matrix[i] = per_variant[v]["pos_mut_freq"]

    fig, ax = plt.subplots(figsize=(0.45 * P + 3.5, 0.9 * len(variants) + 1.8))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="magma", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(non-WT residue)")
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(variants)
    ax.set_xticks(range(P))
    ax.set_xticklabels([f"{i + 1}\n{a}" for i, a in enumerate(C05_CDRH3)], fontsize=8)
    ax.set_xlabel("CDR-H3 position (1-indexed; WT residue beneath)")
    ax.set_title("Per-position mutation frequency in Gibbs samples", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


# --------------------------------------------------------------------- subsetting


def build_early_subset(
    per_variant: Dict[Tuple[str, str], Dict[str, np.ndarray]],
    max_ed: int,
    max_chains: int,
) -> Tuple[Dict[Tuple[str, str], Dict[str, np.ndarray]], List[dict]]:
    """Filter every per (variant, config) cell to rows where n_mutations ≤ max_ed,
    then keep the first ``max_chains`` chain IDs. Empty cells are dropped.

    PLL inference is reused from the full pass; only row-level arrays
    (``gibbs_pll``, ``n_mutations``, ``chain_id``, ``gibbs_step``,
    ``cdrh3_chars``) are sliced. Aggregates (``pos_mut_freq``) are recomputed
    on the subset. ``dms_pll`` / ``dms_*`` / ``wt_pll`` are unchanged.

    Returns the filtered per_variant dict and a list of summary rows for the
    early slice (parallel structure to the full-pass summary rows).
    """
    wt_arr = np.array(list(C05_CDRH3))
    early: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    rows: List[dict] = []

    for key, d in per_variant.items():
        label, config = key
        nm = d["n_mutations"]
        if not len(nm):
            continue
        ed_mask = nm <= max_ed
        if not ed_mask.any():
            log.warning("No Gibbs rows with edit_distance ≤ %d for %s/%s — dropping early cell",
                        max_ed, label, config)
            continue
        chains_e = d["chain_id"][ed_mask]
        keep_chains = sorted(set(int(c) for c in chains_e))[:max_chains]
        keep_set = set(keep_chains)
        full_mask = ed_mask & np.array([int(c) in keep_set for c in d["chain_id"]])
        if not full_mask.any():
            continue

        gibbs_pll_e = d["gibbs_pll"][full_mask]
        cdrh3_chars_e = d["cdrh3_chars"][full_mask]
        n_mutations_e = d["n_mutations"][full_mask]
        pos_mut_freq_e = (cdrh3_chars_e != wt_arr[None, :]).mean(axis=0).astype(np.float32)

        early[key] = {
            "gibbs_pll": gibbs_pll_e,
            "dms_pll": d["dms_pll"],
            "wt_pll": d["wt_pll"],
            "n_mutations": n_mutations_e,
            "pos_mut_freq": pos_mut_freq_e,
            "chain_id": d["chain_id"][full_mask],
            "gibbs_step": d["gibbs_step"][full_mask],
            "cdrh3_chars": cdrh3_chars_e,
            "dms_m22": d["dms_m22"],
            "dms_si06": d["dms_si06"],
        }

        top_alts = []
        for p_idx in range(len(C05_CDRH3)):
            alt_counts = Counter(c for c in cdrh3_chars_e[:, p_idx] if c != C05_CDRH3[p_idx])
            top = alt_counts.most_common(1)
            top_alts.append(f"{top[0][0]}({top[0][1]})" if top else "-")

        rows.append({
            "variant": label,
            "config": config,
            "slice": "early",
            "n_gibbs_samples": int(len(gibbs_pll_e)),
            "edit_dist_median": float(np.median(n_mutations_e)),
            "edit_dist_max": int(n_mutations_e.max()),
            "gibbs_pll_median": float(np.median(gibbs_pll_e)),
            "dms_pll_median": float(np.median(d["dms_pll"])),
            "wt_pll": float(d["wt_pll"]),
            "delta_pll_gibbs_minus_dms": float(np.median(gibbs_pll_e) - np.median(d["dms_pll"])),
            "delta_pll_gibbs_minus_wt": float(np.median(gibbs_pll_e) - float(d["wt_pll"])),
            "max_pos_mut_freq": float(pos_mut_freq_e.max()),
            "top_alt_per_position": " ".join(top_alts),
        })

    return early, rows


# ------------------------------------------------------------------------ main


def parse_variant_spec(spec: str) -> Tuple[str, str, str, str]:
    """LABEL=CHECKPOINT=GIBBS_CSV[=CONFIG]; CONFIG defaults to 'default'."""
    parts = spec.split("=")
    if len(parts) == 3:
        label, checkpoint, csv = parts
        config = "default"
    elif len(parts) == 4:
        label, checkpoint, csv, config = parts
    else:
        raise argparse.ArgumentTypeError(
            f"--gibbs must be LABEL=CHECKPOINT=GIBBS_CSV[=CONFIG], got: {spec!r}"
        )
    return label.strip(), checkpoint.strip(), csv.strip(), config.strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--gibbs",
        action="append",
        required=True,
        type=parse_variant_spec,
        help="LABEL=CHECKPOINT=GIBBS_CSV[=CONFIG]; repeat once per variant×config. "
             "CONFIG defaults to 'default'.",
    )
    p.add_argument("--dms-dataset", default=DEFAULT_DMS_DATASET, choices=sorted(DMS_DATASETS),
                   help="Named DMS dataset whose CSVs should be loaded "
                        "(resolves --dms-m22/--dms-si06 unless those are passed explicitly).")
    p.add_argument("--dms-m22", default=None,
                   help="Override M22 CSV path (defaults to the --dms-dataset entry).")
    p.add_argument("--dms-si06", default=None,
                   help="Override SI06 CSV path (defaults to the --dms-dataset entry).")
    p.add_argument("--max-dms", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--early-max-ed", type=int, default=10,
                   help="Edit-distance-to-WT cap for the early-trajectory plots and summary slice.")
    p.add_argument("--early-max-chains", type=int, default=10,
                   help="Cap on number of chains shown in the early-trajectory plots.")
    p.add_argument("--skip-early", action="store_true",
                   help="Skip the early-trajectory diagnostic plots and summary slice.")
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    dataset_paths = DMS_DATASETS[args.dms_dataset]
    dms_m22_path = args.dms_m22 if args.dms_m22 is not None else dataset_paths.get("m22")
    dms_si06_path = args.dms_si06 if args.dms_si06 is not None else dataset_paths.get("si06")
    if dms_m22_path is None:
        log.warning("DMS dataset %s has no M22 assay — M22 plots will be skipped", args.dms_dataset)
    if dms_si06_path is None:
        log.warning("DMS dataset %s has no SI06 assay — SI06 plots will be skipped", args.dms_dataset)
    log.info("DMS dataset: %s (M22=%s, SI06=%s)", args.dms_dataset, dms_m22_path, dms_si06_path)
    log.info("Loading DMS reference (max %d) …", args.max_dms)
    dms_cdrh3, dms_m22, dms_si06 = load_dms(
        Path(dms_m22_path) if dms_m22_path else None,
        Path(dms_si06_path) if dms_si06_path else None,
        args.max_dms,
    )
    log.info("DMS: %d sequences", len(dms_cdrh3))

    tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_ID)

    # Group entries by label so each model is loaded once; DMS + WT PLL are
    # computed per label and shared across that label's configs.
    by_label: Dict[str, List[Tuple[str, str, str]]] = {}
    label_order: List[str] = []
    for label, checkpoint, gibbs_csv, config in args.gibbs:
        if label not in by_label:
            by_label[label] = []
            label_order.append(label)
        by_label[label].append((checkpoint, gibbs_csv, config))

    per_variant: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    summary_rows: List[dict] = []
    wt_arr = np.array(list(C05_CDRH3))

    for label in label_order:
        entries = by_label[label]
        checkpoint = entries[0][0]
        if any(ck != checkpoint for ck, _, _ in entries):
            log.warning("Multiple checkpoints for label=%s; using %s", label, checkpoint)
        log.info("=== %s (%d configs) ===", label, len(entries))

        model = load_esm_for_mlm(checkpoint).eval().to(device)
        for p_ in model.parameters():
            p_.requires_grad = False
        if device.type == "cuda":
            model = model.half()

        dms_per_pos = per_position_cdr_log_probs(
            model, tokenizer, dms_cdrh3, device, args.batch_size,
        )
        dms_pll = sequence_pll(dms_per_pos)
        wt_per_pos = per_position_cdr_log_probs(
            model, tokenizer, [C05_CDRH3], device, args.batch_size,
        )
        wt_pll = float(sequence_pll(wt_per_pos)[0])

        for _, gibbs_csv, config in entries:
            log.info("--- %s | config=%s ---", label, config)
            gdf = load_gibbs_csv(Path(gibbs_csv))
            gibbs_cdrh3 = gdf["cdrh3"].astype(str).tolist()
            n_mutations = gdf["n_mutations"].to_numpy(dtype=np.int32)
            chain_id = gdf["chain_id"].to_numpy(dtype=np.int32)
            gibbs_step = gdf["gibbs_step"].to_numpy(dtype=np.int32)
            log.info("Gibbs samples: %d (median edit distance from WT = %d)",
                     len(gibbs_cdrh3), int(np.median(n_mutations)) if len(n_mutations) else 0)

            gibbs_per_pos = per_position_cdr_log_probs(
                model, tokenizer, gibbs_cdrh3, device, args.batch_size,
            )
            gibbs_pll = sequence_pll(gibbs_per_pos)

            cdrh3_chars = np.array([list(s) for s in gibbs_cdrh3])  # (N, 24)
            pos_mut_freq = (cdrh3_chars != wt_arr[None, :]).mean(axis=0).astype(np.float32)

            per_variant[(label, config)] = {
                "gibbs_pll": gibbs_pll,
                "dms_pll": dms_pll,
                "wt_pll": np.float32(wt_pll),
                "n_mutations": n_mutations,
                "pos_mut_freq": pos_mut_freq,
                "chain_id": chain_id,
                "gibbs_step": gibbs_step,
                "cdrh3_chars": cdrh3_chars,
                "dms_m22": dms_m22,
                "dms_si06": dms_si06,
            }

            top_alts = []
            for p_idx in range(len(C05_CDRH3)):
                alt_counts = Counter(c for c in cdrh3_chars[:, p_idx] if c != C05_CDRH3[p_idx])
                top = alt_counts.most_common(1)
                top_alts.append(f"{top[0][0]}({top[0][1]})" if top else "-")

            summary_rows.append({
                "variant": label,
                "config": config,
                "slice": "full",
                "n_gibbs_samples": int(len(gibbs_cdrh3)),
                "edit_dist_median": float(np.median(n_mutations)) if len(n_mutations) else 0.0,
                "edit_dist_max": int(n_mutations.max()) if len(n_mutations) else 0,
                "gibbs_pll_median": float(np.median(gibbs_pll)) if len(gibbs_pll) else float("nan"),
                "dms_pll_median": float(np.median(dms_pll)),
                "wt_pll": float(wt_pll),
                "delta_pll_gibbs_minus_dms": (
                    float(np.median(gibbs_pll) - np.median(dms_pll)) if len(gibbs_pll) else float("nan")
                ),
                "delta_pll_gibbs_minus_wt": (
                    float(np.median(gibbs_pll) - wt_pll) if len(gibbs_pll) else float("nan")
                ),
                "max_pos_mut_freq": float(pos_mut_freq.max()) if len(pos_mut_freq) else 0.0,
                "top_alt_per_position": " ".join(top_alts),
            })

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    configs_present = sorted({cfg for _, cfg in per_variant.keys()})
    log.info("Configs present: %s", configs_present)

    def _emit_plots(pv, slice_suffix: str) -> None:
        configs_local = sorted({cfg for _, cfg in pv.keys()})
        labels_local = [lbl for lbl in label_order
                        if any((lbl, cfg) in pv for cfg in configs_local)]
        if not configs_local or not labels_local:
            log.warning("No data for slice=%s — skipping plots", slice_suffix or "full")
            return

        for fkey, flabel in FITNESS_ROWS:
            readout = "M22" if fkey == "dms_m22" else "SI06"
            all_nan = all(
                np.isnan(pv[k][fkey]).all() for k in pv
                if fkey in pv[k] and pv[k][fkey].size
            )
            if all_nan:
                log.info("Skipping %s violin (slice=%s) — no %s values in dataset",
                         readout, slice_suffix or "full", fkey)
                continue
            plot_pll_violin_grid(
                pv, labels_local, configs_local, fkey, flabel,
                args.output_dir / f"gibbs_pll_dist_{readout}{slice_suffix}.png",
            )
        plot_pll_trajectory(pv, labels_local, configs_local,
                            args.output_dir / f"gibbs_pll_trajectory{slice_suffix}.png")
        plot_pairwise_hamming(pv, labels_local, configs_local,
                              args.output_dir / f"gibbs_pairwise_hamming{slice_suffix}.png")
        plot_edit_distance(pv, labels_local, configs_local,
                           args.output_dir / f"gibbs_edit_distance{slice_suffix}.png")

        for cfg in configs_local:
            cfg_subset = {label: pv[(label, cfg)]
                          for label in labels_local if (label, cfg) in pv}
            cfg_part = f"_{cfg}" if len(configs_local) > 1 else ""
            plot_sequence_logo(
                cfg_subset,
                args.output_dir / f"gibbs_sequence_logo{slice_suffix}{cfg_part}.png",
            )
            plot_position_mutation_freq(
                cfg_subset,
                args.output_dir / f"gibbs_position_mutation_freq{slice_suffix}{cfg_part}.png",
            )

    _emit_plots(per_variant, "")

    if not args.skip_early:
        early_pv, early_rows = build_early_subset(
            per_variant, args.early_max_ed, args.early_max_chains,
        )
        if early_pv:
            log.info("Early slice: %d (variant, config) cells with ED ≤ %d (≤ %d chains)",
                     len(early_pv), args.early_max_ed, args.early_max_chains)
            _emit_plots(early_pv, "_early")
            summary_rows.extend(early_rows)
        else:
            log.warning("Early slice empty across all cells — skipping early plots")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = args.output_dir / "gibbs_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    log.info("Wrote %s\n%s", summary_path,
             summary_df.drop(columns=["top_alt_per_position"]).round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
