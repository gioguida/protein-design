"""Notebook-facing figure functions for the paper.

Each `plot_*` takes a list of model keys (order = plot order; add/remove freely)
and returns a matplotlib Figure, so the notebook can display it inline and
`save_fig()` it to report/figures/<name>.pdf.

Reads only the cached artifacts via `registry`; run `preflight()` first to see
what still needs extracting.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import registry

log = logging.getLogger(__name__)

FIGURES_DIR = registry.REPO_ROOT / "report" / "figures"


# ── helpers ─────────────────────────────────────────────────────────────────

def _label(model_key: str) -> str:
    return registry.load_models().get(model_key, {}).get("label", model_key)


def _color(model_key: str, fallback: str) -> str:
    return registry.load_models().get(model_key, {}).get("color", fallback)


def pll_truth_spearman(model_key: str, dataset_key: str) -> tuple[float, int]:
    """Spearman rho between cached PLL and experimental truth, joined on sequence.

    Returns (rho, n). Raises FileNotFoundError if the PLL artifact is missing.
    """
    ds = registry.load_datasets_cfg()["datasets"][dataset_key]
    seq_col = ds["seq_col"]
    pll = registry.load_pll(model_key, dataset_key)
    truth = registry.load_truth(dataset_key)
    joined = pll.merge(truth, on=seq_col, how="inner")
    x = joined["pll"].to_numpy(np.float64)
    y = joined["enrichment"].to_numpy(np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan"), int(mask.sum())
    rho, _ = spearmanr(x[mask], y[mask])
    return float(rho), int(mask.sum())


def save_fig(fig: plt.Figure, name: str) -> Path:
    """Save `fig` to report/figures/<name>.pdf (vector, embedded fonts)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / (name if name.endswith(".pdf") else f"{name}.pdf")
    plt.rcParams["pdf.fonttype"] = 42  # embed TrueType, not Type 3
    fig.savefig(out, format="pdf", bbox_inches="tight")
    print(f"Saved: {out}")
    return out


# ── preflight ────────────────────────────────────────────────────────────────

def preflight(model_keys: list[str], datasets: str | list[str] = "all",
              kind: str = "pll") -> list[tuple[str, str]]:
    """Report which (model, dataset) artifacts are missing and how to make them.

    Prints copy-pasteable sbatch commands and returns the missing pairs so a
    notebook cell can optionally submit them.
    """
    ds_keys = registry.dataset_keys("all") if datasets == "all" else (
        [datasets] if isinstance(datasets, str) else datasets)
    missing = []
    for m in model_keys:
        for d in ds_keys:
            if not registry.artifact_path(m, kind, f"{d}.csv").exists():
                missing.append((m, d))
    if not missing:
        print(f"All {kind} artifacts present for {len(model_keys)} model(s) "
              f"x {len(ds_keys)} dataset(s).")
        return []
    print(f"Missing {len(missing)} {kind} artifact(s). To extract:\n")
    for m in sorted({m for m, _ in missing}):
        ds = ",".join(d for mm, d in missing if mm == m)
        print(f"  sbatch bash_scripts/extract.sbatch --what {kind} "
              f"--model {m} --dataset {ds}")
    return missing


# ── figures ──────────────────────────────────────────────────────────────────

def plot_pll_spearman(model_keys: list[str], datasets: str | list[str] = "all",
                      ax: plt.Axes | None = None) -> plt.Figure:
    """Grouped bars: Spearman(PLL, truth) per dataset, one bar per model.

    The canonical "which model best ranks binders" comparison. Reorder/extend
    `model_keys` to change the bars shown. Missing artifacts are skipped with a
    warning (run `preflight` to extract them first).
    """
    ds_keys = registry.dataset_keys("all") if datasets == "all" else (
        [datasets] if isinstance(datasets, str) else datasets)

    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])
    rhos: dict[str, list[float]] = {}
    for i, m in enumerate(model_keys):
        row = []
        for d in ds_keys:
            try:
                rho, _ = pll_truth_spearman(m, d)
            except FileNotFoundError:
                log.warning("[%s/%s] PLL artifact missing — bar omitted", m, d)
                rho = np.nan
            row.append(rho)
        rhos[m] = row

    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(
        figsize=(1.6 + 1.1 * len(ds_keys), 4.2), constrained_layout=True)
    n_models = len(model_keys)
    width = 0.8 / max(n_models, 1)
    x = np.arange(len(ds_keys))
    for i, m in enumerate(model_keys):
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, rhos[m], width,
               label=_label(m), color=_color(m, palette[i % len(palette)]))

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(ds_keys, rotation=30, ha="right")
    ax.set_ylabel(r"Spearman $\rho$ (PLL vs truth)")
    ax.set_title("CDR-H3 PLL ranks experimental enrichment")
    ax.legend(fontsize=8, ncol=1, frameon=False)
    return fig


def preflight_scorer(datasets: str | list[str] = "all") -> list[str]:
    """Report which scorer artifacts are missing and how to produce them.

    Scorer predictions are computed by score_dms_with_esme.py (flu conda env)
    and stored at $cache_root/scorer_preds/<dataset>/<scorer>.csv.
    """
    ds_keys = registry.dataset_keys("all") if datasets == "all" else (
        [datasets] if isinstance(datasets, str) else datasets)
    missing = [d for d in ds_keys
               if registry.scorer_artifact_path(d) is not None
               and not registry.scorer_artifact_path(d).exists()]
    if not missing:
        print("All scorer artifacts present.")
        return []
    ds_arg = ",".join(missing)
    print(f"Missing scorer predictions for: {', '.join(missing)}\n")
    print(f"  sbatch bash_scripts/utils/score_dms.sbatch --dataset {ds_arg}")
    return missing


def _scorer_truth_spearman(dataset_key: str) -> tuple[float, int]:
    """Spearman rho between scorer predictions and experimental truth."""
    cfg = registry.load_datasets_cfg()
    seq_col = cfg["datasets"][dataset_key]["seq_col"]
    score_df = registry.load_scorer(dataset_key)
    if score_df is None:
        return float("nan"), 0
    truth = registry.load_truth(dataset_key)
    joined = score_df.merge(truth, on=seq_col, how="inner")
    x = joined["score"].to_numpy(np.float64)
    y = joined["enrichment"].to_numpy(np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan"), int(mask.sum())
    rho, _ = spearmanr(x[mask], y[mask])
    return float(rho), int(mask.sum())


def plot_pll_vs_scorer_spearman(
    model_keys: list[str],
    datasets: str | list[str] = "all",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Grouped bars: Spearman(PLL, truth) per model, with scorer-vs-truth baseline.

    The dashed horizontal segment per dataset shows the fixed binder scorer's
    Spearman with experimental truth — a supervised upper bound that doesn't
    vary per model. Missing PLL artifacts are skipped; missing scorer predictions
    produce a gap in the baseline (run preflight_scorer to see what's needed).
    """
    ds_keys = registry.dataset_keys("all") if datasets == "all" else (
        [datasets] if isinstance(datasets, str) else datasets)

    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])
    rhos: dict[str, list[float]] = {}
    for m in model_keys:
        row = []
        for d in ds_keys:
            try:
                rho, _ = pll_truth_spearman(m, d)
            except FileNotFoundError:
                log.warning("[%s/%s] PLL artifact missing — bar omitted", m, d)
                rho = np.nan
            row.append(rho)
        rhos[m] = row

    scorer_rhos = [_scorer_truth_spearman(d)[0] for d in ds_keys]

    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(
        figsize=(1.6 + 1.1 * len(ds_keys), 4.2), constrained_layout=True)
    n_models = len(model_keys)
    width = 0.8 / max(n_models, 1)
    x = np.arange(len(ds_keys))

    for i, m in enumerate(model_keys):
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, rhos[m], width,
               label=_label(m), color=_color(m, palette[i % len(palette)]))

    half_group = 0.42
    for j, (d, srho) in enumerate(zip(ds_keys, scorer_rhos)):
        if np.isfinite(srho):
            ax.plot([x[j] - half_group, x[j] + half_group], [srho, srho],
                    color="black", linewidth=2, linestyle="--",
                    label="Scorer baseline" if j == 0 else "_nolegend_")

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(ds_keys, rotation=30, ha="right")
    ax.set_ylabel(r"Spearman $\rho$ (vs experimental truth)")
    ax.set_title("PLL vs supervised scorer: ranking experimental enrichment")
    ax.legend(fontsize=8, ncol=1, frameon=False)
    return fig


def plot_pseudo_perplexity(
    model_keys: list[str],
    datasets: str | list[str] = "all",
) -> plt.Figure:
    """Violin plots of per-sequence CDR-H3 pseudo-perplexity for each model.

    Pseudo-perplexity PP = exp(-PLL / L) where L is the CDR-H3 sequence length.
    Lower PP means the model assigns higher average probability to each residue.
    Sequences are pooled across the selected datasets (deduped by sequence string).
    Missing artifacts for a model are skipped with a warning.
    """
    ds_keys = registry.dataset_keys("all") if datasets == "all" else (
        [datasets] if isinstance(datasets, str) else datasets)

    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])

    # Collect perplexity values per model (pool + dedup sequences across datasets)
    labels_ordered: list[str] = []
    ppl_arrays: list[np.ndarray] = []
    colors: list[str] = []

    for i, m in enumerate(model_keys):
        frames = []
        for d in ds_keys:
            try:
                df = registry.load_pll(m, d)
                seq_col = registry.load_datasets_cfg()["datasets"][d]["seq_col"]
                tmp = df[[seq_col, "pll"]].copy()
                tmp["seq_len"] = tmp[seq_col].str.len()
                frames.append(tmp[[seq_col, "pll", "seq_len"]])
            except FileNotFoundError:
                log.warning("[%s/%s] PLL artifact missing — skipped", m, d)
        if not frames:
            continue
        pooled = pd.concat(frames).drop_duplicates(subset=[seq_col], keep="first")
        pll_vals = pooled["pll"].to_numpy(np.float64)
        seq_lens = pooled["seq_len"].to_numpy(np.float64)
        ppl = np.exp(-pll_vals / seq_lens)
        ppl = ppl[np.isfinite(ppl)]
        if ppl.size == 0:
            continue
        labels_ordered.append(_label(m))
        ppl_arrays.append(ppl)
        colors.append(_color(m, palette[i % len(palette)]))

    if not ppl_arrays:
        raise RuntimeError("No perplexity data — check that PLL artifacts exist (run preflight).")

    fig, ax = plt.subplots(figsize=(max(3.5, 1.2 * len(ppl_arrays)), 4.0),
                           constrained_layout=True)
    positions = np.arange(1, len(ppl_arrays) + 1)
    parts = ax.violinplot(ppl_arrays, positions=positions, showmedians=True,
                          showextrema=False)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels_ordered, rotation=30, ha="right")
    ax.set_ylabel("CDR-H3 pseudo-perplexity")
    ax.set_title("Per-sequence pseudo-perplexity across models\n"
                 f"(pooled from: {', '.join(ds_keys)})")
    return fig
