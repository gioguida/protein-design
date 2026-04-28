"""Render per-model DMS-only PCA figures.

Reads
-----
- ``per_model_pca_{cdrh3,whole_seq}.npz`` from ``compute_per_model_pca.py``.

Writes (200 DPI, in ``{output_dir}/``)
-------------------------------------
- ``per_model_pc1_pc2_{emb_type}_{fitness}.png`` — one panel per variant,
  PC1 vs PC2 colored by fitness (M22 or SI06).
- ``per_model_pc1_vs_{fitness}_{emb_type}.png`` — one panel per variant,
  PC1 vs fitness regression with Pearson r and Spearman ρ annotations.

Each model variant has its **own** PCA fit on DMS-only embeddings, so PC1
here is the actual dominant variance direction within that model's DMS
representation — not vanilla's PC1 reused. Distances and axes are NOT
comparable across panels by construction.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

EMB_TYPES = ["cdrh3", "whole_seq"]
FITNESS = [
    ("M22_enrich", "M22 binding enrichment", "M22"),
    ("SI06_enrich", "SI06 binding enrichment", "SI06"),
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("plot_per_model_pca")


def load(npz_path: Path) -> Tuple[List[str], Dict[str, Dict[str, np.ndarray]], str]:
    z = np.load(npz_path, allow_pickle=False)
    variants = [str(v) for v in z["model_variants"]]
    per_variant: Dict[str, Dict[str, np.ndarray]] = {}
    for v in variants:
        per_variant[v] = {
            "pca": z[f"{v}__pca"],
            "M22_enrich": z[f"{v}__M22_enrich"],
            "SI06_enrich": z[f"{v}__SI06_enrich"],
            "cdrh3_identity": z[f"{v}__cdrh3_identity"],
            "sequences": z[f"{v}__sequences"],
            "explained_variance": z[f"{v}__pca_explained_variance"],
        }
    dms_dataset = str(z["dms_dataset"][0]) if "dms_dataset" in z.files else "ed2"
    return variants, per_variant, dms_dataset


def _scatter_continuous(ax, coords: np.ndarray, values: np.ndarray):
    """Scatter colored by a continuous value; greys out NaN-valued points."""
    valid_xy = ~np.isnan(coords).any(axis=1)
    valid_v = ~np.isnan(values)
    grey_mask = valid_xy & ~valid_v
    color_mask = valid_xy & valid_v
    if grey_mask.any():
        ax.scatter(coords[grey_mask, 0], coords[grey_mask, 1],
                   s=8, c="lightgrey", alpha=0.35, zorder=2)
    if color_mask.any():
        return ax.scatter(coords[color_mask, 0], coords[color_mask, 1],
                          c=values[color_mask], cmap="viridis",
                          s=14, alpha=0.85, zorder=3)
    return None


def make_pc1_pc2_grid(per_variant, variants, fitness_key, fitness_label, dms_dataset, out_path):
    n_cols = len(variants)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.6 * n_cols, 4.4), squeeze=False)
    for col, v in enumerate(variants):
        d = per_variant[v]
        ax = axes[0, col]
        sc = _scatter_continuous(ax, d["pca"][:, :2], d[fitness_key])
        if sc is not None:
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=fitness_label)
        ev = d["explained_variance"]
        ax.set_title(v, fontsize=11)
        ax.set_xlabel(f"PC1 ({100 * ev[0]:.1f}% var)")
        ax.set_ylabel(f"PC2 ({100 * ev[1]:.1f}% var)" if len(ev) > 1 else "PC2")
    fig.suptitle(
        f"Per-model DMS-only PCA [{dms_dataset.upper()}] — colored by {fitness_label}\n"
        f"(each panel uses its own PCA; axes NOT comparable across panels)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def make_pc1_vs_fitness_grid(per_variant, variants, fitness_key, fitness_label, dms_dataset, out_path):
    n_cols = len(variants)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.6 * n_cols, 4.4), squeeze=False)
    for col, v in enumerate(variants):
        d = per_variant[v]
        x = d["pca"][:, 0]
        y = d[fitness_key]
        m = ~(np.isnan(x) | np.isnan(y))
        x, y = x[m], y[m]

        ax = axes[0, col]
        ax.scatter(x, y, s=14, alpha=0.55, c="tab:blue")
        if len(x) > 1:
            slope, intercept = np.polyfit(x, y, 1)
            xr = np.linspace(x.min(), x.max(), 100)
            ax.plot(xr, slope * xr + intercept, "r-", linewidth=1.5)
            r_p, p_p = pearsonr(x, y)
            r_s, p_s = spearmanr(x, y)
            ax.text(
                0.02, 0.97,
                f"Pearson r = {r_p:.3f} (p = {p_p:.1e})\n"
                f"Spearman ρ = {r_s:.3f} (p = {p_s:.1e})\n"
                f"n = {len(x)}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="grey"),
            )
        ev = d["explained_variance"]
        ax.set_title(v, fontsize=11)
        ax.set_xlabel(f"PC1 ({100 * ev[0]:.1f}% var)")
        ax.set_ylabel(fitness_label)

    fig.suptitle(
        f"Per-model PC1 vs {fitness_label} [{dms_dataset.upper()}] — DMS only, per-model PCA",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--projections-dir",
        type=Path,
        required=True,
        help="Directory containing per_model_pca_{emb_type}.npz",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for emb_type in EMB_TYPES:
        npz_path = args.projections_dir / f"per_model_pca_{emb_type}.npz"
        if not npz_path.exists():
            log.warning("Skipping %s — %s not found", emb_type, npz_path)
            continue
        variants, per_variant, dms_dataset = load(npz_path)
        for fkey, flabel, fshort in FITNESS:
            all_nan = all(
                np.isnan(per_variant[v][fkey]).all() for v in variants
            )
            if all_nan:
                log.warning("Skipping %s/%s — no %s values for any variant in dataset %s",
                            emb_type, fshort, fkey, dms_dataset)
                continue
            make_pc1_pc2_grid(
                per_variant, variants, fkey, flabel, dms_dataset,
                args.output_dir / f"per_model_pc1_pc2_{emb_type}_{fshort}_{dms_dataset}.png",
            )
            make_pc1_vs_fitness_grid(
                per_variant, variants, fkey, flabel, dms_dataset,
                args.output_dir / f"per_model_pc1_vs_{fshort}_{emb_type}_{dms_dataset}.png",
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
