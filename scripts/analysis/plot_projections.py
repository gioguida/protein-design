"""Render PCA/UMAP figures for the embedding-space analysis.

Reads
-----
- {projections_dir}/{whole_seq,cdrh3}.npz  — from compute_projections.py
- {embeddings_dir}/{variant}.npz           — from extract_embeddings.py
  (Figure 6 needs the original 480-D embeddings for cosine-similarity metrics.)

Writes (200 DPI) to {output_dir}/
- pca_grid_{whole_seq,cdrh3}.png             Figure set 1
- umap_grid_{whole_seq,cdrh3}.png            Figure set 2
- pc1_vs_{M22,SI06}_{whole_seq,cdrh3}.png    Figure set 3 (4 files)
- gibbs_trajectory_{variant}.png             Figure set 4 (per variant w/ gibbs)
- pca_vs_umap_{variant}.png                  Figure set 5
- summary_table.{png,csv}                    Figure 6
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr

SOURCE_COLORS = {"wt": "red", "dms": "tab:orange", "oas": "tab:blue", "gibbs": "tab:green"}
COLOR_AXES = [
    ("source", "Source"),
    ("cdrh3_identity", "CDRH3 identity to WT"),
    ("M22_enrich", "M22 binding enrichment"),
    ("SI06_enrich", "SI06 binding enrichment"),
]
WT_STAR = dict(marker="*", s=220, c="red", edgecolors="black", linewidths=0.6, zorder=10)
GIBBS_TRI = dict(marker="^", s=14, alpha=0.45, c="tab:green", zorder=4)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("plot_projections")


# --------------------------------------------------------------------------- I/O


def load_projections(projections_dir: Path, emb_type: str):
    z = np.load(projections_dir / f"{emb_type}.npz", allow_pickle=False)
    variants = [str(v) for v in z["model_variants"]]
    pc1_var = float(z["pca_pc1_variance"])
    pc2_var = float(z["pca_pc2_variance"])
    per_variant: Dict[str, Dict[str, np.ndarray]] = {}
    for v in variants:
        per_variant[v] = {k: z[f"{v}__{k}"] for k in (
            "pca2d", "umap2d", "source_labels", "M22_enrich", "SI06_enrich",
            "cdrh3_identity", "gibbs_step", "chain_id", "sequences",
        )}
    return variants, per_variant, pc1_var, pc2_var


# ------------------------------------------------------------------------ panels


def _scatter_continuous(ax, coords, values, **kw):
    valid = ~np.isnan(values)
    if (~valid).any():
        ax.scatter(coords[~valid, 0], coords[~valid, 1], s=8, c="lightgrey", alpha=0.35, zorder=2)
    if valid.any():
        return ax.scatter(coords[valid, 0], coords[valid, 1], c=values[valid],
                          cmap="viridis", s=12, alpha=0.85, zorder=3, **kw)
    return None


def _scatter_source(ax, coords, src):
    for s, color in SOURCE_COLORS.items():
        if s in ("wt", "gibbs"):
            continue
        m = src == s
        if m.any():
            ax.scatter(coords[m, 0], coords[m, 1], s=10, alpha=0.55, c=color,
                       edgecolors="none", label=s, zorder=3)


def _draw_panel(ax, coords, data, color_axis_key, fig):
    src = data["source_labels"]
    is_wt = src == "wt"
    is_gibbs = src == "gibbs"
    is_bg = ~(is_wt | is_gibbs)

    if color_axis_key == "source":
        _scatter_source(ax, coords, src)
    else:
        sc = _scatter_continuous(ax, coords[is_bg], data[color_axis_key][is_bg])
        if sc is not None:
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    if is_gibbs.any():
        ax.scatter(coords[is_gibbs, 0], coords[is_gibbs, 1], **GIBBS_TRI)
    if is_wt.any():
        ax.scatter(coords[is_wt, 0], coords[is_wt, 1], **WT_STAR)


def _padded_limits(values: np.ndarray, pad: float = 0.05) -> Tuple[float, float]:
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    span = hi - lo
    return lo - pad * span, hi + pad * span


# ----------------------------------------------------------------- Figure sets


def make_grid(per_variant, variants, coord_key, pc1_var, pc2_var,
              out_path, title, share_axes: bool, axis_label_fmt):
    n_rows = len(COLOR_AXES)
    n_cols = len(variants)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 4.0 * n_rows), squeeze=False)

    xlim = ylim = None
    if share_axes:
        all_coords = np.concatenate([per_variant[v][coord_key] for v in variants])
        finite = ~np.isnan(all_coords).any(axis=1)
        if finite.any():
            xlim = _padded_limits(all_coords[finite, 0])
            ylim = _padded_limits(all_coords[finite, 1])

    for col, v in enumerate(variants):
        d = per_variant[v]
        coords = d[coord_key]
        for row, (key, label) in enumerate(COLOR_AXES):
            ax = axes[row, col]
            _draw_panel(ax, coords, d, key, fig)
            if row == 0:
                ax.set_title(v, fontsize=11)
            ax.set_xlabel(axis_label_fmt(0, pc1_var))
            ax.set_ylabel(axis_label_fmt(1, pc2_var))
            if xlim is not None:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            if col == 0:
                ax.text(-0.18, 0.5, label, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=11, fontweight="bold")
        # Add legend for source row only on last column
        axes[0, col].legend(loc="best", fontsize=8, frameon=False)

    legend_handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="red",
               markeredgecolor="black", markeredgewidth=0.6, markersize=15, label="C05 WT"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="tab:green",
               alpha=0.45, markersize=8, label="Gibbs"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=9)
    fig.suptitle(title, fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def make_pc1_vs_fitness(per_variant, variants, pc1_var, pc2_var,
                        fitness_key, fitness_label, out_path):
    n_cols = len(variants)
    fig, axes = plt.subplots(2, n_cols, figsize=(4.4 * n_cols, 8.4), squeeze=False)

    # Shared PC1 limit on bottom row for visual comparability
    all_pca = np.concatenate([per_variant[v]["pca2d"] for v in variants])
    finite = ~np.isnan(all_pca).any(axis=1)
    pc1_lim = _padded_limits(all_pca[finite, 0]) if finite.any() else None

    for col, v in enumerate(variants):
        d = per_variant[v]
        pca = d["pca2d"]
        src = d["source_labels"]
        fit = d[fitness_key]
        is_dms = src == "dms"
        is_wt = src == "wt"

        # Top — PC1 vs PC2 scatter (DMS colored by fitness)
        ax_top = axes[0, col]
        sc = _scatter_continuous(ax_top, pca[is_dms], fit[is_dms])
        if sc is not None:
            fig.colorbar(sc, ax=ax_top, fraction=0.046, pad=0.04)
        if is_wt.any():
            ax_top.scatter(pca[is_wt, 0], pca[is_wt, 1], **WT_STAR)
        ax_top.set_title(v, fontsize=11)
        ax_top.set_xlabel(f"PC1 ({100 * pc1_var:.1f}% var)")
        ax_top.set_ylabel(f"PC2 ({100 * pc2_var:.1f}% var)")

        # Bottom — PC1 vs fitness regression (DMS only)
        ax_bot = axes[1, col]
        x = pca[is_dms, 0]
        y = fit[is_dms]
        m = ~(np.isnan(x) | np.isnan(y))
        x, y = x[m], y[m]
        ax_bot.scatter(x, y, s=10, alpha=0.5, c="tab:blue")
        if len(x) > 1:
            slope, intercept = np.polyfit(x, y, 1)
            xr = np.linspace(x.min(), x.max(), 100)
            ax_bot.plot(xr, slope * xr + intercept, "r-", linewidth=1.5)
            r_p, p_p = pearsonr(x, y)
            r_s, p_s = spearmanr(x, y)
            ax_bot.text(
                0.02, 0.97,
                f"Pearson r = {r_p:.3f}  (p = {p_p:.1e})\n"
                f"Spearman ρ = {r_s:.3f}  (p = {p_s:.1e})\n"
                f"n = {len(x)}",
                transform=ax_bot.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="grey"),
            )
        ax_bot.set_xlabel(f"PC1 ({100 * pc1_var:.1f}% var)")
        ax_bot.set_ylabel(fitness_label)
        if pc1_lim is not None:
            ax_bot.set_xlim(pc1_lim)

    fig.suptitle(f"PC1 vs {fitness_label}  —  Biswas-style validation", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def make_gibbs_trajectories(proj, out_dir: Path):
    variants_ws = proj["whole_seq"][0]
    for v in variants_ws:
        d_ws = proj["whole_seq"][1][v]
        d_cd = proj["cdrh3"][1][v]
        if not (d_ws["source_labels"] == "gibbs").any():
            continue

        fig, axes = plt.subplots(1, 2, figsize=(11, 5), squeeze=False)
        for ax, d, label in zip(axes[0], (d_ws, d_cd), ("whole_seq", "cdrh3")):
            src = d["source_labels"]
            pca = d["pca2d"]
            # Background
            oas = src == "oas"
            dms = src == "dms"
            if oas.any():
                ax.scatter(pca[oas, 0], pca[oas, 1], s=4, alpha=0.15, c="lightgrey", zorder=1)
            if dms.any():
                ax.scatter(pca[dms, 0], pca[dms, 1], s=8, alpha=0.35, c="grey", zorder=2)
            # Gibbs trajectories per chain
            gmask = src == "gibbs"
            chains = d["chain_id"][gmask]
            steps = d["gibbs_step"][gmask]
            coords = pca[gmask]
            unique_chains = sorted(set(int(c) for c in chains))
            cmap = plt.get_cmap("plasma")
            chain_colors = plt.get_cmap("tab10")
            for i, ch in enumerate(unique_chains):
                cm = chains == ch
                pts = coords[cm]
                st = steps[cm]
                order = np.argsort(st)
                pts, st = pts[order], st[order]
                line_color = chain_colors(i % 10) if len(unique_chains) > 1 else "tab:gray"
                ax.plot(pts[:, 0], pts[:, 1], "-", color=line_color, alpha=0.5, linewidth=1.0, zorder=4)
                sc = ax.scatter(pts[:, 0], pts[:, 1], c=st, cmap=cmap, s=18, zorder=5,
                                edgecolors=line_color if len(unique_chains) > 1 else "none",
                                linewidths=0.4)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="gibbs_step")
            wt = src == "wt"
            if wt.any():
                ax.scatter(pca[wt, 0], pca[wt, 1], **WT_STAR)
            ax.set_title(f"{label} — {len(unique_chains)} chain(s)")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
        fig.suptitle(f"Gibbs trajectories in PCA space — {v}", fontsize=12)
        fig.tight_layout()
        out_path = out_dir / f"gibbs_trajectory_{v}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        log.info("Wrote %s", out_path)


def make_pca_vs_umap_per_variant(proj, out_dir: Path, pc1_var: float, pc2_var: float):
    variants = proj["whole_seq"][0]
    all_pca = np.concatenate([proj["whole_seq"][1][v]["pca2d"] for v in variants])
    finite = ~np.isnan(all_pca).any(axis=1)
    xlim = _padded_limits(all_pca[finite, 0]) if finite.any() else None
    ylim = _padded_limits(all_pca[finite, 1]) if finite.any() else None

    for v in variants:
        d = proj["whole_seq"][1][v]
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), squeeze=False)
        for ax, coord_key, label in zip(axes[0], ("pca2d", "umap2d"), ("PCA", "UMAP")):
            _draw_panel(ax, d[coord_key], d, "source", fig)
            ax.legend(loc="best", fontsize=8, frameon=False)
            ax.set_title(label)
            if coord_key == "pca2d":
                ax.set_xlabel(f"PC1 ({100 * pc1_var:.1f}% var)")
                ax.set_ylabel(f"PC2 ({100 * pc2_var:.1f}% var)")
                if xlim is not None:
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
            else:
                ax.set_xlabel("UMAP 1")
                ax.set_ylabel("UMAP 2")
        fig.suptitle(f"{v}  —  whole_seq embeddings", fontsize=12)
        fig.tight_layout()
        out_path = out_dir / f"pca_vs_umap_{v}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        log.info("Wrote %s", out_path)


# ----------------------------------------------------------------- Figure 6


def _mean_cosine(matrix: np.ndarray, ref: np.ndarray) -> float:
    valid = ~np.isnan(matrix).any(axis=1)
    if not valid.any() or np.linalg.norm(ref) == 0:
        return float("nan")
    M = matrix[valid]
    M_norm = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    r_norm = ref / (np.linalg.norm(ref) + 1e-12)
    return float((M_norm @ r_norm).mean())


def _row_cosines(matrix: np.ndarray, ref: np.ndarray) -> np.ndarray:
    M_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    r_norm = ref / (np.linalg.norm(ref) + 1e-12)
    return M_norm @ r_norm


def make_summary_table(proj, embeddings_dir: Path, out_dir: Path):
    rows: List[dict] = []
    variants = proj["whole_seq"][0]
    for v in variants:
        emb = np.load(embeddings_dir / f"{v}.npz", allow_pickle=False)
        whole = emb["whole_seq_embs"]
        cdr = emb["cdrh3_embs"]
        src = emb["source_labels"]
        identity = emb["cdrh3_identity_to_wt"]

        wt_mask = src == "wt"
        dms_mask = src == "dms"
        oas_mask = src == "oas"
        gibbs_mask = src == "gibbs"

        if not wt_mask.any():
            log.warning("No WT row for %s — skipping cosine metrics", v)
            continue
        wt_whole = whole[wt_mask][0]
        wt_cdr = cdr[wt_mask][0]

        row: dict = {"model_variant": v}
        row["cos_DMS_WT_whole"] = _mean_cosine(whole[dms_mask], wt_whole)
        row["cos_DMS_WT_cdrh3"] = _mean_cosine(cdr[dms_mask], wt_cdr)
        row["cos_OAS_WT_whole"] = _mean_cosine(whole[oas_mask], wt_whole)
        row["cos_Gibbs_WT_whole"] = _mean_cosine(whole[gibbs_mask], wt_whole) if gibbs_mask.any() else float("nan")
        row["cos_Gibbs_WT_cdrh3"] = _mean_cosine(cdr[gibbs_mask], wt_cdr) if gibbs_mask.any() else float("nan")

        # Spearman: cdrh3 identity vs cosine distance to WT (cdrh3 emb, DMS only)
        cdr_dms = cdr[dms_mask]
        ident_dms = identity[dms_mask]
        valid = ~np.isnan(cdr_dms).any(axis=1) & ~np.isnan(ident_dms)
        if valid.sum() > 1:
            cos_dist = 1.0 - _row_cosines(cdr_dms[valid], wt_cdr)
            row["spearman_identity_cosdist_cdrh3"] = float(spearmanr(ident_dms[valid], cos_dist)[0])
        else:
            row["spearman_identity_cosdist_cdrh3"] = float("nan")

        # PC1 vs M22/SI06 for each emb_type
        for emb_type in ("whole_seq", "cdrh3"):
            d = proj[emb_type][1][v]
            dms_p = d["source_labels"] == "dms"
            x = d["pca2d"][dms_p, 0]
            for fk, fl in (("M22_enrich", "M22"), ("SI06_enrich", "SI06")):
                y = d[fk][dms_p]
                m = ~(np.isnan(x) | np.isnan(y))
                if m.sum() > 1:
                    row[f"pearson_PC1_{fl}_{emb_type}"] = float(pearsonr(x[m], y[m])[0])
                    row[f"spearman_PC1_{fl}_{emb_type}"] = float(spearmanr(x[m], y[m])[0])
                else:
                    row[f"pearson_PC1_{fl}_{emb_type}"] = float("nan")
                    row[f"spearman_PC1_{fl}_{emb_type}"] = float("nan")

        # Euclidean PC1-PC2: WT → DMS centroid, WT → Gibbs centroid
        for emb_type in ("whole_seq", "cdrh3"):
            d = proj[emb_type][1][v]
            pca = d["pca2d"]
            src_p = d["source_labels"]
            wt_p = pca[src_p == "wt"][0]
            dms_p = pca[src_p == "dms"]
            row[f"euclid_WT_DMS_centroid_{emb_type}"] = float(np.linalg.norm(wt_p - np.nanmean(dms_p, axis=0)))
            gibbs_p = pca[src_p == "gibbs"]
            if len(gibbs_p):
                row[f"euclid_WT_Gibbs_centroid_{emb_type}"] = float(np.linalg.norm(wt_p - np.nanmean(gibbs_p, axis=0)))
            else:
                row[f"euclid_WT_Gibbs_centroid_{emb_type}"] = float("nan")

        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "summary_table.csv"
    df.to_csv(csv_path, index=False)
    log.info("Wrote %s", csv_path)

    cell_text = df.round(3).astype(str).values
    fig_w = max(14.0, 0.9 * len(df.columns))
    fig_h = max(2.0, 0.55 * (len(df) + 2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=df.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.4)
    fig.tight_layout()
    png_path = out_dir / "summary_table.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", png_path)


# ------------------------------------------------------------------------- main


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--projections-dir", type=Path, required=True)
    p.add_argument("--embeddings-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    proj = {emb: load_projections(args.projections_dir, emb) for emb in ("whole_seq", "cdrh3")}
    variants_ws, _, pc1_ws, pc2_ws = proj["whole_seq"]
    variants_cd, _, pc1_cd, pc2_cd = proj["cdrh3"]
    assert variants_ws == variants_cd, "Variant order must match across emb_types"

    # Figure set 1 — PCA grid
    for emb_type in ("whole_seq", "cdrh3"):
        variants, per_variant, pc1, pc2 = proj[emb_type]
        make_grid(per_variant, variants, "pca2d", pc1, pc2,
                  args.output_dir / f"pca_grid_{emb_type}.png",
                  title=f"PCA — {emb_type}  (distances are geometrically meaningful across panels)",
                  share_axes=True,
                  axis_label_fmt=lambda i, var: f"PC{i+1} ({100*var:.1f}% var)")

    # Figure set 2 — UMAP grid (no shared axes)
    for emb_type in ("whole_seq", "cdrh3"):
        variants, per_variant, _, _ = proj[emb_type]
        make_grid(per_variant, variants, "umap2d", 0.0, 0.0,
                  args.output_dir / f"umap_grid_{emb_type}.png",
                  title=f"UMAP — {emb_type}  (cluster structure only — distances and axes NOT comparable across panels)",
                  share_axes=False,
                  axis_label_fmt=lambda i, _var: f"UMAP {i+1}")

    # Figure set 3 — PC1 vs fitness
    for emb_type in ("whole_seq", "cdrh3"):
        variants, per_variant, pc1, pc2 = proj[emb_type]
        for fk, fl in (("M22_enrich", "M22 binding enrichment"),
                       ("SI06_enrich", "SI06 binding enrichment")):
            short = fk.split("_")[0]
            make_pc1_vs_fitness(per_variant, variants, pc1, pc2, fk, fl,
                                args.output_dir / f"pc1_vs_{short}_{emb_type}.png")

    # Figure set 4 — Gibbs trajectories
    make_gibbs_trajectories(proj, args.output_dir)

    # Figure set 5 — PCA vs UMAP per variant
    make_pca_vs_umap_per_variant(proj, args.output_dir, pc1_ws, pc2_ws)

    # Figure 6 — summary table
    make_summary_table(proj, args.embeddings_dir, args.output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
