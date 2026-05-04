"""Project Gibbs samples into each variant's own DMS-only PCA frame.

Why this exists
---------------
The per-model DMS-only PCA from ``compute_per_model_pca.py`` is the right
frame to ask "does the Gibbs sampler collapse onto the high-fitness DMS
region?" — each variant's PCA captures its own DMS variance.

Reads
-----
- ``{per_model_pca_dir}/per_model_pca_{cdrh3,whole_seq}.npz``
- ``{per_model_pca_dir}/pca_dms_{variant}_{emb_type}.pkl``
- ``{embeddings_dir}/{variant}.npz`` — for the gibbs rows.

Writes (200 DPI, in ``{output_dir}/``)
-------------------------------------
- ``gibbs_per_model_pca_{variant}_{emb_type}_{fitness}_{dms_dataset}[_{cfg}][_early].png``
  PC1 vs PC2 in this variant's own DMS-PCA. DMS background coloured by
  enrichment readout, gibbs trajectory overlaid (one line per chain, marker
  shaded by gibbs_step). WT shown as a red star.

  The ``_early`` variant restricts Gibbs samples to those at edit-distance
  ≤ ``--early-max-ed`` from C05 WT (default 10) and shows at most
  ``--early-max-chains`` chains (default 10). It exists because most Gibbs
  chains end up at edit distance ~20 from WT, which makes the full plot
  dominated by the late, off-WT region; the early view shows how chains
  *leave* the WT neighbourhood at the start of sampling.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from protein_design.constants import C05_CDRH3

EMB_TYPES = ["cdrh3", "whole_seq"]
FITNESS = [
    ("M22_enrich", "M22 binding enrichment", "M22"),
    ("SI06_enrich", "SI06 binding enrichment", "SI06"),
]
CDRH3_LEN = len(C05_CDRH3)  # 24

WT_STAR = dict(marker="*", s=240, c="red", edgecolors="black", linewidths=0.6, zorder=10)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("plot_gibbs_per_model_pca")


def load_per_model_npz(
    npz_path: Path,
) -> Tuple[List[str], Dict[str, Dict[str, np.ndarray]], str]:
    z = np.load(npz_path, allow_pickle=False)
    variants = [str(v) for v in z["model_variants"]]
    per_variant: Dict[str, Dict[str, np.ndarray]] = {}
    for v in variants:
        per_variant[v] = {
            "pca": z[f"{v}__pca"],
            "M22_enrich": z[f"{v}__M22_enrich"],
            "SI06_enrich": z[f"{v}__SI06_enrich"],
            "explained_variance": z[f"{v}__pca_explained_variance"],
        }
    dms_dataset = str(z["dms_dataset"][0]) if "dms_dataset" in z.files else "ed2"
    return variants, per_variant, dms_dataset


def load_gibbs_rows(emb_npz: Path, emb_key: str) -> Dict[str, np.ndarray]:
    z = np.load(emb_npz, allow_pickle=False)
    src = z["source_labels"]
    g = src == "gibbs"
    wt = src == "wt"
    if "gibbs_config" in z.files:
        gibbs_config = z["gibbs_config"][g]
    else:
        gibbs_config = np.array(["default"] * int(g.sum()))
    identity = z["cdrh3_identity_to_wt"][g]
    edit_distance = np.rint((1.0 - identity) * CDRH3_LEN).astype(np.int32)
    return {
        "gibbs_emb": z[emb_key][g],
        "gibbs_chain_id": z["chain_id"][g],
        "gibbs_step": z["gibbs_step"][g],
        "gibbs_config": gibbs_config,
        "gibbs_edit_distance": edit_distance,
        "wt_emb": z[emb_key][wt][0] if wt.any() else None,
    }


def _scatter_dms(ax, coords: np.ndarray, values: np.ndarray, label: str):
    valid_xy = ~np.isnan(coords).any(axis=1)
    valid_v = ~np.isnan(values)
    grey_mask = valid_xy & ~valid_v
    color_mask = valid_xy & valid_v
    if grey_mask.any():
        ax.scatter(coords[grey_mask, 0], coords[grey_mask, 1],
                   s=10, c="lightgrey", alpha=0.35, zorder=2)
    sc = None
    if color_mask.any():
        sc = ax.scatter(coords[color_mask, 0], coords[color_mask, 1],
                        c=values[color_mask], cmap="viridis",
                        s=18, alpha=0.85, zorder=3)
    return sc


def plot_one_variant(
    variant: str,
    emb_type: str,
    dms_pca: np.ndarray,
    dms_fitness: Dict[str, np.ndarray],
    gibbs_pc: np.ndarray | None,
    gibbs_chain_id: np.ndarray | None,
    gibbs_step: np.ndarray | None,
    wt_pc: np.ndarray | None,
    explained_variance: np.ndarray,
    dms_dataset: str,
    out_dir: Path,
    config_suffix: str = "",
    view_suffix: str = "",
    title_extra: str = "",
) -> None:
    chain_cmap = plt.get_cmap("tab10")
    step_cmap = plt.get_cmap("plasma")

    for fkey, flabel, fshort in FITNESS:
        if np.isnan(dms_fitness[fkey]).all():
            log.info("Skipping %s/%s/%s — no %s values in dataset %s",
                     variant, emb_type, fshort, fkey, dms_dataset)
            continue
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        sc = _scatter_dms(ax, dms_pca[:, :2], dms_fitness[fkey], flabel)
        if sc is not None:
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=flabel)

        if gibbs_pc is not None and len(gibbs_pc):
            unique_chains = sorted(set(int(c) for c in gibbs_chain_id))
            for i, ch in enumerate(unique_chains):
                cm = gibbs_chain_id == ch
                pts = gibbs_pc[cm]
                st = gibbs_step[cm]
                order = np.argsort(st)
                pts, st = pts[order], st[order]
                line_color = chain_cmap(i % 10)
                ax.plot(pts[:, 0], pts[:, 1], "-", color=line_color,
                        alpha=0.55, linewidth=1.0, zorder=4)
                ax.scatter(pts[:, 0], pts[:, 1], c=st, cmap=step_cmap,
                           s=22, zorder=5, edgecolors=line_color, linewidths=0.5)

        if wt_pc is not None:
            ax.scatter(wt_pc[0], wt_pc[1], **WT_STAR)

        ev = explained_variance
        ax.set_xlabel(f"PC1 ({100 * ev[0]:.1f}% var)")
        ax.set_ylabel(f"PC2 ({100 * ev[1]:.1f}% var)" if len(ev) > 1 else "PC2")
        cfg_extra = f"  [{config_suffix}]" if config_suffix else ""
        view_extra = f"  [{title_extra}]" if title_extra else ""
        ax.set_title(
            f"{variant} — Gibbs trajectory in own DMS-PCA "
            f"[{dms_dataset.upper()}]  ({emb_type}){cfg_extra}{view_extra}\n"
            f"DMS background coloured by {flabel}; WT marked with red star",
            fontsize=11,
        )
        cfg_part = f"_{config_suffix}" if config_suffix else ""
        view_part = f"_{view_suffix}" if view_suffix else ""
        out_path = (
            out_dir
            / f"gibbs_per_model_pca_{variant}_{emb_type}_{fshort}_{dms_dataset}{cfg_part}{view_part}.png"
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        log.info("Wrote %s", out_path)


def plot_per_model_pca_grid(
    per_cell: dict,
    variant_order: list,
    config_order: list,
    fkey: str,
    flabel: str,
    fshort: str,
    emb_type: str,
    dms_dataset: str,
    out_dir: Path,
    view_suffix: str = "",
    title_extra: str = "",
) -> None:
    """Combined grid: rows = configs, cols = variants — all in one figure.

    Each cell shows the DMS background in that variant's own PCA frame,
    coloured by enrichment with a shared scale, plus the Gibbs trajectories.
    Output filename: gibbs_per_model_pca_all_{emb_type}_{fshort}_{dms_dataset}[_{view_suffix}].png
    """
    variants = [v for v in variant_order if any((v, cfg) in per_cell for cfg in config_order)]
    configs = [cfg for cfg in config_order if any((v, cfg) in per_cell for v in variant_order)]
    if not variants or not configs:
        log.info("No data for combined grid %s/%s/%s — skipping", fshort, emb_type, view_suffix or "full")
        return

    # Build global fitness range for a shared colorbar across all cells.
    all_fvals: list[np.ndarray] = []
    for v in variants:
        for cfg in configs:
            key = (v, cfg)
            if key not in per_cell:
                continue
            fvals = per_cell[key]["dms_fitness"][fkey]
            valid = fvals[~np.isnan(fvals)]
            if len(valid):
                all_fvals.append(valid)
    if not all_fvals:
        log.info("Skipping %s combined grid — no %s values in dataset %s", fshort, fkey, dms_dataset)
        return

    all_fcat = np.concatenate(all_fvals)
    norm = plt.Normalize(vmin=float(np.min(all_fcat)), vmax=float(np.max(all_fcat)))
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])

    chain_cmap = plt.get_cmap("tab10")
    step_cmap = plt.get_cmap("plasma")

    n_rows, n_cols = len(configs), len(variants)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.0 * n_cols + 1.4, 3.8 * n_rows + 0.8),
        squeeze=False,
        layout="constrained",
    )

    for row, cfg in enumerate(configs):
        for col, v in enumerate(variants):
            ax = axes[row, col]
            key = (v, cfg)
            if key not in per_cell:
                ax.axis("off")
                continue
            d = per_cell[key]
            dms_pca = d["dms_pca"]
            fvals = d["dms_fitness"][fkey]
            gibbs_pc = d["gibbs_pc"]
            gibbs_chain_id = d["gibbs_chain_id"]
            gibbs_step = d["gibbs_step"]
            wt_pc = d["wt_pc"]
            ev = d["explained_variance"]

            # DMS background
            valid_xy = ~np.isnan(dms_pca[:, :2]).any(axis=1)
            valid_v = ~np.isnan(fvals)
            if (valid_xy & ~valid_v).any():
                ax.scatter(dms_pca[valid_xy & ~valid_v, 0], dms_pca[valid_xy & ~valid_v, 1],
                           s=7, c="lightgrey", alpha=0.3, zorder=2)
            if (valid_xy & valid_v).any():
                ax.scatter(
                    dms_pca[valid_xy & valid_v, 0], dms_pca[valid_xy & valid_v, 1],
                    c=fvals[valid_xy & valid_v], cmap="viridis", norm=norm,
                    s=12, alpha=0.75, zorder=3,
                )

            # Gibbs trajectories
            if gibbs_pc is not None and len(gibbs_pc):
                unique_chains = sorted(set(int(c) for c in gibbs_chain_id))
                for i, ch in enumerate(unique_chains):
                    cm = gibbs_chain_id == ch
                    pts = gibbs_pc[cm]
                    st = gibbs_step[cm]
                    order = np.argsort(st)
                    pts, st = pts[order], st[order]
                    line_color = chain_cmap(i % 10)
                    ax.plot(pts[:, 0], pts[:, 1], "-", color=line_color,
                            alpha=0.55, linewidth=0.9, zorder=4)
                    ax.scatter(pts[:, 0], pts[:, 1], c=st, cmap=step_cmap,
                               s=16, zorder=5, edgecolors=line_color, linewidths=0.35)

            if wt_pc is not None:
                ax.scatter(wt_pc[0], wt_pc[1], **WT_STAR)

            pc2_label = f"PC2 ({100 * ev[1]:.1f}% var)" if len(ev) > 1 else "PC2"
            pc1_label = f"PC1 ({100 * ev[0]:.1f}% var)"
            if row == 0:
                ax.set_title(v, fontsize=9, fontweight="bold")
            ax.set_ylabel((f"{cfg}\n\n{pc2_label}") if col == 0 else pc2_label, fontsize=8)
            ax.set_xlabel(pc1_label, fontsize=7)
            ax.tick_params(labelsize=6)

    fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, aspect=30, pad=0.02, label=flabel)
    view_extra = f"  [{title_extra}]" if title_extra else ""
    fig.suptitle(
        f"Gibbs trajectory in per-model DMS-PCA [{dms_dataset.upper()}]  ({emb_type}){view_extra}\n"
        f"DMS coloured by {flabel}; chains coloured by chain ID; markers shaded by Gibbs step",
        fontsize=11,
    )
    view_part = f"_{view_suffix}" if view_suffix else ""
    out_path = out_dir / f"gibbs_per_model_pca_all_{emb_type}_{fshort}_{dms_dataset}{view_part}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--per-model-pca-dir", type=Path, required=True,
                   help="Directory with per_model_pca_{emb_type}.npz and "
                        "pca_dms_{variant}_{emb_type}.pkl from compute_per_model_pca.py")
    p.add_argument("--embeddings-dir", type=Path, required=True,
                   help="Directory with {variant}.npz from extract_embeddings.py")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--early-max-ed", type=int, default=10,
                   help="Edit-distance-to-WT cap for the early-trajectory plot.")
    p.add_argument("--early-max-chains", type=int, default=10,
                   help="Cap on number of chains shown in the early-trajectory plot.")
    p.add_argument("--skip-full", action="store_true",
                   help="Skip the full-trajectory plot; emit only the early view.")
    p.add_argument("--skip-early", action="store_true",
                   help="Skip the early-trajectory plot; emit only the full view.")
    p.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Optional list of gibbs_config names to include (e.g. gibbs_dist gibbs_fit).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_full and args.skip_early:
        log.error("--skip-full and --skip-early both set; nothing to do.")
        return 1

    for emb_type in EMB_TYPES:
        per_model_npz = args.per_model_pca_dir / f"per_model_pca_{emb_type}.npz"
        if not per_model_npz.exists():
            log.warning("Skipping %s — %s not found", emb_type, per_model_npz)
            continue

        variants, per_variant, dms_dataset = load_per_model_npz(per_model_npz)
        emb_key = f"{emb_type}_embs"

        # Accumulators for the combined grid plot (keyed by (variant, config_name)).
        per_cell_full: dict = {}
        per_cell_early: dict = {}
        variant_order: list = []
        config_name_set: list = []   # ordered unique config names

        for v in variants:
            variant_order.append(v)
            pkl_path = args.per_model_pca_dir / f"pca_dms_{v}_{emb_type}.pkl"
            emb_npz = args.embeddings_dir / f"{v}.npz"
            if not pkl_path.exists():
                log.warning("Skipping %s/%s — pickle %s missing", v, emb_type, pkl_path)
                continue
            if not emb_npz.exists():
                log.warning("Skipping %s/%s — embedding npz %s missing",
                            v, emb_type, emb_npz)
                continue

            with open(pkl_path, "rb") as fh:
                pca = pickle.load(fh)
            grows = load_gibbs_rows(emb_npz, emb_key)

            wt_pc = None
            if grows["wt_emb"] is not None and not np.isnan(grows["wt_emb"]).any():
                wt_pc = pca.transform(grows["wt_emb"][None, :])[0, :2]

            configs = sorted(set(grows["gibbs_config"].tolist())) if len(grows["gibbs_config"]) else []
            if args.configs is not None:
                wanted = set(args.configs)
                configs = [cfg for cfg in configs if cfg in wanted]
            multi_config = len(configs) >= 2

            def _collect_data(emb, chains, steps):
                gibbs_pc_local = None
                chain_ids_local = None
                steps_local = None
                if len(emb):
                    valid = ~np.isnan(emb).any(axis=1)
                    if valid.any():
                        gibbs_pc_local = pca.transform(emb[valid])[:, :2]
                        chain_ids_local = chains[valid]
                        steps_local = steps[valid]
                return {
                    "dms_pca": per_variant[v]["pca"],
                    "dms_fitness": {
                        "M22_enrich": per_variant[v]["M22_enrich"],
                        "SI06_enrich": per_variant[v]["SI06_enrich"],
                    },
                    "gibbs_pc": gibbs_pc_local,
                    "gibbs_chain_id": chain_ids_local,
                    "gibbs_step": steps_local,
                    "wt_pc": wt_pc,
                    "explained_variance": per_variant[v]["explained_variance"],
                }

            def _early_subset(emb, chains, steps, ed):
                # Select up to early_max_chains chains (all available chains).
                unique_chains = sorted(set(int(c) for c in chains))[: args.early_max_chains]
                if not unique_chains:
                    return None
                # For each chain, keep every step from the start through (and
                # including) the first checkpoint where ED > early_max_ed.
                # This guarantees at least one line segment per chain even when
                # sampling is coarse and chains diverge quickly.
                keep_idx: list[np.ndarray] = []
                for cid in unique_chains:
                    cm = chains == cid
                    orig_idx = np.where(cm)[0]
                    order = np.argsort(steps[orig_idx])
                    sorted_idx = orig_idx[order]
                    sorted_ed = ed[sorted_idx]
                    exceeds = np.where(sorted_ed > args.early_max_ed)[0]
                    cutoff = int(exceeds[0]) + 1 if len(exceeds) else len(sorted_idx)
                    keep_idx.append(sorted_idx[:cutoff])
                all_idx = np.concatenate(keep_idx)
                return emb[all_idx], chains[all_idx], steps[all_idx]

            def _emit_for_subset(emb, chains, steps, ed, cfg_suffix, config_name):
                if config_name not in config_name_set:
                    config_name_set.append(config_name)
                cell = _collect_data(emb, chains, steps)
                if not args.skip_full:
                    per_cell_full[(v, config_name)] = cell
                if not args.skip_early:
                    sub = _early_subset(emb, chains, steps, ed)
                    if sub is None:
                        log.warning("No Gibbs chains found for %s/%s/%s — skipping early plot",
                                    v, emb_type, config_name)
                    else:
                        emb_e, chains_e, steps_e = sub
                        per_cell_early[(v, config_name)] = _collect_data(emb_e, chains_e, steps_e)

            if not configs:
                _emit_for_subset(grows["gibbs_emb"], grows["gibbs_chain_id"],
                                 grows["gibbs_step"], grows["gibbs_edit_distance"], "", "default")
            else:
                for cfg in configs:
                    mask = grows["gibbs_config"] == cfg
                    suffix = cfg if multi_config else ""
                    _emit_for_subset(grows["gibbs_emb"][mask],
                                     grows["gibbs_chain_id"][mask],
                                     grows["gibbs_step"][mask],
                                     grows["gibbs_edit_distance"][mask],
                                     suffix, cfg)

        # Combined grid plots (all variants in one figure per fitness × view).
        for fkey, flabel, fshort in FITNESS:
            if per_cell_full:
                plot_per_model_pca_grid(
                    per_cell_full, variant_order, config_name_set,
                    fkey, flabel, fshort, emb_type, dms_dataset, args.output_dir,
                )
            if per_cell_early:
                plot_per_model_pca_grid(
                    per_cell_early, variant_order, config_name_set,
                    fkey, flabel, fshort, emb_type, dms_dataset, args.output_dir,
                    view_suffix="early",
                    title_extra=(
                        f"early: ≤{args.early_max_chains} chains, "
                        f"steps through first ED>{args.early_max_ed}"
                    ),
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
