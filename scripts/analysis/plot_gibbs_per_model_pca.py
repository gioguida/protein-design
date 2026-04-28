"""Project Gibbs samples into each variant's own DMS-only PCA frame.

Why this exists
---------------
``plot_projections.py`` renders Gibbs trajectories in the *shared* PCA fit on
vanilla's combined background (DMS + OAS + WT). Because OAS dominates that
background, the dominant axis is germline diversity rather than within-DMS
variation, so "does the sampler collapse onto the high-fitness DMS region?"
cannot be answered there. The per-model DMS-only PCA from
``compute_per_model_pca.py`` *is* the right frame for that question.

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

        for v in variants:
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
            multi_config = len(configs) >= 2

            def _do_plot(emb, chains, steps, cfg_suffix, view_suffix, view_title):
                gibbs_pc_local = None
                chain_ids_local = None
                steps_local = None
                if len(emb):
                    valid = ~np.isnan(emb).any(axis=1)
                    if valid.any():
                        gibbs_pc_local = pca.transform(emb[valid])[:, :2]
                        chain_ids_local = chains[valid]
                        steps_local = steps[valid]
                plot_one_variant(
                    variant=v,
                    emb_type=emb_type,
                    dms_pca=per_variant[v]["pca"],
                    dms_fitness={
                        "M22_enrich": per_variant[v]["M22_enrich"],
                        "SI06_enrich": per_variant[v]["SI06_enrich"],
                    },
                    gibbs_pc=gibbs_pc_local,
                    gibbs_chain_id=chain_ids_local,
                    gibbs_step=steps_local,
                    wt_pc=wt_pc,
                    explained_variance=per_variant[v]["explained_variance"],
                    dms_dataset=dms_dataset,
                    out_dir=args.output_dir,
                    config_suffix=cfg_suffix,
                    view_suffix=view_suffix,
                    title_extra=view_title,
                )

            def _early_subset(emb, chains, steps, ed):
                ed_mask = ed <= args.early_max_ed
                if not ed_mask.any():
                    return None
                emb_e = emb[ed_mask]
                chains_e = chains[ed_mask]
                steps_e = steps[ed_mask]
                keep = sorted(set(int(c) for c in chains_e))[: args.early_max_chains]
                if not keep:
                    return None
                keep_mask = np.isin(chains_e, keep)
                return emb_e[keep_mask], chains_e[keep_mask], steps_e[keep_mask]

            def _emit_for_subset(emb, chains, steps, ed, cfg_suffix):
                if not args.skip_full:
                    _do_plot(emb, chains, steps, cfg_suffix, "", "")
                if not args.skip_early:
                    sub = _early_subset(emb, chains, steps, ed)
                    if sub is None:
                        log.warning("No Gibbs rows with edit_distance ≤ %d for %s/%s/%s — skipping early plot",
                                    args.early_max_ed, v, emb_type, cfg_suffix or "default")
                    else:
                        emb_e, chains_e, steps_e = sub
                        _do_plot(
                            emb_e, chains_e, steps_e, cfg_suffix, "early",
                            f"early: ED≤{args.early_max_ed}, ≤{args.early_max_chains} chains",
                        )

            if not configs:
                _emit_for_subset(grows["gibbs_emb"], grows["gibbs_chain_id"],
                                 grows["gibbs_step"], grows["gibbs_edit_distance"], "")
            else:
                for cfg in configs:
                    mask = grows["gibbs_config"] == cfg
                    suffix = cfg if multi_config else ""
                    _emit_for_subset(grows["gibbs_emb"][mask],
                                     grows["gibbs_chain_id"][mask],
                                     grows["gibbs_step"][mask],
                                     grows["gibbs_edit_distance"][mask],
                                     suffix)

    return 0


if __name__ == "__main__":
    sys.exit(main())
