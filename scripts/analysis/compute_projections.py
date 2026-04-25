"""Fit PCA + UMAP on vanilla ESM2 embeddings and project all model variants.

Reads
-----
N .npz files written by `extract_embeddings.py`, one per model variant. Reference rows
(wt/dms/oas) are positionally aligned across variants by construction (same seed, same
sampling). Gibbs rows are variant-specific and appended after.

Writes
------
- `whole_seq.npz`, `cdrh3.npz` — projected 2D coordinates per variant (keys prefixed
  `{variant}__`), plus `model_variants`, `pca_explained_variance`, `pca_pc{1,2}_variance`.
- `pca_{emb_type}.pkl`, `umap_{emb_type}.pkl` — fitted projectors. Reusable: load the
  pickle and call `.transform(new_embeddings)` to project new sequences without refitting.
- `scree_{emb_type}.png` — explained variance per PCA component.

Design
------
Both projectors are fit on the **vanilla** model's background (non-gibbs) embeddings only.
All other variants are projected through the same projectors so that PCA distances are
geometrically comparable across panels. UMAP `transform()` is approximate — use only for
cluster structure, not cross-variant distance claims.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.decomposition import PCA

SEED = 42
EMB_TYPES = ["whole_seq", "cdrh3"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("compute_projections")


def load_variants(npz_paths: list[Path]) -> Dict[str, Dict[str, np.ndarray]]:
    """Return {variant_label: {key: array}} for each input .npz."""
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for path in npz_paths:
        z = np.load(path, allow_pickle=False)
        variant = str(z["model_variant"][0])
        if variant in out:
            raise ValueError(f"Duplicate model_variant '{variant}' in inputs")
        out[variant] = {k: z[k] for k in z.files}
        log.info("Loaded %s  (%d rows)  from %s", variant, len(out[variant]["sequences"]), path)
    return out


def background_intersection_mask(
    data: Dict[str, Dict[str, np.ndarray]], emb_key: str, n_bg: int
) -> np.ndarray:
    """Boolean mask over the first n_bg rows: True where embedding is finite for *all* variants."""
    mask = np.ones(n_bg, dtype=bool)
    for v, d in data.items():
        valid = ~np.isnan(d[emb_key][:n_bg]).any(axis=1)
        mask &= valid
    return mask


def fit_and_save_projectors(
    vanilla_bg_emb: np.ndarray, out_dir: Path, emb_type: str
) -> tuple[PCA, "umap.UMAP"]:
    pca = PCA(n_components=50, random_state=SEED)
    pca.fit(vanilla_bg_emb)
    with open(out_dir / f"pca_{emb_type}.pkl", "wb") as fh:
        pickle.dump(pca, fh)

    vanilla_pca50 = pca.transform(vanilla_bg_emb)
    umap_model = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=SEED)
    umap_model.fit(vanilla_pca50)
    with open(out_dir / f"umap_{emb_type}.pkl", "wb") as fh:
        pickle.dump(umap_model, fh)

    return pca, umap_model


def scree_plot(pca: PCA, out_path: Path, emb_type: str) -> None:
    var = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(1, len(var) + 1), var, color="steelblue")
    ax.set_xlabel("PCA component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title(f"Scree — {emb_type}  (cumulative over 50 components: {var.sum():.1%})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def process_emb_type(
    emb_type: str,
    data: Dict[str, Dict[str, np.ndarray]],
    out_dir: Path,
    vanilla_label: str,
) -> None:
    emb_key = f"{emb_type}_embs"
    log.info("=== %s ===", emb_type)

    n_bg = int((data[vanilla_label]["source_labels"] != "gibbs").sum())
    for v, d in data.items():
        v_bg = int((d["source_labels"] != "gibbs").sum())
        if v_bg != n_bg:
            raise ValueError(
                f"Background row count mismatch: {vanilla_label}={n_bg} vs {v}={v_bg}. "
                "Reference set must be sampled identically across variants."
            )

    bg_mask = background_intersection_mask(data, emb_key, n_bg)
    log.info("Shared background mask: %d / %d rows valid across all variants", bg_mask.sum(), n_bg)

    vanilla_bg_emb = data[vanilla_label][emb_key][:n_bg][bg_mask]
    pca, umap_model = fit_and_save_projectors(vanilla_bg_emb, out_dir, emb_type)
    log.info("PCA explained variance: PC1=%.1f%%  PC2=%.1f%%  total(50)=%.1f%%",
             100 * pca.explained_variance_ratio_[0],
             100 * pca.explained_variance_ratio_[1],
             100 * pca.explained_variance_ratio_.sum())

    out: Dict[str, np.ndarray] = {}
    for v, d in data.items():
        emb = d[emb_key]
        n = emb.shape[0]
        valid = ~np.isnan(emb).any(axis=1)
        pca2d = np.full((n, 2), np.nan, dtype=np.float32)
        umap2d = np.full((n, 2), np.nan, dtype=np.float32)
        if valid.any():
            pca50 = pca.transform(emb[valid])
            pca2d[valid] = pca50[:, :2].astype(np.float32)
            umap2d[valid] = umap_model.transform(pca50).astype(np.float32)
        prefix = f"{v}__"
        out[prefix + "pca2d"] = pca2d
        out[prefix + "umap2d"] = umap2d
        out[prefix + "source_labels"] = d["source_labels"]
        out[prefix + "M22_enrich"] = d["M22_binding_enrichment"]
        out[prefix + "SI06_enrich"] = d["SI06_binding_enrichment"]
        out[prefix + "cdrh3_identity"] = d["cdrh3_identity_to_wt"]
        out[prefix + "gibbs_step"] = d["gibbs_step"]
        out[prefix + "chain_id"] = d["chain_id"]
        out[prefix + "sequences"] = d["sequences"]

    out["model_variants"] = np.array(list(data.keys()))
    out["pca_explained_variance"] = pca.explained_variance_ratio_.astype(np.float32)
    out["pca_pc1_variance"] = np.float32(pca.explained_variance_ratio_[0])
    out["pca_pc2_variance"] = np.float32(pca.explained_variance_ratio_[1])

    np.savez(out_dir / f"{emb_type}.npz", **out)
    scree_plot(pca, out_dir / f"scree_{emb_type}.png", emb_type)
    log.info("Wrote %s/%s.npz + scree", out_dir, emb_type)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("npz_files", nargs="+", type=Path, help="Per-variant .npz files from extract_embeddings.py")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--vanilla-label", default="vanilla", help="Variant label whose embeddings fit the projectors")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    data = load_variants(args.npz_files)
    if args.vanilla_label not in data:
        raise SystemExit(
            f"--vanilla-label '{args.vanilla_label}' not among inputs: {list(data.keys())}"
        )

    for emb_type in EMB_TYPES:
        process_emb_type(emb_type, data, args.output_dir, args.vanilla_label)

    return 0


if __name__ == "__main__":
    sys.exit(main())
