"""Per-model DMS-only PCA — fit one PCA per model variant on its DMS embeddings.

Why this exists
---------------
``compute_projections.py`` fits a single PCA on the vanilla model's combined
background (DMS + OAS + WT). Because the ~2000 OAS sequences dominate that fit
numerically, PC1 there describes germline diversity rather than within-DMS
variation, and asking "does PC1 correlate with affinity?" against that PCA is
a guaranteed null result regardless of whether fine-tuning worked.

This script answers the per-model question instead: for each model variant
independently, what is the dominant axis of variance among the DMS sequences,
and does that axis align with affinity? The fitted projectors are saved so
new sequences can be projected later without refitting.

Reads
-----
N .npz files written by ``extract_embeddings.py`` (one per variant).

Writes
------
- ``per_model_pca_{emb_type}.npz`` — per-variant 2D coords + enrichment +
  per-variant explained variance, keyed ``{variant}__{field}``.
- ``pca_dms_{variant}_{emb_type}.pkl`` — fitted ``sklearn.decomposition.PCA``
  per variant. Reusable: load and call ``.transform(new_embeddings)``.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.decomposition import PCA

SEED = 42
EMB_TYPES = ["cdrh3", "whole_seq"]
N_COMPONENTS = 10

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("compute_per_model_pca")


def load_variants(npz_paths: list[Path]) -> Tuple[Dict[str, Dict[str, np.ndarray]], str]:
    """Load all per-variant npz files. Returns (per_variant_dict, dms_dataset).

    All inputs must share the same `dms_dataset`; otherwise the per-model PCA
    would be fitted on inconsistent DMS rows across variants.
    """
    out: Dict[str, Dict[str, np.ndarray]] = {}
    datasets: set[str] = set()
    for path in npz_paths:
        z = np.load(path, allow_pickle=False)
        variant = str(z["model_variant"][0])
        if variant in out:
            raise ValueError(f"Duplicate model_variant '{variant}' in inputs")
        out[variant] = {k: z[k] for k in z.files}
        ds = str(z["dms_dataset"][0]) if "dms_dataset" in z.files else "ed2"
        datasets.add(ds)
        log.info("Loaded %s (%d rows, dms_dataset=%s) from %s",
                 variant, len(out[variant]["sequences"]), ds, path)
    if len(datasets) > 1:
        raise ValueError(f"Input npz files mix DMS datasets {datasets}; "
                         "re-extract embeddings with a single --dms-dataset.")
    dms_dataset = datasets.pop() if datasets else "ed2"
    return out, dms_dataset


def fit_dms_pca(emb: np.ndarray, src: np.ndarray) -> Tuple[PCA, np.ndarray]:
    """Fit PCA on DMS-only rows; return (pca, padded_coords).

    ``padded_coords`` has shape ``(n_dms, n_components)`` and contains NaN for
    DMS rows whose embedding was missing in the input (e.g. CDR3 not locatable).
    """
    is_dms = src == "dms"
    n_dms = int(is_dms.sum())
    if n_dms == 0:
        raise ValueError("No DMS rows found in this variant's embeddings.")

    dms_emb = emb[is_dms]
    valid_local = ~np.isnan(dms_emb).any(axis=1)
    n_valid = int(valid_local.sum())
    if n_valid < 3:
        raise ValueError(f"Need at least 3 valid DMS rows to fit PCA; got {n_valid}")

    n_components = min(N_COMPONENTS, n_valid, emb.shape[1])
    pca = PCA(n_components=n_components, random_state=SEED)
    pca.fit(dms_emb[valid_local])

    padded = np.full((n_dms, n_components), np.nan, dtype=np.float32)
    padded[valid_local] = pca.transform(dms_emb[valid_local]).astype(np.float32)
    return pca, padded


def process_emb_type(
    emb_type: str,
    data: Dict[str, Dict[str, np.ndarray]],
    dms_dataset: str,
    out_dir: Path,
) -> None:
    emb_key = f"{emb_type}_embs"
    log.info("=== %s ===", emb_type)
    out: Dict[str, np.ndarray] = {}

    for variant, d in data.items():
        emb = d[emb_key]
        src = d["source_labels"]
        is_dms = src == "dms"

        pca, coords = fit_dms_pca(emb, src)

        prefix = f"{variant}__"
        out[prefix + "pca"] = coords
        out[prefix + "M22_enrich"] = d["M22_binding_enrichment"][is_dms]
        out[prefix + "SI06_enrich"] = d["SI06_binding_enrichment"][is_dms]
        out[prefix + "cdrh3_identity"] = d["cdrh3_identity_to_wt"][is_dms]
        out[prefix + "sequences"] = d["sequences"][is_dms]
        out[prefix + "pca_explained_variance"] = pca.explained_variance_ratio_.astype(np.float32)

        ev = pca.explained_variance_ratio_
        log.info(
            "%s: fit on %d DMS rows; PC1=%.1f%% PC2=%.1f%% total(%d)=%.1f%%",
            variant,
            int((~np.isnan(coords).any(axis=1)).sum()),
            100 * ev[0],
            100 * ev[1] if len(ev) > 1 else float("nan"),
            len(ev),
            100 * ev.sum(),
        )

        with open(out_dir / f"pca_dms_{variant}_{emb_type}.pkl", "wb") as fh:
            pickle.dump(pca, fh)

    out["model_variants"] = np.array(list(data.keys()))
    out["dms_dataset"] = np.array([dms_dataset])
    out_path = out_dir / f"per_model_pca_{emb_type}.npz"
    np.savez(out_path, **out)
    log.info("Wrote %s", out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "npz_files",
        nargs="+",
        type=Path,
        help="Per-variant .npz files from extract_embeddings.py",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    data, dms_dataset = load_variants(args.npz_files)
    log.info("All inputs share dms_dataset=%s", dms_dataset)
    for emb_type in EMB_TYPES:
        process_emb_type(emb_type, data, dms_dataset, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
