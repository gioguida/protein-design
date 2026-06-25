#!/usr/bin/env python3
"""Seed the model-comparison cache from existing per-model PLL artifacts (no GPU).

`compare_models.py --use-cache --plots-only` reads a scratch cache laid out as:

    <cache_root>/<model_fp>/model_metrics.json         # {schema_version, ppl_wt}
    <cache_root>/<model_fp>/<ds_fp>/scores.npz          # scores_avg, enrichment
    <cache_root>/<model_fp>/<ds_fp>/metrics.json        # spearman/auroc/...
    <cache_root>/<model_fp>/<ds_fp>/meta.json           # {fingerprint: ds_key_payload}

where <model_fp>/<ds_fp> are sha1 fingerprints of the run config. Normally that
cache is produced on the GPU by compare_models_build_cache.sh. This script
produces the *same* files on CPU by reusing the analysis PLL CSVs at
    $ANALYSIS_DIR/<analysis_key>/pll/<dataset_key>.csv   (cols: aa, pll)
so a model that already has PLL artifacts never needs to be re-scored.

It reproduces compare_models.py's CDR-PLL scoring path exactly:
  * the same resolved test-split dataframe (via compare_models._resolve_dataset_path),
  * scores_avg aligned positionally to df rows by merging PLL on the `aa` string,
  * the same Spearman / stratified-Spearman / AUROC metrics (protein_design.eval),
  * WT pseudo-perplexity derived as exp(-PLL(C05_CDRH3)/len) IF the WT CDR-H3 is
    present in one of the model's PLL CSVs (otherwise that model keeps RUN_WT_PPL
    on the GPU; pass --skip-wt or set RUN_WT_PPL=0 in the .sh).

Fingerprints are computed by importing compare_models.py's own helpers, so the
cache keys match the GPU run config exactly.

Usage (mirror the dataset args of compare_models.sh, then map each model to its
analysis key with --pair "LABEL|SIZE|CHECKPOINT::ANALYSIS_KEY"; an empty
ANALYSIS_KEY skips that model so it still gets scored on the GPU):

    uv run python scripts/analysis/seed_comparison_cache_from_pll.py \
        --dms-config conf/data/dms/default.yaml --split-name test \
        --include-dataset ED2 --include-dataset ED5 \
        --include-dataset ED811 --include-dataset EXP \
        --good-threshold 5.190013461 \
        --pair "vanilla|650M|facebook/esm2_t33_650M_UR50D::vanilla_650m" \
        --pair "evotuned|650M|/.../oas_full_evo_650m/oas_full_evo_650m.pt::evo_650m"

Then run compare_models.sh with PLOTS_ONLY=1 (and USE_CACHE=1) to read it back.
Run this under the same environment as the comparison (common_setup.sh) so
SCRATCH_DIR / ANALYSIS_DIR resolve identically, or pass --cache-root explicitly.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from protein_design.analysis import registry  # noqa: E402
from protein_design.constants import C05_CDRH3  # noqa: E402
from protein_design.eval import (  # noqa: E402
    ENRICHMENT_BIMODAL_THRESHOLD,
    evaluate_spearman,
    evaluate_stratified_spearman,
)


def _import_compare_models() -> ModuleType:
    """Import scripts/analysis/compare_models.py so we reuse its exact helpers."""
    path = REPO_ROOT / "scripts" / "analysis" / "compare_models.py"
    spec = importlib.util.spec_from_file_location("compare_models", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_pair(spec: str) -> tuple[str, str]:
    """'LABEL|SIZE|CHECKPOINT::ANALYSIS_KEY' -> (model_spec, analysis_key)."""
    if "::" not in spec:
        raise argparse.ArgumentTypeError(
            f"--pair {spec!r} must be 'LABEL|SIZE|CHECKPOINT::ANALYSIS_KEY'"
        )
    model_spec, analysis_key = spec.rsplit("::", 1)
    return model_spec.strip(), analysis_key.strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pair", action="append", required=True, type=_parse_pair,
                   help="'LABEL|SIZE|CHECKPOINT::ANALYSIS_KEY' (repeatable). "
                        "Empty ANALYSIS_KEY skips the model (GPU-scored later).")
    p.add_argument("--include-dataset", action="append", required=True,
                   choices=["ED2", "ED5", "ED811", "EXP"])
    p.add_argument("--dms-config", type=Path, default=Path("conf/data/dms/default.yaml"))
    p.add_argument("--split-name", choices=("train", "val", "test"), default="test")
    p.add_argument("--ed2-dataset-key", default="ed2_m22")
    p.add_argument("--ed5-dataset-key", default="ed5_m22")
    p.add_argument("--ed811-dataset-key", default="ed811_m22")
    p.add_argument("--exp-dataset-key", default="exp")
    p.add_argument("--good-threshold", type=float, default=5.190013461)
    p.add_argument("--cache-root", type=Path, default=None,
                   help="Defaults to compare_models' own default (SCRATCH-based).")
    p.add_argument("--force-split-rebuild", action="store_true")
    p.add_argument("--skip-wt", action="store_true",
                   help="Do not derive WT PPL even if the WT CDR-H3 is present.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite cache entries that already exist.")
    return p.parse_args()


def _load_pll_map(analysis_key: str, dataset_key: str) -> dict[str, float]:
    """{stripped aa -> pll} from $ANALYSIS_DIR/<analysis_key>/pll/<dataset_key>.csv."""
    path = registry.artifact_path(analysis_key, "pll", f"{dataset_key}.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"PLL artifact missing: {path}\nExtract it first: "
            f"sbatch bash_scripts/extract.sbatch --what pll "
            f"--model {analysis_key} --dataset {dataset_key}"
        )
    df = pd.read_csv(path)
    if "aa" not in df.columns or "pll" not in df.columns:
        raise ValueError(f"{path} must have columns [aa, pll], got {list(df.columns)}")
    out: dict[str, float] = {}
    for aa, pll in zip(df["aa"].astype(str), df["pll"].astype(float)):
        key = aa.strip()
        if key not in out:  # PLL CSVs are deduped, but be defensive
            out[key] = float(pll)
    return out


def _derive_wt_ppl(analysis_key: str, dataset_keys: list[str]) -> float | None:
    """exp(-PLL(C05_CDRH3)/len) using whichever PLL CSV contains the WT CDR-H3."""
    wt = C05_CDRH3.strip()
    for ds_key in dataset_keys:
        path = registry.artifact_path(analysis_key, "pll", f"{ds_key}.csv")
        if not path.exists():
            continue
        df = pd.read_csv(path)
        hit = df[df["aa"].astype(str).str.strip() == wt]
        if not hit.empty:
            pll_wt = float(hit["pll"].iloc[0])
            return float(math.exp(-pll_wt / len(wt)))
    return None


def main() -> int:
    args = parse_args()
    cm = _import_compare_models()

    # Namespace shaped for compare_models._resolve_dataset_path.
    ns = argparse.Namespace(
        dms_config=args.dms_config,
        split_name=args.split_name,
        ed2_dataset_key=args.ed2_dataset_key,
        ed5_dataset_key=args.ed5_dataset_key,
        ed811_dataset_key=args.ed811_dataset_key,
        exp_dataset_key=args.exp_dataset_key,
        force_split_rebuild=args.force_split_rebuild,
    )

    cache_root = (args.cache_root or cm._default_cache_root()).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    print(f"[cache] root: {cache_root}")

    selected = list(dict.fromkeys(args.include_dataset))
    dataset_keys = [str(getattr(ns, f"{ds.lower()}_dataset_key")) for ds in selected]

    # Resolve each dataset once (same split file + dataframe the GPU run uses).
    resolved: dict[str, tuple[Path, pd.DataFrame]] = {}
    for ds in selected:
        path, metric_col = cm._resolve_dataset_path(ds, ns)
        df = cm._load_dataset(path, metric_col)
        resolved[ds] = (path.resolve(), df)
        print(f"[data] {ds}: {len(df)} rows  <-  {path}")

    n_written = n_skipped = 0
    for model_spec, analysis_key in args.pair:
        model_label, model_size, checkpoint = cm._parse_model_spec(model_spec)
        model_display = f"{model_label} ({model_size})"
        if not analysis_key:
            print(f"[skip-model] {model_display}: no analysis key (score on GPU)")
            continue

        model_key_payload = cm._model_cache_key(model_label, model_size, checkpoint)
        model_fp = cm._stable_sha(model_key_payload)
        model_cache_dir = cache_root / model_fp
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[model] {model_display}  ->  analysis_key={analysis_key}  fp={model_fp}")

        # WT PPL (per-model scalar) — needed only if the .sh keeps RUN_WT_PPL=1.
        model_metrics_path = model_cache_dir / "model_metrics.json"
        if not args.skip_wt and (args.force or not model_metrics_path.exists()):
            ppl_wt = _derive_wt_ppl(analysis_key, dataset_keys)
            if ppl_wt is None:
                print(f"  [wt] WT CDR-H3 not found in PLL CSVs — skipping WT PPL. "
                      f"Set RUN_WT_PPL=0 or run that model on the GPU.")
            else:
                model_metrics_path.write_text(json.dumps(
                    {"schema_version": cm.CACHE_SCHEMA_VERSION, "ppl_wt": ppl_wt}, indent=2))
                (model_cache_dir / "model_meta.json").write_text(json.dumps({
                    "schema_version": cm.CACHE_SCHEMA_VERSION,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "fingerprint": model_key_payload,
                    "source": "seeded_from_pll",
                }, indent=2))
                print(f"  [wt] ppl_wt={ppl_wt:.4f}")

        for ds in selected:
            dataset_path, df = resolved[ds]
            dataset_key = str(getattr(ns, f"{ds.lower()}_dataset_key"))
            ds_key_payload = cm._dataset_cache_key(
                model_fp=model_fp, dataset_name=ds, dataset_path=dataset_path,
                dataset_key=dataset_key, split_name=ns.split_name,
            )
            ds_fp = cm._stable_sha(ds_key_payload)
            ds_cache_dir = model_cache_dir / ds_fp
            scores_path = ds_cache_dir / "scores.npz"
            if scores_path.exists() and not args.force:
                print(f"  [skip] {ds}: cache exists ({ds_fp}) — pass --force to overwrite")
                n_skipped += 1
                continue

            pll_map = _load_pll_map(analysis_key, dataset_key)
            seqs = df["aa"].astype(str).str.strip().tolist()
            scores = np.array([pll_map.get(s, np.nan) for s in seqs], dtype=np.float32)
            enrichment = df[cm.ENRICHMENT_COL].to_numpy(dtype=np.float32)

            matched = int(np.isfinite(scores).sum())
            coverage = matched / max(len(seqs), 1)
            if matched == 0:
                raise RuntimeError(
                    f"{model_display}/{ds}: 0/{len(seqs)} sequences found in "
                    f"{analysis_key}/pll/{dataset_key}.csv — the PLL artifact and the "
                    f"resolved test split do not overlap. Check that both were built "
                    f"from the same raw data + split seed."
                )
            if coverage < 0.999:
                print(f"  [warn] {ds}: only {matched}/{len(seqs)} "
                      f"({coverage:.1%}) sequences matched the PLL CSV — "
                      f"missing rows scored as NaN (dropped from metrics).")

            rho_avg, pval_avg = evaluate_spearman(scores, enrichment)
            strat = evaluate_stratified_spearman(
                scores, enrichment, ENRICHMENT_BIMODAL_THRESHOLD)
            metrics = {
                "schema_version": cm.CACHE_SCHEMA_VERSION,
                "spearman_avg": float(rho_avg),
                "spearman_avg_pval": float(pval_avg),
                "spearman_avg_pos": float(strat["spearman_pos"]),
                "spearman_avg_pos_pval": float(strat["spearman_pos_pval"]),
                "spearman_avg_neg": float(strat["spearman_neg"]),
                "spearman_avg_neg_pval": float(strat["spearman_neg_pval"]),
                "n_pos": int(strat["n_pos"]),
                "n_neg": int(strat["n_neg"]),
                "auroc": float(strat["auroc"]),
            }

            ds_cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(scores_path, scores_avg=scores, enrichment=enrichment)
            (ds_cache_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
            (ds_cache_dir / "meta.json").write_text(json.dumps({
                "schema_version": cm.CACHE_SCHEMA_VERSION,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "fingerprint": ds_key_payload,
                "source": "seeded_from_pll",
                "pll_coverage": coverage,
            }, indent=2))
            n_written += 1
            print(f"  [ok]   {ds}: spearman={metrics['spearman_avg']:.4f} "
                  f"auroc={metrics['auroc']:.4f}  ({ds_fp})")

    print(f"\n[done] wrote {n_written} dataset cache entries "
          f"({n_skipped} already present). cache_root={cache_root}")
    print("Now run compare_models.sh with USE_CACHE=1 PLOTS_ONLY=1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
