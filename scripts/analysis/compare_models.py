#!/usr/bin/env python3
"""Compare fine-tuned models on WT/good-sequence PPL and PLL-vs-enrichment metrics.

Outputs include:
- whole-set Spearman bar plot
- stratified Spearman bar plot (ALL/POS/NEG per dataset)
- AUROC bar plot (per dataset)
- per-dataset grouped violin plots (all models) with enrichment-colored strips
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protein_design.checkpoint_loading import load_scorer_from_checkpoint
from protein_design.constants import C05_CDRH3
from protein_design.dms_splitting import dataset_spec, resolve_dataset_split
from protein_design.eval import corpus_perplexity, run_scoring_evaluation
from protein_design.model import ESM2Model

ENRICHMENT_COL = "M22_binding_enrichment_adj"
GOOD_THRESHOLD_DEFAULT = 5.190013461
DATASET_TO_KEY = {"ED2": "ed2_m22", "ED5": "ed5_m22", "ED811": "ed811_m22", "EXP": "exp"}
CACHE_SCHEMA_VERSION = 1


def _parse_model_spec(spec: str) -> tuple[str, str, str]:
    parts = [p.strip() for p in spec.split("|")]
    if len(parts) != 3 or any(not p for p in parts):
        raise argparse.ArgumentTypeError(
            f"Invalid --model spec {spec!r}. Expected: LABEL|SIZE|CHECKPOINT"
        )
    return parts[0], parts[1], parts[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--model", action="append", required=True, type=_parse_model_spec)
    p.add_argument("--include-dataset", action="append", required=True, choices=list(DATASET_TO_KEY))
    p.add_argument("--dms-config", type=Path, default=Path("conf/data/dms/default.yaml"))
    p.add_argument("--split-name", choices=("train", "val", "test"), default="test")
    p.add_argument("--ed2-dataset-key", default="ed2_m22")
    p.add_argument("--ed5-dataset-key", default="ed5_m22")
    p.add_argument("--ed811-dataset-key", default="ed811_m22")
    p.add_argument("--exp-dataset-key", default="exp")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--good-threshold", type=float, default=GOOD_THRESHOLD_DEFAULT)
    p.add_argument("--force-split-rebuild", action="store_true")
    p.add_argument("--run-wt-ppl", action="store_true")
    p.add_argument("--run-good-ppl", action="store_true")
    p.add_argument("--run-spearman", action="store_true")
    p.add_argument("--run-plots", action="store_true")
    p.add_argument("--run-violin-plots", action="store_true")
    p.add_argument("--cache-root", type=Path, default=None)
    p.add_argument("--use-cache", action="store_true")
    p.add_argument("--write-cache", action="store_true")
    p.add_argument("--plots-only", action="store_true")
    p.add_argument("--force-recompute", action="store_true")
    return p.parse_args()


def _load_model(device: torch.device, checkpoint: str) -> ESM2Model:
    return load_scorer_from_checkpoint(checkpoint, device=str(device))


def _resolve_dataset_path(dataset_name: str, args: argparse.Namespace) -> tuple[Path, str]:
    dms_config = args.dms_config if args.dms_config.is_absolute() else REPO_ROOT / args.dms_config
    dataset_key = str(getattr(args, f"{dataset_name.lower()}_dataset_key"))
    spec = dataset_spec(dataset_key, dms_config)
    if spec.no_split:
        # No-split datasets (e.g. EXP) are evaluated on the full file.
        return spec.path, spec.key_metric_col
    path = resolve_dataset_split(
        dataset_key, args.split_name, dms_config, force=bool(args.force_split_rebuild)
    )
    return path, spec.key_metric_col


def _load_dataset(path: Path, metric_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "aa" not in df.columns:
        raise ValueError(f"Missing aa column in {path}")
    if metric_col not in df.columns:
        raise ValueError(f"Missing metric column {metric_col!r} in {path}")
    out = df.copy()
    out["aa"] = out["aa"].astype(str).str.strip()
    out[ENRICHMENT_COL] = pd.to_numeric(out[metric_col], errors="coerce")
    out = out[(out["aa"] != "") & out[ENRICHMENT_COL].notna()].reset_index(drop=True)
    return out


def _dataset_ppl(scores_pll: np.ndarray, sequences: list[str]) -> float:
    valid = np.isfinite(scores_pll)
    if not np.any(valid):
        return float("nan")
    denom = float(sum(len(sequences[i]) for i in range(len(sequences)) if valid[i]))
    if denom <= 0:
        return float("nan")
    pll_sum = float(np.sum(scores_pll[valid]))
    return float(math.exp(-pll_sum / denom))


def _barplot(
    *,
    out_path: Path,
    title: str,
    ylabel: str,
    models: list[str],
    values: np.ndarray,
    legend_labels: list[str] | None = None,
) -> None:
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(12, 6.5))
    n_models, n_groups = values.shape
    x = np.arange(n_groups)
    width = min(0.8 / max(n_models, 1), 0.22)
    offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * width
    cmap = plt.get_cmap("tab10")
    for i, model_label in enumerate(models):
        bars = ax.bar(x + offsets[i], values[i], width=width, alpha=0.9, color=cmap(i % 10), label=model_label)
        for bar, val in zip(bars, values[i]):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.2g}",
                    ha="center", va="bottom", fontsize=7, rotation=90,
                )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if legend_labels is not None:
        ax.set_xticks(x, legend_labels, rotation=0)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _slugify_model_label(label: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in label)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "model"


def _grouped_violin_plot(
    *,
    out_path: Path,
    dataset_name: str,
    model_labels: list[str],
    per_model_pll: list[np.ndarray],
    per_model_enrichment: list[np.ndarray],
) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(max(9.0, 1.6 * len(model_labels)), 6.2))
    cmap = plt.get_cmap("viridis")
    rng = np.random.default_rng(42)
    x_positions = np.arange(1, len(model_labels) + 1, dtype=float)

    # Track one scatter mappable for a shared colorbar.
    color_mappable = None
    had_data = False

    for i, (label, pll_vals_raw, enr_vals_raw) in enumerate(
        zip(model_labels, per_model_pll, per_model_enrichment)
    ):
        pll_vals = np.asarray(pll_vals_raw, dtype=np.float32)
        enr_vals = np.asarray(enr_vals_raw, dtype=np.float32)
        valid = np.isfinite(pll_vals) & np.isfinite(enr_vals)
        x = float(x_positions[i])
        if not np.any(valid):
            continue

        had_data = True
        y = pll_vals[valid]
        c = enr_vals[valid]

        parts = ax.violinplot(
            [y],
            positions=[x],
            widths=0.75,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor("tab:blue")
            body.set_edgecolor("black")
            body.set_alpha(0.28)

        jitter = rng.uniform(-0.20, 0.20, size=len(y))
        color_mappable = ax.scatter(
            np.full(len(y), x, dtype=np.float32) + jitter,
            y,
            c=c,
            cmap=cmap,
            s=10,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.2,
            zorder=3,
        )

    ax.set_title(f"{dataset_name}: PLL distribution across models")
    ax.set_xlabel("Model")
    ax.set_ylabel("CDR-H3 PLL")
    ax.set_xticks(x_positions, model_labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.22, linewidth=0.8)

    if had_data and color_mappable is not None:
        fig.colorbar(color_mappable, ax=ax, fraction=0.028, pad=0.03, label="Binding enrichment")
    else:
        ax.text(0.5, 0.5, "No finite PLL/enrichment pairs", transform=ax.transAxes, ha="center", va="center")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _default_cache_root() -> Path:
    scratch = os.environ.get("SCRATCH_DIR")
    if scratch:
        return Path(scratch) / "protein-design" / "model_comparison_cache"
    user = os.environ.get("USER", "unknown")
    return Path(f"/cluster/scratch/{user}/protein-design/model_comparison_cache")


def _normalize_checkpoint(checkpoint: str) -> str:
    p = Path(checkpoint)
    if p.exists():
        try:
            return str(p.resolve())
        except Exception:
            return str(p)
    return checkpoint.strip()


def _stable_sha(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def _model_cache_key(model_label: str, model_size: str, checkpoint: str) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "model_label": model_label,
        "model_size": model_size,
        "checkpoint": _normalize_checkpoint(checkpoint),
    }


def _dataset_cache_key(
    *,
    model_fp: str,
    dataset_name: str,
    dataset_path: Path,
    dataset_key: str,
    split_name: str,
) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "model_fp": model_fp,
        "dataset_name": dataset_name,
        "dataset_path": str(dataset_path.resolve()),
        "dataset_key": dataset_key,
        "split_name": split_name,
        "scoring_mode": "cdr_pll",
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def main() -> int:
    args = parse_args()
    if not any([args.run_wt_ppl, args.run_good_ppl, args.run_spearman, args.run_plots, args.run_violin_plots]):
        raise ValueError("Enable at least one run switch.")
    if args.run_violin_plots and not args.run_spearman:
        raise ValueError("--run-violin-plots requires --run-spearman.")
    if args.plots_only and not args.use_cache:
        raise ValueError("--plots-only requires --use-cache.")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = args.cache_root if args.cache_root is not None else _default_cache_root()
    cache_root = cache_root.resolve()
    if args.write_cache:
        cache_root.mkdir(parents=True, exist_ok=True)

    selected_datasets = list(dict.fromkeys(args.include_dataset))
    dataset_frames: dict[str, pd.DataFrame] = {}
    dataset_paths: dict[str, Path] = {}
    dataset_rows: dict[str, int] = {}
    dataset_keys: dict[str, str] = {}
    for ds in selected_datasets:
        path, metric_col = _resolve_dataset_path(ds, args)
        df = _load_dataset(path, metric_col)
        dataset_frames[ds] = df
        dataset_paths[ds] = path
        dataset_rows[ds] = int(len(df))
        dataset_keys[ds] = str(getattr(args, f"{ds.lower()}_dataset_key"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []
    violin_payload: dict[str, list[dict[str, Any]]] = {ds: [] for ds in selected_datasets}

    model_specs = args.model
    for model_label, model_size, checkpoint in model_specs:
        model_display = f"{model_label} ({model_size})"
        model_key_payload = _model_cache_key(model_label, model_size, checkpoint)
        model_fp = _stable_sha(model_key_payload)
        model_cache_dir = cache_root / model_fp
        model_metrics_path = model_cache_dir / "model_metrics.json"
        model_meta_path = model_cache_dir / "model_meta.json"
        model = None
        row: dict[str, Any] = {
            "model_label": model_label,
            "model_size": model_size,
            "checkpoint": checkpoint,
            "device": str(device),
        }

        if args.run_wt_ppl:
            wt_loaded = False
            if args.use_cache and not args.force_recompute and model_metrics_path.exists():
                try:
                    mm = _load_json(model_metrics_path)
                    if mm.get("schema_version") == CACHE_SCHEMA_VERSION and "ppl_wt" in mm:
                        row["ppl_wt"] = float(mm["ppl_wt"])
                        wt_loaded = True
                        print(f"[cache-hit] model wt metric: {model_display}")
                except Exception as exc:
                    print(f"[cache-warn] could not load model cache {model_metrics_path} ({exc})")
            if not wt_loaded:
                if args.plots_only:
                    raise RuntimeError(
                        f"Missing cached WT PPL for {model_display} while --plots-only is enabled."
                    )
                if model is None:
                    model = _load_model(device, checkpoint)
                row["ppl_wt"] = float(corpus_perplexity([C05_CDRH3], scorer=model, cdr_only=True))
                if args.write_cache:
                    _write_json(
                        model_metrics_path,
                        {
                            "schema_version": CACHE_SCHEMA_VERSION,
                            "ppl_wt": float(row["ppl_wt"]),
                        },
                    )
                    _write_json(
                        model_meta_path,
                        {
                            "schema_version": CACHE_SCHEMA_VERSION,
                            "created_at": datetime.now().isoformat(timespec="seconds"),
                            "fingerprint": model_key_payload,
                        },
                    )

        need_scoring = bool(args.run_good_ppl or args.run_spearman or args.run_violin_plots)
        for ds in selected_datasets:
            df = dataset_frames[ds]
            scores: np.ndarray | None = None
            enrichment: np.ndarray | None = None
            cached_metrics: dict[str, Any] | None = None
            eval_result: dict[str, Any] | None = None
            ds_key_payload = _dataset_cache_key(
                model_fp=model_fp,
                dataset_name=ds,
                dataset_path=dataset_paths[ds],
                dataset_key=dataset_keys[ds],
                split_name=args.split_name,
            )
            ds_fp = _stable_sha(ds_key_payload)
            ds_cache_dir = model_cache_dir / ds_fp
            ds_scores_path = ds_cache_dir / "scores.npz"
            ds_metrics_path = ds_cache_dir / "metrics.json"
            ds_meta_path = ds_cache_dir / "meta.json"

            if need_scoring and args.use_cache and not args.force_recompute:
                try:
                    if ds_scores_path.exists() and ds_metrics_path.exists() and ds_meta_path.exists():
                        meta = _load_json(ds_meta_path)
                        if meta.get("schema_version") == CACHE_SCHEMA_VERSION and meta.get("fingerprint") == ds_key_payload:
                            arrays = np.load(ds_scores_path)
                            scores = np.asarray(arrays["scores_avg"], dtype=np.float32)
                            enrichment = np.asarray(arrays["enrichment"], dtype=np.float32)
                            cached_metrics = _load_json(ds_metrics_path)
                            print(f"[cache-hit] {model_display} | {ds}")
                except Exception as exc:
                    print(f"[cache-warn] could not load dataset cache for {model_display}/{ds} ({exc})")

            if need_scoring and scores is None:
                if args.plots_only:
                    raise RuntimeError(
                        f"Missing cached dataset score artifact for {model_display} / {ds} while --plots-only is enabled."
                    )
                if model is None:
                    model = _load_model(device, checkpoint)
                eval_result = run_scoring_evaluation(
                    df=df,
                    enrichment_col=ENRICHMENT_COL,
                    batch_size=int(args.batch_size),
                    scorer=model,
                    scoring_mode="cdr_pll",
                )
                scores = np.asarray(eval_result["scores_avg"], dtype=np.float32)
                enrichment = df[ENRICHMENT_COL].to_numpy(dtype=np.float32, copy=False).astype(np.float32, copy=False)
                cached_metrics = {
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "spearman_avg": float(eval_result["spearman_avg"]),
                    "spearman_avg_pval": float(eval_result["spearman_avg_pval"]),
                    "spearman_avg_pos": float(eval_result["spearman_avg_pos"]),
                    "spearman_avg_pos_pval": float(eval_result["spearman_avg_pos_pval"]),
                    "spearman_avg_neg": float(eval_result["spearman_avg_neg"]),
                    "spearman_avg_neg_pval": float(eval_result["spearman_avg_neg_pval"]),
                    "n_pos": int(eval_result["n_pos"]),
                    "n_neg": int(eval_result["n_neg"]),
                    "auroc": float(eval_result["auroc"]),
                }
                if args.write_cache:
                    ds_cache_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        ds_scores_path,
                        scores_avg=scores,
                        enrichment=np.asarray(enrichment, dtype=np.float32),
                    )
                    _write_json(ds_metrics_path, cached_metrics)
                    _write_json(
                        ds_meta_path,
                        {
                            "schema_version": CACHE_SCHEMA_VERSION,
                            "created_at": datetime.now().isoformat(timespec="seconds"),
                            "fingerprint": ds_key_payload,
                        },
                    )

            sequences = df["aa"].astype(str).tolist()

            if args.run_good_ppl:
                if scores is None or enrichment is None:
                    raise RuntimeError(f"Scores unavailable for {model_display}/{ds}.")
                mask_good = np.isfinite(enrichment) & (enrichment > float(args.good_threshold))
                row[f"n_good_{ds.lower()}"] = int(mask_good.sum())
                row[f"ppl_good_{ds.lower()}"] = _dataset_ppl(scores[mask_good], [s for s, m in zip(sequences, mask_good) if m])
            if args.run_spearman:
                if cached_metrics is None:
                    raise RuntimeError(f"Spearman metrics unavailable for {model_display}/{ds}.")
                row[f"spearman_{ds.lower()}"] = float(cached_metrics["spearman_avg"])
                row[f"spearman_pval_{ds.lower()}"] = float(cached_metrics["spearman_avg_pval"])
                row[f"spearman_pos_{ds.lower()}"] = float(cached_metrics["spearman_avg_pos"])
                row[f"spearman_pos_pval_{ds.lower()}"] = float(cached_metrics["spearman_avg_pos_pval"])
                row[f"spearman_neg_{ds.lower()}"] = float(cached_metrics["spearman_avg_neg"])
                row[f"spearman_neg_pval_{ds.lower()}"] = float(cached_metrics["spearman_avg_neg_pval"])
                row[f"n_pos_{ds.lower()}"] = int(cached_metrics["n_pos"])
                row[f"n_neg_{ds.lower()}"] = int(cached_metrics["n_neg"])
                row[f"auroc_{ds.lower()}"] = float(cached_metrics["auroc"])
                if args.run_violin_plots:
                    if scores is None or enrichment is None:
                        raise RuntimeError(f"Violin payload unavailable for {model_display}/{ds}.")
                    violin_payload[ds].append(
                        {
                            "model_display": model_display,
                            "model_slug": _slugify_model_label(model_display),
                            "scores": np.asarray(scores, dtype=np.float32),
                            "enrichment": np.asarray(enrichment, dtype=np.float32),
                        }
                    )

        rows.append(row)
        had_model = model is not None
        if had_model:
            del model
        if had_model and device.type == "cuda":
            torch.cuda.empty_cache()

    summary_df = pd.DataFrame(rows)
    summary_csv = out_dir / "metrics_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    if args.run_plots and not summary_df.empty:
        model_names = [
            f"{r['model_label']} ({r['model_size']})" for _, r in summary_df.iterrows()
        ]
        if args.run_wt_ppl and "ppl_wt" in summary_df.columns:
            values = summary_df[["ppl_wt"]].to_numpy(dtype=float)
            _barplot(
                out_path=out_dir / "plots" / "hist_ppl_wt.png",
                title="WT Pseudo-Perplexity (lower is better)",
                ylabel="Pseudo-Perplexity",
                models=model_names,
                values=values,
                legend_labels=["WT"],
            )
        if args.run_good_ppl:
            cols = [f"ppl_good_{ds.lower()}" for ds in selected_datasets if f"ppl_good_{ds.lower()}" in summary_df]
            if cols:
                values = summary_df[cols].to_numpy(dtype=float)
                _barplot(
                    out_path=out_dir / "plots" / "hist_ppl_good_by_dataset.png",
                    title=f"Pseudo-Perplexity on Good Sequences (enrichment > {args.good_threshold:.4f})",
                    ylabel="Pseudo-Perplexity",
                    models=model_names,
                    values=values,
                    legend_labels=selected_datasets,
                )
        if args.run_spearman:
            cols = [f"spearman_{ds.lower()}" for ds in selected_datasets if f"spearman_{ds.lower()}" in summary_df]
            if cols:
                values = summary_df[cols].to_numpy(dtype=float)
                _barplot(
                    out_path=out_dir / "plots" / "hist_spearman_by_dataset.png",
                    title="Spearman Correlation: PLL vs Binding Enrichment (higher is better)",
                    ylabel="Spearman rho",
                    models=model_names,
                    values=values,
                    legend_labels=selected_datasets,
                )
            strat_cols: list[str] = []
            strat_labels: list[str] = []
            for ds in selected_datasets:
                ds_l = ds.lower()
                key_all = f"spearman_{ds_l}"
                key_pos = f"spearman_pos_{ds_l}"
                key_neg = f"spearman_neg_{ds_l}"
                if key_all in summary_df.columns:
                    strat_cols.append(key_all)
                    strat_labels.append(f"{ds} ALL")
                if key_pos in summary_df.columns:
                    strat_cols.append(key_pos)
                    strat_labels.append(f"{ds} POS")
                if key_neg in summary_df.columns:
                    strat_cols.append(key_neg)
                    strat_labels.append(f"{ds} NEG")
            if strat_cols:
                values = summary_df[strat_cols].to_numpy(dtype=float)
                _barplot(
                    out_path=out_dir / "plots" / "hist_spearman_stratified_by_dataset.png",
                    title="Stratified Spearman: PLL vs Binding Enrichment (higher is better)",
                    ylabel="Spearman rho",
                    models=model_names,
                    values=values,
                    legend_labels=strat_labels,
                )

            auroc_cols = [f"auroc_{ds.lower()}" for ds in selected_datasets if f"auroc_{ds.lower()}" in summary_df]
            if auroc_cols:
                values = summary_df[auroc_cols].to_numpy(dtype=float)
                _barplot(
                    out_path=out_dir / "plots" / "hist_auroc_by_dataset.png",
                    title="AUROC: Binder vs Non-binder by Dataset (higher is better)",
                    ylabel="AUROC",
                    models=model_names,
                    values=values,
                    legend_labels=selected_datasets,
                )
    if args.run_violin_plots:
        violin_dir = out_dir / "plots" / "violin_pll_vs_enrichment"
        for ds in selected_datasets:
            payload = violin_payload.get(ds, [])
            if not payload:
                continue
            model_labels = [p["model_display"] for p in payload]
            per_model_pll = [p["scores"] for p in payload]
            per_model_enrichment = [p["enrichment"] for p in payload]
            out_path = violin_dir / f"violin_pll_vs_enrichment_{ds.lower()}.png"
            _grouped_violin_plot(
                out_path=out_path,
                dataset_name=ds,
                model_labels=model_labels,
                per_model_pll=per_model_pll,
                per_model_enrichment=per_model_enrichment,
            )

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "datasets": {k: str(v) for k, v in dataset_paths.items()},
        "dataset_rows": dataset_rows,
        "good_threshold": float(args.good_threshold),
        "run_switches": {
            "wt_ppl": bool(args.run_wt_ppl),
            "good_ppl": bool(args.run_good_ppl),
            "spearman": bool(args.run_spearman),
            "plots": bool(args.run_plots),
            "violin_plots": bool(args.run_violin_plots),
            "use_cache": bool(args.use_cache),
            "write_cache": bool(args.write_cache),
            "plots_only": bool(args.plots_only),
            "force_recompute": bool(args.force_recompute),
            "cache_root": str(cache_root),
        },
        "rows": rows,
    }
    summary_json = out_dir / "metrics_summary.json"
    summary_json.write_text(json.dumps(payload, indent=2))

    print(f"[done] wrote {summary_csv}")
    print(f"[done] wrote {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

