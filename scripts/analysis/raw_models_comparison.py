#!/usr/bin/env python3
"""Compare zero-shot raw ESM2 models on C05 and M22 enrichment datasets.

Outputs:
- metrics_summary.csv / metrics_summary.json
- One PLL-vs-enrichment figure per dataset with model-wise subplots.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from protein_design.config import build_model_config
from protein_design.constants import C05_CDRH3
from protein_design.dpo.data_processing import ensure_delta_m22_binding_enrichment
from protein_design.dpo.splitting import (
    build_or_load_cluster_split_membership,
    split_membership_keys,
)
from protein_design.eval import corpus_perplexity, run_scoring_evaluation
from protein_design.model import ESM2Model

MODEL_CONF_PATHS: dict[str, str] = {
    "esm2_8m": "conf/model/esm2_8m.yaml",
    "esm2_35m": "conf/model/esm2_35m.yaml",
    "esm2_150m": "conf/model/esm2_150m.yaml",
    "esm2_650m": "conf/model/esm2_650m.yaml",
}

ENRICHMENT_COL = "M22_binding_enrichment_adj"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--include-model", action="append", required=True)
    parser.add_argument("--include-dataset", action="append", required=True)
    parser.add_argument("--ed2-path", required=True, type=Path)
    parser.add_argument("--ed5-path", required=True, type=Path)
    parser.add_argument("--ed811-path", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--split-mode",
        choices=("full", "val_dpo"),
        default="full",
        help="full: use all rows. val_dpo: use only validation split from DPO cluster split.",
    )
    parser.add_argument("--dpo-split-config", type=Path, default=Path("conf/data/dpo/default.yaml"))
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-cache-dir", type=Path, default=None)
    parser.add_argument("--run-c05-ppl", action="store_true")
    parser.add_argument("--run-dataset-ppl", action="store_true")
    parser.add_argument("--run-spearman", action="store_true")
    parser.add_argument("--run-plots", action="store_true")
    return parser.parse_args()


def _load_model_config(repo_root: Path, model_key: str):
    if model_key not in MODEL_CONF_PATHS:
        raise ValueError(f"Unsupported model key: {model_key}")
    conf_path = repo_root / MODEL_CONF_PATHS[model_key]
    if not conf_path.exists():
        raise FileNotFoundError(f"Model config not found: {conf_path}")
    cfg = OmegaConf.load(conf_path)
    if "model" not in cfg:
        raise ValueError(f"Expected 'model' section in config: {conf_path}")
    return cfg


def _load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    required = {"aa", ENRICHMENT_COL}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset {path} missing required columns: {sorted(missing)}")
    out = df.copy()
    out["aa"] = out["aa"].astype(str).str.strip()
    out = out[out["aa"] != ""].copy()
    out[ENRICHMENT_COL] = pd.to_numeric(out[ENRICHMENT_COL], errors="coerce")
    out = out.reset_index(drop=True)
    return out


def _load_dpo_split_params(config_path: Path) -> dict[str, float | int]:
    if not config_path.exists():
        raise FileNotFoundError(f"DPO split config not found: {config_path}")
    cfg = OmegaConf.load(config_path)
    if "data" not in cfg:
        raise ValueError(f"Expected 'data' section in DPO split config: {config_path}")
    split_cfg = cfg.data.get("split") if "split" in cfg.data else None
    return {
        "train_frac": float(cfg.data.get("train_frac", 0.8)),
        "val_frac": float(cfg.data.get("val_frac", 0.1)),
        "test_frac": float(cfg.data.get("test_frac", 0.1)),
        "stratify_bins": int(getattr(split_cfg, "stratify_bins", 10)),
        "hamming_distance": int(getattr(split_cfg, "hamming_distance", 1)),
    }


def _filter_val_split_like_dpo(
    *,
    dataset_name: str,
    dataset_path: Path,
    df: pd.DataFrame,
    split_params: dict[str, float | int],
    split_seed: int,
    cache_root: Path,
) -> pd.DataFrame:
    required = {"aa", "mut", "num_mut", ENRICHMENT_COL}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{dataset_name} must contain {sorted(required)} for DPO-style split. Missing: {sorted(missing)}"
        )

    base_df = ensure_delta_m22_binding_enrichment(df.copy())
    cache_dir = cache_root / dataset_name.lower()
    cache_dir.mkdir(parents=True, exist_ok=True)
    membership = build_or_load_cluster_split_membership(
        base_df=base_df,
        base_csv_path=dataset_path,
        processed_dir=cache_dir,
        train_frac=float(split_params["train_frac"]),
        val_frac=float(split_params["val_frac"]),
        test_frac=float(split_params["test_frac"]),
        seed=int(split_seed),
        force_rebuild=False,
        positive_threshold=0.0,
        stratify_bins=int(split_params["stratify_bins"]),
        hamming_distance=int(split_params["hamming_distance"]),
    )
    val_keys = set(
        membership.loc[membership["split"] == "val", "split_key"].astype(str).tolist()
    )
    row_keys = split_membership_keys(base_df).astype(str)
    return base_df.loc[row_keys.isin(val_keys)].reset_index(drop=True)


def _dataset_cdr_ppl(scores_pll: np.ndarray, sequences: list[str]) -> float:
    valid = np.isfinite(scores_pll)
    if not np.any(valid):
        return float("nan")
    denom = float(sum(len(sequences[i]) for i in range(len(sequences)) if valid[i]))
    if denom <= 0:
        return float("nan")
    pll_sum = float(np.sum(scores_pll[valid]))
    return float(math.exp(-pll_sum / denom))


def _plot_dataset_subplots(
    dataset_name: str,
    model_order: list[str],
    per_model_xy: dict[str, tuple[np.ndarray, np.ndarray]],
    per_model_spearman: dict[str, tuple[float, float]],
    out_path: Path,
) -> None:
    n_models = len(model_order)
    n_cols = 2
    n_rows = int(math.ceil(n_models / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)
    axes_flat = list(axes.flatten())

    for idx, model_key in enumerate(model_order):
        ax = axes_flat[idx]
        x, y = per_model_xy.get(model_key, (np.array([]), np.array([])))
        rho, pval = per_model_spearman.get(model_key, (float("nan"), float("nan")))
        valid = np.isfinite(x) & np.isfinite(y)
        xv = x[valid]
        yv = y[valid]
        if len(xv) == 0:
            ax.text(0.5, 0.5, "No valid points", ha="center", va="center")
            ax.set_title(model_key)
            ax.set_xlabel("CDR-H3 PLL")
            ax.set_ylabel("M22 binding enrichment")
            ax.grid(alpha=0.2)
            continue

        ax.scatter(xv, yv, s=7, alpha=0.35, color="#4e79a7")
        if len(xv) > 1:
            slope, intercept = np.polyfit(xv, yv, 1)
            xline = np.linspace(float(np.min(xv)), float(np.max(xv)), 200)
            ax.plot(xline, slope * xline + intercept, color="black", linewidth=1.0)

        title = f"{model_key} | n={len(xv)} | rho={rho:.3f} (p={pval:.1e})"
        ax.set_title(title)
        ax.set_xlabel("CDR-H3 PLL")
        ax.set_ylabel("M22 binding enrichment")
        ax.grid(alpha=0.2)

    for idx in range(n_models, len(axes_flat)):
        fig.delaxes(axes_flat[idx])

    fig.suptitle(f"PLL vs M22 binding enrichment - {dataset_name}", fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if not any([args.run_c05_ppl, args.run_dataset_ppl, args.run_spearman, args.run_plots]):
        raise ValueError("Select at least one metric switch.")

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_models = list(dict.fromkeys(args.include_model))
    selected_datasets = list(dict.fromkeys(args.include_dataset))
    dataset_paths = {
        "ED2": args.ed2_path,
        "ED5": args.ed5_path,
        "ED811": args.ed811_path,
    }
    for ds in selected_datasets:
        if ds not in dataset_paths:
            raise ValueError(f"Unsupported dataset key: {ds}")

    split_cache_dir = (
        (args.split_cache_dir if args.split_cache_dir is not None else (out_dir / "split_cache"))
    )
    split_params: dict[str, float | int] | None = None
    if args.split_mode == "val_dpo":
        dpo_cfg_path = args.dpo_split_config
        if not dpo_cfg_path.is_absolute():
            dpo_cfg_path = repo_root / dpo_cfg_path
        split_params = _load_dpo_split_params(dpo_cfg_path)

    dataset_frames: dict[str, pd.DataFrame] = {}
    dataset_row_stats: dict[str, dict[str, int]] = {}
    for ds in selected_datasets:
        full_df = _load_dataset(dataset_paths[ds])
        selected_df = full_df
        if args.split_mode == "val_dpo":
            assert split_params is not None
            selected_df = _filter_val_split_like_dpo(
                dataset_name=ds,
                dataset_path=dataset_paths[ds],
                df=full_df,
                split_params=split_params,
                split_seed=int(args.split_seed),
                cache_root=split_cache_dir,
            )
        dataset_frames[ds] = selected_df
        dataset_row_stats[ds] = {
            "full_rows": int(len(full_df)),
            "selected_rows": int(len(selected_df)),
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []
    plot_data: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
        ds: {} for ds in selected_datasets
    }
    plot_stats: dict[str, dict[str, tuple[float, float]]] = {
        ds: {} for ds in selected_datasets
    }

    for model_key in selected_models:
        model_cfg = _load_model_config(repo_root, model_key)
        model = ESM2Model(build_model_config(model_cfg, device=str(device)))
        model.to(device).eval()

        row: dict[str, Any] = {"model": model_key, "device": str(device)}
        if args.run_c05_ppl:
            row["c05_wt_pseudo_perplexity"] = float(
                corpus_perplexity([C05_CDRH3], scorer=model, cdr_only=True)
            )

        per_dataset_ppl: dict[str, float] = {}
        for ds in selected_datasets:
            df = dataset_frames[ds]
            eval_result = run_scoring_evaluation(
                df=df,
                enrichment_col=ENRICHMENT_COL,
                batch_size=int(args.batch_size),
                scorer=model,
                scoring_mode="cdr_pll",
            )
            scores = np.asarray(eval_result["scores_avg"], dtype=np.float32)
            enrichment = df[ENRICHMENT_COL].to_numpy(dtype=np.float32, copy=False)
            sequences = df["aa"].astype(str).tolist()

            if args.run_dataset_ppl:
                ppl = _dataset_cdr_ppl(scores, sequences)
                row[f"ppl_{ds.lower()}"] = ppl
                per_dataset_ppl[ds] = ppl
            if args.run_spearman:
                row[f"spearman_{ds.lower()}"] = float(eval_result["spearman_avg"])
                row[f"spearman_pval_{ds.lower()}"] = float(eval_result["spearman_avg_pval"])

            if args.run_plots:
                plot_data[ds][model_key] = (scores, enrichment)
                plot_stats[ds][model_key] = (
                    float(eval_result["spearman_avg"]),
                    float(eval_result["spearman_avg_pval"]),
                )

        if args.run_dataset_ppl and per_dataset_ppl:
            vals = [v for v in per_dataset_ppl.values() if np.isfinite(v)]
            row["ppl_avg_ed2_ed5_ed811"] = float(np.mean(vals)) if vals else float("nan")

        rows.append(row)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.run_plots:
        for ds in selected_datasets:
            out_path = out_dir / "plots" / f"pll_vs_enrichment_{ds.lower()}.png"
            _plot_dataset_subplots(
                dataset_name=ds,
                model_order=selected_models,
                per_model_xy=plot_data[ds],
                per_model_spearman=plot_stats[ds],
                out_path=out_path,
            )

    summary_df = pd.DataFrame(rows)
    summary_csv = out_dir / "metrics_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(device),
        "models": selected_models,
        "datasets": {ds: str(dataset_paths[ds]) for ds in selected_datasets},
        "run_switches": {
            "c05_ppl": bool(args.run_c05_ppl),
            "dataset_ppl": bool(args.run_dataset_ppl),
            "spearman": bool(args.run_spearman),
            "plots": bool(args.run_plots),
        },
        "data_selection": {
            "split_mode": args.split_mode,
            "dpo_split_config": str(args.dpo_split_config),
            "split_seed": int(args.split_seed),
            "split_cache_dir": str(split_cache_dir),
            "dataset_rows": dataset_row_stats,
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
