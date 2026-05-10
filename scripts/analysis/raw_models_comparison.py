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
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protein_design.config import build_model_config
from protein_design.constants import C05_CDRH3
from protein_design.dms_splitting import dataset_spec, resolve_dataset_split
from protein_design.eval import corpus_perplexity, run_scoring_evaluation
from protein_design.model import ESM2Model

MODEL_CONF_PATHS: dict[str, str] = {
    "esm2_8m": "conf/model/esm2_8m.yaml",
    "esm2_35m": "conf/model/esm2_35m.yaml",
    "esm2_150m": "conf/model/esm2_150m.yaml",
    "esm2_650m": "conf/model/esm2_650m.yaml",
}

ENRICHMENT_COL = "M22_binding_enrichment_adj"
DATASET_TO_DMS_KEY = {
    "ED2": "ed2_m22",
    "ED5": "ed5_m22",
    "ED811": "ed811_m22",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--include-model", action="append", required=True)
    parser.add_argument("--include-dataset", action="append", required=True)
    parser.add_argument("--ed2-path", type=Path, default=None)
    parser.add_argument("--ed5-path", type=Path, default=None)
    parser.add_argument("--ed811-path", type=Path, default=None)
    parser.add_argument("--dms-config", type=Path, default=Path("conf/data/dms/default.yaml"))
    parser.add_argument("--split-name", choices=("train", "val", "test"), default="test")
    parser.add_argument("--ed2-dataset-key", default="ed2_m22")
    parser.add_argument("--ed5-dataset-key", default="ed5_m22")
    parser.add_argument("--ed811-dataset-key", default="ed811_m22")
    parser.add_argument(
        "--max-dataset-rows",
        type=int,
        default=None,
        help="Optional cap per resolved dataset split. Sampling is deterministic and stratified.",
    )
    parser.add_argument("--subsample-seed", type=int, default=42)
    parser.add_argument("--subsample-stratify-bins", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--split-mode",
        choices=("dms_split", "explicit_csv"),
        default="dms_split",
        help="dms_split: resolve shared DMS split CSVs. explicit_csv: use --ed*-path inputs.",
    )
    parser.add_argument("--force-split-rebuild", action="store_true")
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


def _load_dataset(path: Path, enrichment_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    required = {"aa", enrichment_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset {path} missing required columns: {sorted(missing)}")
    out = df.copy()
    out["aa"] = out["aa"].astype(str).str.strip()
    out = out[out["aa"] != ""].copy()
    out[ENRICHMENT_COL] = pd.to_numeric(out[enrichment_col], errors="coerce")
    out = out.reset_index(drop=True)
    return out


def _metric_strata(values: pd.Series, bins: int) -> pd.Series:
    clean = pd.to_numeric(values, errors="coerce")
    fill_value = clean.median()
    if pd.isna(fill_value):
        fill_value = 0.0
    clean = clean.fillna(fill_value)
    bins = max(1, min(int(bins), len(clean)))
    if bins == 1:
        return pd.Series(np.zeros(len(clean), dtype=np.int64), index=clean.index)
    try:
        return pd.qcut(clean, q=bins, labels=False, duplicates="drop").fillna(0).astype(int)
    except ValueError:
        return pd.Series(np.zeros(len(clean), dtype=np.int64), index=clean.index)


def _subsample_dataset(
    df: pd.DataFrame,
    *,
    max_rows: int | None,
    seed: int,
    stratify_bins: int,
) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(df) <= max_rows:
        return df.reset_index(drop=True)

    rng = np.random.default_rng(int(seed))
    working = df.copy()
    working["_metric_bin"] = _metric_strata(working[ENRICHMENT_COL], int(stratify_bins))
    group_cols = ["_metric_bin"]
    if "num_mut" in working.columns:
        group_cols = ["num_mut", "_metric_bin"]

    grouped = working.groupby(group_cols, dropna=False, sort=True)
    group_sizes = grouped.size()
    raw_targets = group_sizes / len(working) * int(max_rows)
    targets = np.floor(raw_targets).astype(int)
    for idx in group_sizes[group_sizes > 0].index:
        if targets.loc[idx] == 0:
            targets.loc[idx] = 1

    overflow = int(targets.sum() - int(max_rows))
    if overflow > 0:
        removable = (targets - 1).sort_values(ascending=False)
        for idx, can_remove in removable.items():
            if overflow <= 0:
                break
            delta = min(int(can_remove), overflow)
            targets.loc[idx] -= delta
            overflow -= delta
    elif overflow < 0:
        remaining = -overflow
        remainders = (raw_targets - np.floor(raw_targets)).sort_values(ascending=False)
        for idx in remainders.index:
            if remaining <= 0:
                break
            capacity = int(group_sizes.loc[idx] - targets.loc[idx])
            if capacity <= 0:
                continue
            targets.loc[idx] += 1
            remaining -= 1

    selected: list[int] = []
    for group_key, group in grouped:
        n = int(targets.loc[group_key])
        if n <= 0:
            continue
        selected.extend(
            rng.choice(group.index.to_numpy(), size=min(n, len(group)), replace=False).tolist()
        )
    if len(selected) > max_rows:
        selected = rng.choice(np.array(selected), size=int(max_rows), replace=False).tolist()
    return df.loc[selected].reset_index(drop=True)


def _resolve_dataset_input(
    *,
    dataset_name: str,
    args: argparse.Namespace,
    repo_root: Path,
) -> tuple[Path, str]:
    dms_config = args.dms_config if args.dms_config.is_absolute() else repo_root / args.dms_config
    if args.split_mode == "dms_split":
        key_attr = f"{dataset_name.lower()}_dataset_key"
        dataset_key = str(getattr(args, key_attr))
        path = resolve_dataset_split(
            dataset_key,
            args.split_name,
            dms_config,
            force=bool(args.force_split_rebuild),
        )
        return path, dataset_spec(dataset_key, dms_config).key_metric_col

    path_attr = f"{dataset_name.lower()}_path"
    path = getattr(args, path_attr)
    if path is None:
        raise ValueError(f"--{dataset_name.lower()}-path is required when --split-mode explicit_csv")
    return path, ENRICHMENT_COL


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
    for ds in selected_datasets:
        if ds not in DATASET_TO_DMS_KEY:
            raise ValueError(f"Unsupported dataset key: {ds}")

    dataset_frames: dict[str, pd.DataFrame] = {}
    dataset_row_stats: dict[str, dict[str, int]] = {}
    dataset_paths: dict[str, Path] = {}
    dataset_metric_cols: dict[str, str] = {}
    for ds in selected_datasets:
        dataset_path, metric_col = _resolve_dataset_input(
            dataset_name=ds,
            args=args,
            repo_root=repo_root,
        )
        selected_df = _load_dataset(dataset_path, metric_col)
        full_rows = int(len(selected_df))
        selected_df = _subsample_dataset(
            selected_df,
            max_rows=args.max_dataset_rows,
            seed=int(args.subsample_seed),
            stratify_bins=int(args.subsample_stratify_bins),
        )
        dataset_frames[ds] = selected_df
        dataset_paths[ds] = dataset_path
        dataset_metric_cols[ds] = metric_col
        dataset_row_stats[ds] = {
            "resolved_rows": full_rows,
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
            "dms_config": str(args.dms_config),
            "split_name": str(args.split_name),
            "max_dataset_rows": args.max_dataset_rows,
            "subsample_seed": int(args.subsample_seed),
            "subsample_stratify_bins": int(args.subsample_stratify_bins),
            "metric_columns": dataset_metric_cols,
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
