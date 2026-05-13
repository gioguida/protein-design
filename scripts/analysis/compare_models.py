#!/usr/bin/env python3
"""Compare fine-tuned models on WT/good-sequence PPL and PLL-vs-enrichment metrics.

Outputs include:
- whole-set Spearman bar plot
- stratified Spearman bar plot (ALL/POS/NEG per dataset)
- AUROC bar plot (per dataset)
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
from transformers import EsmForMaskedLM

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protein_design.config import build_model_config
from protein_design.constants import C05_CDRH3
from protein_design.dms_splitting import dataset_spec, resolve_dataset_split
from protein_design.eval import corpus_perplexity, run_scoring_evaluation
from protein_design.model import ESM2Model

BASE_MODEL_CONF_PATH = "conf/model/esm2_35m.yaml"
ENRICHMENT_COL = "M22_binding_enrichment_adj"
GOOD_THRESHOLD_DEFAULT = 5.190013461
DATASET_TO_KEY = {"ED2": "ed2_m22", "ED5": "ed5_m22", "ED811": "ed811_m22"}


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
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--good-threshold", type=float, default=GOOD_THRESHOLD_DEFAULT)
    p.add_argument("--force-split-rebuild", action="store_true")
    p.add_argument("--run-wt-ppl", action="store_true")
    p.add_argument("--run-good-ppl", action="store_true")
    p.add_argument("--run-spearman", action="store_true")
    p.add_argument("--run-plots", action="store_true")
    return p.parse_args()


def _extract_state_dict(raw: Any) -> dict[str, torch.Tensor]:
    if isinstance(raw, dict):
        for key in ("policy_state_dict", "model_state_dict"):
            if key in raw and isinstance(raw[key], dict):
                return raw[key]
    if isinstance(raw, dict):
        return raw
    raise TypeError(f"Unsupported checkpoint payload type: {type(raw)}")


def _load_pt_into_esm(model: ESM2Model, pt_path: Path) -> None:
    raw = torch.load(pt_path, map_location="cpu")
    state = _extract_state_dict(raw)
    new_state: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        new_state[k[len("model."):] if k.startswith("model.") else k] = v
    missing, unexpected = model.model.load_state_dict(new_state, strict=False)
    non_optional_missing = [
        m for m in missing if not m.startswith("esm.contact_head.") and "position_ids" not in m
    ]
    if non_optional_missing:
        raise RuntimeError(f"Checkpoint missing required keys: {non_optional_missing[:5]}")
    if unexpected:
        print(f"[warn] ignored {len(unexpected)} unexpected keys from {pt_path}")


def _load_model(device: torch.device, checkpoint: str) -> ESM2Model:
    cfg = OmegaConf.load(REPO_ROOT / BASE_MODEL_CONF_PATH)
    model = ESM2Model(build_model_config(cfg, device=str(device)))

    ckpt = Path(checkpoint)
    if ckpt.is_file() and ckpt.suffix == ".pt":
        _load_pt_into_esm(model, ckpt)
    elif ckpt.is_dir():
        if (ckpt / "model.safetensors").exists() or (ckpt / "pytorch_model.bin").exists():
            model.model = EsmForMaskedLM.from_pretrained(str(ckpt))
        else:
            pt_path = next((ckpt / n for n in ("best.pt", "final.pt") if (ckpt / n).exists()), None)
            if pt_path is None:
                raise FileNotFoundError(f"No HF weights, best.pt, or final.pt found at {ckpt}")
            _load_pt_into_esm(model, pt_path)
    elif ckpt.is_absolute():
        raise FileNotFoundError(f"Checkpoint path does not exist: {ckpt}")
    else:
        # Allow HF model IDs.
        model.model = EsmForMaskedLM.from_pretrained(checkpoint)

    model.to(device).eval()
    return model


def _resolve_dataset_path(dataset_name: str, args: argparse.Namespace) -> tuple[Path, str]:
    dms_config = args.dms_config if args.dms_config.is_absolute() else REPO_ROOT / args.dms_config
    dataset_key = str(getattr(args, f"{dataset_name.lower()}_dataset_key"))
    path = resolve_dataset_split(
        dataset_key, args.split_name, dms_config, force=bool(args.force_split_rebuild)
    )
    return path, dataset_spec(dataset_key, dms_config).key_metric_col


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


def main() -> int:
    args = parse_args()
    if not any([args.run_wt_ppl, args.run_good_ppl, args.run_spearman, args.run_plots]):
        raise ValueError("Enable at least one run switch.")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_datasets = list(dict.fromkeys(args.include_dataset))
    dataset_frames: dict[str, pd.DataFrame] = {}
    dataset_paths: dict[str, Path] = {}
    dataset_rows: dict[str, int] = {}
    for ds in selected_datasets:
        path, metric_col = _resolve_dataset_path(ds, args)
        df = _load_dataset(path, metric_col)
        dataset_frames[ds] = df
        dataset_paths[ds] = path
        dataset_rows[ds] = int(len(df))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []

    model_specs = args.model
    for model_label, model_size, checkpoint in model_specs:
        model = _load_model(device, checkpoint)
        row: dict[str, Any] = {
            "model_label": model_label,
            "model_size": model_size,
            "checkpoint": checkpoint,
            "device": str(device),
        }

        if args.run_wt_ppl:
            row["ppl_wt"] = float(corpus_perplexity([C05_CDRH3], scorer=model, cdr_only=True))

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

            if args.run_good_ppl:
                mask_good = np.isfinite(enrichment) & (enrichment > float(args.good_threshold))
                row[f"n_good_{ds.lower()}"] = int(mask_good.sum())
                row[f"ppl_good_{ds.lower()}"] = _dataset_ppl(scores[mask_good], [s for s, m in zip(sequences, mask_good) if m])
            if args.run_spearman:
                row[f"spearman_{ds.lower()}"] = float(eval_result["spearman_avg"])
                row[f"spearman_pval_{ds.lower()}"] = float(eval_result["spearman_avg_pval"])
                row[f"spearman_pos_{ds.lower()}"] = float(eval_result["spearman_avg_pos"])
                row[f"spearman_pos_pval_{ds.lower()}"] = float(eval_result["spearman_avg_pos_pval"])
                row[f"spearman_neg_{ds.lower()}"] = float(eval_result["spearman_avg_neg"])
                row[f"spearman_neg_pval_{ds.lower()}"] = float(eval_result["spearman_avg_neg_pval"])
                row[f"n_pos_{ds.lower()}"] = int(eval_result["n_pos"])
                row[f"n_neg_{ds.lower()}"] = int(eval_result["n_neg"])
                row[f"auroc_{ds.lower()}"] = float(eval_result["auroc"])

        rows.append(row)
        del model
        if device.type == "cuda":
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

