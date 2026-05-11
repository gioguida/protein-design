#!/usr/bin/env python
"""End-to-end TTT + LoRA sweep on C05.

For each of (baseline ESM2, evotuned ESM2, evotuned ESM2 + TTT) score ED2,
ED5, ED8 by sum of masked log-prob at mutated CDR-H3 positions. Picks the
TTT step count that maximizes ED2 Spearman, then reports ED5/ED8 at that
step.

Examples:
    uv run python scripts/run_ttt_sweep.py task=ttt_sweep \\
        finetune=/path/to/evotuned/best.pt

    uv run python scripts/run_ttt_sweep.py task=ttt_sweep \\
        finetune=/path/to/evotuned/best.pt \\
        model=esm2_8m training.max_steps=10 training.snapshot_steps=[5,10]
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from protein_design.config import (
    LoraSpec,
    build_model_config,
    build_run_config,
    build_scoring_config,
    generate_run_name,
)
from protein_design.constants import C05_CDRH3
from protein_design.dms_splitting import (
    dms_config_path_from_cfg,
    resolve_dataset_split,
)
from protein_design.eval import evaluate_spearman, score_sequences_masked_positions
from protein_design.evotuning.config import build_data_config, build_training_config
from protein_design.evotuning.train import run_stage
from protein_design.model import ESM2Model

logger = logging.getLogger(__name__)


DATASET_KEYS = {
    "ED2": ("ed2_m22", None),
    "ED5": ("ed5_m22", None),
    "ED8": ("ed811_m22", 8),  # filter ed811 test split to num_mut == 8
}


def _load_test_dataset(
    dataset_key: str, num_mut_filter: Optional[int], dms_config_path: Path,
    log: logging.Logger, sample_n: Optional[int] = None, seed: int = 42,
) -> pd.DataFrame:
    test_path = resolve_dataset_split(dataset_key, "test", dms_config_path)
    df = pd.read_csv(test_path)
    initial = len(df)
    df = df.dropna(subset=["mut", "M22_binding_enrichment_adj"]).copy()
    df["mut"] = df["mut"].astype(str).str.strip()
    df = df[df["mut"] != ""].reset_index(drop=True)
    if num_mut_filter is not None and "num_mut" in df.columns:
        df = df[df["num_mut"] == num_mut_filter].reset_index(drop=True)
    if sample_n is not None and sample_n > 0 and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=seed).reset_index(drop=True)
    log.info(
        "Loaded %s test split: %d rows (after filter+sample; was %d) from %s",
        dataset_key, len(df), initial, test_path,
    )
    return df


def _spearman_for_model(
    model: ESM2Model, datasets: dict[str, pd.DataFrame], batch_size: int, log: logging.Logger
) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    for name, df in datasets.items():
        scores = score_sequences_masked_positions(
            scorer=model, df=df, wt=C05_CDRH3, batch_size=batch_size,
        )
        enrichment = df["M22_binding_enrichment_adj"].to_numpy(dtype=float)
        rho, pval = evaluate_spearman(scores, enrichment)
        log.info("  %s (n=%d): Spearman rho=%.4f p=%.2e", name, len(df), rho, pval)
        out[name] = float(rho)
    return out


def _build_eval_model(
    base_cfg, device: torch.device, finetune_path: Optional[str], log: logging.Logger,
) -> ESM2Model:
    """Construct a clean (non-LoRA) ESM2Model, optionally loading a finetune ckpt."""
    cfg = replace(base_cfg, lora=None, freeze_lm_head=False)
    m = ESM2Model(cfg)
    if finetune_path:
        log.info("Loading checkpoint: %s", finetune_path)
        state = torch.load(finetune_path, map_location="cpu")
        m.load_state_dict(state["model_state_dict"])
    m.to(device)
    return m


def _build_ttt_eval_model(
    base_cfg, device: torch.device, finetune_path: Optional[str], lora_spec: LoraSpec,
    adapter_path: Path, log: logging.Logger,
) -> ESM2Model:
    """Build (optionally finetuned) base + LoRA + adapter snapshot."""
    cfg = replace(base_cfg, lora=None, freeze_lm_head=False)
    m = ESM2Model(cfg)
    if finetune_path:
        log.info("Loading evotuned base: %s", finetune_path)
        state = torch.load(finetune_path, map_location="cpu")
        m.load_state_dict(state["model_state_dict"])
    m.attach_lora(lora_spec)
    m.freeze_lm_head()
    log.info("Loading LoRA adapter snapshot: %s", adapter_path)
    m.load_state(adapter_path)
    m.to(device)
    return m


def _format_table(
    rows: list[tuple[str, dict[str, float]]],
    columns: list[str],
) -> tuple[str, pd.DataFrame]:
    df = pd.DataFrame(
        [{"model": name, **{c: row[c] for c in columns}} for name, row in rows]
    )
    md_rows = ["| " + " | ".join(["model"] + columns) + " |",
               "| " + " | ".join(["---"] * (len(columns) + 1)) + " |"]
    for name, row in rows:
        cells = [name] + [f"{row[c]:.4f}" for c in columns]
        md_rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(md_rows), df


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("run_ttt_sweep")

    model_cfg = build_model_config(cfg)
    run_cfg = build_run_config(cfg)
    training_cfg = build_training_config(cfg)
    data_cfg = build_data_config(cfg)
    scoring_cfg = build_scoring_config(cfg)

    if not training_cfg.snapshot_steps:
        raise ValueError("ttt_sweep requires training.snapshot_steps to be non-empty.")
    if model_cfg.lora is None:
        raise ValueError("ttt_sweep requires model.lora to be set in the task config.")
    has_evotuned = bool(run_cfg.finetune)
    if not has_evotuned:
        log.warning(
            "run.finetune is unset — evotuned row will mirror baseline, and TTT runs from the HF base."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ------------------------------------------------------------------
    # 1. Build ED2 / ED5 / ED8 test splits via dms_splitting.
    # ------------------------------------------------------------------
    dms_config_path = dms_config_path_from_cfg(cfg)
    # Optional cap on per-dataset eval rows (smoke tests / CPU runs). Pass
    # `+eval_sample_n=200` on the CLI to cap each of ED2/ED5/ED8.
    eval_sample_n = cfg.get("eval_sample_n", None)
    if eval_sample_n is not None:
        eval_sample_n = int(eval_sample_n)
    datasets: dict[str, pd.DataFrame] = {}
    for label, (key, num_mut) in DATASET_KEYS.items():
        datasets[label] = _load_test_dataset(
            key, num_mut, dms_config_path, log,
            sample_n=eval_sample_n, seed=int(cfg.seed),
        )

    eval_batch_size = max(int(scoring_cfg.batch_size or 0), 64)

    # ------------------------------------------------------------------
    # 2. Baseline ESM2 (no finetune, no TTT).
    # ------------------------------------------------------------------
    log.info("=== Evaluating baseline ESM2 ===")
    m_base = _build_eval_model(model_cfg, device, finetune_path=None, log=log)
    base_scores = _spearman_for_model(m_base, datasets, eval_batch_size, log)
    del m_base
    torch.cuda.empty_cache() if device.type == "cuda" else None

    # ------------------------------------------------------------------
    # 3. Evotuned ESM2 (finetune only, no TTT).
    # ------------------------------------------------------------------
    if has_evotuned:
        log.info("=== Evaluating evotuned ESM2 ===")
        m_evo = _build_eval_model(model_cfg, device, run_cfg.finetune, log)
        evo_scores = _spearman_for_model(m_evo, datasets, eval_batch_size, log)
        del m_evo
        torch.cuda.empty_cache() if device.type == "cuda" else None
    else:
        log.info("Skipping evotuned evaluation (no run.finetune); reusing baseline scores.")
        evo_scores = dict(base_scores)

    # ------------------------------------------------------------------
    # 4. TTT sweep: run training (snapshots saved), then evaluate each.
    # ------------------------------------------------------------------
    log.info("=== Running TTT training (snapshots at steps %s) ===",
             sorted(training_cfg.snapshot_steps))
    run_name = generate_run_name(cfg)
    run_stage(
        stage_type="ttt",
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        training_cfg=training_cfg,
        scoring_cfg=scoring_cfg,
        run_cfg=run_cfg,
        run_name=run_name,
        cfg=cfg,
    )

    ttt_run_dir = Path(run_cfg.train_dir) / run_name
    snapshot_dir = ttt_run_dir / "checkpoints"

    ttt_step_scores: dict[int, dict[str, float]] = {}
    for step in sorted(training_cfg.snapshot_steps):
        snap_path = snapshot_dir / f"step_{step}.pt"
        if not snap_path.exists():
            log.warning("Missing TTT snapshot at %s — skipping.", snap_path)
            continue
        log.info("=== Evaluating evotuned+TTT @ step %d ===", step)
        m_ttt = _build_ttt_eval_model(
            model_cfg, device, run_cfg.finetune, model_cfg.lora, snap_path, log,
        )
        ttt_step_scores[step] = _spearman_for_model(m_ttt, datasets, eval_batch_size, log)
        del m_ttt
        torch.cuda.empty_cache() if device.type == "cuda" else None

    if not ttt_step_scores:
        raise RuntimeError("No TTT snapshots evaluated.")

    # ------------------------------------------------------------------
    # 5. Pick best step by ED2 Spearman; build the results table.
    # ------------------------------------------------------------------
    best_step = max(
        ttt_step_scores.keys(),
        key=lambda s: (
            ttt_step_scores[s]["ED2"] if np.isfinite(ttt_step_scores[s]["ED2"]) else -np.inf
        ),
    )
    log.info("Selected best TTT step = %d (argmax ED2 Spearman)", best_step)
    ttt_scores = ttt_step_scores[best_step]

    rows = [
        ("baseline ESM2", base_scores),
        ("evotuned ESM2", evo_scores),
        (f"evotuned + TTT @ step {best_step}", ttt_scores),
    ]
    columns = ["ED2", "ED5", "ED8"]
    md, df = _format_table(rows, columns)

    # Also dump the full per-step sweep so users can inspect ED2 trajectory.
    sweep_df = pd.DataFrame(
        [{"step": s, **{c: ttt_step_scores[s][c] for c in columns}}
         for s in sorted(ttt_step_scores.keys())]
    )

    out_dir = ttt_run_dir
    (out_dir / "results_table.md").write_text(md + "\n")
    df.to_csv(out_dir / "results_table.csv", index=False)
    sweep_df.to_csv(out_dir / "ttt_step_sweep.csv", index=False)
    summary = {
        "ttt_run_dir": str(out_dir),
        "evotuned_ckpt": str(run_cfg.finetune),
        "best_step": int(best_step),
        "baseline": base_scores,
        "evotuned": evo_scores,
        "ttt_per_step": {int(s): v for s, v in ttt_step_scores.items()},
    }
    (out_dir / "results_summary.json").write_text(json.dumps(summary, indent=2))

    log.info("Results table:\n%s", md)
    log.info("Wrote %s, %s, %s, %s",
             out_dir / "results_table.md",
             out_dir / "results_table.csv",
             out_dir / "ttt_step_sweep.csv",
             out_dir / "results_summary.json")


if __name__ == "__main__":
    main()
