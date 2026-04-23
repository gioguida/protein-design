#!/usr/bin/env python3
"""Run DPO for one epoch and track validation Spearman at fixed step intervals."""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protein_design.eval import run_scoring_evaluation
from protein_design.utils import init_wandb, setup_train_logger
from protein_design.dpo.loss import dpo_loss, weighted_dpo_loss
from protein_design.dpo.train import (
    _build_dataloader,
    _build_optimizer_and_scheduler,
    _build_scorers,
    _build_split_pair_dataframes,
    _load_validation_spearman_df,
    _resolve_storage_paths,
    _save_checkpoint,
)
from protein_design.dpo.utils import build_full_run_name, load_hydra_runtime_modules

hydra, OmegaConf, HydraConfig, _ = load_hydra_runtime_modules()


def _get_probe_cfg(cfg: Any) -> Any:
    probe_cfg = getattr(cfg, "probe", None)
    if probe_cfg is None:
        return OmegaConf.create(
            {
                "spearman_interval_steps": 50,
            }
        )
    return probe_cfg


def _run_probe(cfg: Any) -> Path:
    if cfg.run.base_name is None:
        cfg.run.base_name = "dpo_spearman_probe"

    full_run_name = build_full_run_name(cfg, timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"))
    storage = _resolve_storage_paths(full_run_name)
    output_dir = storage["run_dir"]

    os.environ.setdefault("WANDB_DIR", str(storage["wandb_dir"]))
    for key in ("scratch_dir", "train_dir", "wandb_dir", "run_dir"):
        storage[key].mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_train_logger(output_dir=output_dir, level_name=str(cfg.logging.level))
    logger.info("Starting DPO first-epoch Spearman probe")
    logger.info("Full run name: %s", full_run_name)
    logger.info("Run directory: %s", output_dir)

    resolved_cfg_path = output_dir / "resolved_config.yaml"
    run_cfg_path = output_dir / "config.yaml"
    OmegaConf.save(cfg, resolved_cfg_path)
    OmegaConf.save(cfg, run_cfg_path)

    wandb_mod, wandb_run = init_wandb(
        cfg,
        output_dir,
        logger,
        run_name=full_run_name,
        group="dpo_probe_first_epoch",
    )

    train_df, val_df, test_df = _build_split_pair_dataframes(cfg)
    if train_df.empty:
        raise ValueError("No DPO train pairs generated after split. Adjust data pairing settings.")
    if val_df.empty:
        raise ValueError("No DPO validation pairs generated after split. Adjust data pairing settings.")
    if test_df.empty:
        raise ValueError("No DPO test pairs generated after split. Adjust data pairing settings.")
    logger.info("Split sizes | train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

    train_loader = _build_dataloader(
        train_df,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        seed=int(cfg.seed),
        num_workers=int(cfg.training.num_workers),
        pin_memory=bool(getattr(cfg.training, "pin_memory", False)),
        persistent_workers=bool(getattr(cfg.training, "persistent_workers", False)),
        prefetch_factor=getattr(cfg.training, "prefetch_factor", None),
    )

    policy, reference = _build_scorers(cfg, logger)
    optimizer, scheduler = _build_optimizer_and_scheduler(cfg, policy)

    val_spearman_df = _load_validation_spearman_df(cfg, logger)
    if val_spearman_df is None:
        raise ValueError(
            "Validation Spearman set could not be loaded. "
            "This probe requires val_pos/val_neg CSVs with enrichment labels."
        )

    probe_cfg = _get_probe_cfg(cfg)
    interval_steps = int(getattr(probe_cfg, "spearman_interval_steps", 50))
    if interval_steps <= 0:
        raise ValueError("probe.spearman_interval_steps must be > 0.")
    spearman_batch_size = int(getattr(cfg.model, "pll_mask_chunk_size", 64))

    history: List[Dict[str, float]] = []
    global_step = 0
    total_batches = max(1, len(train_loader))
    running_loss = 0.0
    valid_batches = 0
    skipped_batches = 0

    def log_spearman(stage: str, train_step: int) -> None:
        nonlocal history
        metrics = run_scoring_evaluation(
            scorer=policy,
            df=val_spearman_df,
            enrichment_col="M22_binding_enrichment_adj",
            batch_size=spearman_batch_size,
            seed=int(cfg.seed),
            scoring_mode="cdr_pll",
        )
        record = {
            "epoch": 1.0,
            "global_step": float(global_step),
            "train_step": float(train_step),
            "train_progress": float(train_step) / float(total_batches),
            "spearman_avg": float(metrics["spearman_avg"]),
            "spearman_avg_pval": float(metrics["spearman_avg_pval"]),
            "spearman_random": float(metrics["spearman_random"]),
            "spearman_random_pval": float(metrics["spearman_random_pval"]),
        }
        history.append(record)
        logger.info(
            "[%s] step=%d/%d | spearman_avg=%.4f (p=%.2e) | spearman_random=%.4f (p=%.2e)",
            stage,
            train_step,
            total_batches,
            record["spearman_avg"],
            record["spearman_avg_pval"],
            record["spearman_random"],
            record["spearman_random_pval"],
        )
        if wandb_run is not None:
            wandb_mod.log(
                {
                    "probe/stage": stage,
                    "probe/epoch": 1.0,
                    "probe/train_step": float(train_step),
                    "probe/train_progress": record["train_progress"],
                    "probe/spearman_avg": record["spearman_avg"],
                    "probe/spearman_avg_pval": record["spearman_avg_pval"],
                    "probe/spearman_random": record["spearman_random"],
                    "probe/spearman_random_pval": record["spearman_random_pval"],
                },
                step=global_step,
            )

    # Track the pretrained policy before any DPO update.
    log_spearman(stage="pretrained", train_step=0)

    loss_name = str(cfg.training.loss)
    log_every_n_steps = int(cfg.logging.log_every_n_steps)
    grad_clip_norm = float(cfg.training.grad_clip_norm)
    beta = float(cfg.training.beta)
    temperature = float(cfg.training.temperature)

    for step, batch in enumerate(train_loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        try:
            if loss_name == "dpo":
                batch_loss = dpo_loss(
                    batch,
                    beta=beta,
                    scorer=policy,
                    reference=reference,
                    policy_use_grad=True,
                )
            elif loss_name == "weighted_dpo":
                batch_loss = weighted_dpo_loss(
                    batch,
                    beta=beta,
                    temperature=temperature,
                    scorer=policy,
                    reference=reference,
                    policy_use_grad=True,
                )
            else:
                raise ValueError(f"Unsupported loss: {loss_name}")
        except ValueError:
            skipped_batches += 1
            continue

        batch_loss.backward()
        if grad_clip_norm > 0:
            clip_grad_norm_(policy.model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        batch_loss_value = float(batch_loss.item())
        running_loss += batch_loss_value
        valid_batches += 1

        if log_every_n_steps > 0 and step % log_every_n_steps == 0:
            logger.info("step=%d train_loss=%.6f", step, batch_loss_value)
            if wandb_run is not None:
                wandb_mod.log(
                    {
                        "train_step/loss": batch_loss_value,
                        "train_step/epoch": 1.0,
                        "train_step/step_in_epoch": float(step),
                    },
                    step=global_step,
                )

        if step % interval_steps == 0:
            log_spearman(stage="interval", train_step=step)

    # Always track the model at the end of epoch 1.
    last_logged_step = int(history[-1]["train_step"]) if history else -1
    if last_logged_step != total_batches:
        log_spearman(stage="end_first_epoch", train_step=total_batches)

    if scheduler is not None:
        scheduler.step()

    avg_train_loss = running_loss / max(1, valid_batches)
    epoch_summary: Dict[str, float] = {
        "epoch": 1.0,
        "global_step": float(global_step),
        "train_loss": float(avg_train_loss),
        "train_batches": float(valid_batches),
        "train_skipped": float(skipped_batches),
        "spearman_points": float(len(history)),
    }
    final_spearman = history[-1] if history else None
    if final_spearman is not None:
        epoch_summary["final_spearman_avg"] = float(final_spearman["spearman_avg"])
        epoch_summary["final_spearman_random"] = float(final_spearman["spearman_random"])

    logger.info(
        "Epoch 1 complete | train_loss=%.6f | train_batches=%d | skipped=%d | spearman_points=%d",
        float(avg_train_loss),
        valid_batches,
        skipped_batches,
        len(history),
    )

    if wandb_run is not None:
        wandb_mod.log(epoch_summary, step=global_step)

    ckpt_dir = output_dir / str(cfg.checkpointing.dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    first_epoch_ckpt = ckpt_dir / "first_epoch.pt"
    _save_checkpoint(
        first_epoch_ckpt,
        epoch=1,
        policy=policy,
        optimizer=optimizer,
        scheduler=scheduler,
        best_val_loss=float("nan"),
    )

    probe_history_df = pd.DataFrame(history)
    probe_history_csv = output_dir / "spearman_first_epoch_history.csv"
    probe_history_df.to_csv(probe_history_csv, index=False)

    summary_payload: Dict[str, Any] = {
        "run_name": full_run_name,
        "interval_steps": int(interval_steps),
        "num_probe_points": int(len(history)),
        "num_train_batches": int(valid_batches),
        "num_skipped_batches": int(skipped_batches),
        "avg_train_loss": float(avg_train_loss),
        "checkpoint": str(first_epoch_ckpt),
        "history_csv": str(probe_history_csv),
        "final_spearman_avg": None,
        "final_spearman_random": None,
    }
    if final_spearman is not None:
        avg_value = float(final_spearman["spearman_avg"])
        rnd_value = float(final_spearman["spearman_random"])
        summary_payload["final_spearman_avg"] = None if math.isnan(avg_value) else avg_value
        summary_payload["final_spearman_random"] = None if math.isnan(rnd_value) else rnd_value

    summary_path = output_dir / "summary_first_epoch_spearman.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    logger.info("Saved first-epoch checkpoint to %s", first_epoch_ckpt)
    logger.info("Saved Spearman history to %s", probe_history_csv)
    logger.info("Saved summary to %s", summary_path)

    if wandb_run is not None:
        wandb_run.summary.update(summary_payload)
        wandb_run.finish()

    return first_epoch_ckpt


@hydra.main(version_base=None, config_path="../conf", config_name="dpo")
def main(cfg: Any) -> None:
    HydraConfig.get().runtime.output_dir
    _run_probe(cfg)


if __name__ == "__main__":
    main()
