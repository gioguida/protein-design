"""Training loop for ESM2 evotuning (continued MLM pretraining)."""

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Optional

import torch
import wandb
import yaml
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from protein_design.evotuning.data import make_dataloaders
from protein_design.eval import (
    compute_perplexity,
    load_scoring_datasets,
    run_multi_scoring_evaluation,
)
from protein_design.model import ESM2Model
from protein_design.utils import (
    DataConfig,
    ModelConfig,
    RunConfig,
    ScoringConfig,
    TrainingConfig,
    ensure_dir,
)

logger = logging.getLogger(__name__)


def train(
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    training_cfg: TrainingConfig,
    scoring_cfg: ScoringConfig,
    run_cfg: RunConfig,
    run_name: str,
    cfg: Optional[DictConfig] = None,
) -> None:
    """Run the evotuning training loop."""
    # ── Run directory setup ─────────────────────────────────────────────
    run_dir = ensure_dir(f"{run_cfg.train_dir}/{run_name}")
    checkpoint_dir = ensure_dir(f"{run_dir}/checkpoints")

    # Save resolved config snapshot
    snapshot = OmegaConf.to_container(cfg, resolve=True) if cfg is not None else {}
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(snapshot, f, default_flow_style=False, sort_keys=False)
    logger.info("Run directory: %s", run_dir)

    # ── Setup ────────────────────────────────────────────────────────────
    torch.manual_seed(run_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    wandb.init(project=run_cfg.wandb_project, name=run_name, config=snapshot)

    # ── Model ────────────────────────────────────────────────────────────
    model = ESM2Model(model_cfg)

    if run_cfg.finetune:
        logger.info("Loading finetune checkpoint: %s", run_cfg.finetune)
        ckpt = torch.load(run_cfg.finetune, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])

    summary = model.param_summary()
    logger.info(
        "Parameters — total: %s, trainable: %s, frozen: %s",
        f"{summary['total']:,}",
        f"{summary['trainable']:,}",
        f"{summary['frozen']:,}",
    )
    model.to(device)

    # ── Data ─────────────────────────────────────────────────────────────
    train_loader, val_loader = make_dataloaders(
        fasta_path=data_cfg.fasta_path,
        tokenizer_name=model_cfg.esm_model_path,
        max_seq_len=data_cfg.max_seq_len,
        mlm_probability=data_cfg.mlm_probability,
        batch_size=training_cfg.batch_size,
        seed=run_cfg.seed,
    )

    # ── Optimizer + scheduler ────────────────────────────────────────────
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=training_cfg.learning_rate,
        weight_decay=0.01,
    )
    accum_steps = training_cfg.gradient_accumulation_steps
    max_steps = training_cfg.max_steps
    epoch_based_steps = training_cfg.max_epochs * len(train_loader) // accum_steps
    num_training_steps = max_steps if max_steps else epoch_based_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_cfg.warmup_steps,
        num_training_steps=num_training_steps,
    )
    if max_steps:
        logger.info("max_steps=%d — will stop after %d optimizer steps", max_steps, max_steps)

    use_fp16 = training_cfg.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    # ── Scoring data (loaded once, reused at every checkpoint) ─────────
    if scoring_cfg.datasets:
        tokenizer = AutoTokenizer.from_pretrained(model_cfg.esm_model_path)
        scoring_datasets = load_scoring_datasets(
            scoring_cfg.datasets,
            n_samples=scoring_cfg.n_samples,
            seed=run_cfg.seed,
        )
        scoring_batch_size = scoring_cfg.batch_size
        logger.info("Scoring evaluation enabled: %d datasets", len(scoring_datasets))
    else:
        tokenizer = None
        scoring_datasets = None
        scoring_batch_size = scoring_cfg.batch_size
        logger.info("Scoring evaluation disabled (no scoring.datasets in config)")

    # ── Training loop ────────────────────────────────────────────────────
    model.train()
    running_loss = 0.0
    log_steps = 0
    global_step = 0
    optim_step = 0
    best_val_ppl = float("inf")

    save_every_n_steps = training_cfg.save_every_n_steps
    max_epochs = training_cfg.max_epochs
    hit_max_steps = False
    train_start = time.time()
    training_history = []
    scoring_history = []

    for epoch in range(1, max_epochs + 1):
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}")
        for batch in progress:
            global_step += 1
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", enabled=use_fp16):
                outputs = model(**batch)
                loss = outputs.loss / accum_steps

            scaler.scale(loss).backward()
            running_loss += outputs.loss.item()
            log_steps += 1

            if global_step % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                optim_step += 1

            # ── Logging ──────────────────────────────────────────────────
            if global_step % 50 == 0:
                avg_loss = running_loss / log_steps
                lr = scheduler.get_last_lr()[0]
                wandb.log(
                    {"train/loss": avg_loss, "train/lr": lr, "train/epoch": epoch},
                    step=global_step,
                )
                progress.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")
                logger.info(
                    "Epoch %d Step %d — loss: %.4f — lr: %.2e",
                    epoch, global_step, avg_loss, lr,
                )
                training_history.append({
                    "step": global_step,
                    "train_loss": avg_loss,
                    "learning_rate": lr,
                    "epoch": epoch,
                    "wall_time": time.time() - train_start,
                })
                running_loss = 0.0
                log_steps = 0

            # ── Evaluation + checkpoint + scoring ────────────────────────
            if save_every_n_steps and global_step % save_every_n_steps == 0:
                ppl, val_loss = compute_perplexity(model, val_loader, device)
                wandb.log(
                    {"val/loss": val_loss, "val/perplexity": ppl, "train/epoch": epoch},
                    step=global_step,
                )
                logger.info("Step %d — val loss: %.4f — val perplexity: %.2f", global_step, val_loss, ppl)
                training_history.append({
                    "step": global_step,
                    "val_loss": val_loss,
                    "val_perplexity": ppl,
                    "epoch": epoch,
                    "wall_time": time.time() - train_start,
                })

                ckpt_state = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "val_perplexity": ppl,
                }

                ckpt_path = checkpoint_dir / f"step_{global_step}.pt"
                torch.save(ckpt_state, ckpt_path)
                logger.info("Saved checkpoint to %s", ckpt_path)

                # Track best checkpoint
                if ppl < best_val_ppl:
                    best_val_ppl = ppl
                    best_path = run_dir / "best.pt"
                    torch.save(ckpt_state, best_path)
                    logger.info(
                        "New best checkpoint (ppl=%.2f) saved to %s",
                        ppl, best_path,
                    )

                if scoring_datasets is not None:
                    scoring_results = run_multi_scoring_evaluation(
                        model, tokenizer, scoring_datasets,
                        device=device,
                        batch_size=scoring_batch_size,
                        seed=run_cfg.seed,
                    )
                    wandb.log(
                        {f"eval/{k}": v for k, v in scoring_results.items()},
                        step=global_step,
                    )
                    scoring_history.append({"step": global_step, **scoring_results})

                model.train()

            # ── Early termination (max_steps) ────────────────────────────
            if max_steps and optim_step >= max_steps:
                logger.info("Reached max_steps=%d, stopping training.", max_steps)
                hit_max_steps = True
                break

        # ── End of epoch ─────────────────────────────────────────────────
        wandb.log({"train/epoch": epoch}, step=global_step)
        if hit_max_steps:
            break

    # ── Final save ───────────────────────────────────────────────────────
    final_path = checkpoint_dir / "final.pt"

    if len(val_loader.dataset) > 0:
        final_ppl, final_val_loss = compute_perplexity(model, val_loader, device)
        wandb.log(
            {"val/loss": final_val_loss, "val/perplexity": final_ppl},
            step=global_step,
        )
        logger.info("Final val loss: %.4f — val perplexity: %.2f", final_val_loss, final_ppl)
        training_history.append({
            "step": global_step,
            "val_loss": final_val_loss,
            "val_perplexity": final_ppl,
            "wall_time": time.time() - train_start,
        })
    else:
        final_ppl = float("inf")
        logger.info("Skipping final perplexity (empty validation set)")

    final_state = {
        "epoch": max_epochs,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "val_perplexity": final_ppl,
    }
    torch.save(final_state, final_path)
    logger.info("Training complete. Final checkpoint: %s", final_path)

    if final_ppl < best_val_ppl:
        best_val_ppl = final_ppl
        torch.save(final_state, run_dir / "best.pt")
        logger.info("Final checkpoint is also best (ppl=%.2f)", final_ppl)

    # ── Archive best checkpoint to project dir ───────────────────────────
    best_src = run_dir / "best.pt"
    if run_cfg.project_dir and best_src.exists():
        archive_dir = ensure_dir(f"{run_cfg.project_dir}/checkpoints/{run_name}")
        archive_path = archive_dir / "best.pt"
        shutil.copy2(best_src, archive_path)
        logger.info("Archived best checkpoint to %s", archive_path)

    # ── Final scoring evaluation ─────────────────────────────────────────
    if scoring_datasets is not None:
        scoring_results = run_multi_scoring_evaluation(
            model, tokenizer, scoring_datasets,
            device=device,
            batch_size=scoring_batch_size,
            seed=run_cfg.seed,
        )
        wandb.log(
            {f"eval/{k}": v for k, v in scoring_results.items()},
            step=global_step,
        )
        if not scoring_history or scoring_history[-1]["step"] != global_step:
            scoring_history.append({"step": global_step, **scoring_results})

    # ── Save local metrics ──────────────────────────────────────────────
    metrics = {
        "metadata": {
            "total_steps": global_step,
            "total_time_seconds": round(time.time() - train_start, 2),
            "device": str(device),
            "param_summary": summary,
        },
        "training_history": training_history,
        "scoring_history": scoring_history,
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", run_dir / "metrics.json")

    wandb.finish()
