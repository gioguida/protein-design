"""Unified training loop for ESM2 stages (evotuning + TTT).

A single `run_stage` entry point dispatches on `stage_type` to:
- pick the dataloader (corpus FASTA vs. repeated masked copies of one seq)
- pick the optimizer (AdamW + linear warmup vs. SGD, no scheduler)
- pick the eval cadence (every save_every_n_steps vs. once at the end)

Everything else — run-dir setup, config snapshot, wandb, model construction,
finetune load, scoring, metrics.json, checkpoint archiving — is shared.
"""

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from Bio import SeqIO
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling, get_linear_schedule_with_warmup

from protein_design.constants import C05_CDRH3
from protein_design.eval import (
    corpus_perplexity,
    compute_perplexity,
    load_scoring_datasets,
    run_multi_scoring_evaluation,
)
from protein_design.config import ModelConfig, RunConfig, ScoringConfig
from protein_design.evotuning.config import DataConfig, TrainingConfig
from protein_design.evotuning.data import build_train_loader, make_dataloaders
from protein_design.model import ESM2Model
from protein_design.utils import ensure_dir, init_wandb, setup_train_logger

# Module-level logger for orchestration messages (before run_dir exists).
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TTT helpers (private — TTT is the only strategy that needs these)
# ---------------------------------------------------------------------------


def _load_single_sequence(fasta_path: str, log: logging.Logger) -> str:
    record = next(SeqIO.parse(fasta_path, "fasta"))
    seq = str(record.seq)
    log.info("Loaded sequence (%d aa) from %s: %s", len(seq), fasta_path, record.id)
    return seq


def _make_masked_batch(
    tokenized: dict[str, list[int]],
    batch_size: int,
    collator: DataCollatorForLanguageModeling,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    samples = [
        {k: torch.tensor(v) for k, v in tokenized.items()}
        for _ in range(batch_size)
    ]
    batch = collator(samples)
    return {k: v.to(device) for k, v in batch.items()}


# ---------------------------------------------------------------------------
# Strategy: evotuning (continued MLM pretraining on a corpus)
# ---------------------------------------------------------------------------


def _load_resume_checkpoint(
    path: str,
    model: ESM2Model,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    log: logging.Logger,
) -> tuple[int, int, int, float]:
    """Restore full training state from an evotuning checkpoint.

    Returns (epoch, global_step, samples_seen_this_epoch, best_val_ppl).
    The caller is responsible for the epoch-rollover decision when
    samples_seen_this_epoch >= full_train_len.
    """
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    epoch = int(ckpt.get("epoch", 1))
    global_step = int(ckpt.get("global_step", 0))
    samples_seen = int(ckpt.get("samples_seen", 0))
    best_val_ppl = float(ckpt.get("best_val_ppl", ckpt.get("val_perplexity", float("inf"))))
    log.info(
        "Loaded resume checkpoint %s: epoch=%d global_step=%d samples_seen=%d best_val_ppl=%.4f",
        path, epoch, global_step, samples_seen, best_val_ppl,
    )
    return epoch, global_step, samples_seen, best_val_ppl


def _train_evotuning(
    model: ESM2Model,
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    training_cfg: TrainingConfig,
    scoring_cfg: ScoringConfig,
    run_cfg: RunConfig,
    run_dir: Path,
    checkpoint_dir: Path,
    device: torch.device,
    train_start: float,
    log: logging.Logger,
    wandb_mod: Optional[Any],
    log_every_n_steps: int,
) -> tuple[list, list, int, Optional[Path]]:
    """Run corpus-MLM training. Returns (training_history, scoring_history,
    global_step, best_ckpt_path_or_None, final_metrics)."""
    accum_steps = training_cfg.gradient_accumulation_steps
    if training_cfg.save_every_n_steps:
        # Resume relies on optim_step = global_step // accum_steps being exact.
        assert training_cfg.save_every_n_steps % accum_steps == 0, (
            "save_every_n_steps must be divisible by gradient_accumulation_steps "
            f"(got {training_cfg.save_every_n_steps} % {accum_steps} != 0) "
            "to keep optimizer-step accounting exact across resume."
        )

    # Build val/test loaders + cache the train_dataset+collator so that we can
    # rebuild train_loader cheaply per epoch without rescanning the FASTA.
    # The initial train_loader uses skip=0,seed=run_cfg.seed+1 — it is discarded
    # and rebuilt below once we know start_epoch / start_samples_seen.
    _initial_train_loader, val_loader, test_loader, train_dataset, collator, full_train_len = make_dataloaders(
        fasta_path=data_cfg.fasta_path,
        max_seq_len=data_cfg.max_seq_len,
        mlm_probability=data_cfg.mlm_probability,
        batch_size=training_cfg.batch_size,
        split_cfg=data_cfg.split,
        tokenizer=model.tokenizer,
        skip_samples=0,
        epoch_seed=run_cfg.seed + 1,
    )
    del _initial_train_loader

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=training_cfg.learning_rate,
        weight_decay=0.01,
    )
    max_steps = training_cfg.max_steps
    # Size the LR schedule from full_train_len, not len(train_loader): on
    # resume the loader is truncated, but the schedule must be the same shape
    # as the original run so the loaded scheduler state lands on the right LR.
    epoch_based_steps = training_cfg.max_epochs * (full_train_len // training_cfg.batch_size) // accum_steps
    num_training_steps = max_steps if max_steps else epoch_based_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_cfg.warmup_steps,
        num_training_steps=num_training_steps,
    )
    if max_steps:
        log.info("max_steps=%d — will stop after %d optimizer steps", max_steps, max_steps)

    use_fp16 = training_cfg.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    start_epoch = 1
    start_samples_seen = 0
    start_global_step = 0
    best_val_ppl = float("inf")
    if training_cfg.resume_checkpoint:
        start_epoch, start_global_step, start_samples_seen, best_val_ppl = _load_resume_checkpoint(
            training_cfg.resume_checkpoint, model, optimizer, scheduler, scaler, log,
        )
        if start_samples_seen >= full_train_len:
            # Previous run finished an epoch exactly at the checkpoint boundary.
            start_epoch += 1
            start_samples_seen = 0
        log.info(
            "Resuming training at epoch=%d, global_step=%d, samples_seen_this_epoch=%d",
            start_epoch, start_global_step, start_samples_seen,
        )

    tokenizer = model.tokenizer

    if scoring_cfg.datasets:
        scoring_datasets = load_scoring_datasets(
            scoring_cfg.datasets,
            n_samples=scoring_cfg.n_samples,
            seed=run_cfg.seed,
        )
        log.info("Scoring evaluation enabled: %d datasets", len(scoring_datasets))
    else:
        scoring_datasets = None
        log.info("Scoring evaluation disabled (no scoring.datasets in config)")

    model.train()
    running_loss = 0.0
    log_steps = 0
    global_step = start_global_step
    optim_step = global_step // accum_steps
    best_ckpt_path: Optional[Path] = None

    save_every_n_steps = training_cfg.save_every_n_steps
    max_epochs = training_cfg.max_epochs
    hit_max_steps = False
    training_history = []
    scoring_history = []

    for epoch in range(start_epoch, max_epochs + 1):
        skip = start_samples_seen if epoch == start_epoch else 0
        epoch_seed = run_cfg.seed + epoch
        train_loader = build_train_loader(
            train_dataset=train_dataset,
            collator=collator,
            batch_size=training_cfg.batch_size,
            epoch_seed=epoch_seed,
            skip_samples=skip,
        )
        samples_seen_this_epoch = skip
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{max_epochs}",
            initial=skip // training_cfg.batch_size,
        )
        for batch in progress:
            global_step += 1
            samples_seen_this_epoch += training_cfg.batch_size
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

            if global_step % log_every_n_steps == 0:
                avg_loss = running_loss / log_steps
                lr = scheduler.get_last_lr()[0]
                if wandb_mod is not None:
                    wandb_mod.log(
                        {"train/loss": avg_loss, "train/lr": lr, "train/epoch": epoch},
                        step=global_step,
                    )
                progress.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")
                log.info(
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

            if save_every_n_steps and global_step % save_every_n_steps == 0:
                ppl, val_loss = compute_perplexity(model, val_loader, device)
                cdr_ppl = corpus_perplexity([C05_CDRH3], scorer=model, cdr_only=True)
                if wandb_mod is not None:
                    wandb_mod.log(
                        {
                            "val/loss": val_loss,
                            "val/perplexity": ppl,
                            "val/cdr_ppl": cdr_ppl,
                            "train/epoch": epoch,
                        },
                        step=global_step,
                    )
                log.info(
                    "Step %d — val loss: %.4f — val perplexity: %.2f — CDR-H3 ppl: %.2f",
                    global_step, val_loss, ppl, cdr_ppl,
                )
                training_history.append({
                    "step": global_step,
                    "val_loss": val_loss,
                    "val_perplexity": ppl,
                    "val_cdr_ppl": cdr_ppl,
                    "epoch": epoch,
                    "wall_time": time.time() - train_start,
                })

                ckpt_state = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "samples_seen": samples_seen_this_epoch,
                    "epoch_seed": epoch_seed,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "val_perplexity": ppl,
                    "best_val_ppl": best_val_ppl,
                }

                ckpt_path = checkpoint_dir / f"step_{global_step}.pt"
                torch.save(ckpt_state, ckpt_path)
                log.info("Saved checkpoint to %s", ckpt_path)

                if ppl < best_val_ppl:
                    best_val_ppl = ppl
                    best_ckpt_path = run_dir / "best.pt"
                    torch.save(ckpt_state, best_ckpt_path)
                    log.info("New best checkpoint (ppl=%.2f) saved to %s", ppl, best_ckpt_path)

                if scoring_datasets is not None:
                    scoring_results = run_multi_scoring_evaluation(
                        model, tokenizer, scoring_datasets,
                        device=device,
                        batch_size=scoring_cfg.batch_size,
                        seed=run_cfg.seed,
                        flank_ks=scoring_cfg.flank_ks,
                        scorer=model,
                    )
                    if wandb_mod is not None:
                        wandb_mod.log(
                            {f"eval/{k}": v for k, v in scoring_results.items()},
                            step=global_step,
                        )
                    scoring_history.append({"step": global_step, **scoring_results})

                model.train()

            if max_steps and optim_step >= max_steps:
                log.info("Reached max_steps=%d, stopping training.", max_steps)
                hit_max_steps = True
                break

        if wandb_mod is not None:
            wandb_mod.log({"train/epoch": epoch}, step=global_step)
        if hit_max_steps:
            break

    final_path = checkpoint_dir / "final.pt"
    final_metrics: dict = {}
    if len(val_loader.dataset) > 0:
        final_ppl, final_val_loss = compute_perplexity(
            model, val_loader, device, max_batches=max(len(val_loader), 1),
        )
        final_cdr_ppl = corpus_perplexity([C05_CDRH3], scorer=model, cdr_only=True)
        if wandb_mod is not None:
            wandb_mod.log(
                {"val/loss": final_val_loss, "val/perplexity": final_ppl, "val/cdr_ppl": final_cdr_ppl},
                step=global_step,
            )
        log.info(
            "Final val loss: %.4f — val perplexity: %.2f — CDR-H3 ppl: %.2f",
            final_val_loss, final_ppl, final_cdr_ppl,
        )
        training_history.append({
            "step": global_step,
            "val_loss": final_val_loss,
            "val_perplexity": final_ppl,
            "val_cdr_ppl": final_cdr_ppl,
            "wall_time": time.time() - train_start,
        })
        final_metrics["final_val_perplexity"] = float(final_ppl)
        final_metrics["final_val_loss"] = float(final_val_loss)
        final_metrics["final_cdr_pseudo_perplexity"] = float(final_cdr_ppl)
    else:
        final_ppl = float("inf")
        log.info("Skipping final val perplexity (empty validation set)")

    if len(test_loader.dataset) > 0:
        test_ppl, test_loss = compute_perplexity(
            model, test_loader, device, max_batches=max(len(test_loader), 1),
        )
        if wandb_mod is not None:
            wandb_mod.log({"test/loss": test_loss, "test/perplexity": test_ppl}, step=global_step)
        log.info("Final test loss: %.4f — test perplexity: %.2f", test_loss, test_ppl)
        final_metrics["final_test_perplexity"] = float(test_ppl)
        final_metrics["final_test_loss"] = float(test_loss)
    else:
        log.info("Skipping final test perplexity (empty test split)")

    final_state = {
        "epoch": max_epochs,
        "global_step": global_step,
        "samples_seen": full_train_len,
        "epoch_seed": run_cfg.seed + max_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "val_perplexity": final_ppl,
        "best_val_ppl": best_val_ppl,
    }
    torch.save(final_state, final_path)
    log.info("Training complete. Final checkpoint: %s", final_path)

    if final_ppl < best_val_ppl:
        best_ckpt_path = run_dir / "best.pt"
        torch.save(final_state, best_ckpt_path)
        log.info("Final checkpoint is also best (ppl=%.2f)", final_ppl)

    if scoring_datasets is not None:
        scoring_results = run_multi_scoring_evaluation(
            model, tokenizer, scoring_datasets,
            device=device,
            batch_size=scoring_cfg.batch_size,
            seed=run_cfg.seed,
            flank_ks=scoring_cfg.flank_ks,
            scorer=model,
        )
        if wandb_mod is not None:
            wandb_mod.log(
                {f"eval/{k}": v for k, v in scoring_results.items()},
                step=global_step,
            )
        if not scoring_history or scoring_history[-1]["step"] != global_step:
            scoring_history.append({"step": global_step, **scoring_results})

    return training_history, scoring_history, global_step, best_ckpt_path, final_metrics


# ---------------------------------------------------------------------------
# Strategy: TTT (test-time training on a single sequence)
# ---------------------------------------------------------------------------


def _train_ttt(
    model: ESM2Model,
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    training_cfg: TrainingConfig,
    scoring_cfg: ScoringConfig,
    run_cfg: RunConfig,
    run_dir: Path,
    checkpoint_dir: Path,
    device: torch.device,
    train_start: float,
    log: logging.Logger,
    wandb_mod: Optional[Any],
    log_every_n_steps: int,
) -> tuple[list, list, int, Path]:
    """Run TTT on a single sequence. Returns (training_history, scoring_history,
    global_step, final_ckpt_path)."""
    sequence = _load_single_sequence(data_cfg.fasta_path, log)
    tokenizer = model.tokenizer
    tokenized = tokenizer(
        sequence,
        truncation=True,
        max_length=data_cfg.max_seq_len,
        padding=False,
        return_tensors=None,
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=data_cfg.mlm_probability,
        pad_to_multiple_of=8,
    )

    batch_size = training_cfg.batch_size
    accum_steps = training_cfg.gradient_accumulation_steps
    max_steps = training_cfg.max_steps
    if max_steps is None:
        raise ValueError("TTT stage requires training.max_steps to be set.")

    log.info(
        "TTT config: %d steps × %d accum × %d batch = %d forward passes",
        max_steps, accum_steps, batch_size, max_steps * accum_steps,
    )

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=training_cfg.learning_rate,
        momentum=0.0,
        weight_decay=0.0,
    )

    training_history = []
    model.train()

    for step in range(1, max_steps + 1):
        step_loss = 0.0
        for _ in range(accum_steps):
            batch = _make_masked_batch(tokenized, batch_size, collator, device)
            outputs = model(**batch)
            loss = outputs.loss / accum_steps
            loss.backward()
            step_loss += outputs.loss.item()

        optimizer.step()
        optimizer.zero_grad()

        avg_loss = step_loss / accum_steps
        if wandb_mod is not None:
            wandb_mod.log({"train/loss": avg_loss, "train/step": step}, step=step)
        log.info("Step %d/%d — loss: %.4f", step, max_steps, avg_loss)
        training_history.append({
            "step": step,
            "train_loss": avg_loss,
            "learning_rate": training_cfg.learning_rate,
            "wall_time": time.time() - train_start,
        })

    final_state = {
        "global_step": max_steps,
        "model_state_dict": model.state_dict(),
    }
    final_path = checkpoint_dir / "final.pt"
    torch.save(final_state, final_path)
    log.info("Saved final checkpoint to %s", final_path)

    scoring_history = []
    if scoring_cfg.datasets:
        scoring_datasets = load_scoring_datasets(
            scoring_cfg.datasets,
            n_samples=scoring_cfg.n_samples,
            seed=run_cfg.seed,
        )
        scoring_results = run_multi_scoring_evaluation(
            model, tokenizer, scoring_datasets,
            device=device,
            batch_size=scoring_cfg.batch_size,
            seed=run_cfg.seed,
            flank_ks=scoring_cfg.flank_ks,
            scorer=model,
        )
        if wandb_mod is not None:
            wandb_mod.log(
                {f"eval/{k}": v for k, v in scoring_results.items()},
                step=max_steps,
            )
        scoring_history.append({"step": max_steps, **scoring_results})

    return training_history, scoring_history, max_steps, final_path, {}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_stage(
    stage_type: str,
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    training_cfg: TrainingConfig,
    scoring_cfg: ScoringConfig,
    run_cfg: RunConfig,
    run_name: str,
    cfg: Optional[DictConfig] = None,
) -> Path:
    """Run one training stage. Returns the path to the output checkpoint that
    downstream stages should seed from (best.pt for evotuning, final.pt for TTT)."""
    if stage_type not in ("evotuning", "ttt"):
        raise ValueError(f"Unknown stage_type: {stage_type!r} (expected 'evotuning' or 'ttt')")

    run_dir = ensure_dir(f"{run_cfg.train_dir}/{run_name}")
    checkpoint_dir = ensure_dir(f"{run_dir}/checkpoints")

    level_name = "INFO"
    if cfg is not None and hasattr(cfg, "logging"):
        level_name = str(getattr(cfg.logging, "level", "INFO"))
    log_every_n_steps = 50
    if cfg is not None and hasattr(cfg, "logging"):
        log_every_n_steps = int(getattr(cfg.logging, "log_every_n_steps", 50))

    run_log = setup_train_logger(run_dir, level_name=level_name, logger_name=__name__)

    snapshot = OmegaConf.to_container(cfg, resolve=True) if cfg is not None else {}
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(snapshot, f, default_flow_style=False, sort_keys=False)
    run_log.info("Run directory: %s", run_dir)

    torch.manual_seed(run_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_log.info("Device: %s", device)

    wandb_id_path = run_dir / "wandb_id.txt"
    resume_wandb_id: Optional[str] = None
    if training_cfg.resume_checkpoint and wandb_id_path.exists():
        resume_wandb_id = wandb_id_path.read_text().strip() or None
        if resume_wandb_id:
            run_log.info("Resuming W&B run id=%s (from %s)", resume_wandb_id, wandb_id_path)
        else:
            run_log.warning("%s exists but is empty; starting a fresh W&B run.", wandb_id_path)
    elif training_cfg.resume_checkpoint:
        run_log.warning(
            "resume_checkpoint set but %s not found; starting a fresh W&B run.",
            wandb_id_path,
        )

    wandb_mod, wandb_run = init_wandb(
        cfg,
        run_dir,
        run_log,
        run_name=run_name,
        group="evotuning",
        resume_id=resume_wandb_id,
    ) if cfg is not None else (None, None)

    if wandb_run is not None and resume_wandb_id is None:
        # First launch of this run: persist the id so a future resume can attach.
        try:
            wandb_id_path.write_text(str(wandb_run.id))
            run_log.info("Persisted W&B run id to %s", wandb_id_path)
        except Exception as exc:
            run_log.warning("Failed to write %s: %s", wandb_id_path, exc)

    model = ESM2Model(model_cfg)

    if run_cfg.finetune:
        run_log.info("Loading finetune checkpoint: %s", run_cfg.finetune)
        ckpt = torch.load(run_cfg.finetune, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])

    summary = model.param_summary()
    run_log.info(
        "Parameters — total: %s, trainable: %s, frozen: %s",
        f"{summary['total']:,}", f"{summary['trainable']:,}", f"{summary['frozen']:,}",
    )
    model.to(device)

    train_start = time.time()

    if stage_type == "evotuning":
        training_history, scoring_history, global_step, best_ckpt_path, final_metrics = _train_evotuning(
            model, model_cfg, data_cfg, training_cfg, scoring_cfg, run_cfg,
            run_dir, checkpoint_dir, device, train_start,
            log=run_log, wandb_mod=wandb_mod, log_every_n_steps=log_every_n_steps,
        )
        handoff_ckpt = best_ckpt_path if best_ckpt_path is not None else checkpoint_dir / "final.pt"
    else:  # ttt
        training_history, scoring_history, global_step, handoff_ckpt, final_metrics = _train_ttt(
            model, model_cfg, data_cfg, training_cfg, scoring_cfg, run_cfg,
            run_dir, checkpoint_dir, device, train_start,
            log=run_log, wandb_mod=wandb_mod, log_every_n_steps=log_every_n_steps,
        )

    metrics = {
        "metadata": {
            "total_steps": global_step,
            "total_time_seconds": round(time.time() - train_start, 2),
            "device": str(device),
            "param_summary": summary,
        },
        "training_history": training_history,
        "scoring_history": scoring_history,
        **final_metrics,
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    run_log.info("Saved metrics to %s", run_dir / "metrics.json")

    if wandb_run is not None:
        wandb_run.finish()

    return handoff_ckpt
