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
from typing import Optional

import torch
import wandb
import yaml
from Bio import SeqIO
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, get_linear_schedule_with_warmup

from protein_design.eval import (
    compute_cdr_pseudo_perplexity,
    compute_perplexity,
    load_scoring_datasets,
    run_multi_scoring_evaluation,
)
from protein_design.config import ModelConfig, RunConfig, ScoringConfig
from protein_design.evotuning.config import DataConfig, TrainingConfig
from protein_design.evotuning.data import make_dataloaders
from protein_design.model import ESM2Model
from protein_design.utils import ensure_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TTT helpers (private — TTT is the only strategy that needs these)
# ---------------------------------------------------------------------------


def _load_single_sequence(fasta_path: str) -> str:
    record = next(SeqIO.parse(fasta_path, "fasta"))
    seq = str(record.seq)
    logger.info("Loaded sequence (%d aa) from %s: %s", len(seq), fasta_path, record.id)
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
) -> tuple[list, list, int, Optional[Path]]:
    """Run corpus-MLM training. Returns (training_history, scoring_history,
    global_step, best_ckpt_path_or_None, final_metrics)."""
    train_loader, val_loader, test_loader = make_dataloaders(
        fasta_path=data_cfg.fasta_path,
        tokenizer_name=model_cfg.esm_model_path,
        max_seq_len=data_cfg.max_seq_len,
        mlm_probability=data_cfg.mlm_probability,
        batch_size=training_cfg.batch_size,
        split_cfg=data_cfg.split,
    )

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

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.esm_model_path)

    if scoring_cfg.datasets:
        scoring_datasets = load_scoring_datasets(
            scoring_cfg.datasets,
            n_samples=scoring_cfg.n_samples,
            seed=run_cfg.seed,
        )
        logger.info("Scoring evaluation enabled: %d datasets", len(scoring_datasets))
    else:
        scoring_datasets = None
        logger.info("Scoring evaluation disabled (no scoring.datasets in config)")

    model.train()
    running_loss = 0.0
    log_steps = 0
    global_step = 0
    optim_step = 0
    best_val_ppl = float("inf")
    best_ckpt_path: Optional[Path] = None

    save_every_n_steps = training_cfg.save_every_n_steps
    max_epochs = training_cfg.max_epochs
    hit_max_steps = False
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

            if save_every_n_steps and global_step % save_every_n_steps == 0:
                ppl, val_loss = compute_perplexity(model, val_loader, device)
                cdr_ppl = compute_cdr_pseudo_perplexity(model, tokenizer, device)
                wandb.log(
                    {
                        "val/loss": val_loss,
                        "val/perplexity": ppl,
                        "val/cdr_ppl": cdr_ppl,
                        "train/epoch": epoch,
                    },
                    step=global_step,
                )
                logger.info(
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
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "val_perplexity": ppl,
                }

                ckpt_path = checkpoint_dir / f"step_{global_step}.pt"
                torch.save(ckpt_state, ckpt_path)
                logger.info("Saved checkpoint to %s", ckpt_path)

                if ppl < best_val_ppl:
                    best_val_ppl = ppl
                    best_ckpt_path = run_dir / "best.pt"
                    torch.save(ckpt_state, best_ckpt_path)
                    logger.info("New best checkpoint (ppl=%.2f) saved to %s", ppl, best_ckpt_path)

                if scoring_datasets is not None:
                    scoring_results = run_multi_scoring_evaluation(
                        model, tokenizer, scoring_datasets,
                        device=device,
                        batch_size=scoring_cfg.batch_size,
                        seed=run_cfg.seed,
                        flank_ks=scoring_cfg.flank_ks,
                    )
                    wandb.log(
                        {f"eval/{k}": v for k, v in scoring_results.items()},
                        step=global_step,
                    )
                    scoring_history.append({"step": global_step, **scoring_results})

                model.train()

            if max_steps and optim_step >= max_steps:
                logger.info("Reached max_steps=%d, stopping training.", max_steps)
                hit_max_steps = True
                break

        wandb.log({"train/epoch": epoch}, step=global_step)
        if hit_max_steps:
            break

    final_path = checkpoint_dir / "final.pt"
    final_metrics: dict = {}
    if len(val_loader.dataset) > 0:
        # Full-pass val eval (no 50-batch cap) for the headline final number.
        final_ppl, final_val_loss = compute_perplexity(
            model, val_loader, device, max_batches=max(len(val_loader), 1),
        )
        final_cdr_ppl = compute_cdr_pseudo_perplexity(model, tokenizer, device)
        wandb.log(
            {"val/loss": final_val_loss, "val/perplexity": final_ppl, "val/cdr_ppl": final_cdr_ppl},
            step=global_step,
        )
        logger.info(
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
        logger.info("Skipping final val perplexity (empty validation set)")

    if len(test_loader.dataset) > 0:
        test_ppl, test_loss = compute_perplexity(
            model, test_loader, device, max_batches=max(len(test_loader), 1),
        )
        wandb.log({"test/loss": test_loss, "test/perplexity": test_ppl}, step=global_step)
        logger.info("Final test loss: %.4f — test perplexity: %.2f", test_loss, test_ppl)
        final_metrics["final_test_perplexity"] = float(test_ppl)
        final_metrics["final_test_loss"] = float(test_loss)
    else:
        logger.info("Skipping final test perplexity (empty test split)")

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
        best_ckpt_path = run_dir / "best.pt"
        torch.save(final_state, best_ckpt_path)
        logger.info("Final checkpoint is also best (ppl=%.2f)", final_ppl)

    if scoring_datasets is not None:
        scoring_results = run_multi_scoring_evaluation(
            model, tokenizer, scoring_datasets,
            device=device,
            batch_size=scoring_cfg.batch_size,
            seed=run_cfg.seed,
            flank_ks=scoring_cfg.flank_ks,
        )
        wandb.log(
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
) -> tuple[list, list, int, Path]:
    """Run TTT on a single sequence. Returns (training_history, scoring_history,
    global_step, final_ckpt_path)."""
    sequence = _load_single_sequence(data_cfg.fasta_path)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.esm_model_path)
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

    logger.info(
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
        wandb.log({"train/loss": avg_loss, "train/step": step}, step=step)
        logger.info("Step %d/%d — loss: %.4f", step, max_steps, avg_loss)
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
    logger.info("Saved final checkpoint to %s", final_path)

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
        )
        wandb.log(
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

    snapshot = OmegaConf.to_container(cfg, resolve=True) if cfg is not None else {}
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(snapshot, f, default_flow_style=False, sort_keys=False)
    logger.info("Run directory: %s", run_dir)

    torch.manual_seed(run_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    wandb.init(project=run_cfg.wandb_project, name=run_name, config=snapshot)

    model = ESM2Model(model_cfg)

    if run_cfg.finetune:
        logger.info("Loading finetune checkpoint: %s", run_cfg.finetune)
        ckpt = torch.load(run_cfg.finetune, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])

    summary = model.param_summary()
    logger.info(
        "Parameters — total: %s, trainable: %s, frozen: %s",
        f"{summary['total']:,}", f"{summary['trainable']:,}", f"{summary['frozen']:,}",
    )
    model.to(device)

    train_start = time.time()

    if stage_type == "evotuning":
        training_history, scoring_history, global_step, best_ckpt_path, final_metrics = _train_evotuning(
            model, model_cfg, data_cfg, training_cfg, scoring_cfg, run_cfg,
            run_dir, checkpoint_dir, device, train_start,
        )
        handoff_ckpt = best_ckpt_path if best_ckpt_path is not None else checkpoint_dir / "final.pt"
    else:  # ttt
        training_history, scoring_history, global_step, handoff_ckpt, final_metrics = _train_ttt(
            model, model_cfg, data_cfg, training_cfg, scoring_cfg, run_cfg,
            run_dir, checkpoint_dir, device, train_start,
        )

    archive_dir: Optional[Path] = None
    if run_cfg.project_dir and handoff_ckpt.exists():
        archive_dir = ensure_dir(f"{run_cfg.project_dir}/checkpoints/{run_name}")
        archive_path = archive_dir / handoff_ckpt.name
        shutil.copy2(handoff_ckpt, archive_path)
        logger.info("Archived checkpoint to %s", archive_path)

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
    logger.info("Saved metrics to %s", run_dir / "metrics.json")

    if archive_dir is not None:
        shutil.copy2(run_dir / "metrics.json", archive_dir / "metrics.json")
        logger.info("Archived metrics to %s", archive_dir / "metrics.json")

    wandb.finish()
    return handoff_ckpt
