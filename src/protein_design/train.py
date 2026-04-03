"""Training loop for ESM2 evotuning (continued MLM pretraining)."""

import itertools
import logging
import os
from pathlib import Path

import torch
import wandb
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from protein_design.data import make_dataloaders
from protein_design.evaluate import compute_perplexity
from protein_design.model import EvotuningModel

logger = logging.getLogger(__name__)


def _resolve_env_vars(config: dict) -> dict:
    """Replace ${VAR} placeholders in string config values with env vars."""
    resolved = {}
    for k, v in config.items():
        if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            env_key = v[2:-1]
            resolved[k] = os.environ.get(env_key, v)
        else:
            resolved[k] = v
    return resolved


def train(config: dict) -> None:
    """Run the evotuning training loop.

    Args:
        config: Flat dict loaded from the evotuning YAML config file.
    """
    config = _resolve_env_vars(config)

    # ── Setup ────────────────────────────────────────────────────────────
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    wandb.init(project=config["wandb_project"], config=config)

    # ── Model ────────────────────────────────────────────────────────────
    model = EvotuningModel(config)
    summary = model.param_summary()
    logger.info(
        "Parameters — total: %s, trainable: %s, frozen: %s",
        f"{summary['total']:,}",
        f"{summary['trainable']:,}",
        f"{summary['frozen']:,}",
    )
    model.to(device)

    # ── Data ─────────────────────────────────────────────────────────────
    fasta_path = os.path.join(config["checkpoint_dir"], "..", "oas_dedup_rep_seq.fasta")
    data_dir = os.environ.get("SCRATCH_DIR", ".")
    fasta_path = os.path.join(data_dir, "oas_dedup_rep_seq.fasta")
    train_loader, val_loader = make_dataloaders(fasta_path, config)

    # ── Optimizer + scheduler ────────────────────────────────────────────
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["learning_rate"],
        weight_decay=0.01,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=config["max_steps"],
    )

    use_fp16 = config.get("fp16", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    # ── Training loop ────────────────────────────────────────────────────
    model.train()
    train_iter = iter(itertools.cycle(train_loader))
    accum_steps = config["gradient_accumulation_steps"]
    running_loss = 0.0

    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    progress = tqdm(range(1, config["max_steps"] + 1), desc="Training")
    for global_step in progress:
        batch = next(train_iter)
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.amp.autocast("cuda", enabled=use_fp16):
            outputs = model(**batch)
            loss = outputs.loss / accum_steps

        scaler.scale(loss).backward()
        running_loss += loss.item()

        if global_step % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        # ── Logging ──────────────────────────────────────────────────────
        if global_step % 50 == 0:
            avg_loss = running_loss / 50 * accum_steps
            lr = scheduler.get_last_lr()[0]
            wandb.log({"train/loss": avg_loss, "train/lr": lr}, step=global_step)
            progress.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")
            running_loss = 0.0

        if global_step % 100 == 0:
            logger.info(
                "Step %d/%d — loss: %.4f — lr: %.2e",
                global_step, config["max_steps"],
                running_loss / min(global_step % 100 or 100, 100) * accum_steps,
                scheduler.get_last_lr()[0],
            )

        # ── Evaluation ───────────────────────────────────────────────────
        if global_step % config["eval_every_n_steps"] == 0:
            ppl = compute_perplexity(model, val_loader, device)
            wandb.log({"val/perplexity": ppl}, step=global_step)
            logger.info("Step %d — val perplexity: %.2f", global_step, ppl)
            model.train()

        # ── Checkpoint ───────────────────────────────────────────────────
        if global_step % config["save_every_n_steps"] == 0:
            ckpt_path = checkpoint_dir / f"step_{global_step}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                },
                ckpt_path / "checkpoint.pt",
            )
            logger.info("Saved checkpoint to %s", ckpt_path)

    # ── Final save ───────────────────────────────────────────────────────
    final_path = checkpoint_dir / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "global_step": config["max_steps"],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        },
        final_path / "checkpoint.pt",
    )
    logger.info("Training complete. Final checkpoint: %s", final_path)
    wandb.finish()
