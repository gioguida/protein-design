"""Training loop for ESM2 evotuning (continued MLM pretraining)."""

import logging
import shutil
from pathlib import Path

import torch
import wandb
import yaml
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from protein_design.data import make_dataloaders
from protein_design.evaluate import compute_perplexity
from protein_design.model import EvotuningModel
from protein_design.scoring import load_scoring_datasets, run_multi_scoring_evaluation
from protein_design.utils import ensure_dir

logger = logging.getLogger(__name__)


def train(config: dict, run_name: str) -> None:
    """Run the evotuning training loop.

    Args:
        config: Resolved dict loaded from the evotuning YAML config file.
        run_name: Name for this run (used for output directory and W&B).
    """
    # ── Run directory setup ─────────────────────────────────────────────
    train_dir = config["train_dir"]
    run_dir = ensure_dir(f"{train_dir}/{run_name}")
    checkpoint_dir = ensure_dir(f"{run_dir}/checkpoints")

    # Save resolved config snapshot
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info("Run directory: %s", run_dir)

    # ── Setup ────────────────────────────────────────────────────────────
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    wandb.init(project=config["wandb_project"], name=run_name, config=config)

    # ── Model ────────────────────────────────────────────────────────────
    model = EvotuningModel(config)

    # Load finetuning checkpoint (weights only, fresh optimizer/scheduler)
    finetune_path = config.get("finetune")
    if finetune_path:
        logger.info("Loading finetune checkpoint: %s", finetune_path)
        ckpt = torch.load(finetune_path, map_location="cpu")
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
    fasta_path = config.get("fasta_path")
    if not fasta_path:
        scratch_dir = config.get("scratch_dir", ".")
        fasta_path = f"{scratch_dir}/oas_dedup_rep_seq.fasta"
    train_loader, val_loader = make_dataloaders(fasta_path, config)

    # ── Optimizer + scheduler ────────────────────────────────────────────
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["learning_rate"],
        weight_decay=0.01,
    )
    accum_steps = config["gradient_accumulation_steps"]
    max_steps = config.get("max_steps")  # optional: stop after N optimizer steps
    epoch_based_steps = config["max_epochs"] * len(train_loader) // accum_steps
    num_training_steps = max_steps if max_steps else epoch_based_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=num_training_steps,
    )
    if max_steps:
        logger.info("max_steps=%d — will stop after %d optimizer steps", max_steps, max_steps)

    use_fp16 = config.get("fp16", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    # ── Scoring data (loaded once, reused at every checkpoint) ─────────
    scoring_datasets_config = config.get("scoring_datasets")
    if scoring_datasets_config:
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        scoring_datasets = load_scoring_datasets(
            scoring_datasets_config,
            n_samples=config.get("scoring_n_samples", 10000),
            seed=config["seed"],
        )
        scoring_batch_size = config.get("scoring_batch_size", 512)
        logger.info("Scoring evaluation enabled: %d datasets", len(scoring_datasets))
    else:
        scoring_datasets = None
        logger.info("Scoring evaluation disabled (no scoring_datasets in config)")

    # ── Training loop ────────────────────────────────────────────────────
    model.train()
    running_loss = 0.0
    log_steps = 0
    global_step = 0
    optim_step = 0
    best_val_ppl = float("inf")

    save_every_n_steps = config["save_every_n_steps"]
    max_epochs = config["max_epochs"]
    hit_max_steps = False

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
                running_loss = 0.0
                log_steps = 0

            # ── Evaluation + checkpoint + scoring ────────────────────────
            if global_step % save_every_n_steps == 0:
                ppl, val_loss = compute_perplexity(model, val_loader, device)
                wandb.log(
                    {"val/loss": val_loss, "val/perplexity": ppl, "train/epoch": epoch},
                    step=global_step,
                )
                logger.info("Step %d — val loss: %.4f — val perplexity: %.2f", global_step, val_loss, ppl)

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
                        seed=config["seed"],
                    )
                    wandb.log(
                        {f"eval/{k}": v for k, v in scoring_results.items()},
                        step=global_step,
                    )

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
    project_dir = config.get("project_dir")
    best_src = run_dir / "best.pt"
    if project_dir and best_src.exists():
        archive_dir = ensure_dir(f"{project_dir}/checkpoints/{run_name}")
        archive_path = archive_dir / "best.pt"
        shutil.copy2(best_src, archive_path)
        logger.info("Archived best checkpoint to %s", archive_path)

    # ── Final scoring evaluation ─────────────────────────────────────────
    if scoring_datasets is not None:
        scoring_results = run_multi_scoring_evaluation(
            model, tokenizer, scoring_datasets,
            device=device,
            batch_size=scoring_batch_size,
            seed=config["seed"],
        )
        wandb.log(
            {f"eval/{k}": v for k, v in scoring_results.items()},
            step=global_step,
        )

    wandb.finish()
