"""Test-Time Training (TTT) loop for ESM2.

Adapts a pretrained (or evotuned) ESM2 model to a single target protein
sequence using continued masked language modeling for a fixed number of
optimizer steps with SGD.
"""

import json
import logging
import shutil
import time
from pathlib import Path

import torch
import wandb
import yaml
from Bio import SeqIO
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from protein_design.eval import load_scoring_datasets, run_multi_scoring_evaluation
from protein_design.model import ESM2Model
from protein_design.utils import build_model_config
from protein_design.utils import ensure_dir

logger = logging.getLogger(__name__)


def _load_single_sequence(fasta_path: str) -> str:
    """Load the first sequence from a FASTA file."""
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
    """Create a batch of independently masked copies of a single tokenized sequence."""
    samples = [
        {k: torch.tensor(v) for k, v in tokenized.items()}
        for _ in range(batch_size)
    ]
    batch = collator(samples)
    return {k: v.to(device) for k, v in batch.items()}


def train_ttt(config: dict, run_name: str) -> None:
    """Run test-time training on a single protein sequence.

    Args:
        config: Resolved dict loaded from a TTT YAML config file.
        run_name: Name for this run (used for output directory and W&B).
    """
    # ── Run directory setup ─────────────────────────────────────────────
    train_dir = config["train_dir"]
    run_dir = ensure_dir(f"{train_dir}/{run_name}")
    checkpoint_dir = ensure_dir(f"{run_dir}/checkpoints")

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
    model = ESM2Model(build_model_config(config, device=str(device)))

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
    sequence = _load_single_sequence(config["fasta_path"])
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenized = tokenizer(
        sequence,
        truncation=True,
        max_length=config["max_seq_len"],
        padding=False,
        return_tensors=None,
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=config["mlm_probability"],
        pad_to_multiple_of=8,
    )

    batch_size = config["batch_size"]
    accum_steps = config["gradient_accumulation_steps"]
    max_steps = config["max_steps"]

    logger.info(
        "TTT config: %d steps × %d accum × %d batch = %d forward passes",
        max_steps, accum_steps, batch_size, max_steps * accum_steps,
    )

    # ── Optimizer (SGD, no scheduler) ────────────────────────────────────
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["learning_rate"],
        momentum=0.0,
        weight_decay=0.0,
    )

    train_start = time.time()
    training_history = []

    # ── Training loop ────────────────────────────────────────────────────
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
            "learning_rate": config["learning_rate"],
            "wall_time": time.time() - train_start,
        })

    # ── Save final checkpoint ────────────────────────────────────────────
    final_state = {
        "global_step": max_steps,
        "model_state_dict": model.state_dict(),
    }
    final_path = checkpoint_dir / "final.pt"
    torch.save(final_state, final_path)
    logger.info("Saved final checkpoint to %s", final_path)

    project_dir = config.get("project_dir")
    if project_dir:
        archive_dir = ensure_dir(f"{project_dir}/checkpoints/{run_name}")
        archive_path = archive_dir / "final.pt"
        shutil.copy2(final_path, archive_path)
        logger.info("Archived checkpoint to %s", archive_path)

    # ── Scoring evaluation ───────────────────────────────────────────────
    scoring_results = None
    scoring_datasets_config = config.get("scoring_datasets")
    if scoring_datasets_config:
        scoring_datasets = load_scoring_datasets(
            scoring_datasets_config,
            n_samples=config.get("scoring_n_samples", 10000),
            seed=config["seed"],
        )
        scoring_results = run_multi_scoring_evaluation(
            model, tokenizer, scoring_datasets,
            device=device,
            batch_size=config.get("scoring_batch_size", 512),
            seed=config["seed"],
        )
        wandb.log(
            {f"eval/{k}": v for k, v in scoring_results.items()},
            step=max_steps,
        )

    # ── Save local metrics ──────────────────────────────────────────────
    scoring_history = []
    if scoring_results is not None:
        scoring_history.append({"step": max_steps, **scoring_results})

    metrics = {
        "metadata": {
            "total_steps": max_steps,
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
