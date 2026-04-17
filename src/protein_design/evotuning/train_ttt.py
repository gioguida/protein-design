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
from typing import Optional

import torch
import wandb
import yaml
from Bio import SeqIO
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from protein_design.eval import load_scoring_datasets, run_multi_scoring_evaluation
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


def train_ttt(
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    training_cfg: TrainingConfig,
    scoring_cfg: ScoringConfig,
    run_cfg: RunConfig,
    run_name: str,
    cfg: Optional[DictConfig] = None,
) -> None:
    """Run test-time training on a single protein sequence."""
    # ── Run directory setup ─────────────────────────────────────────────
    run_dir = ensure_dir(f"{run_cfg.train_dir}/{run_name}")
    checkpoint_dir = ensure_dir(f"{run_dir}/checkpoints")

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

    logger.info(
        "TTT config: %d steps × %d accum × %d batch = %d forward passes",
        max_steps, accum_steps, batch_size, max_steps * accum_steps,
    )

    # ── Optimizer (SGD, no scheduler) ────────────────────────────────────
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=training_cfg.learning_rate,
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
            "learning_rate": training_cfg.learning_rate,
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

    if run_cfg.project_dir:
        archive_dir = ensure_dir(f"{run_cfg.project_dir}/checkpoints/{run_name}")
        archive_path = archive_dir / "final.pt"
        shutil.copy2(final_path, archive_path)
        logger.info("Archived checkpoint to %s", archive_path)

    # ── Scoring evaluation ───────────────────────────────────────────────
    scoring_results = None
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
