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
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from Bio import SeqIO
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling, get_linear_schedule_with_warmup

from protein_design.constants import C05_CDRH3
from protein_design.dms_splitting import dms_config_path_from_cfg, resolve_dataset_split
from protein_design.eval import (
    corpus_perplexity,
    compute_perplexity,
    ensure_test_eval_csv,
    evaluate_pll_eval_sets,
    evaluate_spearman,
    load_scoring_datasets,
    load_test_pll_eval_sets,
    region_stratified_masked_recovery_accuracy,
    run_multi_scoring_evaluation,
    score_sequences_masked_positions,
)
from protein_design.config import ModelConfig, RunConfig, ScoringConfig
from protein_design.evotuning.config import DataConfig, TrainingConfig
from protein_design.evotuning.data import build_train_loader, make_dataloaders
from protein_design.model import ESM2Model
from protein_design.utils import ensure_dir, init_wandb, setup_train_logger
from protein_design.wandb_plots import (
    RunArtifacts,
    build_summary_table,
    plot_flank_breakdown,
    plot_pll_comparison,
    plot_pll_vs_enrichment_grid,
    plot_spearman_evolution,
    plot_training_curves,
    set_publication_style,
)

# Module-level logger for orchestration messages (before run_dir exists).
logger = logging.getLogger(__name__)

# DMS dataset keys used for TTT snapshot evaluation.
_TTT_EVAL_DATASET_KEYS: dict[str, str] = {
    "ED2":   "ed2_m22",
    "ED5":   "ed5_m22",
    "ED811": "ed811_m22",
}


def _load_ttt_eval_datasets(
    cfg: DictConfig, log: logging.Logger
) -> dict[str, pd.DataFrame]:
    """Load ED2/ED5/ED811 test splits for TTT snapshot evaluation."""
    dms_config_path = dms_config_path_from_cfg(cfg)
    datasets: dict[str, pd.DataFrame] = {}
    for label, dataset_key in _TTT_EVAL_DATASET_KEYS.items():
        test_path = resolve_dataset_split(dataset_key, "test", dms_config_path)
        df = pd.read_csv(test_path)
        df = df.dropna(subset=["mut", "M22_binding_enrichment_adj"]).copy()
        df["mut"] = df["mut"].astype(str).str.strip()
        df = df[df["mut"] != ""].reset_index(drop=True)
        log.info("TTT eval: loaded %s (%d rows) from %s", label, len(df), test_path)
        datasets[label] = df
    return datasets


def _ttt_snapshot_eval(
    model: ESM2Model,
    step: int,
    ttt_eval_datasets: dict[str, pd.DataFrame],
    eval_batch_size: int,
    artifacts: "RunArtifacts",
    log: logging.Logger,
) -> dict[str, float]:
    """Compute CDR PPL + Spearman on all TTT eval datasets; log and return results."""
    model.eval()
    results: dict[str, float] = {}

    cdr_ppl = corpus_perplexity([C05_CDRH3], scorer=model, cdr_only=True)
    results["cdr_ppl"] = cdr_ppl

    spearman_vals: list[float] = []
    for name, df in ttt_eval_datasets.items():
        scores = score_sequences_masked_positions(
            scorer=model, df=df, wt=C05_CDRH3, batch_size=eval_batch_size,
        )
        enrichment = df["M22_binding_enrichment_adj"].to_numpy(dtype=float)
        rho, pval = evaluate_spearman(scores, enrichment)
        results[f"spearman_{name}"] = float(rho)
        spearman_vals.append(float(rho))
        log.info("  step=%d  %s (n=%d): Spearman rho=%.4f  p=%.2e", step, name, len(df), rho, pval)

    results["spearman_mean"] = float(np.mean(spearman_vals)) if spearman_vals else float("nan")
    log.info("  step=%d  CDR PPL=%.4f  spearman_mean=%.4f", step, cdr_ppl, results["spearman_mean"])

    artifacts.log(
        {
            "eval/cdr_ppl": cdr_ppl,
            "eval/spearman_mean": results["spearman_mean"],
            **{f"eval/spearman_{name}": results[f"spearman_{name}"] for name in ttt_eval_datasets},
        },
        step=step,
    )
    model.train()
    return results


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
# Single-mask (PLL-aligned) per-epoch evaluation helpers
# ---------------------------------------------------------------------------


def _load_masked_spearman_datasets(
    datasets_cfg: list[dict],
    log: logging.Logger,
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> list[tuple[str, pd.DataFrame, str]]:
    """Load DMS CSVs for masked-position Spearman (no num_mut filter).

    Keeps every row with finite enrichment and a non-empty `mut` string. When
    `n_samples` is set and a dataset has more rows, a fixed (seeded) subsample
    is taken — used for the per-epoch val tracking so the cost stays bounded
    and the row set is identical across epochs. The full set (n_samples=None)
    is used for the end-of-training test evaluation.
    """
    out: list[tuple[str, pd.DataFrame, str]] = []
    for ds in datasets_cfg:
        col = ds["enrichment_col"]
        df = pd.read_csv(ds["path"])
        enr = pd.to_numeric(df[col], errors="coerce")
        df = df[enr.notna()].copy()
        df[col] = enr[enr.notna()].astype(float)
        df["mut"] = df["mut"].astype(str).str.strip()
        df = df[df["mut"] != ""].reset_index(drop=True)
        if n_samples is not None and len(df) > n_samples:
            df = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)
        log.info("Masked-Spearman dataset %r: %d rows from %s", ds["name"], len(df), ds["path"])
        out.append((ds["name"], df, col))
    return out


def _eval_masked_spearman(
    model: ESM2Model,
    datasets: list[tuple[str, pd.DataFrame, str]],
    batch_size: int,
    log: logging.Logger,
) -> dict[str, float]:
    """Spearman of single-position-masked PLL vs. fitness for each dataset."""
    results: dict[str, float] = {}
    rhos: list[float] = []
    model.eval()
    for name, df, col in datasets:
        scores = score_sequences_masked_positions(
            scorer=model, df=df, wt=C05_CDRH3, batch_size=batch_size,
        )
        rho, pval = evaluate_spearman(scores, df[col].to_numpy(dtype=float))
        results[f"spearman_{name}"] = float(rho)
        rhos.append(float(rho))
        log.info("  %s (n=%d): Spearman rho=%.4f  p=%.2e", name, len(df), rho, pval)
    results["spearman_mean"] = float(np.mean(rhos)) if rhos else float("nan")
    model.train()
    return results


@dataclass
class RecoveryEvalConfig:
    """Fixed inputs for the region-stratified checkpoint selection rule.

    Built once at the start of a run. `reference_framework_accuracy` never
    changes: it is the framework accuracy this run's checkpoints are held
    against, which is the base pretrained model for a batch-masking run and
    the branch-point checkpoint for a single-position continuation.
    """

    corpus: Any
    val_indices: np.ndarray
    reference_framework_accuracy: float
    fr_tolerance_pp: float
    max_seq_len: int
    n_samples: int = 2000
    seed: int = 42
    batch_size: int = 256


def _eval_step_grid(
    fracs: list[float], optim_steps_per_epoch: int, max_epochs: int, log: logging.Logger
) -> list[int]:
    """Turn epoch fractions into absolute optimizer steps.

    A fraction landing below one step is clamped to one step, and duplicates
    are dropped, so a corpus too small to resolve the early fractions just gets
    a shorter grid instead of evaluating the same checkpoint several times.
    """
    steps: set[int] = set()
    for epoch in range(max_epochs):
        base = epoch * optim_steps_per_epoch
        for f in fracs:
            steps.add(base + max(1, int(round(f * optim_steps_per_epoch))))
    grid = sorted(steps)
    log.info(
        "Evaluation grid: %d points over %d optimizer steps/epoch × %d epoch(s) -> %s",
        len(grid), optim_steps_per_epoch, max_epochs, grid,
    )
    if len(grid) < len(fracs) * max_epochs:
        log.info(
            "Some requested fractions collapsed to the same step (corpus too small "
            "to resolve them); the grid was de-duplicated."
        )
    return grid


def _run_periodic_eval(
    *,
    model: ESM2Model,
    val_loader: Any,
    device: torch.device,
    global_step: int,
    optim_step: int,
    epoch: int,
    epoch_frac: float,
    epoch_seed: int,
    samples_seen: int,
    run_dir: Path,
    checkpoint_dir: Path,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    training_history: list,
    eval_points: list,
    artifacts: RunArtifacts,
    log: logging.Logger,
    train_start: float,
    recovery_cfg: "RecoveryEvalConfig",
    best_cdr_accuracy: float,
    best_ckpt_path: Optional[Path],
) -> tuple[float, Optional[Path]]:
    """Evaluate, checkpoint, and update the selected checkpoint.

    Every evaluation point is checkpointed, not just the ones that win. That is
    what lets the tolerance be re-examined afterwards without retraining, and
    it is the only reason the selection rule can be applied honestly: "earliest
    checkpoint maximizing CDR accuracy" cannot be decided until the later
    points exist, so the run always trains to the end and the choice is made
    from the recorded curve.

    Perplexity is recorded as a diagnostic and never used to select anything.
    """
    model.eval()
    ppl, val_loss = compute_perplexity(
        model, val_loader, device, max_batches=max(len(val_loader), 1),
    )
    acc = region_stratified_masked_recovery_accuracy(
        model, recovery_cfg.corpus, recovery_cfg.val_indices, device,
        max_seq_len=recovery_cfg.max_seq_len, batch_size=recovery_cfg.batch_size,
        n_samples=recovery_cfg.n_samples, seed=recovery_cfg.seed,
    )
    cdr_accuracy = acc["cdr_accuracy"]
    framework_accuracy = acc["framework_accuracy"]

    artifacts.log(
        {
            "val/loss": val_loss,
            "val/perplexity": ppl,
            "eval/cdr_accuracy": cdr_accuracy,
            "eval/framework_accuracy": framework_accuracy,
            "eval/epoch_frac": epoch_frac,
            "train/epoch": epoch,
        },
        step=global_step,
    )
    log.info(
        "Eval at %.4f of epoch %d (step %d) — val loss %.4f — val ppl %.2f — "
        "CDR acc %.4f — FR acc %.4f (reference %.4f, tolerance %.2fpp)",
        epoch_frac, epoch, global_step, val_loss, ppl, cdr_accuracy, framework_accuracy,
        recovery_cfg.reference_framework_accuracy, recovery_cfg.fr_tolerance_pp,
    )
    training_history.append({
        "step": global_step, "val_loss": val_loss, "val_perplexity": ppl,
        "epoch": epoch, "wall_time": time.time() - train_start,
    })

    ckpt_state = {
        "epoch": epoch,
        "epoch_frac": epoch_frac,
        "global_step": global_step,
        "optim_step": optim_step,
        "samples_seen": samples_seen,
        "epoch_seed": epoch_seed,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_perplexity": ppl,
        "cdr_accuracy": cdr_accuracy,
        "framework_accuracy": framework_accuracy,
        "pareto_reference_framework_accuracy": recovery_cfg.reference_framework_accuracy,
    }
    ckpt_path = checkpoint_dir / f"step_{global_step}.pt"
    torch.save(ckpt_state, ckpt_path)

    # Two-sided, as specified: a checkpoint drops out whether its framework
    # accuracy has fallen or risen too far from the reference.
    drift_pp = abs(framework_accuracy - recovery_cfg.reference_framework_accuracy) * 100.0
    eligible = drift_pp <= recovery_cfg.fr_tolerance_pp
    selected = False
    if eligible and cdr_accuracy == cdr_accuracy and cdr_accuracy > best_cdr_accuracy:
        best_cdr_accuracy = cdr_accuracy
        best_ckpt_path = run_dir / "selected.pt"
        torch.save(ckpt_state, best_ckpt_path)
        selected = True
        log.info(
            "New selected checkpoint (CDR acc %.4f, FR drift %.3fpp) -> %s",
            cdr_accuracy, drift_pp, best_ckpt_path,
        )
    elif not eligible:
        log.info(
            "Not eligible: framework accuracy drifted %.3fpp from the reference "
            "(tolerance %.2fpp)", drift_pp, recovery_cfg.fr_tolerance_pp,
        )

    eval_points.append({
        "step": global_step,
        "optim_step": optim_step,
        "epoch": epoch,
        "epoch_frac": epoch_frac,
        "val_loss": val_loss,
        "val_perplexity": ppl,
        "cdr_accuracy": cdr_accuracy,
        "framework_accuracy": framework_accuracy,
        "fr_drift_pp": drift_pp,
        "eligible": bool(eligible),
        "selected": bool(selected),
        "checkpoint": str(ckpt_path),
    })
    pd.DataFrame(eval_points).to_csv(run_dir / "eval_points.csv", index=False)

    fig_curves = plot_training_curves(training_history, best_step=None)
    artifacts.log_figure(fig_curves, "figures/training_curves", step=global_step)

    model.train()
    artifacts.flush()
    return best_cdr_accuracy, best_ckpt_path


# ---------------------------------------------------------------------------
# Strategy: evotuning (continued MLM pretraining on a corpus)
# ---------------------------------------------------------------------------


def _reconstruct_histories(artifacts: "RunArtifacts") -> tuple[list, list]:
    """Rebuild training_history and scoring_history from artifacts on disk.

    Maps logged key names back to the dict shapes the plotting helpers expect.
    """
    training_history: list = []
    scoring_history: list = []
    for step in sorted(artifacts._history.keys()):
        row = artifacts._history[step]
        if "train/loss" in row:
            training_history.append({
                "step": step,
                "train_loss": row["train/loss"],
                "learning_rate": row.get("train/lr"),
                "epoch": row.get("train/epoch"),
            })
        if "val/loss" in row:
            training_history.append({
                "step": step,
                "val_loss": row["val/loss"],
                "val_perplexity": row.get("val/perplexity"),
                "epoch": row.get("train/epoch"),
            })
        score_entry: dict = {"step": step, **{
            k[5:]: v for k, v in row.items() if k.startswith("eval/")
        }}
        if len(score_entry) > 1:
            scoring_history.append(score_entry)
    return training_history, scoring_history


def _load_resume_checkpoint(
    path: str,
    model: ESM2Model,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    log: logging.Logger,
) -> tuple[int, int, int, float, Optional[float]]:
    """Restore training state from an evotuning checkpoint.

    Returns (epoch, global_step, samples_seen_this_epoch, framework_accuracy,
    pareto_reference_framework_accuracy). The caller decides what to do about
    an epoch rollover and about a missing reference.
    """
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    epoch = int(ckpt.get("epoch", 1))
    global_step = int(ckpt.get("global_step", 0))
    samples_seen = int(ckpt.get("samples_seen", 0))
    framework_accuracy = float(ckpt.get("framework_accuracy", float("nan")))
    reference = ckpt.get("pareto_reference_framework_accuracy", None)
    reference = float(reference) if reference is not None else None
    log.info(
        "Loaded checkpoint %s: epoch=%d global_step=%d samples_seen=%d "
        "framework_accuracy=%s reference=%s",
        path, epoch, global_step, samples_seen,
        f"{framework_accuracy:.4f}" if framework_accuracy == framework_accuracy else "None",
        f"{reference:.4f}" if reference is not None else "None",
    )
    return epoch, global_step, samples_seen, framework_accuracy, reference


def _base_model_framework_accuracy(
    model_cfg: ModelConfig,
    corpus: Any,
    val_indices: np.ndarray,
    device: torch.device,
    max_seq_len: int,
    n_samples: int,
    seed: int,
    batch_size: int,
    log: logging.Logger,
) -> float:
    """Framework accuracy of the base pretrained model, on the same validation
    subsample every checkpoint in this run is scored against.

    Built as a throwaway instance from the model preset, so it is the true base
    model regardless of which checkpoint this particular run starts from.
    """
    vanilla_cfg = replace(model_cfg, lora=None, freeze_lm_head=False)
    vanilla_model = ESM2Model(vanilla_cfg)
    vanilla_model.to(device)
    result = region_stratified_masked_recovery_accuracy(
        vanilla_model, corpus, val_indices, device,
        max_seq_len=max_seq_len, batch_size=batch_size, n_samples=n_samples, seed=seed,
    )
    log.info(
        "Base model reference: framework accuracy %.4f (CDR accuracy %.4f, "
        "%d framework / %d CDR positions over %d sequences)",
        result["framework_accuracy"], result["cdr_accuracy"],
        result["n_framework_positions"], result["n_cdr_positions"],
        result["n_sequences_used"],
    )
    del vanilla_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return float(result["framework_accuracy"])


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
    artifacts: RunArtifacts,
    log_every_n_steps: int,
) -> tuple[list, list, int, Optional[Path], dict]:
    """Run corpus-MLM training. Returns (training_history, scoring_history,
    global_step, selected_checkpoint_or_None, final_metrics)."""
    accum_steps = training_cfg.gradient_accumulation_steps
    if not data_cfg.pack_path:
        raise ValueError(
            "Evotuning needs data.pack_path (a packed corpus directory). Build "
            "one with scripts/data_prep/pack_corpus.py."
        )

    (
        _initial_train_loader, val_loader, test_loader, train_dataset, collator,
        full_train_len, corpus,
    ) = make_dataloaders(
        pack_path=data_cfg.pack_path,
        max_seq_len=data_cfg.max_seq_len,
        mlm_probability=data_cfg.mlm_probability,
        batch_size=training_cfg.batch_size,
        split_cfg=data_cfg.split,
        policy=data_cfg.policy,
        tokenizer=model.tokenizer,
        skip_samples=0,
        epoch_seed=run_cfg.seed + 1,
        cdr_mask_prob=data_cfg.cdr_mask_prob,
        hybrid_cdr_frac=data_cfg.hybrid_cdr_frac,
        subsample_n=data_cfg.subsample_n,
        subsample_seed=data_cfg.subsample_seed,
        val_max_sequences=data_cfg.val_max_sequences,
    )
    del _initial_train_loader

    batches_per_epoch = full_train_len // training_cfg.batch_size
    optim_steps_per_epoch = max(batches_per_epoch // accum_steps, 1)
    eval_steps = set(
        _eval_step_grid(
            training_cfg.eval_at_epoch_fracs, optim_steps_per_epoch,
            training_cfg.max_epochs, log,
        )
    )

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=training_cfg.learning_rate,
        weight_decay=0.01,
    )
    max_steps = training_cfg.max_steps
    # Size the schedule from the full training length, not the loader's: on
    # resume the loader is truncated, but the schedule has to keep the same
    # shape as the original run so the restored state lands on the right rate.
    epoch_based_steps = training_cfg.max_epochs * optim_steps_per_epoch
    num_training_steps = max_steps if max_steps else epoch_based_steps
    warmup_steps = int(round(training_cfg.warmup_ratio * num_training_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
    log.info(
        "Schedule: %d optimizer steps, %d warmup (ratio %.3f), %d batches/epoch, "
        "effective batch %d",
        num_training_steps, warmup_steps, training_cfg.warmup_ratio,
        batches_per_epoch, training_cfg.batch_size * accum_steps,
    )

    use_bf16 = training_cfg.bf16 and device.type == "cuda"
    if use_bf16:
        # bf16 tensor cores start at compute capability 8.0. On older cards
        # autocast still runs, but on fp32 units and several times slower, and
        # nothing in the logs says so. A sweep that lands on the wrong card by
        # accident should stop here rather than quietly take a week.
        major = torch.cuda.get_device_capability(device)[0]
        name = torch.cuda.get_device_name(device)
        if major < 8:
            raise RuntimeError(
                f"training.bf16 is set but {name} (compute capability "
                f"{major}.x) has no bf16 tensor cores, so training would fall "
                f"back to fp32 and run several times slower. Request a GPU that "
                f"supports it (--gpus=rtx_4090:1 or --gpus=a100_80gb:1), or set "
                f"training.bf16=false to accept the slowdown deliberately."
            )
        log.info("Training in bf16 on %s (compute capability %d.x)", name, major)
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float32

    start_epoch = 1
    start_samples_seen = 0
    start_global_step = 0
    resumed_framework_accuracy: Optional[float] = None
    resumed_reference: Optional[float] = None
    if training_cfg.resume_checkpoint:
        (
            start_epoch, start_global_step, start_samples_seen,
            resumed_framework_accuracy, resumed_reference,
        ) = _load_resume_checkpoint(
            training_cfg.resume_checkpoint, model, optimizer, scheduler, log,
        )
        if start_samples_seen >= full_train_len:
            start_epoch += 1
            start_samples_seen = 0
        # The order is rebuilt per epoch from the seed, so resuming mid-epoch
        # only works if the skip lands on a batch boundary.
        start_samples_seen -= start_samples_seen % training_cfg.batch_size
        log.info(
            "Resuming at epoch=%d, global_step=%d, samples_seen_this_epoch=%d",
            start_epoch, start_global_step, start_samples_seen,
        )

    tokenizer = model.tokenizer
    val_indices = val_loader.dataset.base_indices

    # The reference the selection rule holds this run's checkpoints against.
    # A single-position continuation is measured against its own branch point,
    # which the config carries explicitly; anything else is measured against
    # the base pretrained model.
    if training_cfg.pareto_reference_framework_accuracy is not None:
        reference_fr_accuracy = float(training_cfg.pareto_reference_framework_accuracy)
        log.info(
            "Reference framework accuracy pinned by config: %.4f", reference_fr_accuracy
        )
    elif training_cfg.resume_checkpoint and data_cfg.policy == "single_pool":
        if resumed_framework_accuracy != resumed_framework_accuracy:
            raise ValueError(
                "A single-position continuation is measured against its branch "
                "point's framework accuracy, but the resume checkpoint does not "
                "record one. Pass training.pareto_reference_framework_accuracy "
                "explicitly."
            )
        reference_fr_accuracy = resumed_framework_accuracy
        log.info(
            "Reference framework accuracy taken from the branch point: %.4f",
            reference_fr_accuracy,
        )
    elif resumed_reference is not None:
        reference_fr_accuracy = resumed_reference
        log.info("Reference framework accuracy restored from checkpoint: %.4f",
                 reference_fr_accuracy)
    else:
        reference_fr_accuracy = _base_model_framework_accuracy(
            model_cfg, corpus, val_indices, device,
            max_seq_len=data_cfg.max_seq_len,
            n_samples=training_cfg.recovery_n_samples,
            seed=training_cfg.recovery_seed,
            batch_size=training_cfg.recovery_batch_size, log=log,
        )
    recovery_cfg = RecoveryEvalConfig(
        corpus=corpus,
        val_indices=val_indices,
        reference_framework_accuracy=reference_fr_accuracy,
        fr_tolerance_pp=training_cfg.pareto_fr_tolerance_pp,
        max_seq_len=data_cfg.max_seq_len,
        n_samples=training_cfg.recovery_n_samples,
        seed=training_cfg.recovery_seed,
        batch_size=training_cfg.recovery_batch_size,
    )

    scoring_datasets = None
    if scoring_cfg.datasets:
        scoring_datasets = load_scoring_datasets(
            scoring_cfg.datasets, n_samples=scoring_cfg.n_samples, seed=run_cfg.seed,
        )
        log.info(
            "Loaded %d scoring dataset(s) for the end-of-training report. These "
            "are never used to select a checkpoint.", len(scoring_datasets),
        )

    model.train()
    running_loss = 0.0
    running_masked = 0
    running_seqs = 0
    log_steps = 0
    global_step = start_global_step
    optim_step = global_step // accum_steps
    best_cdr_accuracy = float("-inf")
    best_ckpt_path: Optional[Path] = None

    max_epochs = training_cfg.max_epochs
    hit_max_steps = False
    training_history: list = []
    scoring_history: list = []
    eval_points: list = []
    if training_cfg.resume_checkpoint:
        training_history, scoring_history = _reconstruct_histories(artifacts)

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
            total=batches_per_epoch,
        )
        for batch in progress:
            global_step += 1
            samples_seen_this_epoch += training_cfg.batch_size
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=use_bf16):
                outputs = model(**batch)
                loss = outputs.loss / accum_steps

            loss.backward()
            running_loss += outputs.loss.item()
            # The masking rate differs a lot between policies, and the CDR
            # policies mask far fewer residues per sequence than whole-chain
            # does. Record it so the comparison is legible instead of implied.
            running_masked += int((batch["labels"] != -100).sum().item())
            running_seqs += batch["labels"].size(0)
            log_steps += 1

            if global_step % accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optim_step += 1

                if optim_step in eval_steps:
                    epoch_frac = (
                        (optim_step - (epoch - 1) * optim_steps_per_epoch)
                        / optim_steps_per_epoch
                    )
                    best_cdr_accuracy, best_ckpt_path = _run_periodic_eval(
                        model=model,
                        val_loader=val_loader,
                        device=device,
                        global_step=global_step,
                        optim_step=optim_step,
                        epoch=epoch,
                        epoch_frac=epoch_frac,
                        epoch_seed=epoch_seed,
                        samples_seen=samples_seen_this_epoch,
                        run_dir=run_dir,
                        checkpoint_dir=checkpoint_dir,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        training_history=training_history,
                        eval_points=eval_points,
                        artifacts=artifacts,
                        log=log,
                        train_start=train_start,
                        recovery_cfg=recovery_cfg,
                        best_cdr_accuracy=best_cdr_accuracy,
                        best_ckpt_path=best_ckpt_path,
                    )

            if global_step % log_every_n_steps == 0:
                avg_loss = running_loss / log_steps
                lr = scheduler.get_last_lr()[0]
                masked_per_seq = running_masked / max(running_seqs, 1)
                artifacts.log(
                    {
                        "train/loss": avg_loss,
                        "train/lr": lr,
                        "train/epoch": epoch,
                        "train/masked_per_seq": masked_per_seq,
                    },
                    step=global_step,
                )
                progress.set_postfix(
                    loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}", masked=f"{masked_per_seq:.1f}"
                )
                log.info(
                    "Epoch %d Step %d — loss %.4f — lr %.2e — masked/seq %.2f",
                    epoch, global_step, avg_loss, lr, masked_per_seq,
                )
                training_history.append({
                    "step": global_step,
                    "train_loss": avg_loss,
                    "learning_rate": lr,
                    "epoch": epoch,
                    "wall_time": time.time() - train_start,
                })
                running_loss = 0.0
                running_masked = 0
                running_seqs = 0
                log_steps = 0

            if max_steps and optim_step >= max_steps:
                log.info("Reached max_steps=%d, stopping.", max_steps)
                hit_max_steps = True
                break

        artifacts.log({"train/epoch": epoch}, step=global_step)
        if hit_max_steps:
            break

    final_path = checkpoint_dir / "final.pt"
    final_metrics: dict = {}
    if len(val_loader.dataset) > 0:
        final_ppl, final_val_loss = compute_perplexity(
            model, val_loader, device, max_batches=max(len(val_loader), 1),
        )
        artifacts.log(
            {"val/loss": final_val_loss, "val/perplexity": final_ppl}, step=global_step
        )
        log.info("Final val loss %.4f — val perplexity %.2f", final_val_loss, final_ppl)
        training_history.append({
            "step": global_step,
            "val_loss": final_val_loss,
            "val_perplexity": final_ppl,
            "wall_time": time.time() - train_start,
        })
        final_metrics["val_loss"] = float(final_val_loss)
        final_metrics["val_perplexity"] = float(final_ppl)
    else:
        final_ppl = float("inf")
        log.info("Skipping final val perplexity (empty validation split)")

    torch.save(
        {
            "epoch": max_epochs,
            "global_step": global_step,
            "optim_step": optim_step,
            "samples_seen": full_train_len,
            "epoch_seed": run_cfg.seed + max_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_perplexity": final_ppl,
            "pareto_reference_framework_accuracy": reference_fr_accuracy,
        },
        final_path,
    )
    log.info("Training complete. Final checkpoint: %s", final_path)

    final_metrics["pareto_reference_framework_accuracy"] = float(reference_fr_accuracy)
    if best_ckpt_path is not None:
        selected = next(p for p in reversed(eval_points) if p["selected"])
        final_metrics["selected_cdr_accuracy"] = float(best_cdr_accuracy)
        final_metrics["selected_framework_accuracy"] = float(selected["framework_accuracy"])
        final_metrics["selected_epoch_frac"] = float(selected["epoch_frac"])
        final_metrics["selected_step"] = int(selected["step"])
        final_metrics["selected_ckpt"] = str(best_ckpt_path)
        log.info(
            "Selected checkpoint: %.4f of epoch %d (step %d), CDR accuracy %.4f -> %s",
            selected["epoch_frac"], selected["epoch"], selected["step"],
            best_cdr_accuracy, best_ckpt_path,
        )
    else:
        log.warning(
            "No checkpoint was eligible: framework accuracy left the %.2fpp band "
            "around the reference at every evaluation point. Nothing was selected.",
            training_cfg.pareto_fr_tolerance_pp,
        )

    _run_end_of_training_eval(
        model=model,
        tokenizer=tokenizer,
        test_loader=test_loader,
        scoring_datasets=scoring_datasets,
        scoring_cfg=scoring_cfg,
        run_cfg=run_cfg,
        device=device,
        global_step=global_step,
        training_history=training_history,
        scoring_history=scoring_history,
        artifacts=artifacts,
        log=log,
        final_metrics=final_metrics,
        masked_final_datasets=None,
    )

    return training_history, scoring_history, global_step, best_ckpt_path, final_metrics


def _run_end_of_training_eval(
    *,
    model: ESM2Model,
    tokenizer: Any,
    test_loader: Any,
    scoring_datasets: Optional[list],
    scoring_cfg: ScoringConfig,
    run_cfg: RunConfig,
    device: torch.device,
    global_step: int,
    training_history: list,
    scoring_history: list,
    artifacts: RunArtifacts,
    log: logging.Logger,
    final_metrics: dict,
    masked_final_datasets: Optional[list] = None,
) -> None:
    """Run test FASTA PPL + test CDR PLL + D2 Spearman/flank + D5 PLL pos/neg/wt.

    When `masked_final_datasets` is provided (single-mask variants), also score
    those held-out CSVs via masked-position Spearman and log `test/spearman_*`.

    Mutates `final_metrics` in place; logs scalars + figures + summary table.
    """
    # 0. Single-mask held-out (test) masked-position Spearman.
    if masked_final_datasets:
        log.info("Running end-of-training masked-position Spearman on held-out test set(s)")
        test_spear = _eval_masked_spearman(
            model, masked_final_datasets, scoring_cfg.batch_size, log,
        )
        artifacts.log({f"test/{k}": v for k, v in test_spear.items()}, step=global_step)
        for k, v in test_spear.items():
            final_metrics[f"test_{k}"] = float(v)

    # 1. Test FASTA perplexity (existing behavior, kept).
    if len(test_loader.dataset) > 0:
        test_ppl, test_loss = compute_perplexity(
            model, test_loader, device, max_batches=max(len(test_loader), 1),
        )
        artifacts.log({"test/loss": test_loss, "test/perplexity": test_ppl}, step=global_step)
        log.info("Final test loss: %.4f — test perplexity: %.2f", test_loss, test_ppl)
        final_metrics["test_loss"] = float(test_loss)
        final_metrics["test_perplexity"] = float(test_ppl)
    else:
        log.info("Skipping final test perplexity (empty test split)")

    # 2. Test C05 CDR-H3 PLL (single sanity number; framework-conditioned).
    test_cdr_ppl = corpus_perplexity([C05_CDRH3], scorer=model, cdr_only=True)
    artifacts.log({"test/cdr_ppl": test_cdr_ppl}, step=global_step)
    final_metrics["test_cdr_ppl"] = float(test_cdr_ppl)
    log.info("Final test CDR-H3 PLL perplexity: %.4f", test_cdr_ppl)

    # 3. Full D2 Spearman + flank breakdown on the same scoring datasets.
    if scoring_datasets is not None:
        log.info("Running end-of-training D2 Spearman + flank evaluation")
        full_results = run_multi_scoring_evaluation(
            model, tokenizer, scoring_datasets,
            device=device,
            batch_size=scoring_cfg.batch_size,
            seed=run_cfg.seed,
            flank_ks=scoring_cfg.flank_ks,
            scorer=model,
            live_only=False,
            return_payload=True,
        )
        payload = full_results.pop("_payload", {})

        # Aggregate spearman_mean for the W&B headline.
        rhos = [v for k, v in full_results.items() if k.startswith("spearman_avg_") and "_pval_" not in k and isinstance(v, float) and np.isfinite(v) and not any(s in k for s in ("left", "right", "pos", "neg"))]
        spearman_mean = float(np.mean(rhos)) if rhos else float("nan")
        final_metrics["test_spearman_mean"] = spearman_mean
        artifacts.log({"test/spearman_mean": spearman_mean}, step=global_step)

        # Persist full per-dataset Spearman + flank into final_metrics for metrics.json.
        for k, v in full_results.items():
            final_metrics[f"test_{k}"] = (float(v) if isinstance(v, (int, float)) else v)

        dataset_names = [d[0] for d in scoring_datasets]

        # Write per-dataset (PLL, enrichment, num_mut) CSVs for offline figure regen.
        if scoring_cfg.persist_test_scores:
            for name, df, enrichment_col in scoring_datasets:
                if name not in payload:
                    continue
                p = payload[name]
                cols: dict = {}
                if "aa" in df.columns:
                    cols["aa"] = df["aa"].to_numpy()
                if "mut" in df.columns:
                    cols["mut"] = df["mut"].to_numpy()
                if "num_mut" in df.columns:
                    cols["num_mut"] = df["num_mut"].to_numpy()
                cols["pll"] = p["scores"]
                cols["enrichment"] = p["enrichment"]
                out_df = pd.DataFrame(cols)
                out_path = artifacts.write_scores_csv(name, out_df, suffix="test")
                log.info("Wrote per-sequence test scores: %s", out_path)

        # Final figures: scatter grid (headline), flank breakdown, training curves redux.
        if payload:
            fig_scatter = plot_pll_vs_enrichment_grid(payload)
            artifacts.log_figure(fig_scatter, "figures/test_pll_vs_enrichment", step=global_step)

        fig_flank = plot_flank_breakdown(full_results, dataset_names, scoring_cfg.flank_ks)
        artifacts.log_figure(fig_flank, "figures/test_flank_breakdown", step=global_step)

        fig_curves = plot_training_curves(training_history)
        artifacts.log_figure(fig_curves, "figures/training_curves_final", step=global_step)

        fig_evol = plot_spearman_evolution(scoring_history, dataset_names)
        artifacts.log_figure(fig_evol, "figures/spearman_evolution_final", step=global_step)

    # 4. D5 (ED5) PLL pos/neg/wt — only if test_eval is configured.
    test_eval_cfg = scoring_cfg.test_eval
    if test_eval_cfg is not None and test_eval_cfg.raw_ed5_path:
        d5_path = ensure_test_eval_csv(
            raw_ed5_path=Path(test_eval_cfg.raw_ed5_path),
            processed_dir=Path(test_eval_cfg.processed_dir or (artifacts.run_dir / "d5_processed")),
            log=log,
        )
        eval_sets = load_test_pll_eval_sets(d5_path, log, pos_threshold=test_eval_cfg.pos_threshold)
        if eval_sets is not None:
            ppl_metrics = evaluate_pll_eval_sets(model=model, eval_sets=eval_sets)
            # Rename keys so they sit under `test/` consistently with the rest.
            renamed = {f"test/{k.replace('ppl/', 'ppl_')}": v for k, v in ppl_metrics.items()}
            artifacts.log(renamed, step=global_step)
            for k, v in renamed.items():
                final_metrics[k.replace("test/", "test_")] = float(v)
            log.info(
                "Final test D5 PLL — pos: %.4f, neg: %.4f, wt: %.4f",
                ppl_metrics.get("ppl/test_pos", float("nan")),
                ppl_metrics.get("ppl/test_neg", float("nan")),
                ppl_metrics.get("ppl/test_wt", float("nan")),
            )
            fig_pll = plot_pll_comparison(ppl_metrics, label="test")
            artifacts.log_figure(fig_pll, "figures/test_pll_comparison", step=global_step)

    # 5. Headline summary table — flat dict mirroring wandb.run.summary.
    summary = {k: v for k, v in final_metrics.items() if isinstance(v, (int, float))}
    if artifacts.wandb_mod is not None:
        try:
            table = build_summary_table(summary, artifacts.wandb_mod)
            artifacts.wandb_mod.log({"summary/results_table": table}, step=global_step)
        except Exception as exc:
            log.warning("Failed to log W&B summary table: %s", exc)

    artifacts.flush()


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
    artifacts: RunArtifacts,
    log_every_n_steps: int,
    ttt_eval_datasets: Optional[dict[str, pd.DataFrame]] = None,
    eval_batch_size: int = 64,
) -> tuple[list, list, int, Path, dict]:
    """Run TTT on a single sequence. Returns (training_history, eval_history,
    global_step, final_ckpt_path, final_metrics)."""
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

    training_history: list[dict] = []
    eval_history: list[dict] = []
    model.train()

    snapshot_steps = set(int(s) for s in (training_cfg.snapshot_steps or []))
    if snapshot_steps:
        log.info("Will snapshot + eval at TTT steps: %s", sorted(snapshot_steps))

    # --- Step 0: evaluate the starting model before any gradient ---
    if ttt_eval_datasets:
        log.info("=== TTT step 0 eval (pre-training) ===")
        step0_results = _ttt_snapshot_eval(
            model, step=0, ttt_eval_datasets=ttt_eval_datasets,
            eval_batch_size=eval_batch_size, artifacts=artifacts, log=log,
        )
        eval_history.append({"step": 0, **step0_results})
        artifacts.flush()

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

        # Per-step label-free signal on the WT CDR-H3.
        model.eval()
        wt_cdr_ppl = corpus_perplexity([C05_CDRH3], scorer=model, cdr_only=True)
        model.train()

        artifacts.log(
            {"train/loss": avg_loss, "train/step": step, "train/wt_cdr_ppl": wt_cdr_ppl},
            step=step,
        )
        log.info("Step %d/%d — loss: %.4f — wt CDR-H3 PPL: %.4f", step, max_steps, avg_loss, wt_cdr_ppl)
        training_history.append({
            "step": step,
            "train_loss": avg_loss,
            "wt_cdr_ppl": wt_cdr_ppl,
            "learning_rate": training_cfg.learning_rate,
            "wall_time": time.time() - train_start,
        })

        if step in snapshot_steps:
            snap_path = checkpoint_dir / f"step_{step}.pt"
            model.save_state(snap_path, extra={"global_step": step})
            log.info("Saved TTT snapshot: %s", snap_path)

            if ttt_eval_datasets:
                log.info("=== TTT step %d eval ===", step)
                snap_results = _ttt_snapshot_eval(
                    model, step=step, ttt_eval_datasets=ttt_eval_datasets,
                    eval_batch_size=eval_batch_size, artifacts=artifacts, log=log,
                )
                eval_history.append({"step": step, **snap_results})

            artifacts.flush()

    final_path = checkpoint_dir / "final.pt"
    model.save_state(final_path, extra={"global_step": max_steps})
    log.info("Saved final TTT checkpoint: %s", final_path)

    final_metrics: dict = {}
    if eval_history:
        # Write per-step eval history CSV.
        eval_df = pd.DataFrame(eval_history)
        eval_csv = run_dir / "ttt_eval_history.csv"
        eval_df.to_csv(eval_csv, index=False)
        log.info("Wrote TTT eval history: %s", eval_csv)

        # Best step by ED2 Spearman (if available), else by spearman_mean.
        rank_col = "spearman_ED2" if "spearman_ED2" in eval_df.columns else "spearman_mean"
        snap_df = eval_df[eval_df["step"] > 0]
        if not snap_df.empty:
            best_idx = snap_df[rank_col].map(lambda v: v if np.isfinite(v) else -np.inf).idxmax()
            best_step = int(snap_df.loc[best_idx, "step"])
            best_row = snap_df.loc[best_idx]
            log.info(
                "Best TTT step = %d (%s=%.4f)", best_step, rank_col, best_row[rank_col],
            )
            final_metrics["best_ttt_step"] = best_step
            for col in eval_df.columns:
                if col != "step":
                    final_metrics[f"best_{col}"] = float(best_row[col])

        # Markdown summary table.
        cols = [c for c in eval_df.columns if c != "step"]
        header = "| step | " + " | ".join(cols) + " |"
        sep = "| --- | " + " | ".join(["---"] * len(cols)) + " |"
        rows = [header, sep]
        for _, row in eval_df.iterrows():
            cells = [str(int(row["step"]))] + [f"{row[c]:.4f}" for c in cols]
            rows.append("| " + " | ".join(cells) + " |")
        md = "\n".join(rows)
        (run_dir / "ttt_eval_table.md").write_text(md + "\n")
        log.info("Results table:\n%s", md)

    return training_history, eval_history, max_steps, final_path, final_metrics


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
    downstream stages should seed from: the checkpoint the selection rule picked
    for evotuning, final.pt for TTT."""
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

    # Build the model WITHOUT LoRA first so we can load any base/evotuned
    # finetune checkpoint directly. LoRA (if requested) attaches afterwards.
    deferred_lora = getattr(model_cfg, "lora", None)
    deferred_freeze_lm_head = bool(getattr(model_cfg, "freeze_lm_head", False))
    base_cfg = replace(model_cfg, lora=None, freeze_lm_head=False)
    model = ESM2Model(base_cfg)

    if run_cfg.finetune:
        run_log.info("Loading finetune checkpoint: %s", run_cfg.finetune)
        ckpt = torch.load(run_cfg.finetune, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])

    if deferred_lora is not None:
        model.attach_lora(deferred_lora)
    if deferred_freeze_lm_head:
        model.freeze_lm_head()

    summary = model.param_summary()
    run_log.info(
        "Parameters — total: %s, trainable: %s, frozen: %s",
        f"{summary['total']:,}", f"{summary['trainable']:,}", f"{summary['frozen']:,}",
    )
    model.to(device)

    train_start = time.time()
    set_publication_style()
    artifacts = RunArtifacts(run_dir=run_dir, wandb_mod=wandb_mod)

    if stage_type == "evotuning":
        training_history, scoring_history, global_step, best_ckpt_path, final_metrics = _train_evotuning(
            model, model_cfg, data_cfg, training_cfg, scoring_cfg, run_cfg,
            run_dir, checkpoint_dir, device, train_start,
            log=run_log, artifacts=artifacts, log_every_n_steps=log_every_n_steps,
        )
        handoff_ckpt = best_ckpt_path if best_ckpt_path is not None else checkpoint_dir / "final.pt"
    else:  # ttt
        ttt_eval_datasets: dict[str, pd.DataFrame] = {}
        if cfg is not None:
            try:
                ttt_eval_datasets = _load_ttt_eval_datasets(cfg, run_log)
            except Exception as exc:
                run_log.warning("Could not load TTT eval datasets: %s", exc)
        ttt_eval_batch_size = max(int(scoring_cfg.batch_size or 0), 64)
        training_history, scoring_history, global_step, handoff_ckpt, final_metrics = _train_ttt(
            model, model_cfg, data_cfg, training_cfg, scoring_cfg, run_cfg,
            run_dir, checkpoint_dir, device, train_start,
            log=run_log, artifacts=artifacts, log_every_n_steps=log_every_n_steps,
            ttt_eval_datasets=ttt_eval_datasets or None,
            eval_batch_size=ttt_eval_batch_size,
        )

    # ------------------------------------------------------------------
    # On-disk artifacts: split layout (metrics.json + summary.json + history.csv).
    # `scoring_history` is *not* embedded — its scalars are captured in
    # history.csv by RunArtifacts.log; per-sequence test scores live in
    # scores/*.csv. metrics.json stays small (metadata + final scalars).
    # ------------------------------------------------------------------
    final_scalars = {k: float(v) for k, v in final_metrics.items() if isinstance(v, (int, float))}
    metrics = {
        "metadata": {
            "run_name": run_name,
            "total_steps": global_step,
            "total_time_seconds": round(time.time() - train_start, 2),
            "device": str(device),
            "param_summary": summary,
        },
        "final": final_scalars,
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    run_log.info("Saved metrics to %s", run_dir / "metrics.json")

    summary_payload = {"run_name": run_name, **final_scalars}
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary_payload, f, indent=2)
    run_log.info("Saved summary to %s", run_dir / "summary.json")

    artifacts.flush()
    if artifacts.history_path.exists():
        run_log.info("Saved history to %s", artifacts.history_path)

    if wandb_run is not None:
        try:
            wandb_run.summary.update(summary_payload)
        except Exception as exc:
            run_log.warning("Failed to update W&B run summary: %s", exc)
        wandb_run.finish()

    return handoff_ckpt
