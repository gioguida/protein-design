"""Hydra-driven DPO training entrypoint.

This script:
1) builds preference pairs from clustered D2 data
2) splits pairs into train/val/test
3) trains policy ESM2 with DPO against a frozen reference ESM2
4) logs metrics to console/file and optionally Weights & Biases
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict

import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

if __package__:
    from .dataset import build_split_pair_dataframes_from_raw, default_data_paths
    from .loss import batch_monitoring_metrics, dpo_loss
    from .model import ESM2PLLScorer
    from .utils import (
        ModelConfig,
        init_wandb_run,
        load_hydra_runtime_modules,
        log_pair_diagnostics,
        setup_train_logger,
    )
else:  # pragma: no cover
    from dataset import build_split_pair_dataframes_from_raw, default_data_paths
    from loss import batch_monitoring_metrics, dpo_loss
    from model import ESM2PLLScorer
    from utils import (
        ModelConfig,
        init_wandb_run,
        load_hydra_runtime_modules,
        log_pair_diagnostics,
        setup_train_logger,
    )


hydra, OmegaConf, HydraConfig, to_absolute_path = load_hydra_runtime_modules()


class PairMember(TypedDict):
    aa: str
    score: float


PairTuple = Tuple[PairMember, PairMember]


class PairDataset(Dataset):
    """Minimal dataset of scored (chosen, rejected) sequence pairs."""

    def __init__(self, pairs_df: pd.DataFrame):
        self.pairs: List[PairTuple] = [
            (
                {"aa": str(chosen_aa), "score": float(chosen_delta)},
                {"aa": str(rejected_aa), "score": float(rejected_delta)},
            )
            for chosen_aa, rejected_aa, chosen_delta, rejected_delta in zip(
                pairs_df["chosen_sequence"],
                pairs_df["rejected_sequence"],
                pairs_df["chosen_delta"],
                pairs_df["rejected_delta"],
            )
        ]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> PairTuple:
        return self.pairs[idx]


def _pair_collate(batch: Sequence[PairTuple]) -> List[PairTuple]:
    return list(batch)


def _build_split_pair_dataframes(cfg: Any) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    defaults = default_data_paths()

    raw_csv_path = (
        defaults["raw_m22"]
        if cfg.data.raw_csv is None
        else Path(to_absolute_path(str(cfg.data.raw_csv)))
    )
    processed_dir = (
        defaults["processed_dir"]
        if cfg.data.processed_dir is None
        else Path(to_absolute_path(str(cfg.data.processed_dir)))
    )

    return build_split_pair_dataframes_from_raw(
        pairing_strategy=str(cfg.data.pairing_strategy),
        include_views=[str(v) for v in cfg.data.include_views],
        raw_csv_path=raw_csv_path,
        processed_dir=processed_dir,
        force_rebuild=bool(cfg.data.force_rebuild),
        min_positive_delta=float(cfg.data.min_positive_delta),
        min_delta_margin=float(cfg.data.min_delta_margin),
        gap=float(getattr(cfg.data, "gap", 0.5)),
        wt_pairs_frac=float(getattr(cfg.data, "wt_pairs_frac", 0.1)),
        deduplicate_across_views=bool(cfg.data.deduplicate_across_views),
        train_frac=float(cfg.data.train_frac),
        val_frac=float(cfg.data.val_frac),
        test_frac=float(cfg.data.test_frac),
        seed=int(cfg.seed),
    )


def _build_dataloader(
    pairs_df: pd.DataFrame,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = None,
) -> DataLoader:
    dataset = PairDataset(pairs_df)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader_kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "collate_fn": _pair_collate,
        "generator": generator,
        "drop_last": False,
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(persistent_workers) and int(num_workers) > 0,
    }
    if int(num_workers) > 0 and prefetch_factor is not None and int(prefetch_factor) > 0:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    return DataLoader(
        **loader_kwargs,
    )


def _build_scorers(cfg: Any) -> Tuple[ESM2PLLScorer, ESM2PLLScorer]:
    model_cfg = ModelConfig(
        esm_model_path=str(cfg.model.esm_model_path),
        device=str(cfg.training.device),
        use_context=bool(cfg.model.use_context),
        pll_mask_chunk_size=int(getattr(cfg.model, "pll_mask_chunk_size", 64)),
    )

    policy = ESM2PLLScorer(model_cfg)
    reference = ESM2PLLScorer(model_cfg)

    for param in reference.model.parameters():
        param.requires_grad_(False)

    return policy, reference


def _build_optimizer_and_scheduler(
    cfg: Any,
    policy: ESM2PLLScorer,
) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
    trainable_params = [p for p in policy.model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable policy parameters found.")

    optimizer = torch.optim.Adam(
        trainable_params,
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )

    scheduler = None
    if bool(cfg.training.scheduler.enabled):
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg.training.scheduler.step_size),
            gamma=float(cfg.training.scheduler.gamma),
        )

    return optimizer, scheduler


def _save_checkpoint(
    path: Path,
    epoch: int,
    policy: ESM2PLLScorer,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    best_val_loss: float,
) -> None:
    state = {
        "epoch": int(epoch),
        "policy_state_dict": policy.model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "best_val_loss": float(best_val_loss),
    }
    torch.save(state, path)


def _load_checkpoint(
    checkpoint_path: Path,
    policy: ESM2PLLScorer,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
) -> Tuple[int, float]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    policy.model.load_state_dict(ckpt["policy_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    next_epoch = int(ckpt.get("epoch", 0)) + 1
    best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
    return next_epoch, best_val_loss


def _export_checkpoint(
    checkpoint_path: Path,
    export_dir: Optional[str],
    export_filename: str,
    checkpoint_label: str,
    logger: logging.Logger,
) -> Optional[Path]:
    if export_dir is None:
        return None
    if not checkpoint_path.exists():
        return None

    target_dir = Path(to_absolute_path(str(export_dir)))
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / str(export_filename)
    shutil.copy2(checkpoint_path, target_path)
    logger.info("Exported %s checkpoint to %s", checkpoint_label, target_path)
    return target_path


def _run_epoch(
    policy: ESM2PLLScorer,
    reference: ESM2PLLScorer,
    dataloader: DataLoader,
    beta: float,
    optimizer: Optional[torch.optim.Optimizer],
    grad_clip_norm: float,
    logger: logging.Logger,
    log_every_n_steps: int,
    track_metrics: bool,
    global_step: int = 0,
    wandb_mod: Optional[Any] = None,
    epoch: int = 0,
) -> Tuple[Dict[str, float], int]:
    is_train = optimizer is not None

    if is_train:
        policy.model.train()
    else:
        policy.model.eval()
    reference.model.eval()

    total_loss = 0.0
    num_batches = 0
    skipped_batches = 0
    monitored_pairs = 0.0
    reward_accuracy_sum = 0.0
    reward_margin_sum = 0.0
    implicit_kl_sum = 0.0

    for step, batch in enumerate(dataloader, start=1):
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        try:
            loss = dpo_loss(
                batch,
                beta=beta,
                scorer=policy,
                reference=reference,
                policy_use_grad=is_train,
            )
            if track_metrics:
                with torch.no_grad():
                    batch_metrics = batch_monitoring_metrics(
                        batch,
                        beta=beta,
                        scorer=policy,
                        reference=reference,
                    )
        except ValueError:
            skipped_batches += 1
            continue

        if is_train:
            loss.backward()
            if grad_clip_norm > 0:
                clip_grad_norm_(policy.model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1
        if track_metrics:
            num_pairs = float(batch_metrics["num_pairs"])
            monitored_pairs += num_pairs
            reward_accuracy_sum += float(batch_metrics["reward_accuracy"]) * num_pairs
            reward_margin_sum += float(batch_metrics["reward_margin"]) * num_pairs
            implicit_kl_sum += float(batch_metrics["implicit_kl"]) * num_pairs

        if is_train and log_every_n_steps > 0 and step % log_every_n_steps == 0:
            step_record = {
                "step": global_step,
                "epoch": epoch,
                "train_step/loss": float(loss.item()),
            }
            if track_metrics:
                logger.info(
                    "step=%d train_loss=%.6f reward_acc=%.4f reward_margin=%.4f implicit_kl=%.4f",
                    step,
                    float(loss.item()),
                    float(batch_metrics["reward_accuracy"]),
                    float(batch_metrics["reward_margin"]),
                    float(batch_metrics["implicit_kl"]),
                )
                step_record.update({
                    "train_step/reward_accuracy": float(batch_metrics["reward_accuracy"]),
                    "train_step/reward_margin": float(batch_metrics["reward_margin"]),
                    "train_step/implicit_kl": float(batch_metrics["implicit_kl"]),
                })
            else:
                logger.info("step=%d train_loss=%.6f", step, float(loss.item()))
            
            if wandb_mod is not None:
                wandb_mod.log(step_record, step=global_step)

    avg_loss = total_loss / max(1, num_batches)
    if track_metrics and monitored_pairs > 0:
        avg_reward_accuracy = reward_accuracy_sum / monitored_pairs
        avg_reward_margin = reward_margin_sum / monitored_pairs
        avg_implicit_kl = implicit_kl_sum / monitored_pairs
    else:
        avg_reward_accuracy = float("nan")
        avg_reward_margin = float("nan")
        avg_implicit_kl = float("nan")

    return {
        "loss": float(avg_loss),
        "reward_accuracy": float(avg_reward_accuracy),
        "reward_margin": float(avg_reward_margin),
        "implicit_kl": float(avg_implicit_kl),
        "num_batches": float(num_batches),
        "num_pairs": float(monitored_pairs),
        "skipped_batches": float(skipped_batches),
    }, global_step


def _compute_chosen_perplexity(dataloader: DataLoader, scorer: ESM2PLLScorer) -> float:
    """Compute corpus-level CDR perplexity with full-sequence context.

    This uses:
        perplexity = exp(total_negative_log_likelihood / total_scored_tokens)
    where the scored tokens are CDR positions only, while model conditioning
    still uses left/right context (when scorer is configured with use_context=True).
    """
    scorer.model.eval()
    total_pll = 0.0
    total_scored_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            chosen_seqs = [str(pair[0]["aa"]) for pair in batch]
            if not chosen_seqs:
                continue

            # scorer.pseudo_log_likelihood requires equal lengths per call.
            # Group inside each batch so we don't skip mixed-length batches.
            by_len: Dict[int, List[str]] = {}
            for seq in chosen_seqs:
                by_len.setdefault(len(seq), []).append(seq)

            for cdr_len, seq_group in by_len.items():
                if cdr_len <= 0:
                    continue
                pll_scores = scorer.pseudo_log_likelihood(
                    seq_group,
                    cdr_only=True,
                    use_grad=False,
                )
                total_pll += float(pll_scores.sum().item())
                total_scored_tokens += int(cdr_len) * len(seq_group)

    if total_scored_tokens <= 0:
        raise ValueError("No valid CDR tokens were evaluated for perplexity.")

    avg_nll = -total_pll / float(total_scored_tokens)
    return float(math.exp(avg_nll))


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: Any) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_train_logger(output_dir=output_dir, level_name=str(cfg.logging.level))
    logger.info("Starting DPO training run")

    resolved_cfg_path = output_dir / "resolved_config.yaml"
    OmegaConf.save(cfg, resolved_cfg_path)
    logger.info("Saved resolved config to %s", resolved_cfg_path)

    wandb_mod, wandb_run = init_wandb_run(cfg, output_dir, logger, OmegaConf)

    train_df, val_df, test_df = _build_split_pair_dataframes(cfg)
    if train_df.empty:
        raise ValueError("No DPO train pairs generated after split. Adjust data pairing settings.")
    if val_df.empty:
        raise ValueError("No DPO validation pairs generated after split. Adjust data pairing settings.")
    if test_df.empty:
        raise ValueError("No DPO test pairs generated after split. Adjust data pairing settings.")

    logger.info("Train split pair diagnostics:")
    log_pair_diagnostics(logger, train_df, preview_count=int(cfg.logging.preview_count))
    logger.info("Validation split pair diagnostics:")
    log_pair_diagnostics(logger, val_df, preview_count=int(cfg.logging.preview_count))
    logger.info("Test split pair diagnostics:")
    log_pair_diagnostics(logger, test_df, preview_count=int(cfg.logging.preview_count))

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
    val_loader = _build_dataloader(
        val_df,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        seed=int(cfg.seed) + 1,
        num_workers=int(cfg.training.num_workers),
        pin_memory=bool(getattr(cfg.training, "pin_memory", False)),
        persistent_workers=bool(getattr(cfg.training, "persistent_workers", False)),
        prefetch_factor=getattr(cfg.training, "prefetch_factor", None),
    )
    test_loader = _build_dataloader(
        test_df,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        seed=int(cfg.seed) + 2,
        num_workers=int(cfg.training.num_workers),
        pin_memory=bool(getattr(cfg.training, "pin_memory", False)),
        persistent_workers=bool(getattr(cfg.training, "persistent_workers", False)),
        prefetch_factor=getattr(cfg.training, "prefetch_factor", None),
    )

    policy, reference = _build_scorers(cfg)
    optimizer, scheduler = _build_optimizer_and_scheduler(cfg, policy)

    ckpt_dir = output_dir / str(cfg.checkpointing.dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / str(cfg.checkpointing.best_filename)
    last_ckpt = ckpt_dir / str(cfg.checkpointing.last_filename)
    best_export_dir = None if cfg.checkpointing.best_export_dir is None else str(cfg.checkpointing.best_export_dir)
    best_export_filename = str(cfg.checkpointing.best_export_filename)
    last_export_dir = None if cfg.checkpointing.last_export_dir is None else str(cfg.checkpointing.last_export_dir)
    last_export_filename = str(cfg.checkpointing.last_export_filename)

    start_epoch = 1
    best_val_loss = float("inf")
    if cfg.training.resume_checkpoint is not None:
        resume_path = Path(to_absolute_path(str(cfg.training.resume_checkpoint)))
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        start_epoch, best_val_loss = _load_checkpoint(
            checkpoint_path=resume_path,
            policy=policy,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        logger.info("Resumed from %s at epoch %d", resume_path, start_epoch)

    history: List[Dict[str, float]] = []
    epochs_without_improvement = 0
    global_step = 0

    for epoch in range(start_epoch, int(cfg.training.num_epochs) + 1):
        train_metrics, global_step = _run_epoch(
            policy=policy,
            reference=reference,
            dataloader=train_loader,
            beta=float(cfg.training.beta),
            optimizer=optimizer,
            grad_clip_norm=float(cfg.training.grad_clip_norm),
            logger=logger,
            log_every_n_steps=int(cfg.logging.log_every_n_steps),
            track_metrics=False,
            global_step=global_step,
            wandb_mod=wandb_mod if wandb_run is not None else None,
            epoch=epoch,
        )

        val_metrics, _ = _run_epoch(
            policy=policy,
            reference=reference,
            dataloader=val_loader,
            beta=float(cfg.training.beta),
            optimizer=None,
            grad_clip_norm=0.0,
            logger=logger,
            log_every_n_steps=0,
            track_metrics=True,
            global_step=global_step,
            wandb_mod=None, # Typically don't log step-level val metrics online this way
            epoch=epoch,
        )

        if scheduler is not None:
            scheduler.step()

        val_perplexity = _compute_chosen_perplexity(val_loader, policy)

        lr = float(optimizer.param_groups[0]["lr"])
        epoch_record = {
            "epoch": float(epoch),
            "lr": lr,
            "train_loss": float(train_metrics["loss"]),
            "train_reward_accuracy": float(train_metrics["reward_accuracy"]),
            "train_reward_margin": float(train_metrics["reward_margin"]),
            "train_implicit_kl": float(train_metrics["implicit_kl"]),
            "val_loss": float(val_metrics["loss"]),
            "val_reward_accuracy": float(val_metrics["reward_accuracy"]),
            "val_reward_margin": float(val_metrics["reward_margin"]),
            "val_implicit_kl": float(val_metrics["implicit_kl"]),
            "val_perplexity": float(val_perplexity),
            "train_batches": float(train_metrics["num_batches"]),
            "train_pairs": float(train_metrics["num_pairs"]),
            "val_batches": float(val_metrics["num_batches"]),
            "val_pairs": float(val_metrics["num_pairs"]),
            "train_skipped": float(train_metrics["skipped_batches"]),
            "val_skipped": float(val_metrics["skipped_batches"]),
        }
        history.append(epoch_record)

        logger.info(
            "Epoch %d | lr=%.3e | train_loss=%.6f | val_loss=%.6f | val_acc=%.4f | val_margin=%.4f | val_kl=%.4f | val_ppl=%.4f | train_batches=%d | val_batches=%d",
            epoch,
            lr,
            epoch_record["train_loss"],
            epoch_record["val_loss"],
            epoch_record["val_reward_accuracy"],
            epoch_record["val_reward_margin"],
            epoch_record["val_implicit_kl"],
            epoch_record["val_perplexity"],
            int(epoch_record["train_batches"]),
            int(epoch_record["val_batches"]),
        )

        if wandb_run is not None:
            wandb_mod.log(epoch_record, step=global_step)

        improved = epoch_record["val_loss"] < best_val_loss
        if improved:
            best_val_loss = epoch_record["val_loss"]
            epochs_without_improvement = 0
            _save_checkpoint(
                best_ckpt,
                epoch=epoch,
                policy=policy,
                optimizer=optimizer,
                scheduler=scheduler,
                best_val_loss=best_val_loss,
            )
            _export_checkpoint(
                checkpoint_path=best_ckpt,
                export_dir=best_export_dir,
                export_filename=best_export_filename,
                checkpoint_label="best",
                logger=logger,
            )
        else:
            epochs_without_improvement += 1

        _save_checkpoint(
            last_ckpt,
            epoch=epoch,
            policy=policy,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_loss=best_val_loss,
        )
        _export_checkpoint(
            checkpoint_path=last_ckpt,
            export_dir=last_export_dir,
            export_filename=last_export_filename,
            checkpoint_label="last",
            logger=logger,
        )

        if int(cfg.training.patience) > 0 and epochs_without_improvement >= int(cfg.training.patience):
            logger.info("Early stopping after %d epochs without val improvement.", epochs_without_improvement)
            break

    if bool(cfg.training.evaluate_best_checkpoint) and best_ckpt.exists():
        _load_checkpoint(best_ckpt, policy=policy, optimizer=None, scheduler=None)
        logger.info("Loaded best checkpoint for final test evaluation: %s", best_ckpt)

    _export_checkpoint(
        checkpoint_path=best_ckpt,
        export_dir=best_export_dir,
        export_filename=best_export_filename,
        checkpoint_label="best",
        logger=logger,
    )

    _export_checkpoint(
        checkpoint_path=last_ckpt,
        export_dir=last_export_dir,
        export_filename=last_export_filename,
        checkpoint_label="last",
        logger=logger,
    )

    test_metrics, _ = _run_epoch(
        policy=policy,
        reference=reference,
        dataloader=test_loader,
        beta=float(cfg.training.beta),
        optimizer=None,
        grad_clip_norm=0.0,
        logger=logger,
        log_every_n_steps=0,
        track_metrics=True,
        global_step=global_step,
        wandb_mod=None,
        epoch=0,
    )
    logger.info(
        "Test metrics | loss=%.6f | reward_acc=%.4f | reward_margin=%.4f | implicit_kl=%.4f | batches=%d | skipped=%d",
        float(test_metrics["loss"]),
        float(test_metrics["reward_accuracy"]),
        float(test_metrics["reward_margin"]),
        float(test_metrics["implicit_kl"]),
        int(test_metrics["num_batches"]),
        int(test_metrics["skipped_batches"]),
    )

    logger.info("Computing perplexity on test set (chosen sequences)...")
    avg_test_perplexity = _compute_chosen_perplexity(test_loader, policy)
    logger.info("Test Chosen Perplexity: %.4f", avg_test_perplexity)

    history_df = pd.DataFrame(history)
    history_csv_path = output_dir / str(cfg.logging.history_csv)
    history_df.to_csv(history_csv_path, index=False)

    summary = {
        "best_val_loss": float(best_val_loss),
        "test_loss": float(test_metrics["loss"]),
        "test_reward_accuracy": float(test_metrics["reward_accuracy"]),
        "test_reward_margin": float(test_metrics["reward_margin"]),
        "test_implicit_kl": float(test_metrics["implicit_kl"]),
        "test_perplexity": float(avg_test_perplexity),
        "test_batches": int(test_metrics["num_batches"]),
        "test_pairs": int(test_metrics["num_pairs"]),
        "test_skipped_batches": int(test_metrics["skipped_batches"]),
        "num_train_pairs": int(len(train_df)),
        "num_val_pairs": int(len(val_df)),
        "num_test_pairs": int(len(test_df)),
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved history to %s", history_csv_path)
    logger.info("Saved summary to %s", summary_path)

    if wandb_run is not None:
        wandb_mod.log({f"test/{k}": v for k, v in summary.items()})
        wandb_run.summary.update(summary)
        wandb_run.finish()


if __name__ == "__main__":
    main()
