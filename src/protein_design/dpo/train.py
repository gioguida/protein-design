"""Hydra-driven DPO training entrypoint.

This script:
1) builds preference pairs from clustered D2 data
2) splits pairs into train/val/test
3) trains policy ESM2Model with DPO against a frozen reference ESM2Model
4) logs metrics to console/file and optionally Weights & Biases
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict

import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset


from protein_design.eval import run_scoring_evaluation
from protein_design.utils import init_wandb, setup_train_logger
from .dataset import (
    DELTA_BASED_COMPONENTS,
    build_split_pair_dataframes_from_raw,
    default_data_paths,
    validate_delta_based_components,
)
from .data_processing import build_clean_ed5_csv, build_validation_perplexity_csvs
from .loss import batch_monitoring_metrics, dpo_loss, weighted_dpo_loss
from protein_design.config import ModelConfig
from protein_design.constants import WILD_TYPE
from protein_design.model import ESM2Model
from .utils import (
    build_full_run_name,
    load_hydra_runtime_modules,
    log_pair_diagnostics,
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

    pairing_strategy = str(cfg.data.pairing_strategy)
    delta_cfg = getattr(cfg.data, "delta_based", None)

    def _delta_value(name: str, default: float) -> float:
        if delta_cfg is not None and getattr(delta_cfg, name, None) is not None:
            return float(getattr(delta_cfg, name))
        return float(getattr(cfg.data, name, default))

    if pairing_strategy == "delta_based":
        if delta_cfg is None:
            raise ValueError(
                "data.delta_based is required when data.pairing_strategy='delta_based'."
            )
        components = validate_delta_based_components(
            [str(component) for component in delta_cfg.components]
        )
    else:
        components = DELTA_BASED_COMPONENTS

    return build_split_pair_dataframes_from_raw(
        pairing_strategy=pairing_strategy,
        include_views=[str(v) for v in cfg.data.include_views],
        raw_csv_path=raw_csv_path,
        processed_dir=processed_dir,
        force_rebuild=bool(cfg.data.force_rebuild),
        min_positive_delta=float(cfg.data.min_positive_delta),
        min_delta_margin=float(cfg.data.min_delta_margin),
        delta_components=components,
        gap=_delta_value("gap", 0.5),
        wt_pairs_frac=_delta_value("wt_pairs_frac", 0.1),
        cross_pairs_frac=_delta_value("cross_pairs_frac", 0.1),
        strong_pos_threshold=_delta_value("strong_pos_threshold", 1.0),
        strong_neg_threshold=_delta_value("strong_neg_threshold", -5.0),
        min_score_margin=_delta_value("min_score_margin", 0.1),
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


def _resolve_model_init_checkpoint(cfg: Any) -> Optional[Path]:
    init_cfg = getattr(cfg.model, "init", None)
    if init_cfg is None:
        return None

    source = str(getattr(init_cfg, "source", "base")).strip().lower()
    checkpoint = getattr(init_cfg, "checkpoint", None)
    if source == "base":
        return None
    if source != "checkpoint":
        raise ValueError(
            f"Unsupported model.init.source={source!r}. Expected 'base' or 'checkpoint'."
        )
    if checkpoint is None:
        raise ValueError(
            "model.init.checkpoint is required when model.init.source='checkpoint'."
        )
    checkpoint_path = Path(to_absolute_path(str(checkpoint)))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model init checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def _load_model_init_state_dict(checkpoint_path: Path) -> Tuple[str, Dict[str, torch.Tensor]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise TypeError(
            f"Checkpoint at {checkpoint_path} must contain a dict payload, got {type(ckpt)}."
        )
    if "model_state_dict" in ckpt:
        return "model_state_dict", ckpt["model_state_dict"]
    if "policy_state_dict" in ckpt:
        return "policy_state_dict", ckpt["policy_state_dict"]
    raise KeyError(
        f"Checkpoint at {checkpoint_path} does not include 'model_state_dict' or 'policy_state_dict'."
    )


def _build_scorers(cfg: Any, logger: logging.Logger) -> Tuple[ESM2Model, ESM2Model]:
    model_cfg = ModelConfig(
        esm_model_path=str(cfg.model.esm_model_path),
        device=str(cfg.training.device),
        use_context=bool(cfg.model.use_context),
        pll_mask_chunk_size=int(getattr(cfg.model, "pll_mask_chunk_size", 64)),
    )

    policy = ESM2Model(model_cfg)
    reference = ESM2Model(model_cfg)

    init_checkpoint = _resolve_model_init_checkpoint(cfg)
    if init_checkpoint is not None:
        state_key, state_dict = _load_model_init_state_dict(init_checkpoint)
        if state_key == "model_state_dict":
            policy.load_state_dict(state_dict)
            reference.load_state_dict(state_dict)
        else:
            policy.model.load_state_dict(state_dict)
            reference.model.load_state_dict(state_dict)
        logger.info(
            "Initialized policy/reference from checkpoint: %s",
            init_checkpoint,
        )

    policy.to(policy.device)
    reference.to(reference.device)

    for param in reference.model.parameters():
        param.requires_grad_(False)
    reference.eval()

    return policy, reference


def _build_optimizer_and_scheduler(
    cfg: Any,
    policy: ESM2Model,
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
    policy: ESM2Model,
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
    policy: ESM2Model,
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


def _resolve_storage_paths(full_run_name: str) -> Dict[str, Path]:
    """Resolve storage tiers from env vars with cluster-safe defaults."""
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"
    scratch_dir = Path(
        os.environ.get("SCRATCH_DIR", f"/cluster/scratch/{user}/protein-design")
    )
    train_dir = Path(os.environ.get("TRAIN_DIR", str(scratch_dir / "train")))
    wandb_dir = Path(os.environ.get("WANDB_DIR", str(scratch_dir / "wandb")))
    project_dir = Path(
        os.environ.get("PROJECT_DIR", f"/cluster/project/infk/krause/{user}/protein-design")
    )
    return {
        "scratch_dir": scratch_dir,
        "train_dir": train_dir,
        "wandb_dir": wandb_dir,
        "project_dir": project_dir,
        "run_dir": train_dir / full_run_name,
        "archive_dir": project_dir / "checkpoints" / full_run_name,
    }


def _run_epoch(
    policy: ESM2Model,
    reference: ESM2Model,
    dataloader: DataLoader,
    loss: str, 
    beta: float,
    temperature: float,
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
    loss_name = str(loss)

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
            if loss_name == "dpo":
                batch_loss = dpo_loss(
                    batch,
                    beta=beta,
                    scorer=policy,
                    reference=reference,
                    policy_use_grad=is_train,
                )
            elif loss_name == "weighted_dpo":
                batch_loss = weighted_dpo_loss(
                    batch,
                    beta=beta,
                    temperature=temperature,
                    scorer=policy,
                    reference=reference,
                    policy_use_grad=is_train,
                )
            else:
                raise ValueError(f"Unsupported loss: {loss_name}")
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
            batch_loss.backward()
            if grad_clip_norm > 0:
                clip_grad_norm_(policy.model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

        total_loss += float(batch_loss.item())
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
                "train_step/loss": float(batch_loss.item()),
            }
            if track_metrics:
                logger.info(
                    "step=%d train_loss=%.6f reward_acc=%.4f reward_margin=%.4f implicit_kl=%.4f",
                    step,
                    float(batch_loss.item()),
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
                logger.info("step=%d train_loss=%.6f", step, float(batch_loss.item()))
            
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


def _compute_chosen_perplexity(dataloader: DataLoader, scorer: ESM2Model) -> float:
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


def _resolve_processed_dir(cfg: Any) -> Path:
    defaults = default_data_paths()
    return (
        defaults["processed_dir"]
        if cfg.data.processed_dir is None
        else Path(to_absolute_path(str(cfg.data.processed_dir)))
    )


def _resolve_raw_csv_path(cfg: Any) -> Path:
    defaults = default_data_paths()
    return (
        defaults["raw_m22"]
        if cfg.data.raw_csv is None
        else Path(to_absolute_path(str(cfg.data.raw_csv)))
    )


def _resolve_ed5_raw_csv_path(cfg: Any) -> Path:
    defaults = default_data_paths()
    fallback = defaults["raw_m22"].parent / "ED5_M22_enrichment.csv"
    test_cfg = getattr(cfg.data, "test", None)
    ed5_csv = None if test_cfg is None else getattr(test_cfg, "ed5_csv", None)
    return fallback if ed5_csv is None else Path(to_absolute_path(str(ed5_csv)))


def _ensure_validation_eval_csvs(cfg: Any, logger: logging.Logger) -> bool:
    """Ensure val_pos/val_neg CSVs exist for validation perplexity and Spearman tracking."""
    processed_dir = _resolve_processed_dir(cfg)
    val_pos_path = processed_dir / "val_pos.csv"
    val_neg_path = processed_dir / "val_neg.csv"
    if val_pos_path.exists() and val_neg_path.exists():
        return True

    raw_csv_path = _resolve_raw_csv_path(cfg)
    try:
        outputs = build_validation_perplexity_csvs(
            raw_csv_path=raw_csv_path,
            processed_dir=processed_dir,
            cfg=cfg,
            seed=int(cfg.seed),
            force=False,
            verbose=False,
        )
    except Exception as exc:
        logger.warning(
            "Could not build validation eval CSVs at %s from %s (%s).",
            processed_dir,
            raw_csv_path,
            exc,
        )
        return False

    if (not val_pos_path.exists()) or (not val_neg_path.exists()):
        logger.warning(
            "Validation eval CSV build finished but files are still missing at %s and %s.",
            val_pos_path,
            val_neg_path,
        )
        return False

    logger.info(
        "Prepared validation eval CSVs: val_pos=%s val_neg=%s",
        outputs["val_pos"],
        outputs["val_neg"],
    )
    return True


def _ensure_test_eval_csv(cfg: Any, logger: logging.Logger) -> Optional[Path]:
    """Ensure processed ED5 CSV exists for test perplexity logging."""
    processed_dir = _resolve_processed_dir(cfg)
    d5_path = processed_dir / "D5.csv"
    raw_ed5_path = _resolve_ed5_raw_csv_path(cfg)

    try:
        output_path = build_clean_ed5_csv(
            raw_csv_path=raw_ed5_path,
            processed_dir=processed_dir,
            force=bool(getattr(cfg.data, "force_rebuild", False)),
            verbose=False,
        )
    except Exception as exc:
        logger.warning(
            "Could not build ED5 processed CSV at %s from %s (%s). Skipping test perplexity logging.",
            d5_path,
            raw_ed5_path,
            exc,
        )
        return None

    if not output_path.exists():
        logger.warning(
            "ED5 processed CSV build finished but file is still missing at %s. Skipping test perplexity logging.",
            output_path,
        )
        return None

    return output_path


def _load_sequences_from_csv(csv_path: Path, logger: logging.Logger) -> List[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")

    df = pd.read_csv(csv_path)
    if "aa" not in df.columns:
        raise ValueError(f"{csv_path} is missing required column 'aa'.")

    sequences = [str(seq).strip() for seq in df["aa"].dropna().astype(str).tolist()]
    sequences = [seq for seq in sequences if seq]
    if len(sequences) == 0:
        logger.warning("No valid sequences found in %s.", csv_path)
    return sequences


def _load_validation_pll_eval_sets(cfg: Any, logger: logging.Logger) -> Optional[Dict[str, List[str]]]:
    _ensure_validation_eval_csvs(cfg, logger)
    processed_dir = _resolve_processed_dir(cfg)
    val_pos_path = processed_dir / "val_pos.csv"
    val_neg_path = processed_dir / "val_neg.csv"

    if (not val_pos_path.exists()) or (not val_neg_path.exists()):
        logger.warning(
            "Validation perplexity sets missing (expected %s and %s). Skipping validation perplexity logging.",
            val_pos_path,
            val_neg_path,
        )
        return None

    try:
        val_pos = _load_sequences_from_csv(val_pos_path, logger)
        val_neg = _load_sequences_from_csv(val_neg_path, logger)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        logger.warning(
            "Could not load validation perplexity sets (%s). Skipping validation perplexity logging.",
            exc,
        )
        return None

    return {
        "ppl/val_pos": val_pos,
        "ppl/val_neg": val_neg,
        "ppl/val_wt": [WILD_TYPE],
    }


def _load_validation_spearman_df(cfg: Any, logger: logging.Logger) -> Optional[pd.DataFrame]:
    """Load validation scoring rows used for Spearman tracking."""
    _ensure_validation_eval_csvs(cfg, logger)
    processed_dir = _resolve_processed_dir(cfg)
    val_pos_path = processed_dir / "val_pos.csv"
    val_neg_path = processed_dir / "val_neg.csv"

    if (not val_pos_path.exists()) or (not val_neg_path.exists()):
        logger.warning(
            "Validation Spearman sets missing (expected %s and %s). Skipping Spearman logging.",
            val_pos_path,
            val_neg_path,
        )
        return None

    try:
        val_pos_df = pd.read_csv(val_pos_path)
        val_neg_df = pd.read_csv(val_neg_path)
    except (FileNotFoundError, pd.errors.ParserError) as exc:
        logger.warning("Could not read validation Spearman sets (%s). Skipping Spearman logging.", exc)
        return None

    val_df = pd.concat([val_pos_df, val_neg_df], ignore_index=True)
    required_cols = {"mut", "M22_binding_enrichment_adj"}
    missing_cols = required_cols.difference(val_df.columns)
    if missing_cols:
        logger.warning(
            "Validation Spearman set missing columns (%s). Skipping Spearman logging.",
            ", ".join(sorted(missing_cols)),
        )
        return None

    if "num_mut" in val_df.columns:
        val_df = val_df[val_df["num_mut"] == 2].copy()

    enrichment = pd.to_numeric(val_df["M22_binding_enrichment_adj"], errors="coerce")
    val_df = val_df.loc[enrichment.notna()].copy()
    val_df["M22_binding_enrichment_adj"] = enrichment.loc[enrichment.notna()].astype(float)
    val_df["mut"] = val_df["mut"].astype(str).str.strip()
    val_df = val_df[val_df["mut"] != ""].copy()

    if len(val_df) < 3:
        logger.warning("Validation Spearman set too small (%d rows). Skipping Spearman logging.", len(val_df))
        return None

    return val_df.reset_index(drop=True)


def _load_test_pll_eval_sets(cfg: Any, logger: logging.Logger) -> Optional[Dict[str, List[str]]]:
    d5_path = _ensure_test_eval_csv(cfg, logger)
    if d5_path is None:
        return None

    try:
        d5_df = pd.read_csv(d5_path)
    except pd.errors.ParserError as exc:
        logger.warning("Could not parse %s (%s). Skipping test perplexity logging.", d5_path, exc)
        return None

    required_cols = {"aa", "M22_binding_enrichment_adj"}
    missing_cols = required_cols.difference(d5_df.columns)
    if missing_cols:
        logger.warning(
            "%s missing required columns (%s). Skipping test perplexity logging.",
            d5_path,
            ", ".join(sorted(missing_cols)),
        )
        return None

    pos_threshold = float(getattr(getattr(cfg.data, "test", None), "pos_threshold", 0.0))
    clean_scores = pd.to_numeric(d5_df["M22_binding_enrichment_adj"], errors="coerce")
    clean_df = d5_df.loc[clean_scores.notna()].copy()
    clean_df["M22_binding_enrichment_adj"] = clean_scores.loc[clean_scores.notna()].astype(float)
    clean_df["aa"] = clean_df["aa"].astype(str).str.strip()
    clean_df = clean_df[clean_df["aa"] != ""].copy()

    test_pos = clean_df[clean_df["M22_binding_enrichment_adj"] > pos_threshold]["aa"].tolist()
    test_neg = clean_df[clean_df["M22_binding_enrichment_adj"] < 0.0]["aa"].tolist()

    return {
        "ppl/test_pos": test_pos,
        "ppl/test_neg": test_neg,
        "ppl/test_wt": [WILD_TYPE],
    }


def _corpus_perplexity_for_sequences(
    scorer: ESM2Model,
    sequences: Sequence[str],
    batch_size: int,
) -> float:
    valid_sequences = [str(seq).strip() for seq in sequences if str(seq).strip()]
    if not valid_sequences:
        return float("nan")

    grouped_by_len: Dict[int, List[str]] = {}
    for seq in valid_sequences:
        if len(seq) <= 0:
            continue
        grouped_by_len.setdefault(len(seq), []).append(seq)

    if not grouped_by_len:
        return float("nan")

    scorer.model.eval()
    total_pll = 0.0
    total_scored_tokens = 0
    batch_size = max(1, int(batch_size))

    with torch.no_grad():
        for cdr_len, seq_group in grouped_by_len.items():
            for start in range(0, len(seq_group), batch_size):
                batch = seq_group[start : start + batch_size]
                pll_scores = scorer.pseudo_log_likelihood(
                    batch,
                    cdr_only=True,
                    use_grad=False,
                )
                total_pll += float(pll_scores.sum().item())
                total_scored_tokens += int(cdr_len) * len(batch)

    if total_scored_tokens <= 0:
        return float("nan")

    avg_nll = -total_pll / float(total_scored_tokens)
    return float(math.exp(avg_nll))


def evaluate_perplexity(
    model: ESM2Model,
    eval_sets: Dict[str, Sequence[str]],
    device: str,
) -> Dict[str, float]:
    del device  # Device is already configured inside the ESM2Model wrapper.
    sequence_batch_size = int(getattr(model, "pll_mask_chunk_size", 64))
    metrics: Dict[str, float] = {}
    for metric_name, sequences in eval_sets.items():
        metrics[metric_name] = _corpus_perplexity_for_sequences(
            scorer=model,
            sequences=sequences,
            batch_size=sequence_batch_size,
        )
    return metrics


def run_dpo(cfg: Any) -> Path:
    full_run_name = build_full_run_name(cfg, timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"))
    storage = _resolve_storage_paths(full_run_name)
    output_dir = storage["run_dir"]

    # Keep W&B cache/artifacts on scratch by default.
    os.environ.setdefault("WANDB_DIR", str(storage["wandb_dir"]))

    for key in ("scratch_dir", "train_dir", "wandb_dir", "run_dir"):
        storage[key].mkdir(parents=True, exist_ok=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_train_logger(output_dir=output_dir, level_name=str(cfg.logging.level))
    logger.info("Starting DPO training run")
    logger.info("Full run name: %s", full_run_name)
    logger.info("Run directory: %s", output_dir)
    logger.info("Archive directory: %s", storage["archive_dir"])

    resolved_cfg_path = output_dir / "resolved_config.yaml"
    run_cfg_path = output_dir / "config.yaml"
    OmegaConf.save(cfg, resolved_cfg_path)
    OmegaConf.save(cfg, run_cfg_path)
    logger.info("Saved resolved config to %s", resolved_cfg_path)
    logger.info("Saved run config to %s", run_cfg_path)

    wandb_mod, wandb_run = init_wandb(
        cfg,
        output_dir,
        logger,
        run_name=full_run_name,
        group="dpo",
    )

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

    policy, reference = _build_scorers(cfg, logger)
    optimizer, scheduler = _build_optimizer_and_scheduler(cfg, policy)

    ckpt_dir = output_dir / str(cfg.checkpointing.dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / str(cfg.checkpointing.best_filename)
    last_ckpt = ckpt_dir / str(cfg.checkpointing.last_filename)
    root_best_ckpt = output_dir / "best.pt"
    final_ckpt = ckpt_dir / "final.pt"
    step_prefix = str(getattr(cfg.checkpointing, "step_prefix", "step"))

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
    _ensure_validation_eval_csvs(cfg, logger)
    val_spearman_df = _load_validation_spearman_df(cfg, logger)
    val_spearman_batch_size = int(getattr(cfg.model, "pll_mask_chunk_size", 64))

    for epoch in range(start_epoch, int(cfg.training.num_epochs) + 1):
        train_metrics, global_step = _run_epoch(
            policy=policy,
            reference=reference,
            dataloader=train_loader,
            loss=str(cfg.training.loss),
            beta=float(cfg.training.beta),
            temperature=float(cfg.training.temperature),
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
            loss=str(cfg.training.loss),
            beta=float(cfg.training.beta),
            temperature=float(cfg.training.temperature),
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

        if val_spearman_df is not None:
            try:
                val_spearman = run_scoring_evaluation(
                    model=policy.model,
                    tokenizer=policy.tokenizer,
                    df=val_spearman_df,
                    enrichment_col="M22_binding_enrichment_adj",
                    device=policy.device,
                    batch_size=val_spearman_batch_size,
                    seed=int(cfg.seed),
                )
                epoch_record.update(
                    {
                        "val_spearman_avg": float(val_spearman["spearman_avg"]),
                        "val_spearman_avg_pval": float(val_spearman["spearman_avg_pval"]),
                        "val_spearman_random": float(val_spearman["spearman_random"]),
                        "val_spearman_random_pval": float(val_spearman["spearman_random_pval"]),
                    }
                )
                logger.info(
                    "Validation Spearman | avg=%.4f (p=%.2e) | random=%.4f (p=%.2e)",
                    float(val_spearman["spearman_avg"]),
                    float(val_spearman["spearman_avg_pval"]),
                    float(val_spearman["spearman_random"]),
                    float(val_spearman["spearman_random_pval"]),
                )
            except Exception as exc:
                logger.warning("Validation Spearman evaluation failed (%s). Skipping this metric.", exc)

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

        val_eval_sets = _load_validation_pll_eval_sets(cfg, logger)
        if val_eval_sets is not None:
            val_ppl_metrics = evaluate_perplexity(
                model=policy,
                eval_sets=val_eval_sets,
                device=str(cfg.training.device),
            )
            epoch_record.update(val_ppl_metrics)
            logger.info(
                "Validation Perplexity | ppl/val_pos=%.4f | ppl/val_neg=%.4f | ppl/val_wt=%.4f",
                float(val_ppl_metrics["ppl/val_pos"]),
                float(val_ppl_metrics["ppl/val_neg"]),
                float(val_ppl_metrics["ppl/val_wt"]),
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
            shutil.copy2(best_ckpt, root_best_ckpt)
            logger.info("Updated run best checkpoint at %s", root_best_ckpt)
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
        if global_step > 0:
            step_ckpt = ckpt_dir / f"{step_prefix}_{global_step}.pt"
            shutil.copy2(last_ckpt, step_ckpt)
            logger.info("Saved step checkpoint to %s", step_ckpt)

        if int(cfg.training.patience) > 0 and epochs_without_improvement >= int(cfg.training.patience):
            logger.info("Early stopping after %d epochs without val improvement.", epochs_without_improvement)
            break

    if bool(cfg.training.evaluate_best_checkpoint) and best_ckpt.exists():
        _load_checkpoint(best_ckpt, policy=policy, optimizer=None, scheduler=None)
        logger.info("Loaded best checkpoint for final test evaluation: %s", best_ckpt)

    if best_ckpt.exists():
        shutil.copy2(best_ckpt, root_best_ckpt)
    if last_ckpt.exists():
        shutil.copy2(last_ckpt, final_ckpt)
        logger.info("Saved final checkpoint to %s", final_ckpt)

    test_metrics, _ = _run_epoch(
        policy=policy,
        reference=reference,
        dataloader=test_loader,
        beta=float(cfg.training.beta),
        temperature=float(cfg.training.temperature),
        loss=str(cfg.training.loss),
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

    test_ppl_metrics: Dict[str, float] = {}
    test_eval_sets = _load_test_pll_eval_sets(cfg, logger)
    if test_eval_sets is not None:
        test_ppl_metrics = evaluate_perplexity(
            model=policy,
            eval_sets=test_eval_sets,
            device=str(cfg.training.device),
        )
        logger.info(
            "Test Perplexity (ED5) | ppl/test_pos=%.4f | ppl/test_neg=%.4f | ppl/test_wt=%.4f",
            float(test_ppl_metrics["ppl/test_pos"]),
            float(test_ppl_metrics["ppl/test_neg"]),
            float(test_ppl_metrics["ppl/test_wt"]),
        )

    history_df = pd.DataFrame(history)
    history_csv_path = output_dir / str(cfg.logging.history_csv)
    history_df.to_csv(history_csv_path, index=False)

    summary = {
        "run_name": full_run_name,
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

    best_epoch = min(history, key=lambda row: float(row["val_loss"])) if history else None
    val_ppl = None if best_epoch is None else best_epoch.get("val_perplexity")
    if isinstance(val_ppl, float) and math.isnan(val_ppl):
        val_ppl = None
    val_spearman_m22 = None if best_epoch is None else best_epoch.get("val_spearman_avg")
    if isinstance(val_spearman_m22, float) and math.isnan(val_spearman_m22):
        val_spearman_m22 = None

    metrics_payload = {
        "run_name": full_run_name,
        "model": str(cfg.model.esm_model_path),
        "val_ppl": val_ppl,
        "spearman_M22": val_spearman_m22,
        "spearman_SI06": None,
        "spearman_exp": None,
        "notes": None if cfg.wandb.notes is None else str(cfg.wandb.notes),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    archive_dir = storage["archive_dir"]
    archive_dir.mkdir(parents=True, exist_ok=True)
    if root_best_ckpt.exists():
        archived_best = archive_dir / "best.pt"
        shutil.copy2(root_best_ckpt, archived_best)
        logger.info("Archived best checkpoint to %s", archived_best)
    else:
        logger.warning("Best checkpoint not found at %s; skipping archive copy.", root_best_ckpt)

    archived_metrics = archive_dir / "metrics.json"
    shutil.copy2(metrics_path, archived_metrics)
    logger.info("Archived metrics to %s", archived_metrics)

    logger.info("Saved history to %s", history_csv_path)
    logger.info("Saved summary to %s", summary_path)
    logger.info("Saved metrics to %s", metrics_path)

    if wandb_run is not None:
        if test_ppl_metrics:
            wandb_mod.log(test_ppl_metrics, step=global_step)
        wandb_mod.log({f"test/{k}": v for k, v in summary.items()})
        wandb_mod.log({f"metrics/{k}": v for k, v in metrics_payload.items() if k != "run_name"})
        wandb_run.summary.update(summary)
        wandb_run.finish()

    if root_best_ckpt.exists():
        return root_best_ckpt
    return final_ckpt


@hydra.main(version_base=None, config_path="../../../conf", config_name="dpo")
def main(cfg: Any) -> None:
    # Ensure Hydra runtime is initialized (used by Hydra internals/logging).
    HydraConfig.get().runtime.output_dir
    run_dpo(cfg)


if __name__ == "__main__":
    main()



