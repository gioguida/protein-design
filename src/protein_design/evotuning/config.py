"""Evotuning-specific config dataclasses and builders.

These mirror the MLM training loop's shape (`cfg.data`, `cfg.training`).
DPO will add its own `DpoConfig` alongside.
"""

from dataclasses import dataclass, field
from typing import Optional

from omegaconf import DictConfig

from protein_design.evotuning.splits import SplitConfig, cdr_windows_cache_path


@dataclass
class DataConfig:
    """Dataset / tokenizer-side configuration consumed by MLM training."""

    fasta_path: str = ""
    max_seq_len: int = 256
    mlm_probability: float = 0.15
    split: SplitConfig = field(default_factory=SplitConfig)
    # Single-position masking mode:
    #   None (legacy 80/10/10 MLM) | "random15" | "cdr" | "cdr_mix".
    masking: Optional[str] = None
    cdr_flank: int = 3
    cdr_windows_cache: Optional[str] = None
    # CDR modes only: per-epoch keep-prob for CDR-window positions, and (for
    # "cdr_mix") for framework positions. See SingleMaskDataset.
    cdr_mask_prob: float = 0.5
    framework_mask_prob: float = 0.05
    # "always" (default, [MASK] every time) | "bert_80_10_10" (80% [MASK],
    # 10% random amino acid, 10% keep true residue). See SingleMaskDataset.
    mask_replace_strategy: str = "always"
    # "hybrid" mode only: per-example probability of CDR-window masking
    # (vs. whole-chain masking) for that example. See HybridMaskDataset.
    hybrid_cdr_sample_prob: float = 0.8


@dataclass
class TrainingConfig:
    """Training-loop hyperparameters (mirrors `cfg.training.*`)."""

    learning_rate: float = 2.0e-5
    warmup_steps: int = 0
    max_epochs: int = 1
    max_steps: Optional[int] = None
    batch_size: int = 128
    gradient_accumulation_steps: int = 1
    save_every_n_steps: Optional[int] = None
    # When set, overrides save_every_n_steps: derived at runtime from
    # full_train_len (see _train_evotuning) so the eval fraction of an epoch
    # is the same regardless of masking mode. None -> use save_every_n_steps
    # as-is (existing behavior).
    eval_every_epoch_frac: Optional[float] = None
    fp16: bool = False
    resume_checkpoint: Optional[str] = None
    # When True, run eval + checkpoint at the end of every epoch (instead of
    # the step-based save_every_n_steps cadence). Used by the single-mask
    # C05 variants for clean per-epoch comparison.
    eval_per_epoch: bool = False
    # Early stopping (requires eval_per_epoch): stop after this many consecutive
    # per-epoch evals without a val-perplexity improvement of at least
    # `early_stopping_min_delta`. None disables early stopping.
    early_stopping_patience: Optional[int] = None
    early_stopping_min_delta: float = 0.0
    # Which best-checkpoint tracker(s) to run at each eval point. Not a
    # stopping condition — every run always trains the full capped epoch
    # regardless of what's listed here. "perplexity" -> best.pt (existing,
    # always effectively on). "pareto" -> best_pareto.pt, the region-
    # stratified CDR/framework masked-recovery Pareto-knee rule (see
    # report/evotuning.md, "Stopping rule for evotuning").
    checkpoint_trackers: list[str] = field(default_factory=lambda: ["perplexity"])
    # Pareto tracker only: a checkpoint is eligible to become the new best
    # only if its framework accuracy is within this many percentage points
    # of the reference (vanilla model) framework accuracy.
    pareto_fr_tolerance_pp: float = 0.1
    # TTT-only: optimizer-step indices at which to snapshot the model
    # (LoRA adapter weights if LoRA is active, else full state_dict).
    # Empty list = no snapshots beyond final.pt.
    snapshot_steps: list[int] = field(default_factory=list)


def build_data_config(cfg: DictConfig) -> DataConfig:
    split_node = cfg.data.get("split") if "split" in cfg.data else None
    if split_node is None:
        split_cfg = SplitConfig()
    else:
        split_cfg = SplitConfig(
            salt=str(split_node.get("salt", "oas-v1")),
            train_pct=int(split_node.get("train_pct", 90)),
            val_pct=int(split_node.get("val_pct", 5)),
            test_pct=int(split_node.get("test_pct", 5)),
        )
    masking = cfg.data.get("masking", None)
    masking = str(masking) if masking is not None else None
    cdr_flank = int(cfg.data.get("cdr_flank", 3))
    cdr_cache = cfg.data.get("cdr_windows_cache", None)
    cdr_cache = str(cdr_cache) if cdr_cache is not None else None
    # When the cache path is not explicitly pinned in YAML, auto-derive it from
    # (fasta_path, cdr_flank) using the same naming convention as the cache
    # builder. This makes sweeping cdr_flank / swapping fasta_path "just work"
    # (as long as the matching cache has been built once) without per-flank
    # cdr_windows_cache= overrides. Unconditional (not gated on masking mode):
    # the Pareto checkpoint tracker needs a CDR-window cache regardless of
    # which masking mode is actually training, e.g. for the plain
    # whole-chain (masking=null) condition.
    if cdr_cache is None:
        cdr_cache = cdr_windows_cache_path(
            str(cfg.paths.scratch_dir), str(cfg.data.fasta_path), cdr_flank
        )
    return DataConfig(
        fasta_path=cfg.data.fasta_path,
        max_seq_len=int(cfg.data.max_seq_len),
        mlm_probability=float(cfg.data.mlm_probability),
        split=split_cfg,
        masking=masking,
        cdr_flank=cdr_flank,
        cdr_windows_cache=cdr_cache,
        cdr_mask_prob=float(cfg.data.get("cdr_mask_prob", 0.5)),
        framework_mask_prob=float(cfg.data.get("framework_mask_prob", 0.05)),
        mask_replace_strategy=str(cfg.data.get("mask_replace_strategy", "always")),
        hybrid_cdr_sample_prob=float(cfg.data.get("hybrid_cdr_sample_prob", 0.8)),
    )


def build_training_config(cfg: DictConfig) -> TrainingConfig:
    t = cfg.training
    return TrainingConfig(
        learning_rate=float(t.learning_rate),
        warmup_steps=int(t.warmup_steps),
        max_epochs=int(t.max_epochs),
        max_steps=t.max_steps if t.max_steps is not None else None,
        batch_size=int(t.batch_size),
        gradient_accumulation_steps=int(t.gradient_accumulation_steps),
        save_every_n_steps=t.save_every_n_steps if t.save_every_n_steps is not None else None,
        eval_every_epoch_frac=(
            float(t.get("eval_every_epoch_frac"))
            if t.get("eval_every_epoch_frac", None) is not None
            else None
        ),
        fp16=bool(t.fp16),
        resume_checkpoint=t.get("resume_checkpoint", None),
        snapshot_steps=[int(s) for s in t.get("snapshot_steps", []) or []],
        eval_per_epoch=bool(t.get("eval_per_epoch", False)),
        early_stopping_patience=(
            int(t.get("early_stopping_patience"))
            if t.get("early_stopping_patience", None) is not None
            else None
        ),
        early_stopping_min_delta=float(t.get("early_stopping_min_delta", 0.0)),
        checkpoint_trackers=[str(x) for x in (t.get("checkpoint_trackers", ["perplexity"]) or ["perplexity"])],
        pareto_fr_tolerance_pp=float(t.get("pareto_fr_tolerance_pp", 0.1)),
    )
