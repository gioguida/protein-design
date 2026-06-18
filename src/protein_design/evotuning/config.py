"""Evotuning-specific config dataclasses and builders.

These mirror the MLM training loop's shape (`cfg.data`, `cfg.training`).
DPO will add its own `DpoConfig` alongside.
"""

from dataclasses import dataclass, field
from typing import Optional

from omegaconf import DictConfig

from protein_design.evotuning.splits import SplitConfig


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
    cdr_cache = cfg.data.get("cdr_windows_cache", None)
    return DataConfig(
        fasta_path=cfg.data.fasta_path,
        max_seq_len=int(cfg.data.max_seq_len),
        mlm_probability=float(cfg.data.mlm_probability),
        split=split_cfg,
        masking=masking,
        cdr_flank=int(cfg.data.get("cdr_flank", 3)),
        cdr_windows_cache=str(cdr_cache) if cdr_cache is not None else None,
        cdr_mask_prob=float(cfg.data.get("cdr_mask_prob", 0.5)),
        framework_mask_prob=float(cfg.data.get("framework_mask_prob", 0.05)),
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
    )
