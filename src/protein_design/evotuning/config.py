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
    return DataConfig(
        fasta_path=cfg.data.fasta_path,
        max_seq_len=int(cfg.data.max_seq_len),
        mlm_probability=float(cfg.data.mlm_probability),
        split=split_cfg,
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
    )
