"""Evotuning-specific config dataclasses and builders.

These mirror the MLM training loop's shape (`cfg.data`, `cfg.training`).
DPO will add its own `DpoConfig` alongside.
"""

from dataclasses import dataclass
from typing import Optional

from omegaconf import DictConfig


@dataclass
class DataConfig:
    """Dataset / tokenizer-side configuration consumed by MLM training."""

    fasta_path: str = ""
    max_seq_len: int = 256
    mlm_probability: float = 0.15


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


def build_data_config(cfg: DictConfig) -> DataConfig:
    return DataConfig(
        fasta_path=cfg.data.fasta_path,
        max_seq_len=int(cfg.data.max_seq_len),
        mlm_probability=float(cfg.data.mlm_probability),
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
    )
