"""Evotuning-specific config dataclasses and builders.

These mirror the MLM training loop's shape (`cfg.data`, `cfg.training`).
DPO adds its own `DpoConfig` alongside.
"""

from dataclasses import dataclass, field
from typing import Optional

from omegaconf import DictConfig

from protein_design.evotuning.splits import SplitConfig

# Evaluation points as fractions of an epoch. Dense at the start because the
# checkpoint worth keeping can appear well before the first tenth of an epoch
# on a corpus this size, then regular for the rest of the run.
DEFAULT_EVAL_FRACS = [
    0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
]


@dataclass
class DataConfig:
    """Dataset / tokenizer-side configuration consumed by MLM training."""

    # Packed corpus directory (see scripts/data_prep/pack_corpus.py). Required
    # for evotuning.
    pack_path: str = ""
    # Single-sequence FASTA, used by the TTT stage only.
    fasta_path: str = ""
    max_seq_len: int = 256
    mlm_probability: float = 0.15
    split: SplitConfig = field(default_factory=SplitConfig)
    # "wc15" | "cdr50" | "hybrid" | "single_pool". See evotuning/data.py.
    policy: str = "wc15"
    # Share of CDR-H3 residues masked by the CDR policies.
    cdr_mask_prob: float = 0.5
    # Hybrid only: share of the samples in every batch that take CDR masking.
    hybrid_cdr_frac: float = 0.8
    # Shrinks the training split only, for the learning-rate sweep. None uses
    # the whole split.
    subsample_n: Optional[int] = None
    subsample_seed: int = 0
    # Caps the validation split so in-loop metrics cost the same on any corpus.
    val_max_sequences: Optional[int] = 5000


@dataclass
class TrainingConfig:
    """Training-loop hyperparameters (mirrors `cfg.training.*`)."""

    learning_rate: float = 1.0e-5
    # Warmup as a share of total steps. Tracks corpus size and stays comparable
    # across the learning-rate grid, unlike a fixed step count.
    warmup_ratio: float = 0.05
    max_epochs: int = 1
    max_steps: Optional[int] = None
    batch_size: int = 64
    gradient_accumulation_steps: int = 8
    bf16: bool = True
    resume_checkpoint: Optional[str] = None
    # Evaluation points, as fractions of an epoch. Each is converted to an
    # optimizer step, clamped to at least one step, and de-duplicated, so a
    # small corpus simply gets a shorter grid.
    eval_at_epoch_fracs: list[float] = field(
        default_factory=lambda: list(DEFAULT_EVAL_FRACS)
    )
    # A checkpoint is eligible for selection only while its framework accuracy
    # stays within this many percentage points of the reference, in either
    # direction.
    pareto_fr_tolerance_pp: float = 0.1
    # Sequences sampled for the region-stratified recovery metric at each
    # evaluation point, and the seed fixing which ones.
    recovery_n_samples: int = 2000
    recovery_seed: int = 42
    # Framework accuracy of the checkpoint this run branched from. Set on the
    # single-position phase, whose reference is its own branch point rather
    # than the base pretrained model.
    pareto_reference_framework_accuracy: Optional[float] = None
    # TTT-only: optimizer-step indices at which to snapshot the model.
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
    subsample_n = cfg.data.get("subsample_n", None)
    val_max = cfg.data.get("val_max_sequences", 5000)
    return DataConfig(
        pack_path=str(cfg.data.get("pack_path", "") or ""),
        fasta_path=str(cfg.data.get("fasta_path", "") or ""),
        max_seq_len=int(cfg.data.max_seq_len),
        mlm_probability=float(cfg.data.get("mlm_probability", 0.15)),
        split=split_cfg,
        policy=str(cfg.data.get("policy", "wc15")),
        cdr_mask_prob=float(cfg.data.get("cdr_mask_prob", 0.5)),
        hybrid_cdr_frac=float(cfg.data.get("hybrid_cdr_frac", 0.8)),
        subsample_n=int(subsample_n) if subsample_n is not None else None,
        subsample_seed=int(cfg.data.get("subsample_seed", 0)),
        val_max_sequences=int(val_max) if val_max is not None else None,
    )


def build_training_config(cfg: DictConfig) -> TrainingConfig:
    t = cfg.training
    fracs = t.get("eval_at_epoch_fracs", None)
    pareto_ref = t.get("pareto_reference_framework_accuracy", None)
    return TrainingConfig(
        learning_rate=float(t.learning_rate),
        warmup_ratio=float(t.get("warmup_ratio", 0.05)),
        max_epochs=int(t.get("max_epochs", 1)),
        max_steps=t.max_steps if t.get("max_steps", None) is not None else None,
        batch_size=int(t.batch_size),
        gradient_accumulation_steps=int(t.gradient_accumulation_steps),
        bf16=bool(t.get("bf16", True)),
        resume_checkpoint=t.get("resume_checkpoint", None),
        eval_at_epoch_fracs=(
            [float(x) for x in fracs] if fracs is not None else list(DEFAULT_EVAL_FRACS)
        ),
        pareto_fr_tolerance_pp=float(t.get("pareto_fr_tolerance_pp", 0.1)),
        recovery_n_samples=int(t.get("recovery_n_samples", 2000)),
        recovery_seed=int(t.get("recovery_seed", 42)),
        pareto_reference_framework_accuracy=(
            float(pareto_ref) if pareto_ref is not None else None
        ),
        snapshot_steps=[int(s) for s in t.get("snapshot_steps", []) or []],
    )
