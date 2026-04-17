"""Shared utilities: constants, config helpers, and filesystem utilities."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple, TypedDict

from omegaconf import DictConfig, OmegaConf

# ---------------------------------------------------------------------------
# C05 antibody constants (shared across evotuning and DPO pipelines)
# ---------------------------------------------------------------------------

C05_CDRH3 = "HMSMQQVVSAGWERADLVGDAFDV"
WILD_TYPE = C05_CDRH3  # alias used by DPO code

# VH framework flanking the CDR-H3
LEFT_CONTEXT = (
    "EVQLQESGGGLVQPGESLRLSCVGSGSSFGESTLSYYAVSWVRQAPGKGLEWLSIINAGGGDIDYADSVEG"
    "RFTISRDNSKETLYLQMTNLRVEDTGVYYCAK"
)
# Framework 4 + CH1 domain (extends beyond VH terminus)
RIGHT_CONTEXT = (
    "WGQGTMVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSS"
    "GLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEPKSC"
)

# Full C05 VH sequence (framework only, no CH1)
C05_VH = LEFT_CONTEXT + C05_CDRH3 + RIGHT_CONTEXT[:11]
C05_CDRH3_START = len(LEFT_CONTEXT)  # 103
C05_CDRH3_END = C05_CDRH3_START + len(C05_CDRH3)  # 127


def add_context(cdr: str) -> str:
    """Add fixed heavy-chain context around a CDR sequence."""
    return LEFT_CONTEXT + cdr + RIGHT_CONTEXT


@dataclass
class ModelConfig:
    """Configuration for ESM2Model (unified trainable MLM + PLL scorer)."""

    esm_model_path: str = "facebook/esm2_t12_35M_UR50D"
    device: str = "cuda"
    use_context: bool = True
    pll_mask_chunk_size: int = 64
    freeze_embeddings: bool = False
    freeze_first_n_layers: int = 0


@dataclass
class DataConfig:
    """Dataset / tokenizer-side configuration consumed by training."""

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


@dataclass
class ScoringConfig:
    """Scoring / evaluation configuration."""

    n_samples: int = 10000
    batch_size: int = 512
    datasets: Optional[List[dict]] = None


@dataclass
class RunConfig:
    """Top-level run context (paths, seed, wandb, finetune checkpoint)."""

    train_dir: str = ""
    project_dir: Optional[str] = None
    wandb_project: str = "protein-design"
    seed: int = 42
    finetune: Optional[str] = None


class PairMember(TypedDict):
    aa: str
    score: float


PairTuple = Tuple[PairMember, PairMember]


def get_mask_token_idx(token_source: Any) -> int:
    """Resolve mask token index across tokenizer/alphabet API variants."""
    for attr in ("mask_idx", "mask_index", "mask_token_id"):
        if hasattr(token_source, attr):
            return int(getattr(token_source, attr))

    for attr in ("tok_to_idx", "token_to_idx", "stoi", "vocab"):
        mapping = getattr(token_source, attr, None)
        if isinstance(mapping, dict):
            if "<mask>" in mapping:
                return int(mapping["<mask>"])
            if "<mask_token>" in mapping:
                return int(mapping["<mask_token>"])

    raise AttributeError("Could not find mask token index in provided token source.")


def setup_train_logger(
    output_dir: Path, level_name: str, logger_name: str = "dpo_train"
) -> logging.Logger:
    """Create a console + file logger for training."""
    _logger = logging.getLogger(logger_name)
    _logger.setLevel(getattr(logging, str(level_name).upper(), logging.INFO))
    _logger.propagate = False

    if _logger.handlers:
        _logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    _logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    _logger.addHandler(file_handler)

    return _logger


def build_model_config(cfg: DictConfig, device: str | None = None) -> "ModelConfig":
    """Build a ModelConfig from a nested Hydra config.

    Evotuning trains on raw sequences, so `use_context` defaults to False here.
    """
    resolved_device = device or ("cuda" if _cuda_available() else "cpu")
    return ModelConfig(
        esm_model_path=cfg.model.name,
        device=resolved_device,
        use_context=False,
        freeze_embeddings=bool(cfg.model.freeze_embeddings),
        freeze_first_n_layers=int(cfg.model.freeze_first_n_layers),
    )


def build_data_config(cfg: DictConfig) -> "DataConfig":
    return DataConfig(
        fasta_path=cfg.data.fasta_path,
        max_seq_len=int(cfg.data.max_seq_len),
        mlm_probability=float(cfg.data.mlm_probability),
    )


def build_training_config(cfg: DictConfig) -> "TrainingConfig":
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


def build_scoring_config(cfg: DictConfig) -> "ScoringConfig":
    datasets = OmegaConf.to_container(cfg.scoring.datasets, resolve=True) if cfg.scoring.datasets else None
    return ScoringConfig(
        n_samples=int(cfg.scoring.n_samples),
        batch_size=int(cfg.scoring.batch_size),
        datasets=datasets or None,
    )


def build_run_config(cfg: DictConfig) -> "RunConfig":
    return RunConfig(
        train_dir=cfg.paths.train_dir,
        project_dir=cfg.paths.get("project_dir"),
        wandb_project=cfg.wandb.project,
        seed=int(cfg.seed),
        finetune=cfg.get("finetune"),
    )


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def generate_run_name(cfg: DictConfig) -> str:
    """Generate a timestamped run name from config or auto-generate one."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if cfg.run_name:
        return f"{cfg.run_name}_{timestamp}"
    return f"run_{timestamp}"


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
