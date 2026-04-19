"""Shared Hydra-backed config dataclasses and builders.

Evotuning- and DPO-specific configs (DataConfig/TrainingConfig, DpoConfig, ...)
live in their respective subpackages.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple, TypedDict

from omegaconf import DictConfig, OmegaConf


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
class ScoringConfig:
    """Scoring / evaluation configuration."""

    n_samples: int = 10000
    batch_size: int = 512
    datasets: Optional[List[dict]] = None
    flank_ks: List[int] = field(default_factory=lambda: [1, 3, 5])


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


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def build_model_config(cfg: DictConfig, device: str | None = None) -> ModelConfig:
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


def build_scoring_config(cfg: DictConfig) -> ScoringConfig:
    datasets = (
        OmegaConf.to_container(cfg.scoring.datasets, resolve=True)
        if cfg.scoring.datasets
        else None
    )
    flank_ks_cfg = cfg.scoring.get("flank_ks", [1, 3, 5])
    flank_ks = [int(k) for k in flank_ks_cfg]
    return ScoringConfig(
        n_samples=int(cfg.scoring.n_samples),
        batch_size=int(cfg.scoring.batch_size),
        datasets=datasets or None,
        flank_ks=flank_ks,
    )


def build_run_config(cfg: DictConfig) -> RunConfig:
    return RunConfig(
        train_dir=cfg.paths.train_dir,
        project_dir=cfg.paths.get("project_dir"),
        wandb_project=cfg.wandb.project,
        seed=int(cfg.seed),
        finetune=cfg.get("finetune"),
    )


def generate_run_name(cfg: DictConfig) -> str:
    """Generate a timestamped run name from config or auto-generate one."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if cfg.run_name:
        return f"{cfg.run_name}_{timestamp}"
    return f"run_{timestamp}"
