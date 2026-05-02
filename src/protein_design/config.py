"""Shared Hydra-backed config dataclasses and builders.

Evotuning- and DPO-specific configs (DataConfig/TrainingConfig, DpoConfig, ...)
live in their respective subpackages.
"""

from dataclasses import dataclass, field
from datetime import datetime
import re
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
    """Top-level run context (paths, seed, finetune checkpoint)."""

    train_dir: str = ""
    project_dir: Optional[str] = None
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
    """Build a ModelConfig from a nested Hydra config."""
    resolved_device = device or ("cuda" if _cuda_available() else "cpu")
    return ModelConfig(
        esm_model_path=cfg.model.name,
        device=resolved_device,
        use_context=bool(cfg.model.get("use_context", True)),
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
    finetune = cfg.get("finetune")
    model_init = cfg.get("model", {}).get("init") if "model" in cfg else None
    if model_init is not None:
        source = str(model_init.get("source", "huggingface")).strip().lower()
        checkpoint = model_init.get("checkpoint")
        if source == "checkpoint":
            if checkpoint is None:
                raise ValueError(
                    "model.init.checkpoint is required when model.init.source='checkpoint'."
                )
            finetune = str(checkpoint)
        elif source in {"huggingface", "base"}:
            finetune = None
        else:
            raise ValueError(
                f"Unsupported model.init.source={source!r}. "
                "Expected 'huggingface' or 'checkpoint'."
            )

    return RunConfig(
        train_dir=cfg.paths.train_dir,
        project_dir=cfg.paths.get("project_dir"),
        seed=int(cfg.seed),
        finetune=finetune,
    )


def generate_run_name(cfg: DictConfig) -> str:
    """Generate a timestamped run name from config or auto-generate one."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if cfg.run_name:
        return f"{cfg.run_name}_{timestamp}"

    task_name = (
        str(cfg.task.get("name", "")).strip().lower()
        if "task" in cfg and cfg.task is not None
        else ""
    )
    if task_name == "unlikelihood":
        training_cfg = cfg.get("training", {})
        data_cfg = cfg.get("data", {})
        model_cfg = cfg.get("model", {})

        model_name = str(model_cfg.get("name", "model")).split("/")[-1]
        lr = training_cfg.get("learning_rate", None)
        batch_size = training_cfg.get("batch_size", None)
        alpha = training_cfg.get("alpha", None)
        max_epochs = training_cfg.get("max_epochs", None)
        max_steps = training_cfg.get("max_steps", None)
        enrichment_threshold = data_cfg.get("enrichment_threshold", None)

        def _token(value: object, default: str = "na") -> str:
            if value is None:
                return default
            text = str(value).strip()
            if text == "":
                return default
            return re.sub(r"[^A-Za-z0-9._-]+", "-", text)

        def _num_token(value: object, default: str = "na") -> str:
            if value is None:
                return default
            try:
                number = float(value)
            except (TypeError, ValueError):
                return _token(value, default=default)
            if number.is_integer():
                return str(int(number))
            return f"{number:.3g}".replace("+", "")

        steps_or_epochs = (
            f"steps-{_num_token(max_steps)}"
            if max_steps is not None
            else f"ep-{_num_token(max_epochs)}"
        )
        base = "_".join(
            [
                "ul",
                _token(model_name),
                steps_or_epochs,
                f"lr-{_num_token(lr)}",
                f"bs-{_num_token(batch_size)}",
                f"a-{_num_token(alpha)}",
                f"thr-{_num_token(enrichment_threshold)}",
            ]
        )
        return f"{base}_{timestamp}"
    return f"run_{timestamp}"
