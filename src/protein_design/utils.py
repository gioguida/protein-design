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


def model_config_dict(cfg: DictConfig) -> dict:
    """Extract flat model config dict (for scripts that need a mutable dict)."""
    return {
        "model_name": cfg.model.name,
        "freeze_embeddings": cfg.model.freeze_embeddings,
        "freeze_first_n_layers": cfg.model.freeze_first_n_layers,
    }


def build_model_config(cfg: Any, device: str | None = None) -> "ModelConfig":
    """Build a ModelConfig from a flat training dict or nested Hydra config.

    Accepts either:
      - flat dict with keys `model_name`, `freeze_embeddings`, `freeze_first_n_layers`
      - DictConfig with `.model.name`, `.model.freeze_embeddings`, ...
    Evotuning trains on raw sequences, so `use_context` defaults to False here.
    """
    if isinstance(cfg, DictConfig):
        esm_path = cfg.model.name
        freeze_emb = bool(cfg.model.freeze_embeddings)
        freeze_n = int(cfg.model.freeze_first_n_layers)
    else:
        esm_path = cfg["model_name"]
        freeze_emb = bool(cfg.get("freeze_embeddings", False))
        freeze_n = int(cfg.get("freeze_first_n_layers", 0))

    resolved_device = device or ("cuda" if _cuda_available() else "cpu")
    return ModelConfig(
        esm_model_path=esm_path,
        device=resolved_device,
        use_context=False,
        freeze_embeddings=freeze_emb,
        freeze_first_n_layers=freeze_n,
    )


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def flatten_config(cfg: DictConfig) -> dict:
    """Flatten nested Hydra config to the flat dict that train()/train_ttt() expect.

    This is a transitional adapter. A follow-up PR will refactor the training
    functions to accept nested DictConfig directly.
    """
    flat = OmegaConf.to_container(cfg, resolve=True)

    result = {}
    # Model
    result["model_name"] = flat["model"]["name"]
    result["freeze_embeddings"] = flat["model"]["freeze_embeddings"]
    result["freeze_first_n_layers"] = flat["model"]["freeze_first_n_layers"]
    # Data
    result["fasta_path"] = flat["data"]["fasta_path"]
    result["max_seq_len"] = flat["data"]["max_seq_len"]
    result["mlm_probability"] = flat["data"]["mlm_probability"]
    # Training
    result.update(flat["training"])
    # Paths
    result["train_dir"] = flat["paths"]["train_dir"]
    result["project_dir"] = flat["paths"]["project_dir"]
    # W&B
    result["wandb_project"] = flat["wandb"]["project"]
    # Top-level
    result["seed"] = flat["seed"]
    result["finetune"] = flat.get("finetune")
    # Scoring
    result["scoring_n_samples"] = flat["scoring"]["n_samples"]
    result["scoring_batch_size"] = flat["scoring"]["batch_size"]
    result["scoring_datasets"] = flat["scoring"]["datasets"] or None
    return result


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
