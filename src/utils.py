from dataclasses import dataclass
import importlib
import logging
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import pandas as pd
from esme.alphabet import Alphabet, tokenize

LEFT_CONTEXT = (
	"EVQLQESGGGLVQPGESLRLSCVGSGSSFGESTLSYYAVSWVRQAPGKGLEWLSIINAGGGDIDYADSVEGRFTISRDNSKETLYLQMTNLRVEDTGVYYCAK"
)
RIGHT_CONTEXT = (
	"WGQGTMVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKRVEPKSC"
)


@dataclass
class ModelConfig:
	esm_model_path: str = (
        "/cluster/project/krause/flohmann/mgm/oracle_assets/esm2_8m.safetensors"
    )	
	device: str = "cuda"
	use_context: bool = True # If True, expects sequences with context and extracts positions 104:-115; if False, standard slicing 1:-1
	pll_mask_chunk_size: int = 64


@dataclass
class TrainingConfig:
    """Training runtime configuration."""

    batch_size: int = 32
    num_epochs: int = 50
    lr: float = 1e-5
    num_ensembles: int = 1
    subsample: float = 1.0
    patience: int = 20
    device: str = "cuda"
    max_weight: float = float("inf")
    max_train_mutations: Optional[int] = None


def add_context(cdr: str) -> str:
	"""Add fixed heavy-chain context around a CDR sequence."""
	return LEFT_CONTEXT + cdr + RIGHT_CONTEXT


def get_mask_token_idx(alphabet: Alphabet) -> int:
	"""Resolve mask token index across possible Alphabet API variants."""
	for attr in ("mask_idx", "mask_index", "mask_token_id"):
		if hasattr(alphabet, attr):
			return int(getattr(alphabet, attr))

	for attr in ("tok_to_idx", "token_to_idx", "stoi"):
		mapping = getattr(alphabet, attr, None)
		if isinstance(mapping, dict) and "<mask>" in mapping:
			return int(mapping["<mask>"])

	raise AttributeError("Could not find mask token index in Alphabet.")


def load_hydra_runtime_modules() -> Tuple[Any, Any, Any, Any]:
	"""Dynamically load Hydra/OmegaConf utilities used by training scripts."""
	try:
		hydra = importlib.import_module("hydra")
		omegaconf = importlib.import_module("omegaconf")
		hydra_config_mod = importlib.import_module("hydra.core.hydra_config")
		hydra_utils_mod = importlib.import_module("hydra.utils")
	except ModuleNotFoundError as exc:  # pragma: no cover
		raise ModuleNotFoundError(
			"Hydra/OmegaConf is required for train_dpo.py. Install hydra-core and omegaconf."
		) from exc

	return hydra, omegaconf.OmegaConf, hydra_config_mod.HydraConfig, hydra_utils_mod.to_absolute_path


def setup_train_logger(output_dir: Path, level_name: str, logger_name: str = "dpo_train") -> logging.Logger:
	"""Create a console + file logger for training."""
	logger = logging.getLogger(logger_name)
	logger.setLevel(getattr(logging, str(level_name).upper(), logging.INFO))
	logger.propagate = False

	if logger.handlers:
		logger.handlers.clear()

	formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(formatter)
	logger.addHandler(stream_handler)

	file_handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)

	return logger


def init_wandb_run(cfg: Any, output_dir: Path, logger: logging.Logger, omegaconf_cls: Any) -> Tuple[Optional[Any], Optional[Any]]:
	"""Initialize Weights & Biases run if enabled in config."""
	if not bool(cfg.wandb.enabled):
		return None, None

	try:
		wandb = importlib.import_module("wandb")
	except ModuleNotFoundError:  # pragma: no cover
		logger.warning("wandb is not available; continuing without Weights & Biases logging.")
		return None, None

	run = wandb.init(
		project=str(cfg.wandb.project),
		entity=None if cfg.wandb.entity is None else str(cfg.wandb.entity),
		name=None if cfg.wandb.run_name is None else str(cfg.wandb.run_name),
		tags=None if cfg.wandb.tags is None else list(cfg.wandb.tags),
		notes=None if cfg.wandb.notes is None else str(cfg.wandb.notes),
		dir=str(output_dir),
		config=omegaconf_cls.to_container(cfg, resolve=True),
	)
	return wandb, run


def log_pair_diagnostics(logger: logging.Logger, pairs_df: pd.DataFrame, preview_count: int = 5) -> None:
	"""Log basic pair quality stats and top/bottom margin examples."""
	if pairs_df.empty:
		logger.warning("No pairs available after preprocessing.")
		return

	margins = pairs_df["delta_margin"].astype(float)
	logger.info(
		"Pair stats | n=%d | margin mean=%.4f median=%.4f min=%.4f max=%.4f",
		len(pairs_df),
		float(margins.mean()),
		float(margins.median()),
		float(margins.min()),
		float(margins.max()),
	)

	by_view = pairs_df.groupby("source_view").size().sort_values(ascending=False)
	logger.info("Pairs per view: %s", ", ".join(f"{k}:{int(v)}" for k, v in by_view.items()))

	show_n = min(int(preview_count), len(pairs_df))
	if show_n <= 0:
		return

	top_examples = pairs_df.nlargest(show_n, "delta_margin")
	low_examples = pairs_df.nsmallest(show_n, "delta_margin")

	logger.info("Top margin examples:")
	for _, row in top_examples.iterrows():
		logger.info(
			"  view=%s cluster=%s margin=%.4f chosen=%s rejected=%s",
			row["source_view"],
			row["cluster_idx"],
			float(row["delta_margin"]),
			row["chosen_sequence"],
			row["rejected_sequence"],
		)

	logger.info("Bottom margin examples:")
	for _, row in low_examples.iterrows():
		logger.info(
			"  view=%s cluster=%s margin=%.4f chosen=%s rejected=%s",
			row["source_view"],
			row["cluster_idx"],
			float(row["delta_margin"]),
			row["chosen_sequence"],
			row["rejected_sequence"],
		)
