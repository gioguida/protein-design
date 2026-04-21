from datetime import datetime
import importlib
import logging
from pathlib import Path
import re
from typing import Any, List, Optional, Tuple, TypedDict

import pandas as pd

from protein_design.constants import WILD_TYPE
from protein_design.utils import get_mask_token_idx, init_wandb, setup_train_logger

__all__ = [
    "get_mask_token_idx",
    "init_wandb",
    "setup_train_logger",
    "load_hydra_runtime_modules",
    "resolve_base_run_name",
    "build_full_run_name",
    "log_pair_diagnostics",
]


class PairMember(TypedDict):
    aa: str
    score: float


PairTuple = Tuple[PairMember, PairMember]


def load_hydra_runtime_modules() -> Tuple[Any, Any, Any, Any]:
    """Dynamically load Hydra/OmegaConf utilities used by training scripts."""
    try:
        hydra = importlib.import_module("hydra")
        omegaconf = importlib.import_module("omegaconf")
        hydra_config_mod = importlib.import_module("hydra.core.hydra_config")
        hydra_utils_mod = importlib.import_module("hydra.utils")
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "Hydra/OmegaConf is required for DPO training. Install hydra-core and omegaconf."
        ) from exc

    return hydra, omegaconf.OmegaConf, hydra_config_mod.HydraConfig, hydra_utils_mod.to_absolute_path


def _infer_model_label(esm_model_path: Any) -> str:
    """Infer model label as esm2-<millions>M from model path when possible."""
    model_name = Path(str(esm_model_path)).stem
    model_name_lc = model_name.lower()

    match = re.search(r"esm2[^/]*?_(\d+(?:\.\d+)?)([mb])(?=[^a-z0-9]|$)", model_name_lc)
    if match is None:
        match = re.search(r"(\d+(?:\.\d+)?)([mb])(?=[^a-z0-9]|$)", model_name_lc)

    if match is None:
        return "esm2-unknownM"

    value = float(match.group(1))
    unit = match.group(2)
    millions = value if unit == "m" else value * 1000.0

    if float(millions).is_integer():
        million_text = str(int(millions))
    else:
        million_text = f"{millions:g}"

    return f"esm2-{million_text}M"


def _default_wandb_run_name(cfg: Any) -> str:
    """Create default base run name used by W&B and local/archive run directories."""
    model_label = _infer_model_label(cfg.model.esm_model_path)
    pairing = str(cfg.data.pairing_strategy)
    if pairing == "delta_based":
        components = list(cfg.data.delta_based.components)
        component_tag = "".join(
            c for c, flag in [("c", "cross"), ("w", "wt_anchors"), ("p", "within_pos"), ("n", "within_neg")]
            if flag in components
        )
        pairing = f"delta_based-{component_tag}"
    loss_name = str(cfg.training.loss)
    epochs = int(cfg.training.num_epochs)
    batch_size = int(cfg.training.batch_size)
    lr = float(cfg.training.lr)
    beta = float(cfg.training.beta)
    temp_part = ""
    if loss_name == "weighted_dpo":
        temp_part = f"__loss-{float(cfg.training.temperature):g}"

    return (
        f"{model_label}"
        f"__{pairing}"
        f"__{loss_name}"
        f"{temp_part}"
        f"__ep-{epochs}"
        f"__bs-{batch_size}"
        f"__lr-{lr:g}"
        f"__beta-{beta:g}"
    )


def resolve_base_run_name(cfg: Any) -> str:
    """Resolve base run name from config, falling back to auto naming logic."""
    configured_base = None
    if hasattr(cfg, "run") and getattr(cfg.run, "base_name", None) is not None:
        configured_base = str(cfg.run.base_name).strip()
    if not configured_base and getattr(cfg.wandb, "run_name", None) is not None:
        configured_base = str(cfg.wandb.run_name).strip()
    return configured_base or _default_wandb_run_name(cfg)


def build_full_run_name(cfg: Any, timestamp: Optional[str] = None) -> str:
    """Build canonical run name: <base_run_name>_<YYYYMMDD_HHMMSS>."""
    base_name = resolve_base_run_name(cfg).replace("/", "-").replace("\\", "-").replace(" ", "_")
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{ts}"


def log_pair_diagnostics(
    logger: logging.Logger, pairs_df: pd.DataFrame, preview_count: int = 5
) -> None:
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


def _gap_pairs(
    sorted_df: pd.DataFrame,
    delta_col: str,
    seq_col: str,
    gap: float,
) -> List[PairTuple]:
    """Pair with uniform rank gap across all pairs."""
    n = len(sorted_df)
    if n < 2:
        return []

    k = 1 + int(gap * (n // 2 - 1))
    block_size = 2 * k

    pairs: List[PairTuple] = []
    for block_start in range(0, n, block_size):
        for i in range(k):
            if block_start + k + i >= n:
                break
            winner_row = sorted_df.iloc[block_start + i]
            loser_row = sorted_df.iloc[block_start + k + i]
            winner = {"aa": winner_row[seq_col], "score": float(winner_row[delta_col])}
            loser = {"aa": loser_row[seq_col], "score": float(loser_row[delta_col])}
            pairs.append((winner, loser))

    return pairs
