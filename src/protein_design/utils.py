"""Shared filesystem, tokenizer, and training helpers."""

import logging
from pathlib import Path
from typing import Any, Optional, Tuple


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
    output_dir: Path, level_name: str, logger_name: str = "train"
) -> logging.Logger:
    """Create a console + file logger that writes to {output_dir}/train.log."""
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


def init_wandb(
    cfg: Any,
    output_dir: Path,
    logger: logging.Logger,
    run_name: str,
    group: Optional[str] = None,
) -> Tuple[Optional[Any], Optional[Any]]:
    """Initialize a W&B run from a config with wandb.* keys.

    Expects cfg.wandb to have: enabled, project, entity, init_timeout,
    fallback_mode, tags, notes.

    Returns (wandb_module, wandb_run). Both are None when W&B is disabled,
    unavailable, or all fallback modes are exhausted.
    """
    wandb_cfg = getattr(cfg, "wandb", None)
    if wandb_cfg is None or not bool(wandb_cfg.enabled):
        return None, None

    try:
        import wandb as wandb_mod
    except ModuleNotFoundError:
        logger.warning("wandb is not installed; continuing without W&B logging.")
        return None, None

    from omegaconf import OmegaConf

    init_timeout = int(getattr(wandb_cfg, "init_timeout", 120))
    fallback_mode = str(getattr(wandb_cfg, "fallback_mode", "offline")).strip().lower()
    tags = None if wandb_cfg.tags is None else list(wandb_cfg.tags)
    notes = None if wandb_cfg.notes is None else str(wandb_cfg.notes)

    init_kwargs = {
        "project": str(wandb_cfg.project),
        "entity": None if wandb_cfg.entity is None else str(wandb_cfg.entity),
        "name": run_name,
        "group": group,
        "tags": tags,
        "notes": notes,
        "dir": str(output_dir),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "settings": wandb_mod.Settings(init_timeout=init_timeout),
    }

    logger.info("W&B run: %s (project=%s, group=%s)", run_name, init_kwargs["project"], group)

    try:
        run = wandb_mod.init(**init_kwargs)
        return wandb_mod, run
    except Exception as exc:
        logger.warning("W&B online init failed (%s). Fallback mode: '%s'.", exc, fallback_mode)

    if fallback_mode == "offline":
        try:
            run = wandb_mod.init(mode="offline", **init_kwargs)
            logger.warning("Continuing with W&B in offline mode.")
            return wandb_mod, run
        except Exception as exc:
            logger.warning("W&B offline init failed (%s). Disabling W&B logging.", exc)
            return None, None

    if fallback_mode in {"disable", "disabled", "none"}:
        logger.warning("Continuing without W&B logging.")
        return None, None

    logger.warning(
        "Unknown wandb.fallback_mode='%s'. Expected 'offline' or 'disable'. "
        "Continuing without W&B logging.",
        fallback_mode,
    )
    return None, None


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
