"""Shared filesystem and tokenizer helpers."""

import logging
from pathlib import Path
from typing import Any


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


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
