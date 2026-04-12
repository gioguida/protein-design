"""Shared utilities: config loading, env var resolution, run naming."""

import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv


def _resolve_env_vars(obj):
    """Recursively resolve ${VAR} and ${VAR:-default} in string values."""
    if isinstance(obj, str):
        def _replace(match):
            var_name = match.group(1)
            fallback = match.group(2)
            value = os.environ.get(var_name)
            if value is not None:
                return value
            if fallback is not None:
                return fallback
            raise KeyError(
                f"Environment variable '{var_name}' is not set and has no "
                f"fallback. Set it in your .env file (see .env.template)."
            )
        return re.sub(r'\$\{(\w+)(?::-([^}]*))?\}', _replace, obj)
    elif isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_env_vars(v) for v in obj]
    return obj


def load_config(path: str) -> dict:
    """Load YAML config with env var interpolation. Loads .env automatically."""
    load_dotenv()
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return _resolve_env_vars(cfg)


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
