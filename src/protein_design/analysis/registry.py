"""Registry, artifact-path, and provenance helpers for the analysis pipeline.

Everything that reads/writes per-model artifacts goes through here so the
extraction scripts and the plotting notebook agree on layout and skip-logic.

Layout (writable, independent of where the checkpoint lives):
    $ANALYSIS_DIR/<model_key>/<kind>/<name>          e.g. evo_35m/pll/ed2_m22.csv
    $ANALYSIS_DIR/<model_key>/<kind>/<name>.meta.json provenance sidecar

Skip-logic (`needs_recompute`):
    - no artifact            -> recompute
    - artifact, no meta.json -> reuse (legacy cache; warn once), never auto-recompute
    - artifact + meta.json   -> recompute only if recorded provenance != expected
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_YAML = REPO_ROOT / "conf" / "analysis" / "models.yaml"
DATASETS_YAML = REPO_ROOT / "conf" / "analysis" / "dms_datasets.yaml"


# ── env / paths ───────────────────────────────────────────────────────────

def _load_dotenv_value(key: str) -> str | None:
    """Read KEY from the environment, falling back to .env.local.

    sbatch jobs already have ANALYSIS_DIR exported (common_setup.sh sources
    .env.local); a plain `jupyter`/`uv run` may not, so we parse the file too.
    """
    if key in os.environ:
        return os.environ[key]
    env_file = REPO_ROOT / ".env.local"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith(f"{key}=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip()
    return None


def analysis_dir() -> Path:
    """Root of the writable per-model artifact tree ($ANALYSIS_DIR)."""
    val = _load_dotenv_value("ANALYSIS_DIR")
    if val:
        return Path(val)
    project = _load_dotenv_value("PROJECT_DIR")
    if project:
        return Path(project) / "analysis"
    raise RuntimeError(
        "ANALYSIS_DIR not set and PROJECT_DIR unavailable — set ANALYSIS_DIR "
        "in .env.local (see .env.template)."
    )


def model_dir(model_key: str) -> Path:
    return analysis_dir() / model_key


def artifact_path(model_key: str, kind: str, name: str) -> Path:
    """Path to one artifact, e.g. artifact_path('evo_35m', 'pll', 'ed2_m22.csv')."""
    return model_dir(model_key) / kind / name


def meta_path(artifact: Path) -> Path:
    return artifact.with_suffix(artifact.suffix + ".meta.json")


# ── registries ────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_models() -> dict:
    with MODELS_YAML.open() as f:
        return yaml.safe_load(f)["models"]


@lru_cache(maxsize=1)
def load_datasets_cfg() -> dict:
    with DATASETS_YAML.open() as f:
        return yaml.safe_load(f)


def resolve_model(model_key: str, *, checkpoint: str | None = None,
                  base_model: str | None = None) -> dict:
    """Resolve a model spec from the registry, allowing CLI overrides.

    A key absent from the registry is allowed only if `checkpoint`/`base_model`
    are supplied explicitly (ad-hoc models); we still write under `model_key`.
    """
    models = load_models()
    if model_key in models:
        spec = dict(models[model_key])
    elif checkpoint is not None or base_model is not None:
        spec = {"checkpoint": None, "base_model": None,
                "label": model_key, "color": "#444444", "family": "adhoc"}
    else:
        raise KeyError(
            f"Unknown model {model_key!r}. Known: {sorted(models)}. "
            f"For an ad-hoc model pass --checkpoint/--base-model explicitly."
        )
    if checkpoint is not None:
        spec["checkpoint"] = checkpoint
    if base_model is not None:
        spec["base_model"] = base_model
    if not spec.get("base_model"):
        raise ValueError(f"Model {model_key!r} has no base_model set.")
    return spec


def dataset_keys(which: str = "all") -> list[str]:
    datasets = load_datasets_cfg()["datasets"]
    if which == "all":
        return list(datasets)
    keys = [k.strip() for k in which.split(",") if k.strip()]
    unknown = [k for k in keys if k not in datasets]
    if unknown:
        raise KeyError(f"Unknown dataset(s) {unknown}. Known: {list(datasets)}")
    return keys


# ── provenance / skip-logic ────────────────────────────────────────────────

def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def write_meta(artifact: Path, **fields) -> None:
    """Write the provenance sidecar next to an artifact."""
    meta = {"git_sha": git_sha(),
            "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **fields}
    meta_path(artifact).write_text(json.dumps(meta, indent=2, sort_keys=True))


def read_meta(artifact: Path) -> dict | None:
    mp = meta_path(artifact)
    if not mp.exists():
        return None
    try:
        return json.loads(mp.read_text())
    except json.JSONDecodeError:
        return None


def needs_recompute(artifact: Path, expected: dict, *, force: bool = False) -> bool:
    """Decide whether `artifact` must be (re)computed.

    `expected` holds the provenance keys that matter (e.g. checkpoint, dataset
    path). Keys absent from a legacy meta.json are ignored — only keys present
    in BOTH that disagree count as stale, so adding a new provenance field
    doesn't invalidate every existing artifact.
    """
    if force:
        return True
    if not artifact.exists():
        return True
    meta = read_meta(artifact)
    if meta is None:
        return False  # legacy cache, no sidecar -> trust it
    return any(k in meta and meta[k] != v for k, v in expected.items())


# ── loaders (figures read through these) ────────────────────────────────────

def scorer_artifact_path(dataset_key: str) -> Path | None:
    """Path to the scorer-predictions CSV for a dataset, or None if no scorer configured.

    Written by score_dms_with_esme.py (flu conda env) at:
        $cache_root/scorer_preds/<dataset>/<scorer>.csv
    """
    cfg = load_datasets_cfg()
    ds = cfg["datasets"][dataset_key]
    scorer_name = ds.get("scorer")
    if not scorer_name:
        return None
    cache_root = Path(cfg["paths"]["cache_root"])
    return cache_root / "scorer_preds" / dataset_key / f"{scorer_name}.csv"


def load_scorer(dataset_key: str) -> pd.DataFrame | None:
    """Return scorer predictions [seq_col, score] for a dataset, or None if missing/unconfigured."""
    cfg = load_datasets_cfg()
    ds = cfg["datasets"][dataset_key]
    seq_col = ds["seq_col"]
    path = scorer_artifact_path(dataset_key)
    if path is None or not path.exists():
        return None
    return (pd.read_csv(path)[[seq_col, "score"]]
            .drop_duplicates(subset=[seq_col], keep="first"))


def load_pll(model_key: str, dataset_key: str) -> pd.DataFrame:
    """Return the cached PLL frame (cols: <seq_col>, pll) for one (model, dataset)."""
    path = artifact_path(model_key, "pll", f"{dataset_key}.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"PLL artifact missing: {path}\n"
            f"Extract it: sbatch bash_scripts/extract.sbatch --what pll "
            f"--model {model_key} --dataset {dataset_key}"
        )
    return pd.read_csv(path)


def load_truth(dataset_key: str) -> pd.DataFrame:
    """Return [seq_col, enrichment] for a dataset, deduped on the sequence."""
    ds = load_datasets_cfg()["datasets"][dataset_key]
    seq_col, enrich_col = ds["seq_col"], ds["enrichment_col"]
    df = pd.read_csv(ds["path"])[[seq_col, enrich_col]]
    df = df.drop_duplicates(subset=[seq_col], keep="first")
    return df.rename(columns={enrich_col: "enrichment"})
