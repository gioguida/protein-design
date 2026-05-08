#!/usr/bin/env python3
"""Config-driven orchestrator for the full embedding analysis pipeline."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protein_design.dms_splitting import dataset_spec, load_dms_config, resolve_dataset_split

BUILTIN_DMS_KEYS = {"ed2", "ed5", "ed811"}
DATASET_PLOTS = {
    "per_model_pca",
    "gibbs_per_model_pca",
    "beam_per_model_pca",
    "diff_vectors_pca",
    "pll_pca",
    "pll_vs_enrichment_overlays",
}
OVERLAY_PLOTS = {"gibbs_per_model_pca", "beam_per_model_pca", "pll_vs_enrichment_overlays"}

# New beam eval plots that need DMS data (triggered inside the dataset loop).
BEAM_EVAL_DMS_PLOTS = {"beam_pll_vs_dms_histogram", "beam_aa_heatmap"}
# New beam eval plots that do NOT need DMS data (triggered after dataset loop).
BEAM_EVAL_NODMS_PLOTS = {"beam_diversity_diagnostics", "beam_pll_vs_nmut"}


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    checkpoint_path: str


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str
    strategy_type: str
    generate: bool
    csv_path: Path
    params: dict[str, Any]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("conf/analysis/full_analysis.yaml"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def fail(msg: str) -> None:
    raise ValueError(msg)


def _expand(path_str: str, repo_root: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(path_str))
    p = Path(expanded)
    if not p.is_absolute():
        p = repo_root / p
    return p


def _render_path_template(template: str, repo_root: Path, **kwargs: str) -> Path:
    try:
        rendered = template.format(**kwargs)
    except KeyError as exc:
        fail(f"path template {template!r} references unknown placeholder: {exc}")
    return _expand(rendered, repo_root)


def _dataset_arg(dataset_key: str) -> str:
    return dataset_key if dataset_key in BUILTIN_DMS_KEYS else "ed2"


def _run(cmd: list[str], dry_run: bool, step: str) -> None:
    print(f"[{step}] {shlex.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _mkdir(path: Path, dry_run: bool, step: str) -> None:
    if dry_run:
        print(f"[{step}] mkdir -p {path}")
        return
    path.mkdir(parents=True, exist_ok=True)


def _append_flag(cmd: list[str], name: str, value: Any) -> None:
    flag = f"--{name.replace('_', '-')}"
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
        return
    cmd.extend([flag, str(value)])


def _subsample_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(((cfg.get("datasets", {}) or {}).get("subsample", {}) or {}))


def _subsample_enabled(cfg: dict[str, Any]) -> bool:
    scfg = _subsample_cfg(cfg)
    return bool(scfg.get("enabled", False)) and scfg.get("max_rows") is not None


def _analysis_subsample_dir(cfg: dict[str, Any], repo_root: Path) -> Path:
    scfg = _subsample_cfg(cfg)
    return _expand(str(scfg.get("output_dir", "data/processed/analysis_subsamples")), repo_root)


def _subsample_path_for(
    cfg: dict[str, Any],
    repo_root: Path,
    dataset_key: str,
    split_name: str,
    *,
    source_dataset_key: str | None = None,
) -> Path:
    scfg = _subsample_cfg(cfg)
    max_rows = int(scfg.get("max_rows", 0))
    seed = int(scfg.get("seed", 42))
    suffix = f"{split_name}_max{max_rows}_seed{seed}.csv"
    out_key = source_dataset_key or dataset_key
    return _analysis_subsample_dir(cfg, repo_root) / out_key / dataset_key / suffix


def _metric_strata(values: pd.Series, bins: int) -> pd.Series:
    clean = pd.to_numeric(values, errors="coerce")
    fill_value = clean.median()
    if pd.isna(fill_value):
        fill_value = 0.0
    clean = clean.fillna(fill_value)
    bins = max(1, min(int(bins), len(clean)))
    if bins == 1:
        return pd.Series(np.zeros(len(clean), dtype=np.int64), index=clean.index)
    try:
        return pd.qcut(clean, q=bins, labels=False, duplicates="drop").fillna(0).astype(int)
    except ValueError:
        return pd.Series(np.zeros(len(clean), dtype=np.int64), index=clean.index)


def _representative_sample_indices(
    df: pd.DataFrame,
    *,
    metric_col: str,
    max_rows: int,
    seed: int,
    stratify_bins: int,
) -> pd.Index:
    if len(df) <= max_rows:
        return df.index
    rng = np.random.default_rng(seed)
    working = df.copy()
    working["_metric_bin"] = _metric_strata(working[metric_col], stratify_bins)
    if "num_mut" in working.columns:
        group_cols = ["num_mut", "_metric_bin"]
    else:
        group_cols = ["_metric_bin"]

    grouped = working.groupby(group_cols, dropna=False, sort=True)
    group_sizes = grouped.size()
    raw_targets = group_sizes / len(working) * max_rows
    base_targets = np.floor(raw_targets).astype(int)
    nonzero_groups = group_sizes[group_sizes > 0].index
    for idx in nonzero_groups:
        if base_targets.loc[idx] == 0:
            base_targets.loc[idx] = 1
    overflow = int(base_targets.sum() - max_rows)
    if overflow > 0:
        removable = (base_targets - 1).sort_values(ascending=False)
        for idx, can_remove in removable.items():
            if overflow <= 0:
                break
            delta = min(int(can_remove), overflow)
            base_targets.loc[idx] -= delta
            overflow -= delta
    elif overflow < 0:
        remainder = (raw_targets - np.floor(raw_targets)).sort_values(ascending=False)
        remaining = -overflow
        for idx in remainder.index:
            if remaining <= 0:
                break
            capacity = int(group_sizes.loc[idx] - base_targets.loc[idx])
            if capacity <= 0:
                continue
            base_targets.loc[idx] += 1
            remaining -= 1

    selected: list[int] = []
    for group_key, group in grouped:
        n = int(base_targets.loc[group_key])
        if n <= 0:
            continue
        indices = group.index.to_numpy()
        selected.extend(rng.choice(indices, size=min(n, len(indices)), replace=False).tolist())
    if len(selected) > max_rows:
        selected = rng.choice(np.array(selected), size=max_rows, replace=False).tolist()
    return pd.Index(selected)


def _metadata_matches(path: Path, expected: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
    except Exception:
        return False
    return all(existing.get(k) == v for k, v in expected.items())


def _resolve_analysis_split(
    cfg: dict[str, Any],
    dataset_key: str,
    split_name: str,
    *,
    force: bool,
    materialize: bool,
    repo_root: Path,
    sequence_filter: set[str] | None = None,
    source_dataset_key: str | None = None,
) -> tuple[str, set[str] | None]:
    dms_cfg = (cfg.get("datasets", {}) or {}).get("dms_config", "conf/data/dms/default.yaml")
    if materialize:
        full_path = resolve_dataset_split(dataset_key, split_name, dms_cfg, force=force)
    else:
        dms_config = load_dms_config(dms_cfg)
        full_path = dms_config.split.output_dir / dataset_key / f"{split_name}.csv"

    if not _subsample_enabled(cfg):
        return str(full_path), sequence_filter

    spec = dataset_spec(dataset_key, dms_cfg)
    if source_dataset_key is None and spec.split_source:
        source_dataset_key = spec.split_source
    if materialize and sequence_filter is None and spec.split_source:
        _, sequence_filter = _resolve_analysis_split(
            cfg,
            spec.split_source,
            split_name,
            force=force,
            materialize=materialize,
            repo_root=repo_root,
        )
    out_path = _subsample_path_for(
        cfg,
        repo_root,
        dataset_key,
        split_name,
        source_dataset_key=source_dataset_key,
    )
    if not materialize:
        return str(out_path), sequence_filter

    scfg = _subsample_cfg(cfg)
    max_rows = int(scfg["max_rows"])
    seed = int(scfg.get("seed", 42))
    stratify_bins = int(scfg.get("stratify_bins", 10))
    source_stat = full_path.stat()
    expected = {
        "version": 1,
        "source_path": str(full_path),
        "source_mtime": source_stat.st_mtime,
        "dataset_key": dataset_key,
        "split_name": split_name,
        "sequence_col": spec.sequence_col,
        "key_metric_col": spec.key_metric_col,
        "max_rows": max_rows,
        "seed": seed,
        "stratify_bins": stratify_bins,
        "sequence_filter_size": None if sequence_filter is None else len(sequence_filter),
        "source_dataset_key": source_dataset_key,
    }
    meta_path = out_path.with_suffix(".meta.json")
    if not force and _metadata_matches(meta_path, expected):
        if sequence_filter is None:
            existing = pd.read_csv(out_path, usecols=[spec.sequence_col])
            return str(out_path), set(existing[spec.sequence_col].astype(str))
        return str(out_path), sequence_filter

    df = pd.read_csv(full_path)
    if sequence_filter is not None:
        df = df[df[spec.sequence_col].astype(str).isin(sequence_filter)].copy()
    if len(df) > max_rows:
        indices = _representative_sample_indices(
            df,
            metric_col=spec.key_metric_col,
            max_rows=max_rows,
            seed=seed,
            stratify_bins=stratify_bins,
        )
        df = df.loc[indices].copy()
    df = df.reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump({**expected, "rows": int(len(df))}, handle, indent=2, sort_keys=True)
    if sequence_filter is None:
        sequence_filter = set(df[spec.sequence_col].astype(str))
    return str(out_path), sequence_filter


def _resolve_param_value(
    value: Any,
    cfg: dict[str, Any],
    force: bool,
    materialize: bool,
    repo_root: Path,
) -> Any:
    if not isinstance(value, str) or not value.startswith("dms_split:"):
        return value
    parts = value.split(":")
    if len(parts) != 3:
        fail(f"Invalid dms_split reference {value!r}; expected dms_split:DATASET_KEY:SPLIT")
    path, _ = _resolve_analysis_split(
        cfg,
        parts[1],
        parts[2],
        force=force,
        materialize=materialize,
        repo_root=repo_root,
    )
    return path


def _resolve_models(cfg: dict[str, Any], repo_root: Path) -> list[ModelSpec]:
    models_cfg = cfg.get("models", {})
    catalog = models_cfg.get("catalog", {})
    selected = models_cfg.get("selected_models", [])
    if not selected:
        fail("models.selected_models must be non-empty")
    missing = [mid for mid in selected if mid not in catalog]
    if missing:
        fail(f"models.selected_models references unknown model(s): {missing}")
    resolved: list[ModelSpec] = []
    for mid in selected:
        entry = catalog[mid] or {}
        display_name = str(entry.get("display_name", "")).strip() or mid
        raw_ckpt = str(entry.get("checkpoint_path", "") or "")
        checkpoint_path = str(_expand(raw_ckpt, repo_root)) if raw_ckpt else ""
        resolved.append(ModelSpec(model_id=mid, display_name=display_name, checkpoint_path=checkpoint_path))
    return resolved


def _resolve_strategies(cfg: dict[str, Any], repo_root: Path, models: list[ModelSpec]) -> list[StrategySpec]:
    sampling_cfg = cfg.get("sampling", {})
    catalog = sampling_cfg.get("strategies", {})
    selected = sampling_cfg.get("selected_sampling_strategies", [])
    if not selected:
        return []
    missing = [sid for sid in selected if sid not in catalog]
    if missing:
        fail(f"sampling.selected_sampling_strategies references unknown strategy(ies): {missing}")

    resolved: list[StrategySpec] = []
    # Pre-resolve for first model; per-model rendering is repeated later.
    model0 = models[0] if models else ModelSpec("model", "model", "")
    for sid in selected:
        entry = catalog[sid] or {}
        stype = str(entry.get("type", "")).strip().lower()
        if stype not in {"gibbs", "beam"}:
            fail(f"sampling.strategies.{sid}.type must be one of ['gibbs', 'beam']")
        generate = bool(entry.get("generate", False))
        params = dict(entry.get("params", {}) or {})
        if generate:
            template = entry.get("output_csv_path")
            if not template:
                fail(f"sampling.strategies.{sid}.output_csv_path is required when generate=true")
        else:
            template = entry.get("existing_csv_path")
            if not template:
                fail(f"sampling.strategies.{sid}.existing_csv_path is required when generate=false")
        csv_path = _render_path_template(
            str(template),
            repo_root,
            model_id=model0.model_id,
            model_label=model0.display_name,
            strategy_id=sid,
        )
        resolved.append(
            StrategySpec(
                strategy_id=sid,
                strategy_type=stype,
                generate=generate,
                csv_path=csv_path,
                params=params,
            )
        )
    return resolved


def _validate_plots(cfg: dict[str, Any], datasets_catalog: dict[str, Any], selected_strategies: set[str]) -> None:
    plots = cfg.get("plots", {})
    for name, pcfg in plots.items():
        # Flat boolean toggles (beam eval plots) have no sub-keys to validate.
        if isinstance(pcfg, bool):
            continue
        enabled = bool((pcfg or {}).get("enabled", False))
        if not enabled:
            continue
        datasets = list((pcfg or {}).get("datasets", []) or [])
        if name in DATASET_PLOTS and not datasets:
            fail(f"plots.{name}.enabled=true requires a non-empty datasets list")
        unknown_datasets = [d for d in datasets if d not in datasets_catalog]
        if unknown_datasets:
            fail(f"plots.{name}.datasets references unknown dataset(s): {unknown_datasets}")

        overlay = list((pcfg or {}).get("overlay_sampling", []) or [])
        if name in OVERLAY_PLOTS and not overlay:
            fail(f"plots.{name}.enabled=true requires non-empty overlay_sampling")
        unknown_overlay = [s for s in overlay if s not in selected_strategies]
        if unknown_overlay:
            fail(
                f"plots.{name}.overlay_sampling references non-selected strategy(ies): {unknown_overlay}. "
                "Add them to sampling.selected_sampling_strategies."
            )


def _load_and_validate(cfg_path: Path, repo_root: Path) -> dict[str, Any]:
    if not cfg_path.exists():
        fail(f"config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if cfg.get("version") != 1:
        fail("config version must be exactly 1")
    datasets_catalog = ((cfg.get("datasets") or {}).get("catalog") or {})
    if not datasets_catalog:
        fail("datasets.catalog must be non-empty")
    for dkey, dval in datasets_catalog.items():
        if not dval or (not dval.get("m22_csv") and not dval.get("m22_dataset")):
            fail(f"datasets.catalog.{dkey} requires m22_csv or m22_dataset")

    models = _resolve_models(cfg, repo_root)
    strategies = _resolve_strategies(cfg, repo_root, models)
    _validate_plots(cfg, datasets_catalog, {s.strategy_id for s in strategies})
    return cfg


def _strategy_csv_path(
    entry: dict[str, Any],
    strategy_id: str,
    model: ModelSpec,
    repo_root: Path,
) -> Path:
    generate = bool(entry.get("generate", False))
    key = "output_csv_path" if generate else "existing_csv_path"
    template = entry.get(key)
    if not template:
        fail(f"sampling.strategies.{strategy_id}.{key} is required")
    return _render_path_template(
        str(template),
        repo_root,
        model_id=model.model_id,
        model_label=model.display_name,
        strategy_id=strategy_id,
    )


def _dataset_paths(
    cfg: dict[str, Any],
    dataset_key: str,
    repo_root: Path,
    *,
    force: bool = False,
    materialize: bool = True,
) -> tuple[str, str | None, str, str]:
    d = cfg["datasets"]["catalog"][dataset_key]
    dms_cfg = (cfg.get("datasets", {}) or {}).get("dms_config", "conf/data/dms/default.yaml")
    split_name = str(((cfg.get("datasets", {}) or {}).get("split_name", "test")))
    m22_dataset = d.get("m22_dataset")
    si06_dataset = d.get("si06_dataset")
    if m22_dataset:
        m22, selected_sequences = _resolve_analysis_split(
            cfg,
            str(m22_dataset),
            split_name,
            force=force,
            materialize=materialize,
            repo_root=repo_root,
        )
        m22_col = dataset_spec(str(m22_dataset), dms_cfg).key_metric_col
    else:
        m22 = str(_expand(str(d["m22_csv"]), repo_root))
        m22_col = str(d.get("m22_metric_col", "M22_binding_enrichment_adj"))
        selected_sequences = None
    if si06_dataset:
        si06, _ = _resolve_analysis_split(
            cfg,
            str(si06_dataset),
            split_name,
            force=force,
            materialize=materialize,
            repo_root=repo_root,
            sequence_filter=selected_sequences,
            source_dataset_key=str(m22_dataset) if m22_dataset else None,
        )
        si06_col = dataset_spec(str(si06_dataset), dms_cfg).key_metric_col
    else:
        si06_raw = d.get("si06_csv")
        si06 = str(_expand(str(si06_raw), repo_root)) if si06_raw else None
        si06_col = str(d.get("si06_metric_col", "SI06_binding_enrichment_adj"))
    return m22, si06, m22_col, si06_col


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = args.config if args.config.is_absolute() else (repo_root / args.config)
    cfg = _load_and_validate(cfg_path, repo_root)

    run_cfg = cfg.get("run", {})
    dry_run = bool(run_cfg.get("dry_run", False)) or bool(args.dry_run)
    force = bool(run_cfg.get("force", False))
    reuse_existing = bool(run_cfg.get("reuse_existing_artifacts", False))
    plots_only = bool(run_cfg.get("plots_only", False))
    max_dms = int(run_cfg.get("max_dms", 500))
    max_oas = int(run_cfg.get("max_oas", 2000))
    max_gibbs = int(run_cfg.get("max_gibbs", 200))
    cka_threshold = float(run_cfg.get("cka_threshold", 0.5))

    scratch_base = _expand(str(run_cfg.get("scratch_base", "embedding_analysis")), repo_root)
    project_base = _expand(str(run_cfg.get("project_base", "reports/embedding_analysis")), repo_root)

    models = _resolve_models(cfg, repo_root)
    sampling_cfg = cfg.get("sampling", {})
    strategy_catalog: dict[str, Any] = sampling_cfg.get("strategies", {})
    selected_strategy_ids: list[str] = list(sampling_cfg.get("selected_sampling_strategies", []) or [])
    selected_entries = {sid: strategy_catalog[sid] for sid in selected_strategy_ids}
    plots_cfg = cfg.get("plots", {})

    # 1) Optional: generate sampler CSVs from YAML.
    if plots_only:
        print("[sample] plots_only=true -> skip sampler generation")
    else:
        for model in models:
            for sid in selected_strategy_ids:
                sentry = selected_entries[sid]
                stype = str(sentry.get("type", "")).lower()
                generate = bool(sentry.get("generate", False))
                csv_path = _strategy_csv_path(sentry, sid, model, repo_root)
                if not generate:
                    if not csv_path.exists() and not dry_run:
                        fail(f"existing_csv_path for strategy={sid}, model={model.model_id} does not exist: {csv_path}")
                    continue
                _mkdir(csv_path.parent, dry_run, f"sample:{sid}:{model.model_id}")
                if reuse_existing and csv_path.exists():
                    print(f"[sample:{sid}:{model.model_id}] reuse existing {csv_path}")
                    continue
                script = "scripts/gibbs_sampling.py" if stype == "gibbs" else "scripts/stochastic_beam_search.py"
                cmd = [
                    "uv",
                    "run",
                    "python",
                    script,
                    "--model-variant",
                    model.model_id,
                    "--output-path",
                    str(csv_path),
                ]
                if model.checkpoint_path:
                    cmd.extend(["--checkpoint-path", model.checkpoint_path])
                for k, v in (sentry.get("params", {}) or {}).items():
                    _append_flag(cmd, k, _resolve_param_value(v, cfg, force, not dry_run, repo_root))
                _run(cmd, dry_run, f"sample:{sid}:{model.model_id}")

    # 2) Optional: OAS extraction + plotting (dataset-agnostic).
    oas_plot_cfg = plots_cfg.get("oas_umap", {}) or {}
    oas_enabled = bool(oas_plot_cfg.get("enabled", False))
    oas_npz_paths: list[Path] = []
    oas_dir = scratch_base / "embeddings" / "_oas"
    if oas_enabled:
        _mkdir(oas_dir, dry_run, "oas")
        oas = (cfg.get("datasets", {}) or {}).get("oas", {}) or {}
        oas_fasta = _expand(str(oas.get("fasta_path", "")), repo_root)
        oas_meta = _expand(str(oas.get("meta_csv_gz_path", "")), repo_root)
        for model in models:
            oas_npz = oas_dir / f"{model.model_id}.npz"
            oas_npz_paths.append(oas_npz)
            if plots_only:
                continue
            cmd = [
                "uv",
                "run",
                "python",
                "scripts/analysis/extract_oas_embeddings.py",
                "--model-variant",
                model.display_name,
                "--output-path",
                str(oas_npz),
                "--max-oas",
                str(max_oas),
                "--oas-fasta",
                str(oas_fasta),
                "--oas-meta",
                str(oas_meta),
            ]
            if model.checkpoint_path:
                cmd.extend(["--checkpoint-path", model.checkpoint_path])
            if not force:
                cmd.append("--skip-if-current")
            if reuse_existing and oas_npz.exists():
                print(f"[extract_oas:{model.model_id}] reuse existing {oas_npz}")
            else:
                _run(cmd, dry_run, f"extract_oas:{model.model_id}")

    def _beam_plot_enabled(name: str) -> bool:
        """Return True for flat-boolean beam eval plot keys set to true."""
        val = plots_cfg.get(name, False)
        if isinstance(val, bool):
            return val
        return bool((val or {}).get("enabled", False))

    # 3) Dataset-driven stages.
    required_datasets: list[str] = []
    for plot_name, pcfg in plots_cfg.items():
        if isinstance(pcfg, bool):
            continue  # flat boolean toggles handled separately below
        if not bool((pcfg or {}).get("enabled", False)):
            continue
        if plot_name not in DATASET_PLOTS:
            continue
        for ds in list((pcfg or {}).get("datasets", []) or []):
            if ds not in required_datasets:
                required_datasets.append(ds)

    # DMS-dependent beam eval plots run for every active dataset.
    # If any such plot is enabled, ensure all catalog datasets are processed.
    datasets_catalog = ((cfg.get("datasets") or {}).get("catalog") or {})
    if any(_beam_plot_enabled(p) for p in BEAM_EVAL_DMS_PLOTS):
        for dkey in datasets_catalog:
            if dkey not in required_datasets:
                required_datasets.append(dkey)

    for ds in required_datasets:
        dms_arg = _dataset_arg(ds)
        dms_m22, dms_si06, dms_m22_col, dms_si06_col = _dataset_paths(
            cfg,
            ds,
            repo_root,
            force=force,
            materialize=not dry_run,
        )

        emb_dir = scratch_base / "embeddings" / ds
        beam_emb_dir = scratch_base / "embeddings_beam" / ds
        per_model_dir = scratch_base / "per_model_pca" / ds
        diff_dir = scratch_base / "diff_pca" / ds
        procrustes_dir = scratch_base / "procrustes" / ds
        pll_dir = scratch_base / "pll_pca" / ds

        plots_root = project_base / "plots" / ds
        cka_dir = plots_root / "cka"
        gibbs_diag_dir = plots_root / "gibbs_diagnostics"
        beam_diag_dir = plots_root / "beam_diagnostics"
        beam_plot_dir = plots_root / "per_model_pca_beam"
        pll_overlay_dir = plots_root / "pll_vs_enrichment"

        for p in [
            emb_dir,
            beam_emb_dir,
            per_model_dir,
            diff_dir,
            procrustes_dir,
            pll_dir,
            cka_dir,
            gibbs_diag_dir,
            beam_diag_dir,
            beam_plot_dir,
            pll_overlay_dir,
        ]:
            _mkdir(p, dry_run, f"dataset:{ds}")

        embed_npz: list[Path] = []
        gibbs_diag_args: list[str] = []
        beam_diag_args: list[str] = []
        pll_variant_args: list[str] = []

        for model in models:
            label = model.display_name
            ckpt = model.checkpoint_path
            npz_path = emb_dir / f"{model.model_id}.npz"
            beam_npz_path = beam_emb_dir / f"{model.model_id}.npz"
            embed_npz.append(npz_path)
            pll_variant_args.extend(["--variant", f"{label}={ckpt}"])

            gibbs_paths: list[str] = []
            beam_paths: list[str] = []

            for sid in selected_strategy_ids:
                sentry = selected_entries[sid]
                csv_path = _strategy_csv_path(sentry, sid, model, repo_root)
                if not csv_path.exists() and not dry_run:
                    print(f"[warn] CSV missing for strategy={sid}, model={model.model_id}: {csv_path}")
                    continue
                stype = str(sentry.get("type", "")).lower()
                if stype == "gibbs":
                    gibbs_paths.extend(["--gibbs-path", f"{sid}={csv_path}"])
                    gibbs_diag_args.extend(["--gibbs", f"{label}={ckpt}={csv_path}={sid}"])
                elif stype == "beam":
                    beam_paths.extend(["--gibbs-path", f"{sid}={csv_path}"])
                    beam_diag_args.extend(["--gibbs", f"{label}={ckpt}={csv_path}={sid}"])

            if plots_only:
                continue

            cmd = [
                "uv",
                "run",
                "python",
                "scripts/analysis/extract_embeddings.py",
                "--model-variant",
                label,
                "--output-path",
                str(npz_path),
                "--dms-dataset",
                dms_arg,
                "--dms-m22",
                dms_m22,
                "--dms-m22-col",
                dms_m22_col,
                "--max-dms",
                str(max_dms),
                "--max-gibbs",
                str(max_gibbs),
            ]
            if dms_si06:
                cmd.extend(["--dms-si06", dms_si06, "--dms-si06-col", dms_si06_col])
            cmd.extend(gibbs_paths)
            if ckpt:
                cmd.extend(["--checkpoint-path", ckpt])
            if not force:
                cmd.append("--skip-if-current")
            if reuse_existing and npz_path.exists():
                print(f"[extract_embeddings:{ds}:{model.model_id}] reuse existing {npz_path}")
            else:
                _run(cmd, dry_run, f"extract_embeddings:{ds}:{model.model_id}")

            if beam_paths:
                beam_cmd = [
                    "uv",
                    "run",
                    "python",
                    "scripts/analysis/extract_embeddings.py",
                    "--model-variant",
                    label,
                    "--output-path",
                    str(beam_npz_path),
                    "--dms-dataset",
                    dms_arg,
                    "--dms-m22",
                    dms_m22,
                    "--dms-m22-col",
                    dms_m22_col,
                    "--max-dms",
                    str(max_dms),
                    "--max-gibbs",
                    str(max_gibbs),
                ]
                if dms_si06:
                    beam_cmd.extend(["--dms-si06", dms_si06, "--dms-si06-col", dms_si06_col])
                beam_cmd.extend(beam_paths)
                if ckpt:
                    beam_cmd.extend(["--checkpoint-path", ckpt])
                if not force:
                    beam_cmd.append("--skip-if-current")
                if reuse_existing and beam_npz_path.exists():
                    print(f"[extract_beam_embeddings:{ds}:{model.model_id}] reuse existing {beam_npz_path}")
                else:
                    _run(beam_cmd, dry_run, f"extract_beam_embeddings:{ds}:{model.model_id}")

        pll_out = pll_dir / "pll_pca.npz"
        if not plots_only:
            per_model_out = per_model_dir / "per_model_pca_cdrh3.npz"
            if reuse_existing and per_model_out.exists():
                print(f"[compute_per_model_pca:{ds}] reuse existing {per_model_out}")
            else:
                _run(
                    ["uv", "run", "python", "scripts/analysis/compute_per_model_pca.py", *[str(p) for p in embed_npz], "--output-dir", str(per_model_dir)],
                    dry_run,
                    f"compute_per_model_pca:{ds}",
                )

            diff_out = diff_dir / "diff_pca_cdrh3.npz"
            if reuse_existing and diff_out.exists():
                print(f"[compute_diff_vectors_pca:{ds}] reuse existing {diff_out}")
            else:
                _run(
                    ["uv", "run", "python", "scripts/analysis/compute_diff_vectors_pca.py", *[str(p) for p in embed_npz], "--output-dir", str(diff_dir)],
                    dry_run,
                    f"compute_diff_vectors_pca:{ds}",
                )

            cka_out = cka_dir / "cka_cdrh3.csv"
            if reuse_existing and cka_out.exists():
                print(f"[compute_cka:{ds}] reuse existing {cka_out}")
            else:
                _run(
                    ["uv", "run", "python", "scripts/analysis/compute_cka.py", *[str(p) for p in embed_npz], "--output-dir", str(cka_dir)],
                    dry_run,
                    f"compute_cka:{ds}",
                )

            procrustes_out = procrustes_dir / "procrustes_summary_cdrh3.csv"
            if reuse_existing and procrustes_out.exists():
                print(f"[compute_procrustes:{ds}] reuse existing {procrustes_out}")
            else:
                _run(
                    [
                        "uv",
                        "run",
                        "python",
                        "scripts/analysis/compute_procrustes_displacement.py",
                        *[str(p) for p in embed_npz],
                        "--cka-dir",
                        str(cka_dir),
                        "--cka-threshold",
                        str(cka_threshold),
                        "--output-dir",
                        str(procrustes_dir),
                    ],
                    dry_run,
                    f"compute_procrustes:{ds}",
                )

            if force or not pll_out.exists() or dry_run or not reuse_existing:
                pll_cmd = [
                    "uv",
                    "run",
                    "python",
                    "scripts/analysis/compute_pll_pca.py",
                    *pll_variant_args,
                    "--dms-m22",
                    dms_m22,
                    "--dms-m22-col",
                    dms_m22_col,
                    "--max-dms",
                    str(max_dms),
                    "--output-path",
                    str(pll_out),
                ]
                if dms_si06:
                    pll_cmd.extend(["--dms-si06", dms_si06, "--dms-si06-col", dms_si06_col])
                _run(pll_cmd, dry_run, f"compute_pll_pca:{ds}")
            else:
                print(f"[compute_pll_pca:{ds}] skip existing {pll_out}")

            if gibbs_diag_args:
                marker = gibbs_diag_dir / "gibbs_pll_trajectory.png"
                if force or not marker.exists() or dry_run or not reuse_existing:
                    cmd = [
                        "uv",
                        "run",
                        "python",
                        "scripts/analysis/gibbs_diagnostics.py",
                        *gibbs_diag_args,
                        "--dms-dataset",
                        dms_arg,
                        "--dms-m22",
                        dms_m22,
                        "--dms-m22-col",
                        dms_m22_col,
                        "--max-dms",
                        str(max_dms),
                        "--sampler-label",
                        "gibbs",
                        "--output-dir",
                        str(gibbs_diag_dir),
                    ]
                    if dms_si06:
                        cmd.extend(["--dms-si06", dms_si06, "--dms-si06-col", dms_si06_col])
                    _run(cmd, dry_run, f"gibbs_diagnostics:{ds}")
                else:
                    print(f"[gibbs_diagnostics:{ds}] skip existing {marker}")

            if beam_diag_args:
                marker = beam_diag_dir / "beam_pll_trajectory.png"
                if force or not marker.exists() or dry_run or not reuse_existing:
                    cmd = [
                        "uv",
                        "run",
                        "python",
                        "scripts/analysis/gibbs_diagnostics.py",
                        *beam_diag_args,
                        "--dms-dataset",
                        dms_arg,
                        "--dms-m22",
                        dms_m22,
                        "--dms-m22-col",
                        dms_m22_col,
                        "--max-dms",
                        str(max_dms),
                        "--sampler-label",
                        "beam",
                        "--output-dir",
                        str(beam_diag_dir),
                    ]
                    if dms_si06:
                        cmd.extend(["--dms-si06", dms_si06, "--dms-si06-col", dms_si06_col])
                    _run(cmd, dry_run, f"beam_diagnostics:{ds}")
                else:
                    print(f"[beam_diagnostics:{ds}] skip existing {marker}")

        def _plot_enabled(name: str) -> bool:
            p = plots_cfg.get(name, {}) or {}
            return bool(p.get("enabled", False)) and ds in list(p.get("datasets", []) or [])

        if _plot_enabled("per_model_pca"):
            _run(
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/analysis/plot_per_model_pca.py",
                    "--projections-dir",
                    str(per_model_dir),
                    "--output-dir",
                    str(plots_root / "per_model_pca"),
                ],
                dry_run,
                f"plot_per_model_pca:{ds}",
            )

        if _plot_enabled("gibbs_per_model_pca"):
            overlay = list((plots_cfg.get("gibbs_per_model_pca", {}) or {}).get("overlay_sampling", []) or [])
            _run(
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/analysis/plot_gibbs_per_model_pca.py",
                    "--per-model-pca-dir",
                    str(per_model_dir),
                    "--embeddings-dir",
                    str(emb_dir),
                    "--output-dir",
                    str(plots_root / "per_model_pca"),
                    "--configs",
                    *overlay,
                ],
                dry_run,
                f"plot_gibbs_per_model_pca:{ds}",
            )

        if _plot_enabled("beam_per_model_pca"):
            overlay = list((plots_cfg.get("beam_per_model_pca", {}) or {}).get("overlay_sampling", []) or [])
            _run(
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/analysis/plot_gibbs_per_model_pca.py",
                    "--per-model-pca-dir",
                    str(per_model_dir),
                    "--embeddings-dir",
                    str(beam_emb_dir),
                    "--output-dir",
                    str(beam_plot_dir),
                    "--configs",
                    *overlay,
                ],
                dry_run,
                f"plot_beam_per_model_pca:{ds}",
            )

        if _plot_enabled("diff_vectors_pca"):
            _run(
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/analysis/plot_diff_vectors_pca.py",
                    "--projections-dir",
                    str(diff_dir),
                    "--output-dir",
                    str(plots_root / "diff_pca"),
                ],
                dry_run,
                f"plot_diff_vectors_pca:{ds}",
            )

        if _plot_enabled("pll_pca"):
            if not pll_out.exists() and not dry_run:
                print(f"[plot_pll_pca:{ds}] skip missing input {pll_out}")
            else:
                _run(
                    [
                        "uv",
                        "run",
                        "python",
                        "scripts/analysis/plot_pll_pca.py",
                        "--input",
                        str(pll_out),
                        "--output-dir",
                        str(plots_root / "pll_pca"),
                    ],
                    dry_run,
                    f"plot_pll_pca:{ds}",
                )

        if _plot_enabled("pll_vs_enrichment_overlays"):
            overlay_plot_cfg = plots_cfg.get("pll_vs_enrichment_overlays", {}) or {}
            overlay = list(overlay_plot_cfg.get("overlay_sampling", []) or [])
            params = dict(overlay_plot_cfg.get("params", {}) or {})
            cmd = [
                "uv",
                "run",
                "python",
                "scripts/analysis/plot_pll_vs_enrichment_overlays.py",
                *pll_variant_args,
                "--datasets",
                ds,
                "--output-dir",
                str(pll_overlay_dir),
                "--include-samplers",
                *overlay,
            ]
            for model in models:
                for sid in overlay:
                    sentry = selected_entries[sid]
                    csv_path = _strategy_csv_path(sentry, sid, model, repo_root)
                    cmd.extend(["--overlay-csv", f"{sid}={model.display_name}={csv_path}"])
            m22, si06, m22_col, si06_col = _dataset_paths(
                cfg,
                ds,
                repo_root,
                force=force,
                materialize=not dry_run,
            )
            dataset_spec = f"{ds}={m22}"
            if si06:
                dataset_spec = f"{dataset_spec}={si06}"
            cmd.extend([
                "--dataset-spec",
                dataset_spec,
                "--dms-m22-col",
                m22_col,
                "--dms-si06-col",
                si06_col,
            ])
            for k, v in params.items():
                _append_flag(cmd, k, v)
            _run(cmd, dry_run, f"plot_pll_vs_enrichment_overlays:{ds}")

        # ----- New beam eval plots (DMS-dependent) -----
        # These run per (model × beam strategy × dataset).
        for model in models:
            for sid in selected_strategy_ids:
                sentry = selected_entries[sid]
                if str(sentry.get("type", "")).lower() != "beam":
                    continue
                csv_path = _strategy_csv_path(sentry, sid, model, repo_root)
                if not csv_path.exists() and not dry_run:
                    continue
                beam_eval_dir = plots_root / "beam_eval" / sid / model.model_id
                _mkdir(beam_eval_dir, dry_run, f"beam_eval:{ds}:{sid}:{model.model_id}")

                if _beam_plot_enabled("beam_pll_vs_dms_histogram"):
                    cmd = [
                        "uv", "run", "python",
                        "scripts/analysis/plot_beam_pll_vs_dms.py",
                        "--beam-csv", str(csv_path),
                        "--model-variant", model.display_name,
                        "--dms-m22", dms_m22,
                        "--dms-m22-col", dms_m22_col,
                        "--max-dms", str(max_dms),
                        "--output-dir", str(beam_eval_dir),
                    ]
                    if model.checkpoint_path:
                        cmd.extend(["--checkpoint-path", model.checkpoint_path])
                    if dms_si06:
                        cmd.extend(["--dms-si06", dms_si06, "--dms-si06-col", dms_si06_col])
                    _run(cmd, dry_run, f"beam_pll_vs_dms_histogram:{ds}:{sid}:{model.model_id}")

                if _beam_plot_enabled("beam_aa_heatmap"):
                    cmd = [
                        "uv", "run", "python",
                        "scripts/analysis/plot_beam_aa_heatmap.py",
                        "--beam-csv", str(csv_path),
                        "--model-variant", model.display_name,
                        "--dms-m22", dms_m22,
                        "--dms-m22-col", dms_m22_col,
                        "--max-dms", str(max_dms),
                        "--output-dir", str(beam_eval_dir),
                    ]
                    if dms_si06:
                        cmd.extend(["--dms-si06", dms_si06, "--dms-si06-col", dms_si06_col])
                    _run(cmd, dry_run, f"beam_aa_heatmap:{ds}:{sid}:{model.model_id}")

    # ----- New beam eval plots (no DMS dependency) -----
    # These run once per (model × beam strategy), independent of active datasets.
    if any(_beam_plot_enabled(p) for p in BEAM_EVAL_NODMS_PLOTS):
        for model in models:
            for sid in selected_strategy_ids:
                sentry = selected_entries[sid]
                if str(sentry.get("type", "")).lower() != "beam":
                    continue
                csv_path = _strategy_csv_path(sentry, sid, model, repo_root)
                if not csv_path.exists() and not dry_run:
                    continue
                nodata_dir = project_base / "plots" / "beam_eval" / sid / model.model_id
                _mkdir(nodata_dir, dry_run, f"beam_eval_nodata:{sid}:{model.model_id}")

                if _beam_plot_enabled("beam_diversity_diagnostics"):
                    cmd = [
                        "uv", "run", "python",
                        "scripts/analysis/plot_beam_diversity.py",
                        "--beam-csv", str(csv_path),
                        "--model-variant", model.display_name,
                        "--output-dir", str(nodata_dir),
                    ]
                    _run(cmd, dry_run, f"beam_diversity_diagnostics:{sid}:{model.model_id}")

                if _beam_plot_enabled("beam_pll_vs_nmut"):
                    cmd = [
                        "uv", "run", "python",
                        "scripts/analysis/plot_beam_pll_vs_nmut.py",
                        "--beam-csv", str(csv_path),
                        "--model-variant", model.display_name,
                        "--output-dir", str(nodata_dir),
                    ]
                    if model.checkpoint_path:
                        cmd.extend(["--checkpoint-path", model.checkpoint_path])
                    _run(cmd, dry_run, f"beam_pll_vs_nmut:{sid}:{model.model_id}")

    # 4) OAS UMAP plots.
    if oas_enabled and oas_npz_paths:
        oas_npz_paths = [p for p in oas_npz_paths if dry_run or p.exists()]
        if not oas_npz_paths:
            print("[plot_oas_umap] skip: no existing OAS embedding artifacts found")
            print("[done] config-driven analysis completed")
            return 0
        modes = list(oas_plot_cfg.get("color_modes", []) or [])
        if not modes:
            modes = ["germline_family", "j_gene", "vgene_within_family", "shm_within_family", "cdr3_length"]
        filters = dict(oas_plot_cfg.get("filter_family", {}) or {})
        oas_plot_dir = project_base / "plots" / "oas_umap"
        _mkdir(oas_plot_dir, dry_run, "plot_oas_umap")
        for mode in modes:
            cmd = [
                "uv",
                "run",
                "python",
                "scripts/analysis/plot_oas_umap.py",
                *[str(p) for p in oas_npz_paths],
                "--output-dir",
                str(oas_plot_dir),
                "--color-by",
                mode,
            ]
            if mode in {"vgene_within_family", "shm_within_family"}:
                family = filters.get(mode)
                if family:
                    cmd.extend(["--filter-family", str(family)])
            _run(cmd, dry_run, f"plot_oas_umap:{mode}")

    print("[done] config-driven analysis completed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"[config-error] {exc}", file=sys.stderr)
        raise SystemExit(2)
