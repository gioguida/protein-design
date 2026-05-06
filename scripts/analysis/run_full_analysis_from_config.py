#!/usr/bin/env python3
"""Config-driven orchestrator for the full embedding analysis pipeline."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

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
        if not dval or not dval.get("m22_csv"):
            fail(f"datasets.catalog.{dkey}.m22_csv is required")

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


def _dataset_paths(cfg: dict[str, Any], dataset_key: str, repo_root: Path) -> tuple[str, str | None]:
    d = cfg["datasets"]["catalog"][dataset_key]
    m22 = str(_expand(str(d["m22_csv"]), repo_root))
    si06_raw = d.get("si06_csv")
    si06 = str(_expand(str(si06_raw), repo_root)) if si06_raw else None
    return m22, si06


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path = args.config if args.config.is_absolute() else (repo_root / args.config)
    cfg = _load_and_validate(cfg_path, repo_root)

    run_cfg = cfg.get("run", {})
    dry_run = bool(run_cfg.get("dry_run", False)) or bool(args.dry_run)
    force = bool(run_cfg.get("force", False))
    reuse_existing = bool(run_cfg.get("reuse_existing_artifacts", False))
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
                _append_flag(cmd, k, v)
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

    # 3) Dataset-driven stages.
    required_datasets: list[str] = []
    for plot_name, pcfg in plots_cfg.items():
        if not bool((pcfg or {}).get("enabled", False)):
            continue
        if plot_name not in DATASET_PLOTS:
            continue
        for ds in list((pcfg or {}).get("datasets", []) or []):
            if ds not in required_datasets:
                required_datasets.append(ds)

    for ds in required_datasets:
        dms_arg = _dataset_arg(ds)
        dms_m22, dms_si06 = _dataset_paths(cfg, ds, repo_root)

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
                "--max-dms",
                str(max_dms),
                "--max-gibbs",
                str(max_gibbs),
            ]
            if dms_si06:
                cmd.extend(["--dms-si06", dms_si06])
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
                    "--max-dms",
                    str(max_dms),
                    "--max-gibbs",
                    str(max_gibbs),
                ]
                if dms_si06:
                    beam_cmd.extend(["--dms-si06", dms_si06])
                beam_cmd.extend(beam_paths)
                if ckpt:
                    beam_cmd.extend(["--checkpoint-path", ckpt])
                if not force:
                    beam_cmd.append("--skip-if-current")
                if reuse_existing and beam_npz_path.exists():
                    print(f"[extract_beam_embeddings:{ds}:{model.model_id}] reuse existing {beam_npz_path}")
                else:
                    _run(beam_cmd, dry_run, f"extract_beam_embeddings:{ds}:{model.model_id}")

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

        pll_out = pll_dir / "pll_pca.npz"
        if force or not pll_out.exists() or dry_run or not reuse_existing:
            _run(
                [
                    "uv",
                    "run",
                    "python",
                    "scripts/analysis/compute_pll_pca.py",
                    *pll_variant_args,
                    "--max-dms",
                    str(max_dms),
                    "--output-path",
                    str(pll_out),
                ],
                dry_run,
                f"compute_pll_pca:{ds}",
            )
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
                    "--max-dms",
                    str(max_dms),
                    "--output-dir",
                    str(gibbs_diag_dir),
                ]
                if dms_si06:
                    cmd.extend(["--dms-si06", dms_si06])
                _run(cmd, dry_run, f"gibbs_diagnostics:{ds}")
            else:
                print(f"[gibbs_diagnostics:{ds}] skip existing {marker}")

        if beam_diag_args:
            marker = beam_diag_dir / "gibbs_pll_trajectory.png"
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
                    "--max-dms",
                    str(max_dms),
                    "--output-dir",
                    str(beam_diag_dir),
                ]
                if dms_si06:
                    cmd.extend(["--dms-si06", dms_si06])
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
            m22, si06 = _dataset_paths(cfg, ds, repo_root)
            dataset_spec = f"{ds}={m22}"
            if si06:
                dataset_spec = f"{dataset_spec}={si06}"
            cmd.extend(["--dataset-spec", dataset_spec])
            for k, v in params.items():
                _append_flag(cmd, k, v)
            _run(cmd, dry_run, f"plot_pll_vs_enrichment_overlays:{ds}")

    # 4) OAS UMAP plots.
    if oas_enabled and oas_npz_paths:
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
