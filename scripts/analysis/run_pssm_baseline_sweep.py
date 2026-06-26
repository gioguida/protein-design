"""Run a temperature sweep for the train-split CDR-H3 PSSM baseline sampler."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protein_design import constants
from protein_design.dms_splitting import dataset_spec
from protein_design.pssm_baseline import resolve_train_split_with_fallback


def _resolve_enrichment_threshold(value):
    """Resolve the config's enrichment_threshold.

    A number (or numeric string) is used as-is. A non-numeric string is treated
    as the name of a constant in ``protein_design.constants`` (so configs can
    say ``WT_M22_BINDING_ENRICHMENT`` instead of hardcoding the number). None
    disables the filter.
    """
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            if not hasattr(constants, value):
                raise SystemExit(
                    f"enrichment_threshold {value!r} is not a numeric value nor a "
                    f"name in protein_design.constants"
                )
            return float(getattr(constants, value))
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("conf/analysis/pssm_baseline_sweep.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


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


def _temp_label(temp: float) -> str:
    return str(temp)


def _expand_path(path_str: str, repo_root: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(path_str))
    path = Path(expanded)
    if not path.is_absolute():
        path = repo_root / path
    return path


def main() -> int:
    args = parse_args()
    cfg_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    if not cfg_path.exists():
        print(f"[config-error] config file not found: {cfg_path}", file=sys.stderr)
        return 2

    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    dry_run = bool(args.dry_run)
    sweep_cfg = dict(cfg.get("sweep", {}))
    plots_cfg = dict(cfg.get("plots", {}))
    summary_cfg = dict(cfg.get("summary", {}))
    output_cfg = dict(cfg.get("output", {}))
    dms_reference_cfg = dict(cfg.get("dms_reference", {}))

    temperatures = [float(temp) for temp in sweep_cfg.get("temperatures", [1.0])]
    n_sequences = int(sweep_cfg.get("n_sequences", 500))
    seed = int(sweep_cfg.get("seed", 42))
    enrichment_threshold = _resolve_enrichment_threshold(sweep_cfg.get("enrichment_threshold"))

    dataset_key = str(cfg.get("dataset", "ed2_m22"))
    dms_config = str(cfg.get("dms_config", "conf/data/dms/default.yaml"))
    local_splits_dir = str(cfg.get("local_splits_dir", "data/dms_splits"))
    max_dms = int(dms_reference_cfg.get("max_dms", 500))
    spec = dataset_spec(dataset_key, dms_config)

    base_dir = _expand_path(str(output_cfg.get("base_dir", "outputs/baseline_sampler")), REPO_ROOT)
    plots_dir = _expand_path(str(output_cfg.get("plots_dir", "outputs/baseline_sampler/plots")), REPO_ROOT)

    reference_split_path, reference_source = resolve_train_split_with_fallback(
        dataset_key=dataset_key,
        dms_config_path=dms_config,
        local_splits_dir=local_splits_dir,
    )
    print(f"[dms-reference] using train split ({reference_source}): {reference_split_path}")

    temp_csv_map: dict[float, Path] = {}
    for temp in temperatures:
        t_label = _temp_label(temp)
        run_dir = base_dir / f"temp_{t_label}"
        per_temp_plot_dir = plots_dir / f"temp_{t_label}"
        _mkdir(run_dir, dry_run, f"sweep:T={t_label}")
        _mkdir(per_temp_plot_dir, dry_run, f"plots:T={t_label}")

        csv_path = run_dir / "pssm_output.csv"
        temp_csv_map[temp] = csv_path

        sample_cmd = [
            "uv", "run", "python",
            "scripts/pssm_sampling.py",
            "--dataset-key", dataset_key,
            "--dms-config", dms_config,
            "--local-splits-dir", local_splits_dir,
            "--temperature", str(temp),
            "--n-sequences", str(n_sequences),
            "--seed", str(seed),
            "--output-path", str(csv_path),
        ]
        if enrichment_threshold is not None:
            sample_cmd.extend(["--enrichment-threshold", str(enrichment_threshold)])
        _run(sample_cmd, dry_run, f"pssm:T={t_label}")

        need_model_free_diag = any(
            bool(plots_cfg.get(key, False))
            for key in ("sequence_logo", "edit_distance", "pairwise_hamming", "summary_csv")
        )
        if need_model_free_diag:
            diag_cmd = [
                "uv", "run", "python",
                "scripts/analysis/gibbs_diagnostics.py",
                "--gibbs", f"pssm=unused={csv_path}",
                "--sampler-label", "pssm",
                "--skip-pll",
                "--skip-early",
                "--output-dir", str(per_temp_plot_dir),
            ]
            if not bool(plots_cfg.get("pairwise_hamming", False)):
                diag_cmd.append("--skip-pairwise-hamming")
            if not bool(plots_cfg.get("edit_distance", False)):
                diag_cmd.append("--skip-edit-distance")
            if not bool(plots_cfg.get("sequence_logo", False)):
                diag_cmd.append("--skip-sequence-logo")
            diag_cmd.append("--skip-position-mutation-freq")
            if not bool(plots_cfg.get("summary_csv", False)):
                diag_cmd.append("--skip-summary-csv")
            _run(diag_cmd, dry_run, f"model_free_diagnostics:T={t_label}")

        if bool(plots_cfg.get("diversity_diagnostics", False)):
            diversity_cmd = [
                "uv", "run", "python",
                "scripts/analysis/plot_beam_diversity.py",
                "--beam-csv", str(csv_path),
                "--model-variant", f"pssm_T{t_label}",
                "--sampler-label", "pssm",
                "--output-name", "pssm_diversity_diagnostics.png",
                "--output-dir", str(per_temp_plot_dir),
            ]
            _run(diversity_cmd, dry_run, f"diversity_diagnostics:T={t_label}")

        if bool(plots_cfg.get("aa_heatmap", False)):
            aa_heatmap_cmd = [
                "uv", "run", "python",
                "scripts/analysis/plot_beam_aa_heatmap.py",
                "--beam-csv", str(csv_path),
                "--dms-m22", str(reference_split_path),
                "--dms-m22-col", spec.key_metric_col,
                "--max-dms", str(max_dms),
                "--model-variant", f"pssm_T{t_label}",
                "--sampler-label", "pssm",
                "--output-name", "pssm_aa_heatmap.png",
                "--output-dir", str(per_temp_plot_dir),
            ]
            _run(aa_heatmap_cmd, dry_run, f"aa_heatmap:T={t_label}")

    summary_dir = plots_dir / "summary"
    _mkdir(summary_dir, dry_run, "summary")
    temp_csv_args: list[str] = []
    for temp, csv_path in temp_csv_map.items():
        temp_csv_args.extend(["--temp-csv", f"{temp}={csv_path}"])

    if bool(summary_cfg.get("entropy_heatmap", False)):
        entropy_cmd = [
            "uv", "run", "python",
            "scripts/analysis/plot_temp_entropy_heatmap.py",
            *temp_csv_args,
            "--model-variant", "pssm",
            "--sampler-label", "pssm",
            "--output-name", "pssm_entropy_heatmap.png",
            "--output-dir", str(summary_dir),
        ]
        _run(entropy_cmd, dry_run, "summary_entropy_heatmap")

    if bool(summary_cfg.get("jsd_vs_temp", False)):
        jsd_cmd = [
            "uv", "run", "python",
            "scripts/analysis/plot_temp_jsd_vs_temp.py",
            *temp_csv_args,
            "--model-variant", "pssm",
            "--dms-m22", str(reference_split_path),
            "--max-dms", str(max_dms),
            "--sampler-label", "pssm",
            "--output-name", "pssm_jsd_vs_temp.png",
            "--output-dir", str(summary_dir),
        ]
        _run(jsd_cmd, dry_run, "summary_jsd_vs_temp")

    print("[done] PSSM baseline temperature sweep completed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
