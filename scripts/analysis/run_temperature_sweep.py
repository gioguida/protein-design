"""Temperature sweep orchestrator for SBS analysis.

Reads conf/analysis/temperature_sweep.yaml (or --config path) and:
  1. Runs stochastic_beam_search.py for each temperature × model combination.
  2. Runs enabled per-temperature plots (beam_pll_trajectory, beam_pll_vs_dms_histogram,
     beam_diversity_diagnostics, beam_pll_vs_nmut, beam_aa_heatmap).
  3. After all temperatures, produces cross-temperature summary plots:
       - plot_temp_pll_vs_diversity.py   (PLL vs diversity tradeoff)
       - plot_temp_pll_distributions.py  (overlaid PLL histograms)

Usage
-----
uv run python scripts/analysis/run_temperature_sweep.py \
    --config conf/analysis/temperature_sweep.yaml [--dry-run]
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config", type=Path,
        default=Path("conf/analysis/temperature_sweep.yaml"),
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


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
    """e.g. 1.0 -> '1.0', 0.5 -> '0.5'"""
    return str(temp)


def main() -> int:
    args = parse_args()
    repo_root = REPO_ROOT
    cfg_path = args.config if args.config.is_absolute() else (repo_root / args.config)

    if not cfg_path.exists():
        print(f"[config-error] config file not found: {cfg_path}", file=sys.stderr)
        return 2
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    dry_run = bool(cfg.get("run", {}).get("dry_run", False)) or bool(args.dry_run)

    sweep = dict(cfg.get("sweep", {}))
    temperatures = [float(t) for t in sweep.get("temperatures", [1.0])]
    beam_size = int(sweep.get("beam_size", 5))
    n_steps = int(sweep.get("n_steps", 5))
    snapshot_every = int(sweep.get("snapshot_every", 1))
    start_mode = str(sweep.get("start_mode", "wt"))
    n_chains = int(sweep.get("n_chains", 10))

    models_cfg = list(cfg.get("models", []))
    if not models_cfg:
        print("[config-error] models list is empty", file=sys.stderr)
        return 2

    plots_cfg = dict(cfg.get("plots", {}))
    summary_cfg = dict(cfg.get("summary", {}))
    top_k_pll = int(summary_cfg.get("top_k_for_pll", 50))

    out_cfg = dict(cfg.get("output", {}))
    base_dir = Path(str(out_cfg.get("base_dir", "outputs/temperature_sweep")))
    if not base_dir.is_absolute():
        base_dir = repo_root / base_dir
    plots_dir = Path(str(out_cfg.get("plots_dir", "reports/temperature_sweep")))
    if not plots_dir.is_absolute():
        plots_dir = repo_root / plots_dir

    dms_cfg = dict(cfg.get("dms", {}))
    dms_m22 = dms_cfg.get("m22_path")
    dms_si06 = dms_cfg.get("si06_path")
    dms_m22_col = str(dms_cfg.get("m22_col", "M22_binding_enrichment_adj"))
    dms_si06_col = str(dms_cfg.get("si06_col", "SI06_binding_enrichment_adj"))
    max_dms = int(dms_cfg.get("max_dms", 500))

    for model_spec in models_cfg:
        model_name = str(model_spec.get("name", "model"))
        checkpoint = str(model_spec.get("checkpoint") or "")
        model_variant = str(model_spec.get("model_variant", model_name))

        # Collect CSV paths per temperature for summary plots.
        temp_csv_map: dict[float, Path] = {}

        for temp in temperatures:
            t_label = _temp_label(temp)
            run_dir = base_dir / model_name / f"temp_{t_label}"
            _mkdir(run_dir, dry_run, f"sweep:{model_name}:T={t_label}")

            csv_path = run_dir / "beam_output.csv"
            temp_csv_map[temp] = csv_path

            # --- Run SBS ---
            cmd = [
                "uv", "run", "python",
                "scripts/stochastic_beam_search.py",
                "--model-variant", model_variant,
                "--beam-size", str(beam_size),
                "--n-steps", str(n_steps),
                "--snapshot-every", str(snapshot_every),
                "--temperature", str(temp),
                "--start-mode", start_mode,
                "--output-path", str(csv_path),
            ]
            if checkpoint:
                cmd.extend(["--checkpoint-path", checkpoint])
            if start_mode in ("dms", "top_dms"):
                if dms_m22:
                    cmd.extend(["--dms-m22-path", str(dms_m22)])
                if dms_si06:
                    cmd.extend(["--dms-si06-path", str(dms_si06)])
            if start_mode == "top_dms":
                cmd.extend(["--top-k-dms", str(n_chains)])
            _run(cmd, dry_run, f"sbs:{model_name}:T={t_label}")

            # --- Per-temperature plots ---
            per_temp_dir = plots_dir / model_name / f"temp_{t_label}"
            _mkdir(per_temp_dir, dry_run, f"plots:{model_name}:T={t_label}")

            mv_with_temp = f"{model_variant}_T{t_label}"

            if bool(plots_cfg.get("beam_pll_trajectory", False)):
                # Routes to the existing gibbs_diagnostics.py.
                gibbs_spec = f"{model_variant}={checkpoint}={csv_path}"
                pll_traj_cmd = [
                    "uv", "run", "python",
                    "scripts/analysis/gibbs_diagnostics.py",
                    "--gibbs", gibbs_spec,
                    "--sampler-label", "beam",
                    "--output-dir", str(per_temp_dir),
                ]
                if dms_m22:
                    pll_traj_cmd.extend([
                        "--dms-m22", str(dms_m22),
                        "--dms-m22-col", dms_m22_col,
                        "--max-dms", str(max_dms),
                    ])
                if dms_si06:
                    pll_traj_cmd.extend([
                        "--dms-si06", str(dms_si06),
                        "--dms-si06-col", dms_si06_col,
                    ])
                _run(pll_traj_cmd, dry_run, f"beam_pll_trajectory:{model_name}:T={t_label}")

            if bool(plots_cfg.get("beam_pll_vs_dms_histogram", False)) and dms_m22:
                pll_hist_cmd = [
                    "uv", "run", "python",
                    "scripts/analysis/plot_beam_pll_vs_dms.py",
                    "--beam-csv", str(csv_path),
                    "--model-variant", mv_with_temp,
                    "--dms-m22", str(dms_m22),
                    "--dms-m22-col", dms_m22_col,
                    "--max-dms", str(max_dms),
                    "--output-dir", str(per_temp_dir),
                ]
                if checkpoint:
                    pll_hist_cmd.extend(["--checkpoint-path", checkpoint])
                if dms_si06:
                    pll_hist_cmd.extend([
                        "--dms-si06", str(dms_si06),
                        "--dms-si06-col", dms_si06_col,
                    ])
                _run(pll_hist_cmd, dry_run, f"beam_pll_vs_dms:{model_name}:T={t_label}")

            if bool(plots_cfg.get("beam_diversity_diagnostics", False)):
                div_cmd = [
                    "uv", "run", "python",
                    "scripts/analysis/plot_beam_diversity.py",
                    "--beam-csv", str(csv_path),
                    "--model-variant", mv_with_temp,
                    "--output-dir", str(per_temp_dir),
                ]
                _run(div_cmd, dry_run, f"beam_diversity:{model_name}:T={t_label}")

            if bool(plots_cfg.get("beam_pll_vs_nmut", False)):
                nmut_cmd = [
                    "uv", "run", "python",
                    "scripts/analysis/plot_beam_pll_vs_nmut.py",
                    "--beam-csv", str(csv_path),
                    "--model-variant", mv_with_temp,
                    "--output-dir", str(per_temp_dir),
                ]
                if checkpoint:
                    nmut_cmd.extend(["--checkpoint-path", checkpoint])
                _run(nmut_cmd, dry_run, f"beam_pll_vs_nmut:{model_name}:T={t_label}")

            if bool(plots_cfg.get("beam_aa_heatmap", False)) and dms_m22:
                aa_cmd = [
                    "uv", "run", "python",
                    "scripts/analysis/plot_beam_aa_heatmap.py",
                    "--beam-csv", str(csv_path),
                    "--model-variant", mv_with_temp,
                    "--dms-m22", str(dms_m22),
                    "--dms-m22-col", dms_m22_col,
                    "--max-dms", str(max_dms),
                    "--output-dir", str(per_temp_dir),
                ]
                if dms_si06:
                    aa_cmd.extend([
                        "--dms-si06", str(dms_si06),
                        "--dms-si06-col", dms_si06_col,
                    ])
                _run(aa_cmd, dry_run, f"beam_aa_heatmap:{model_name}:T={t_label}")

        # --- Summary plots across temperatures ---
        summary_dir = plots_dir / model_name / "summary"
        _mkdir(summary_dir, dry_run, f"summary:{model_name}")

        # Build --temp-csv T=PATH arguments for summary scripts.
        temp_csv_args: list[str] = []
        for temp, csv_p in temp_csv_map.items():
            temp_csv_args.extend(["--temp-csv", f"{temp}={csv_p}"])

        if bool(summary_cfg.get("pll_vs_diversity_tradeoff", False)):
            tradeoff_cmd = [
                "uv", "run", "python",
                "scripts/analysis/plot_temp_pll_vs_diversity.py",
                *temp_csv_args,
                "--model-variant", model_variant,
                "--top-k", str(top_k_pll),
                "--output-dir", str(summary_dir),
            ]
            if checkpoint:
                tradeoff_cmd.extend(["--checkpoint-path", checkpoint])
            _run(tradeoff_cmd, dry_run, f"summary_pll_vs_diversity:{model_name}")

        if bool(summary_cfg.get("pll_distribution_by_temp", False)):
            dist_cmd = [
                "uv", "run", "python",
                "scripts/analysis/plot_temp_pll_distributions.py",
                *temp_csv_args,
                "--model-variant", model_variant,
                "--output-dir", str(summary_dir),
            ]
            if checkpoint:
                dist_cmd.extend(["--checkpoint-path", checkpoint])
            _run(dist_cmd, dry_run, f"summary_pll_distributions:{model_name}")

    print("[done] temperature sweep completed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
