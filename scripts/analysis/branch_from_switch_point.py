#!/usr/bin/env python
"""Launch the single-position continuation of a finished batch-masking run.

The switch point is not known until a base run has finished its epoch, because
the rule picks the earliest checkpoint that maximises CDR recovery and that
cannot be decided while later points are still missing. This script reads the
recorded evaluation curve, applies the rule, and submits the continuation from
the checkpoint it picks.

The continuation is measured against its own branch point rather than against
the base pretrained model, so the branch point's framework accuracy is passed
through explicitly.

Usage:
  uv run scripts/analysis/branch_from_switch_point.py --run-dir $TRAIN_DIR/<run> --dry-run
  uv run scripts/analysis/branch_from_switch_point.py --manifest bash_scripts/logs/sweep_*.csv
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv(".env.local")
load_dotenv()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir", help="A finished base run directory.")
    g.add_argument(
        "--manifest", help="A sweep manifest CSV; branches every run listed in it."
    )
    p.add_argument(
        "--tolerance-pp", type=float, default=None,
        help=(
            "Re-apply the rule at a different framework tolerance. Defaults to "
            "the one the run itself used."
        ),
    )
    p.add_argument("--data", default=None, help="Data config for the continuation.")
    p.add_argument("--gpu-mem", default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def select_switch_point(run_dir: Path, tolerance_pp: float | None) -> dict:
    """Earliest evaluation point maximising CDR accuracy among the eligible."""
    points = pd.read_csv(run_dir / "eval_points.csv")
    if tolerance_pp is None:
        eligible = points[points["eligible"]]
    else:
        eligible = points[points["fr_drift_pp"] <= tolerance_pp]
    if eligible.empty:
        raise SystemExit(
            f"{run_dir}: no evaluation point stayed inside the framework "
            f"tolerance, so there is no switch point to branch from."
        )
    best = eligible["cdr_accuracy"].max()
    # Earliest, so ties resolve towards less training rather than more.
    return eligible[eligible["cdr_accuracy"] == best].iloc[0].to_dict()


def branch_one(run_dir: Path, args: argparse.Namespace) -> None:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    point = select_switch_point(run_dir, args.tolerance_pp)

    pack_path = str(cfg["data"]["pack_path"])
    data_group = args.data or (
        "evo/c05_wt_similar_single_pool"
        if "c05_wt_similar" in pack_path
        else "evo/oas_single_pool"
    )
    run_name = f"{cfg.get('run_name') or run_dir.name}_single"

    overrides = [
        f"data={data_group}",
        f"model.name={cfg['model']['name']}",
        f"training.learning_rate={cfg['training']['learning_rate']}",
        f"training.resume_checkpoint={point['checkpoint']}",
        f"training.pareto_reference_framework_accuracy={point['framework_accuracy']}",
        f"run_name={run_name}",
    ]
    cmd = ["sbatch"]
    if args.gpu_mem:
        cmd.append(f"--gres=gpumem:{args.gpu_mem}")
    cmd += ["bash_scripts/train.sbatch", "evotuning", *overrides]

    print(
        f"{run_dir.name}: switch point at {point['epoch_frac']:.4f} of epoch "
        f"{int(point['epoch'])} (step {int(point['step'])}), CDR accuracy "
        f"{point['cdr_accuracy']:.4f}, framework drift {point['fr_drift_pp']:.3f}pp"
    )
    if args.dry_run:
        print("  [dry-run]", " ".join(cmd))
        return
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()
    print(f"  {out}  [{run_name}]")


def main() -> None:
    args = parse_args()
    if args.run_dir:
        branch_one(Path(args.run_dir), args)
        return

    train_dir = Path(os.environ.get("TRAIN_DIR", "."))
    manifest = pd.read_csv(args.manifest)
    missing = []
    for run_name in manifest["run_name"]:
        matches = sorted(train_dir.glob(f"{run_name}*"))
        matches = [m for m in matches if (m / "eval_points.csv").exists()]
        if not matches:
            missing.append(run_name)
            continue
        branch_one(matches[-1], args)
    if missing:
        print(
            f"\n{len(missing)} run(s) have no evaluation curve yet and were skipped:",
            file=sys.stderr,
        )
        for m in missing:
            print(f"  {m}", file=sys.stderr)


if __name__ == "__main__":
    main()
