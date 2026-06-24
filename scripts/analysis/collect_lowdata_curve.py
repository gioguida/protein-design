#!/usr/bin/env python
"""Collect low-data DPO sweep results into a learning-curve table.

Each sweep run writes summary.json (with test_spearman_avg on ed5) to
$TRAIN_DIR/<base_name>_<timestamp>/. This script resolves every run by its
base_name prefix (lowdata_<model>_n<N>_s<seed>), reads the ed5 test Spearman,
and writes a tidy CSV plus a per-(model, n_train) aggregate (mean/std over seeds)
ready for the learning-curve figure.

Usage:
    uv run python scripts/analysis/collect_lowdata_curve.py
    uv run python scripts/analysis/collect_lowdata_curve.py --manifest bash_scripts/logs/dpo_lowdata_sweep_<ts>.csv
    uv run python scripts/analysis/collect_lowdata_curve.py --out report/figures/dpo_lowdata_curve.csv
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

BASE_NAME_RE = re.compile(r"^lowdata_(?P<model>.+)_n(?P<n>\d+)_s(?P<seed>\d+)$")
METRIC = "test_spearman_avg"


def _train_dir() -> Path:
    user = os.environ.get("USER", "unknown")
    scratch = os.environ.get("SCRATCH_DIR", f"/cluster/scratch/{user}/protein-design")
    return Path(os.environ.get("TRAIN_DIR", str(Path(scratch) / "train")))


def _latest_run_dir(train_dir: Path, base_name: str) -> Path | None:
    """Newest <base_name>_<timestamp> dir containing a summary.json."""
    candidates = sorted(
        (p for p in train_dir.glob(f"{base_name}_*") if (p / "summary.json").exists()),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _base_names_from_manifest(manifest: Path) -> list[str]:
    df = pd.read_csv(manifest)
    return df["base_name"].astype(str).tolist()


def _base_names_from_glob(train_dir: Path) -> list[str]:
    seen: dict[str, None] = {}
    for p in train_dir.glob("lowdata_*_n*_s*"):
        base = re.sub(r"_\d{8}_\d{6}$", "", p.name)
        if BASE_NAME_RE.match(base):
            seen.setdefault(base, None)
    return list(seen)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--manifest", type=Path, default=None,
                   help="sweep manifest CSV (default: glob $TRAIN_DIR for lowdata_* runs)")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "report" / "figures" / "dpo_lowdata_curve.csv")
    p.add_argument("--metric", default=METRIC, help=f"summary.json key to collect (default: {METRIC})")
    args = p.parse_args()

    train_dir = _train_dir()
    if not train_dir.exists():
        raise SystemExit(f"TRAIN_DIR does not exist: {train_dir}")

    base_names = (
        _base_names_from_manifest(args.manifest)
        if args.manifest is not None
        else _base_names_from_glob(train_dir)
    )
    if not base_names:
        raise SystemExit(f"No lowdata_* runs found under {train_dir} (manifest={args.manifest}).")

    rows: list[dict] = []
    missing: list[str] = []
    for base in base_names:
        m = BASE_NAME_RE.match(base)
        if m is None:
            continue
        run_dir = _latest_run_dir(train_dir, base)
        if run_dir is None:
            missing.append(base)
            continue
        summary = json.loads((run_dir / "summary.json").read_text())
        value = summary.get(args.metric)
        rows.append({
            "model": m.group("model"),
            "n_train": int(m.group("n")),
            "seed": int(m.group("seed")),
            args.metric: value,
            "run_dir": str(run_dir),
        })

    if missing:
        print(f"WARNING: {len(missing)} runs have no summary.json yet (still running?): "
              f"{', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''}")

    if not rows:
        raise SystemExit("No completed runs with summary.json found.")

    df = pd.DataFrame(rows).sort_values(["model", "n_train", "seed"]).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved per-run table: {args.out}  ({len(df)} runs)")

    finite = df[pd.to_numeric(df[args.metric], errors="coerce").notna()]
    if not finite.empty:
        agg = (
            finite.groupby(["model", "n_train"])[args.metric]
            .agg(mean="mean", std="std", n_repeats="size")
            .reset_index()
        )
        agg_path = args.out.with_name(args.out.stem + "_agg.csv")
        agg.to_csv(agg_path, index=False)
        print(f"Saved aggregate (mean/std over seeds): {agg_path}")
        print(agg.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
