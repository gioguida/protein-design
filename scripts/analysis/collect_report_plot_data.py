#!/usr/bin/env python3
"""Collect GPU-backed JSON artifacts consumed by report/report_plots.ipynb."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from protein_design.analysis import report_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="conf/analysis/report_plots.yaml")
    parser.add_argument("--sections", default="all", help="Comma-separated: functional,preference,generation or all.")
    parser.add_argument("--force", action="store_true", help="Recompute and overwrite selected report JSON artifacts.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = report_data.load_report_config(args.config)
    sections = {item.strip() for item in args.sections.split(",") if item.strip()}
    if "all" in sections:
        sections = {"functional", "preference", "generation"}
    unknown = sections.difference({"functional", "preference", "generation"})
    if unknown:
        raise SystemExit(f"Unknown section(s): {sorted(unknown)}")

    if "functional" in sections:
        print("[section] functional")
        print(report_data.collect_functional_metrics(config, force=args.force))
    if "preference" in sections:
        print("[section] preference")
        print(report_data.collect_preference_metrics(config, force=args.force))
    if "generation" in sections:
        print("[section] generation")
        for path in report_data.collect_generation_libraries(config, force=args.force):
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
