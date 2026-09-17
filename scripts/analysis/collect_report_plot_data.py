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
    parser.add_argument("--sections", default=None, help="Compatibility mode: functional,preference,generation or all.")
    parser.add_argument("--work-unit", choices=[
        "functional-model", "functional-reduce", "preference-model", "preference-reduce",
        "generation-baselines", "generation-sample", "generation-native-score",
        "generation-common-score", "generation-reduce",
    ], help="One independently schedulable collection stage.")
    parser.add_argument("--model", help="Report model key required by model-specific work units.")
    parser.add_argument("--sampler", choices=["gibbs", "stochastic_beam"],
                        help="Sampler required by generation-sample.")
    parser.add_argument("--profile", choices=["rtx_4090", "a100_80gb"],
                        help="Hardware execution profile; supplied by the GPU-specific launcher.")
    parser.add_argument("--force", action="store_true", help="Recompute and overwrite selected report JSON artifacts.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = report_data.load_report_config(args.config)
    if args.work_unit:
        unit = args.work_unit
        if unit in {"functional-model", "preference-model", "generation-sample", "generation-native-score"} and not args.model:
            raise SystemExit(f"--work-unit {unit} requires --model")
        if unit == "generation-sample" and not args.sampler:
            raise SystemExit("--work-unit generation-sample requires --sampler")
        if unit in {"functional-model", "preference-model", "generation-sample", "generation-native-score", "generation-common-score"} and not args.profile:
            raise SystemExit(f"--work-unit {unit} requires --profile")
        dispatch = {
            "functional-model": lambda: report_data.collect_functional_model(config, args.model, profile=args.profile, force=args.force),
            "functional-reduce": lambda: report_data.reduce_functional_metrics(config),
            "preference-model": lambda: report_data.collect_preference_model(config, args.model, profile=args.profile, force=args.force),
            "preference-reduce": lambda: report_data.reduce_preference_metrics(config),
            "generation-baselines": lambda: report_data.collect_generation_baselines(config, force=args.force),
            "generation-sample": lambda: report_data.collect_generation_sample(config, args.model, args.sampler, profile=args.profile, force=args.force),
            "generation-native-score": lambda: report_data.collect_generation_native_scores(config, args.model, profile=args.profile, force=args.force),
            "generation-common-score": lambda: report_data.collect_generation_common_scores(config, profile=args.profile, force=args.force),
            "generation-reduce": lambda: report_data.reduce_generation_libraries(config),
        }
        print(dispatch[unit]())
        return 0

    if args.sections is None:
        raise SystemExit("Provide --work-unit (recommended) or --sections (serial compatibility mode).")
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
