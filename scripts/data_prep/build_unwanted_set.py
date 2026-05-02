#!/usr/bin/env python
"""Build per-position unwanted amino-acid sets from ED2 enrichment data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protein_design.constants import C05_CDRH3
from protein_design.dpo.dataset import default_data_paths
from protein_design.unlikelihood.preprocessing import build_unwanted_set


def _build_arg_parser() -> argparse.ArgumentParser:
    defaults = default_data_paths()
    parser = argparse.ArgumentParser(
        description="Build unwanted amino-acid sets for unlikelihood training.",
    )
    parser.add_argument(
        "--raw-csv",
        type=Path,
        default=defaults["raw_m22"],
        help="Path to ED2 enrichment CSV.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=defaults["processed_dir"],
        help="Output directory for processed artifacts.",
    )
    parser.add_argument(
        "--enrichment-col",
        type=str,
        default="M22_binding_enrichment_adj",
        help="Enrichment column used for substitution statistics.",
    )
    parser.add_argument(
        "--wt-seq",
        type=str,
        default=C05_CDRH3,
        help="Wild-type CDRH3 sequence (positions are 1-indexed).",
    )
    parser.add_argument(
        "--min-total-reads",
        type=int,
        default=10,
        help="Minimum total reads filter.",
    )
    parser.add_argument(
        "--min-observations",
        type=int,
        default=30,
        help="Minimum observations required to flag a substitution as unwanted.",
    )
    parser.add_argument(
        "--summary-csv-name",
        type=str,
        default="unwanted_substitution_enrichment.csv",
        help="Output CSV filename for full per-substitution statistics.",
    )
    parser.add_argument(
        "--unwanted-json-name",
        type=str,
        default="unwanted_set.json",
        help="Output JSON filename for position -> unwanted amino acids.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    summary_csv_path, unwanted_json_path = build_unwanted_set(
        raw_csv_path=Path(args.raw_csv),
        processed_dir=Path(args.processed_dir),
        enrichment_col=str(args.enrichment_col),
        wt_seq=str(args.wt_seq),
        min_total_reads=int(args.min_total_reads),
        min_observations=int(args.min_observations),
        summary_csv_name=str(args.summary_csv_name),
        unwanted_json_name=str(args.unwanted_json_name),
    )
    print(f"Wrote substitution summary CSV: {summary_csv_path}")
    print(f"Wrote unwanted-set JSON: {unwanted_json_path}")


if __name__ == "__main__":
    main()
