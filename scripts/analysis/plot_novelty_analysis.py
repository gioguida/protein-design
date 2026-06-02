"""Novelty analysis: fraction of generated CDRH3s already present in DMS datasets.

For each temperature, count how many unique generated CDRH3 sequences appear
in the existing DMS datasets, then produce a stacked bar chart:

  - bottom segment (blue)  : sequences already present in at least one dataset
  - top segment   (green)  : novel sequences not seen in any dataset
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protein_design.analysis.novelty import build_reference_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("plot_novelty_analysis")

COLOR_FOUND = "#4878d0"
COLOR_NOVEL = "#6acc65"


def _parse_temp_csv(spec: str) -> tuple[float, Path]:
    parts = spec.split("=", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"--temp-csv must be T=CSV_PATH, got {spec!r}")
    return float(parts[0]), Path(parts[1])


def plot_novelty(
    temperatures: list[float],
    n_found: list[int],
    n_novel: list[int],
    model_variant: str,
    out_path: Path,
) -> None:
    n_total = [f + v for f, v in zip(n_found, n_novel)]
    x = list(range(len(temperatures)))

    fig, ax = plt.subplots(figsize=(max(8.0, len(temperatures) * 1.1), 5.5))

    ax.bar(x, n_found, color=COLOR_FOUND, label="In dataset")
    ax.bar(x, n_novel, bottom=n_found, color=COLOR_NOVEL, label="Novel (not in dataset)")

    for xi, (tot, nv) in enumerate(zip(n_total, n_novel)):
        frac = nv / tot if tot > 0 else 0.0
        ax.text(
            xi,
            tot,
            f" {frac:.1%}\n novel",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in temperatures])
    ax.set_xlabel("Temperature", fontsize=11)
    ax.set_ylabel("Unique CDRH3 sequences", fontsize=11)
    ax.set_title(
        f"Generated sequence novelty by temperature  (model: {model_variant})",
        fontsize=12,
    )
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, max(n_total) * 1.18 if n_total else 1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--temp-csv", action="append", required=True, help="T=CSV_PATH; repeat once per temperature.")
    p.add_argument("--model-variant", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    temp_csv_pairs = [_parse_temp_csv(s) for s in args.temp_csv]
    temp_csv_pairs.sort(key=lambda x: x[0])

    log.info("Building reference set from configured and local DMS datasets ...")
    ref_set = frozenset(build_reference_index(REPO_ROOT, logger=log))
    log.info("Reference set: %d unique CDRH3 sequences total", len(ref_set))

    temperatures: list[float] = []
    n_found: list[int] = []
    n_novel: list[int] = []

    log.info("%-8s  %10s  %10s  %10s  %8s", "Temp", "Total", "In-dataset", "Novel", "% novel")
    for temp, csv_path in temp_csv_pairs:
        if not csv_path.exists():
            log.warning("T=%s: output CSV not found (%s) - skipping.", temp, csv_path)
            continue
        try:
            col = pd.read_csv(csv_path, usecols=["cdrh3"])["cdrh3"].dropna().astype(str)
        except Exception as exc:
            log.warning("T=%s: could not read %s: %s - skipping.", temp, csv_path, exc)
            continue

        seqs = frozenset(seq.strip() for seq in col.tolist() if seq.strip())
        found = len(seqs & ref_set)
        novel = len(seqs - ref_set)
        total = found + novel

        log.info(
            "%-8s  %10d  %10d  %10d  %7.1f%%",
            temp,
            total,
            found,
            novel,
            100.0 * novel / total if total else 0.0,
        )

        temperatures.append(temp)
        n_found.append(found)
        n_novel.append(novel)

    if not temperatures:
        log.error("No temperature points with valid data - cannot produce plot.")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "novelty_by_temperature.png"
    plot_novelty(temperatures, n_found, n_novel, args.model_variant, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
