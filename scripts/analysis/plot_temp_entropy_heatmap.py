"""Position-wise CDR-H3 entropy heatmap across sweep temperatures."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from protein_design.analysis.entropy import position_entropy
from protein_design.analysis.report_figures import (
    DOUBLE_COL_WIDTH,
    apply_report_style,
    style_axes,
)
from protein_design.constants import C05_CDRH3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("plot_temp_entropy_heatmap")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--temp-csv", action="append", required=True, help="T=CSV_PATH")
    p.add_argument("--model-variant", required=True)
    p.add_argument("--sampler-label", default="sampler")
    p.add_argument("--sweep-label", default="temperature")
    p.add_argument("--output-name", default="temp_entropy_heatmap.png")
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def _parse_temp_csv(spec: str) -> tuple[float, Path]:
    t, p = spec.split("=", 1)
    return float(t), Path(p)


def _load_final_step(csv_path: Path) -> list[str]:
    df = pd.read_csv(csv_path)
    df = df[df["cdrh3"].astype(str).str.len() == len(C05_CDRH3)].copy()
    if df.empty:
        return []
    final_step = int(df["gibbs_step"].max())
    return df.loc[df["gibbs_step"] == final_step, "cdrh3"].astype(str).tolist()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    apply_report_style()

    pairs = sorted((_parse_temp_csv(s) for s in args.temp_csv), key=lambda x: x[0])
    temps: list[float] = []
    rows: list[np.ndarray] = []
    for temp, csv_path in pairs:
        seqs = _load_final_step(csv_path)
        if not seqs:
            log.warning("No final-step sequences for T=%s (%s); skipping.", temp, csv_path)
            continue
        temps.append(temp)
        rows.append(position_entropy(seqs, expected_length=len(C05_CDRH3)))

    if not rows:
        raise ValueError("No valid temperature data for entropy heatmap.")

    mat = np.vstack(rows)
    L = len(C05_CDRH3)
    positions = np.arange(1, L + 1)

    fig, ax = plt.subplots(
        figsize=(DOUBLE_COL_WIDTH, max(3.1, 2.1 + 0.2 * len(temps))),
        constrained_layout=True,
    )
    im = ax.imshow(mat, aspect="auto", cmap="viridis", interpolation="nearest")
    shown = np.arange(0, L, 2)
    ax.set_xticks(shown)
    ax.set_xticklabels(
        [f"{positions[i]}\n{C05_CDRH3[i]}" for i in shown], fontsize=8.5
    )
    ax.set_yticks(np.arange(len(temps)))
    ax.set_yticklabels([str(t) for t in temps])
    ax.set_xlabel("CDR-H3 position (WT residue)")
    ax.set_ylabel(args.sweep_label.title())
    ax.set_title(f"CDR-H3 entropy across {args.sweep_label.lower()}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Shannon entropy (bits)")
    style_axes(ax, grid=False)

    out_path = args.output_dir / args.output_name
    fig.savefig(out_path, dpi=400, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    log.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
