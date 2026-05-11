"""Position-wise CDR-H3 entropy heatmap across sweep temperatures."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from protein_design.constants import C05_CDRH3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("plot_temp_entropy_heatmap")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--temp-csv", action="append", required=True, help="T=CSV_PATH")
    p.add_argument("--model-variant", required=True)
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


def _position_entropy(seqs: list[str]) -> np.ndarray:
    L = len(C05_CDRH3)
    ent = np.zeros(L, dtype=np.float32)
    if not seqs:
        return ent
    n = len(seqs)
    for i in range(L):
        counts: dict[str, int] = {}
        for s in seqs:
            aa = s[i]
            counts[aa] = counts.get(aa, 0) + 1
        probs = np.array([c / n for c in counts.values()], dtype=np.float32)
        probs = probs[probs > 0]
        ent[i] = float(-(probs * np.log2(probs)).sum())
    return ent


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = sorted((_parse_temp_csv(s) for s in args.temp_csv), key=lambda x: x[0])
    temps: list[float] = []
    rows: list[np.ndarray] = []
    for temp, csv_path in pairs:
        seqs = _load_final_step(csv_path)
        if not seqs:
            log.warning("No final-step sequences for T=%s (%s); skipping.", temp, csv_path)
            continue
        temps.append(temp)
        rows.append(_position_entropy(seqs))

    if not rows:
        raise ValueError("No valid temperature data for entropy heatmap.")

    mat = np.vstack(rows)
    L = len(C05_CDRH3)
    positions = np.arange(1, L + 1)

    fig, ax = plt.subplots(figsize=(11.5, 4.5 + 0.2 * len(temps)))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xticks(np.arange(L))
    ax.set_xticklabels([f"{i}\n{aa}" for i, aa in zip(positions, C05_CDRH3)], fontsize=8)
    ax.set_yticks(np.arange(len(temps)))
    ax.set_yticklabels([str(t) for t in temps])
    ax.set_xlabel("CDR-H3 position (WT residue)")
    ax.set_ylabel("Temperature")
    ax.set_title(f"Position-wise CDR-H3 entropy across temperatures (model: {args.model_variant})")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Shannon entropy (bits)")
    fig.tight_layout()

    out_path = args.output_dir / "temp_entropy_heatmap.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

