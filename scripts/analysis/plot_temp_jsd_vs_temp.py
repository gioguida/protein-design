"""JSD (SBS vs DMS) across temperatures using per-position AA frequencies."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from protein_design.constants import C05_CDRH3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("plot_temp_jsd_vs_temp")

AA20 = list("ACDEFGHIKLMNPQRSTVWY")
AA_IDX = {aa: i for i, aa in enumerate(AA20)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--temp-csv", action="append", required=True, help="T=CSV_PATH")
    p.add_argument("--model-variant", required=True)
    p.add_argument("--dms-m22", type=Path, required=True)
    p.add_argument("--max-dms", type=int, default=500)
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


def _load_dms(path: Path, max_dms: int) -> list[str]:
    df = pd.read_csv(path)
    if "aa" not in df.columns:
        raise ValueError(f"DMS file missing 'aa' column: {path}")
    seqs = df["aa"].astype(str)
    seqs = seqs[seqs.str.len() == len(C05_CDRH3)]
    seqs = seqs.drop_duplicates()
    if len(seqs) > max_dms:
        seqs = seqs.sample(n=max_dms, random_state=42)
    return seqs.tolist()


def _pos_freqs(seqs: list[str]) -> np.ndarray:
    L = len(C05_CDRH3)
    out = np.zeros((L, len(AA20)), dtype=np.float64)
    if not seqs:
        return out
    for i in range(L):
        for s in seqs:
            aa = s[i]
            if aa in AA_IDX:
                out[i, AA_IDX[aa]] += 1.0
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return out / row_sums


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dms_seqs = _load_dms(args.dms_m22, args.max_dms)
    if not dms_seqs:
        raise ValueError("No DMS sequences loaded.")
    dms_freqs = _pos_freqs(dms_seqs)

    pairs = sorted((_parse_temp_csv(s) for s in args.temp_csv), key=lambda x: x[0])
    temps: list[float] = []
    mean_jsd: list[float] = []
    per_pos_all: list[np.ndarray] = []

    for temp, csv_path in pairs:
        sbs_seqs = _load_final_step(csv_path)
        if not sbs_seqs:
            log.warning("No final-step sequences for T=%s (%s); skipping.", temp, csv_path)
            continue
        sbs_freqs = _pos_freqs(sbs_seqs)
        pos_jsd = np.array([_jsd(sbs_freqs[i], dms_freqs[i]) for i in range(len(C05_CDRH3))], dtype=np.float64)
        temps.append(temp)
        per_pos_all.append(pos_jsd)
        mean_jsd.append(float(pos_jsd.mean()))

    if not temps:
        raise ValueError("No valid temperature points for JSD plot.")

    mat = np.vstack(per_pos_all)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.2, 1.8]})

    ax0.plot(temps, mean_jsd, marker="o", color="black", linewidth=1.4)
    ax0.set_xlabel("Temperature")
    ax0.set_ylabel("Mean JSD across CDR-H3 positions")
    ax0.set_title("Mean JSD vs temperature")
    ax0.grid(alpha=0.2)

    pos_colors = plt.get_cmap("turbo")(np.linspace(0.0, 1.0, len(C05_CDRH3)))
    for i in range(len(C05_CDRH3)):
        ax1.plot(temps, mat[:, i], linewidth=1.0, alpha=0.9, color=pos_colors[i])
    ax1.set_xlabel("Temperature")
    ax1.set_ylabel("Per-position JSD")
    ax1.set_title("Per-position JSD trajectories (24 positions)")
    ax1.grid(alpha=0.2)

    fig.suptitle(f"SBS vs DMS divergence across temperatures (model: {args.model_variant})")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])

    out_path = args.output_dir / "temp_jsd_vs_temp.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

