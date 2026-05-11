"""Temperature-sweep summary plots for PLL, enrichment, and WT success rate.

This script is meant to be called from the temperature-sweep orchestrator. It
reuses the existing beam CSV format and the model scoring helpers from
gibbs_diagnostics.py to produce three publication-style summary figures:

  1. fraction above WT PLL across temperatures
  2. PLL vs M22 enrichment scatter, one panel per temperature
  3. top-k PLL recovery: mean DMS enrichment of the top-k PLL sequences

All plots are based on the final-step beam sequences for each temperature.
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer

from gibbs_diagnostics import (
    ESM2_MODEL_ID,
    load_esm_for_mlm,
    load_gibbs_csv,
    per_position_cdr_log_probs,
    sequence_pll,
)
from protein_design.constants import C05_CDRH3, WT_M22_BINDING_ENRICHMENT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("plot_temp_pll_fitness_summary")

SEED = 42
FITNESS_COL = "M22_binding_enrichment_adj"


@dataclass
class TemperatureResult:
    temperature: float
    beam_total: int
    beam_matched: int
    above_wt_fraction: float
    matched_pearson_r: float
    matched_spearman_rho: float
    matched_df: pd.DataFrame
    topk_total: int
    topk_matched: int
    topk_mean_enrichment: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--temp-csv",
        action="append",
        required=True,
        help="T=CSV_PATH; repeat once per temperature.",
    )
    p.add_argument("--checkpoint-path", default="")
    p.add_argument("--model-variant", required=True)
    p.add_argument("--dms-m22", type=Path, default=None)
    p.add_argument("--dms-si06", type=Path, default=None)
    p.add_argument("--dms-m22-col", default=FITNESS_COL)
    p.add_argument("--dms-si06-col", default="SI06_binding_enrichment_adj")
    p.add_argument("--max-dms", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--fraction-above-wt", action="store_true")
    p.add_argument("--pll-vs-enrichment-scatter", action="store_true")
    p.add_argument("--top-k-enrichment-recovery", action="store_true")
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def _parse_temp_csv(spec: str) -> tuple[float, Path]:
    parts = spec.split("=", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"--temp-csv must be T=CSV_PATH, got {spec!r}")
    return float(parts[0]), Path(parts[1])


def _temp_label(temp: float) -> str:
    return str(temp)


def _load_final_step(csv_path: Path) -> list[str]:
    df = load_gibbs_csv(csv_path)
    if df.empty:
        return []
    final_step = int(df["gibbs_step"].max())
    return df.loc[df["gibbs_step"] == final_step, "cdrh3"].astype(str).tolist()


def _safe_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 2 or len(y) < 2:
        return float("nan"), float("nan")
    if np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
        return float("nan"), float("nan")
    try:
        rp, _ = pearsonr(x, y)
    except Exception:
        rp = float("nan")
    try:
        rho, _ = spearmanr(x, y)
    except Exception:
        rho = float("nan")
    return float(rp), float(rho)


def _load_dms_reference(
    *,
    m22_path: Path | None,
    si06_path: Path | None,
    m22_col: str,
    si06_col: str,
    max_dms: int,
) -> pd.DataFrame:
    if m22_path is None:
        raise ValueError("--dms-m22 is required for the enrichment plots")

    frames: list[pd.DataFrame] = []
    m22_df = pd.read_csv(m22_path)
    missing = {"aa", m22_col}.difference(m22_df.columns)
    if missing:
        raise ValueError(f"M22 dataset at {m22_path} missing required columns: {sorted(missing)}")
    frames.append(m22_df[["aa", m22_col]].rename(columns={m22_col: FITNESS_COL}))

    if si06_path is not None:
        si06_df = pd.read_csv(si06_path)
        missing = {"aa", si06_col}.difference(si06_df.columns)
        if missing:
            raise ValueError(f"SI06 dataset at {si06_path} missing required columns: {sorted(missing)}")
        frames.append(si06_df[["aa", si06_col]].rename(columns={si06_col: "SI06_binding_enrichment_adj"}))

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="aa", how="outer")

    merged = merged[merged["aa"].astype(str).str.len() == len(C05_CDRH3)].copy()
    merged["aa"] = merged["aa"].astype(str).str.strip()
    merged = merged[merged["aa"] != ""].drop_duplicates(subset=["aa"], keep="first").reset_index(drop=True)

    if len(merged) > max_dms:
        merged = merged.sample(n=max_dms, random_state=SEED).reset_index(drop=True)
    return merged


def _score_sequences(
    model,
    tokenizer,
    sequences: list[str],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if not sequences:
        return np.zeros(0, dtype=np.float32)
    per_pos = per_position_cdr_log_probs(model, tokenizer, sequences, device, batch_size)
    return sequence_pll(per_pos)


def _build_temperature_result(
    *,
    temperature: float,
    csv_path: Path,
    model,
    tokenizer,
    device: torch.device,
    batch_size: int,
    dms_lookup: pd.DataFrame,
    wt_pll: float,
    top_k: int,
) -> TemperatureResult:
    beam_seqs = _load_final_step(csv_path)
    beam_total = len(beam_seqs)
    beam_pll = _score_sequences(model, tokenizer, beam_seqs, device, batch_size)

    beam_df = pd.DataFrame({"cdrh3": beam_seqs, "pll": beam_pll})
    beam_df["above_wt"] = beam_df["pll"] > wt_pll
    above_wt_fraction = float(beam_df["above_wt"].mean()) if not beam_df.empty else float("nan")

    matched_df = beam_df.merge(dms_lookup, on="cdrh3", how="inner")
    matched_x = matched_df["pll"].to_numpy(dtype=np.float32)
    matched_y = matched_df[FITNESS_COL].to_numpy(dtype=np.float32)
    matched_pearson_r, matched_spearman_rho = _safe_corr(matched_x, matched_y)

    topk_total = min(top_k, beam_total)
    if beam_df.empty:
        topk_matched_df = beam_df
    else:
        topk_df = beam_df.sort_values("pll", ascending=False).head(topk_total)
        topk_matched_df = topk_df.merge(dms_lookup, on="cdrh3", how="inner")
    topk_mean_enrichment = (
        float(topk_matched_df[FITNESS_COL].mean()) if not topk_matched_df.empty else float("nan")
    )

    return TemperatureResult(
        temperature=temperature,
        beam_total=beam_total,
        beam_matched=len(matched_df),
        above_wt_fraction=above_wt_fraction,
        matched_pearson_r=matched_pearson_r,
        matched_spearman_rho=matched_spearman_rho,
        matched_df=matched_df,
        topk_total=topk_total,
        topk_matched=len(topk_matched_df),
        topk_mean_enrichment=topk_mean_enrichment,
    )


def _plot_fraction_above_wt(results: list[TemperatureResult], model_variant: str, out_path: Path) -> None:
    temps = np.array([r.temperature for r in results], dtype=np.float32)
    values = np.array([r.above_wt_fraction for r in results], dtype=np.float32) * 100.0

    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    cmap = plt.get_cmap("plasma")
    colors = [cmap(i / max(1, len(results) - 1)) for i in range(len(results))]

    if len(temps) > 1:
        sorted_temps = np.unique(np.sort(temps))
        spacing = float(np.min(np.diff(sorted_temps))) if len(sorted_temps) > 1 else 0.1
        width = max(0.03, min(0.08, spacing * 0.72))
    else:
        width = 0.06

    ax.bar(temps, values, width=width, color=colors, alpha=0.86, edgecolor="black", linewidth=0.35)
    ax.plot(temps, values, color="black", linewidth=1.2, marker="o", markersize=4.5, zorder=3)
    ax.axhline(50.0, color="grey", linestyle="--", linewidth=1.0, alpha=0.8)

    for x, y in zip(temps, values):
        ax.text(x, y + 1.6, f"{y:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Temperature")
    ax.set_ylabel("Final-step sequences above WT PLL (%)")
    ax.set_title(f"Fraction of final-step beam sequences above WT PLL\n(model: {model_variant})")
    ax.grid(axis="y", alpha=0.18)
    ax.set_xticks(temps)
    ax.set_xlim(float(temps.min()) - 0.06, float(temps.max()) + 0.06)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def _plot_pll_vs_enrichment_grid(
    results: list[TemperatureResult],
    dms_df: pd.DataFrame,
    wt_pll: float,
    model_variant: str,
    out_path: Path,
) -> None:
    n_panels = len(results)
    n_cols = 3 if n_panels > 3 else n_panels
    n_rows = int(math.ceil(n_panels / n_cols))

    bg = dms_df[["pll", FITNESS_COL]].dropna().copy()
    if bg.empty:
        raise ValueError("No DMS background points available for the enrichment scatter")

    bg_x = bg["pll"].to_numpy(dtype=np.float32)
    bg_y = bg[FITNESS_COL].to_numpy(dtype=np.float32)
    if len(bg_x) >= 2 and not np.allclose(np.std(bg_x), 0.0):
        slope, intercept = np.polyfit(bg_x, bg_y, 1)
    else:
        slope, intercept = None, None

    beam_x_all = [r.matched_df["pll"].to_numpy(dtype=np.float32) for r in results if not r.matched_df.empty]
    x_candidates = [bg_x, np.array([wt_pll], dtype=np.float32)] + beam_x_all
    x_min = float(min(arr.min() for arr in x_candidates if arr.size))
    x_max = float(max(arr.max() for arr in x_candidates if arr.size))
    y_candidates = [bg_y, np.array([WT_M22_BINDING_ENRICHMENT], dtype=np.float32)]
    for result in results:
        if not result.matched_df.empty:
            y_candidates.append(result.matched_df[FITNESS_COL].to_numpy(dtype=np.float32))
    y_min = float(min(arr.min() for arr in y_candidates if arr.size))
    y_max = float(max(arr.max() for arr in y_candidates if arr.size))
    x_pad = 0.04 * (x_max - x_min if x_max > x_min else 1.0)
    y_pad = 0.04 * (y_max - y_min if y_max > y_min else 1.0)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 4.2 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    cmap = plt.get_cmap("plasma")
    colors = [cmap(i / max(1, n_panels - 1)) for i in range(n_panels)]

    for idx, result in enumerate(results):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]

        ax.scatter(bg_x, bg_y, s=9, color="lightgrey", alpha=0.42, edgecolors="none", zorder=1)
        if slope is not None and intercept is not None:
            x_line = np.linspace(x_min, x_max, 200)
            ax.plot(x_line, slope * x_line + intercept, color="black", linewidth=1.0, zorder=2)

        matched = result.matched_df
        color = colors[idx]
        if not matched.empty:
            ax.scatter(
                matched["pll"],
                matched[FITNESS_COL],
                s=34,
                color=color,
                edgecolors="black",
                linewidths=0.35,
                alpha=0.88,
                zorder=3,
            )
            if len(matched) >= 2 and not np.allclose(np.std(matched["pll"].to_numpy(dtype=np.float32)), 0.0):
                x = matched["pll"].to_numpy(dtype=np.float32)
                y = matched[FITNESS_COL].to_numpy(dtype=np.float32)
                xr = np.linspace(float(x.min()), float(x.max()), 100)
                ax.plot(xr, np.polyval(np.polyfit(x, y, 1), xr), color=color, linewidth=1.2, zorder=3.5)

            panel_text = (
                f"matched {result.beam_matched}/{result.beam_total}\n"
                f"rho={result.matched_spearman_rho:.2f}, r={result.matched_pearson_r:.2f}"
            )
        else:
            panel_text = f"matched 0/{result.beam_total}\nno exact DMS matches"
            ax.text(
                0.5,
                0.52,
                "no exact DMS matches",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="#666666",
            )

        ax.text(
            0.03,
            0.97,
            panel_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", alpha=0.88, edgecolor="#bbbbbb"),
        )

        ax.scatter(
            [wt_pll],
            [WT_M22_BINDING_ENRICHMENT],
            s=180,
            marker="*",
            color="#d62728",
            edgecolors="black",
            linewidths=0.8,
            zorder=4,
        )
        ax.annotate(
            "WT",
            (wt_pll, WT_M22_BINDING_ENRICHMENT),
            textcoords="offset points",
            xytext=(7, 6),
            fontsize=8.5,
            fontweight="bold",
            color="#8b0000",
        )

        ax.set_title(f"T={_temp_label(result.temperature)}", fontsize=10.5)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.grid(alpha=0.14)

    for idx in range(n_panels, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row][col].axis("off")

    fig.suptitle(f"PLL vs M22 enrichment across temperatures\n(model: {model_variant})", fontsize=12)
    fig.supxlabel("CDR-H3 PLL")
    fig.supylabel("M22 binding enrichment")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def _plot_topk_recovery(results: list[TemperatureResult], model_variant: str, out_path: Path) -> None:
    temps = np.array([r.temperature for r in results], dtype=np.float32)
    values = np.array([r.topk_mean_enrichment for r in results], dtype=np.float32)
    matched_counts = np.array([r.topk_matched for r in results], dtype=np.int32)
    topk_total = results[0].topk_total if results else 0

    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    cmap = plt.get_cmap("plasma")
    colors = [cmap(i / max(1, len(results) - 1)) for i in range(len(results))]

    finite = np.isfinite(values)
    if finite.any():
        ax.plot(temps[finite], values[finite], color="black", linewidth=1.2, marker="o", markersize=4.5, zorder=2)

    for temp, value, color, matched in zip(temps, values, colors, matched_counts):
        if np.isfinite(value):
            ax.scatter(temp, value, s=72, color=color, edgecolors="black", linewidths=0.35, zorder=3)
            ax.annotate(
                f"{matched}/{topk_total}",
                (temp, value),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8.5,
            )
        else:
            ax.scatter(temp, WT_M22_BINDING_ENRICHMENT, s=60, marker="x", color=color, linewidths=1.4, zorder=3)
            ax.annotate(
                f"0/{topk_total}",
                (temp, WT_M22_BINDING_ENRICHMENT),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8.5,
            )

    ax.axhline(WT_M22_BINDING_ENRICHMENT, color="#d62728", linestyle="--", linewidth=1.1, label="WT M22 enrichment")
    ax.set_xlabel("Temperature")
    ax.set_ylabel(f"Mean M22 enrichment of top-{topk_total} PLL sequences")
    ax.set_title(f"Top-k enrichment recovery across temperatures\n(model: {model_variant})")
    ax.grid(axis="y", alpha=0.18)
    ax.set_xticks(temps)
    ax.legend(loc="best", fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    temp_csv_pairs = [_parse_temp_csv(spec) for spec in args.temp_csv]
    temp_csv_pairs.sort(key=lambda item: item[0])
    if not temp_csv_pairs:
        log.error("No temperature inputs provided.")
        return 2

    if not (args.fraction_above_wt or args.pll_vs_enrichment_scatter or args.top_k_enrichment_recovery):
        log.error("No outputs selected. Enable at least one summary plot flag.")
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    log.info("Loading model %s ...", args.model_variant)
    model = load_esm_for_mlm(args.checkpoint_path).eval().to(device)
    for param in model.parameters():
        param.requires_grad = False
    if device.type == "cuda":
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_ID)

    dms_df = _load_dms_reference(
        m22_path=args.dms_m22,
        si06_path=args.dms_si06,
        m22_col=args.dms_m22_col,
        si06_col=args.dms_si06_col,
        max_dms=args.max_dms,
    )
    log.info("Loaded DMS reference: %d rows", len(dms_df))

    dms_pll = _score_sequences(model, tokenizer, dms_df["aa"].astype(str).tolist(), device, args.batch_size)
    dms_bg = dms_df.copy()
    dms_bg["pll"] = dms_pll
    dms_lookup = dms_bg[["aa", FITNESS_COL]].rename(columns={"aa": "cdrh3"})

    wt_pll = float(_score_sequences(model, tokenizer, [C05_CDRH3], device, args.batch_size)[0])

    results: list[TemperatureResult] = []
    for temperature, csv_path in temp_csv_pairs:
        log.info("Scoring temperature T=%s from %s", temperature, csv_path)
        results.append(
            _build_temperature_result(
                temperature=temperature,
                csv_path=csv_path,
                model=model,
                tokenizer=tokenizer,
                device=device,
                batch_size=args.batch_size,
                dms_lookup=dms_lookup,
                wt_pll=wt_pll,
                top_k=args.top_k,
            )
        )

    if args.fraction_above_wt:
        _plot_fraction_above_wt(results, args.model_variant, args.output_dir / "temp_fraction_above_wt.png")

    if args.pll_vs_enrichment_scatter:
        _plot_pll_vs_enrichment_grid(results, dms_bg, wt_pll, args.model_variant, args.output_dir / "temp_pll_vs_enrichment.png")

    if args.top_k_enrichment_recovery:
        _plot_topk_recovery(results, args.model_variant, args.output_dir / "temp_topk_enrichment_recovery.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())