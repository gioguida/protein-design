"""
Produce the four DMS correlation plots for one (model, dataset) pair:
    1. pll_vs_scorer.pdf         — colored by experimental log-enrichment ("truth")
    2. pll_vs_truth.pdf          — single color
    3. scorer_vs_truth.pdf       — single color
    4. pll_scorer_truth_3d.pdf   — 3D scatter colored by truth

"truth" = the experimentally-measured log-enrichment column for the dataset
(M22/SI06 binding-enrichment, or expression enrichment for the exp split).

Each scatter shows OLS fit + Pearson r, Spearman rho, and n.

Inputs are read from the caches written by compute_pll.py and
score_dms_with_esme.py. The dataset registry conf/analysis/dms_datasets.yaml
maps dataset → CSV path, sequence column, ground-truth column, and scorer.

Usage:
    uv run python scripts/analysis/plot_dms_correlations.py \
        --model-name evodpo_4ep_step1376 \
        --dataset ed2_m22

Pass --dataset all to plot every dataset whose PLL cache exists for the model.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from protein_design.wandb_plots import set_publication_style  # noqa: E402

CONFIG_PATH = REPO_ROOT / "conf" / "analysis" / "dms_datasets.yaml"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _stats(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    x_c, y_c = x[mask], y[mask]
    r, r_p = pearsonr(x_c, y_c)
    rho, rho_p = spearmanr(x_c, y_c)
    return r, r_p, rho, rho_p, int(mask.sum())


def _stats_text(x: np.ndarray, y: np.ndarray) -> str:
    r, r_p, rho, rho_p, n = _stats(x, y)
    return (rf"Pearson $r$ = {r:.3f}  ($p$ = {r_p:.1e})"
            "\n"
            rf"Spearman $\rho$ = {rho:.3f}  ($p$ = {rho_p:.1e})"
            "\n"
            rf"$n$ = {n}")


def _plot_2d(
    x: np.ndarray, y: np.ndarray, *,
    out_path: Path, x_label: str, y_label: str, title: str,
    color_values: np.ndarray | None = None, color_label: str | None = None,
    cmap: str = "coolwarm",
) -> None:
    mask = np.isfinite(x) & np.isfinite(y)
    if color_values is not None:
        mask &= np.isfinite(color_values)
    x_c, y_c = x[mask], y[mask]

    fig, ax = plt.subplots(figsize=(5.6, 4.5), constrained_layout=True)
    if color_values is not None:
        c = color_values[mask]
        if cmap in {"coolwarm", "RdBu_r", "RdBu", "bwr", "seismic"}:
            vmax = float(np.nanpercentile(np.abs(c), 99))
            vmin = -vmax
        else:
            vmin = float(np.nanpercentile(c, 1))
            vmax = float(np.nanpercentile(c, 99))
        sc = ax.scatter(x_c, y_c, c=c, s=8, alpha=0.6, cmap=cmap,
                        vmin=vmin, vmax=vmax, linewidths=0)
        fig.colorbar(sc, ax=ax).set_label(color_label or "color")
    else:
        ax.scatter(x_c, y_c, s=8, alpha=0.35, color="#648FFF", linewidths=0)

    coeffs = np.polyfit(x_c, y_c, 1)
    x_line = np.linspace(x_c.min(), x_c.max(), 200)
    ax.plot(x_line, np.polyval(coeffs, x_line),
            color="#DC267F", linewidth=1.5, label="OLS fit")

    ax.text(0.03, 0.97, _stats_text(x_c, y_c), ha="left", va="top",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.8, edgecolor="none"))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(fontsize=9, loc="lower right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def _plot_3d(
    pll: np.ndarray, scorer: np.ndarray, truth: np.ndarray, *,
    out_path: Path, title: str,
    pll_label: str, scorer_label: str, truth_label: str,
    cmap: str = "coolwarm",
) -> None:
    mask = np.isfinite(pll) & np.isfinite(scorer) & np.isfinite(truth)
    p, s, t = pll[mask], scorer[mask], truth[mask]

    fig = plt.figure(figsize=(7.0, 5.6), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    vmax = float(np.nanpercentile(np.abs(t), 99))
    sc = ax.scatter(p, s, t, c=t, cmap=cmap, vmin=-vmax, vmax=vmax,
                    s=6, alpha=0.5, linewidths=0)
    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1).set_label(truth_label)

    ax.set_xlabel(pll_label)
    ax.set_ylabel(scorer_label)
    ax.set_zlabel(truth_label)
    ax.set_title(title)

    r_ps, _, rho_ps, _, n = _stats(p, s)
    r_pt, _, rho_pt, _, _ = _stats(p, t)
    r_st, _, rho_st, _, _ = _stats(s, t)
    txt = (f"PLL vs scorer:  r={r_ps:.3f}  ρ={rho_ps:.3f}\n"
           f"PLL vs truth:   r={r_pt:.3f}  ρ={rho_pt:.3f}\n"
           f"scorer vs truth: r={r_st:.3f}  ρ={rho_st:.3f}\n"
           f"n = {n}")
    ax.text2D(0.02, 0.98, txt, ha="left", va="top",
              transform=ax.transAxes, fontsize=8, family="monospace",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                        alpha=0.8, edgecolor="none"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def _load_joined(cfg: dict, model_name: str, dataset_key: str) -> tuple[pd.DataFrame, dict]:
    """Return a DataFrame with columns [seq, pll, score?, enrichment] joined on seq."""
    ds = cfg["datasets"][dataset_key]
    seq_col = ds["seq_col"]
    cache_root = Path(cfg["paths"]["cache_root"])

    pll_csv = cache_root / "pll" / model_name / f"{dataset_key}.csv"
    if not pll_csv.exists():
        raise FileNotFoundError(
            f"PLL cache missing: {pll_csv}\n"
            f"Run: scripts/analysis/compute_pll.py --model-name {model_name} "
            f"--dataset {dataset_key} --checkpoint <ckpt>"
        )
    pll_df = pd.read_csv(pll_csv)

    enrich_df = pd.read_csv(ds["path"])
    enrich_df = enrich_df[[seq_col, ds["enrichment_col"]]].drop_duplicates(
        subset=[seq_col], keep="first")

    joined = pll_df.merge(enrich_df, on=seq_col, how="inner")
    joined.rename(columns={ds["enrichment_col"]: "enrichment"}, inplace=True)

    scorer_name = ds.get("scorer")
    if scorer_name:
        score_csv = cache_root / "scorer_preds" / dataset_key / f"{scorer_name}.csv"
        if not score_csv.exists():
            log.warning("Scorer cache missing for %s/%s — skipping scorer plots",
                        dataset_key, scorer_name)
            scorer_name = None
        else:
            score_df = pd.read_csv(score_csv).drop_duplicates(subset=[seq_col], keep="first")
            joined = joined.merge(score_df[[seq_col, "score"]], on=seq_col, how="inner")

    return joined, {"seq_col": seq_col, "scorer_name": scorer_name,
                    "enrichment_col": ds["enrichment_col"],
                    "split": ds.get("split", "val")}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-name", required=True,
                   help="Cache key used by compute_pll.py (e.g. evodpo_4ep_step1376).")
    p.add_argument("--dataset", required=True,
                   help="Dataset key from conf/analysis/dms_datasets.yaml, or 'all'.")
    return p.parse_args()


def make_plots(cfg: dict, model_name: str, dataset_key: str) -> None:
    plots_root = Path(cfg["paths"]["plots_root"]) / model_name / dataset_key
    df, meta = _load_joined(cfg, model_name, dataset_key)
    log.info("[%s] joined n=%d  cols=%s", dataset_key, len(df), list(df.columns))
    if len(df) == 0:
        log.warning("[%s] empty join — nothing to plot", dataset_key)
        return

    pll = df["pll"].to_numpy(np.float64)
    truth = df["enrichment"].to_numpy(np.float64)
    truth_label = "log-enrichment (experimental)"
    pll_label = "CDR-H3 PLL"
    split = meta["split"]
    tag = f"{model_name} / {dataset_key} ({split})"
    ds_tag = f"{dataset_key} ({split})"

    scorer_name = meta["scorer_name"]
    if scorer_name is not None:
        scorer = df["score"].to_numpy(np.float64)
        scorer_label = f"{scorer_name.upper()} scorer prediction"
        _plot_2d(
            pll, scorer,
            out_path=plots_root / "pll_vs_scorer.pdf",
            x_label=pll_label, y_label=scorer_label,
            title=f"PLL vs {scorer_name.upper()} scorer — {tag}",
            color_values=truth, color_label=truth_label, cmap="coolwarm",
        )
        _plot_2d(
            scorer, truth,
            out_path=plots_root / "scorer_vs_truth.pdf",
            x_label=scorer_label, y_label=truth_label,
            title=f"{scorer_name.upper()} scorer vs truth — {ds_tag}",
        )
        _plot_3d(
            pll, scorer, truth,
            out_path=plots_root / "pll_scorer_truth_3d.pdf",
            title=f"PLL × {scorer_name.upper()} scorer × truth — {tag}",
            pll_label=pll_label, scorer_label=scorer_label, truth_label=truth_label,
        )
    else:
        log.info("[%s] no scorer configured — skipping scorer plots", dataset_key)

    _plot_2d(
        pll, truth,
        out_path=plots_root / "pll_vs_truth.pdf",
        x_label=pll_label, y_label=truth_label,
        title=f"PLL vs truth — {tag}",
    )


def main() -> None:
    args = parse_args()
    set_publication_style()
    with CONFIG_PATH.open() as f:
        cfg = yaml.safe_load(f)

    if args.dataset == "all":
        datasets = list(cfg["datasets"].keys())
    else:
        if args.dataset not in cfg["datasets"]:
            raise SystemExit(f"Unknown dataset {args.dataset!r}. "
                             f"Known: {list(cfg['datasets'])}")
        datasets = [args.dataset]

    for key in datasets:
        try:
            make_plots(cfg, args.model_name, key)
        except FileNotFoundError as e:
            log.warning("[%s] %s", key, e)


if __name__ == "__main__":
    main()
