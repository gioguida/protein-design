#!/usr/bin/env python
"""Generate comparison plots across evotuning / C05 / TTT runs.

Produces:
    spearman_bar.png              Grouped bar chart of Spearman rho per
                                  {model} x {dataset}, with p-value significance.
    test_perplexity_comparison.png   Headline: final test-set MLM perplexity per model.
    val_perplexity_comparison.png    Diagnostic: final val-set MLM perplexity per model.
    cdr_perplexity_comparison.png    CDR-H3 pseudo-perplexity on the C05 reference VH.
    train_val_loss_curves.png     One subplot per run: train loss vs. val loss over steps.
    metrics_summary.csv           One row per (model, dataset, strategy): rho, pval, ppls.

For each run dir passed in, the script:
  1. Reads metrics.json if present (contains final_val_perplexity, final_test_perplexity).
  2. If the run lacks scoring or final perplexities (or --force-rescore/--force-reppl),
     re-scores best.pt (or final.pt) on the D2 datasets and recomputes val/test ppl
     using the run's hash-based split config.

Usage (repeat run/label pairs via Hydra list overrides):
    python scripts/analysis/plot_results.py scoring=d2 data=oas_full \\
        '+runs=[/path/to/run1,/path/to/run2]' \\
        '+labels=[evotuned,+C05]' \\
        +out_dir=\${HOME}/protein-design/plots/meeting
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Sequence

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from protein_design.eval import (
    compute_cdr_pseudo_perplexity,
    compute_perplexity,
    load_scoring_datasets,
    run_multi_scoring_evaluation,
)
from protein_design.evotuning.data import make_dataloaders
from protein_design.evotuning.splits import SplitConfig
from protein_design.model import ESM2Model
from protein_design.config import build_model_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_LABEL = "base ESM2"
STRATEGY = "avg"  # headline strategy for the Spearman bar plot
# Matches `spearman_{strategy}[_{side}{k}]_{dataset}` with optional flank slice.
_SCORING_KEY_RE = re.compile(
    r"^spearman_(?P<strategy>[a-zA-Z]+)(?:_(?P<side>left|right)(?P<k>\d+))?_(?P<dataset>[^_]+)$"
)


def _split_cfg_from(cfg: DictConfig) -> SplitConfig:
    """Read the hash-split policy from a Hydra config, with safe defaults."""
    split_node = cfg.data.get("split") if "split" in cfg.data else None
    if split_node is None:
        return SplitConfig()
    return SplitConfig(
        salt=str(split_node.get("salt", "oas-v1")),
        train_pct=int(split_node.get("train_pct", 90)),
        val_pct=int(split_node.get("val_pct", 5)),
        test_pct=int(split_node.get("test_pct", 5)),
    )


def _batch_size_from(cfg: DictConfig) -> int:
    if "training" in cfg:
        return int(cfg.training.batch_size)
    return int(cfg.scoring.batch_size)


def _make_eval_loaders(fasta_path: str, cfg: DictConfig):
    """Build (val_loader, test_loader) using the run's exact split policy.

    Uses make_dataloaders so the split identity is bit-for-bit identical to
    what training used — no more reconstruction drift.
    """
    split_cfg = _split_cfg_from(cfg)
    _train, val_loader, test_loader = make_dataloaders(
        fasta_path=fasta_path,
        tokenizer_name=cfg.model.name,
        max_seq_len=int(cfg.data.max_seq_len),
        mlm_probability=float(cfg.data.mlm_probability),
        batch_size=_batch_size_from(cfg),
        split_cfg=split_cfg,
    )
    return val_loader, test_loader


def _find_checkpoint(run_dir: Path) -> Path:
    """Prefer best.pt; fall back to final.pt (in root or checkpoints/)."""
    for candidate in [
        run_dir / "best.pt",
        run_dir / "final.pt",
        run_dir / "checkpoints" / "final.pt",
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No best.pt or final.pt in {run_dir}")


def _load_run_config(run_dir: Path) -> DictConfig:
    """Load the run's resolved config.yaml snapshot as a DictConfig."""
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config.yaml in {run_dir}")
    return OmegaConf.load(cfg_path)


def _score_model(model, tokenizer, datasets, device, batch_size, seed,
                 flank_ks: Sequence[int] = ()) -> dict:
    return run_multi_scoring_evaluation(
        model, tokenizer, datasets, device=device, batch_size=batch_size, seed=seed,
        flank_ks=flank_ks,
    )


def _extract_final_scoring(metrics: dict) -> dict | None:
    history = metrics.get("scoring_history", [])
    return history[-1] if history else None


def evaluate_run(run_dir: Path, datasets, tokenizer, device, scoring_batch_size,
                 seed, force_rescore: bool, force_reppl: bool,
                 fallback_config: DictConfig | None = None,
                 flank_ks: Sequence[int] = ()):
    """Return (scoring, val_ppl, test_ppl, cdr_ppl, training_history) for this run."""
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    try:
        cfg = _load_run_config(run_dir)
    except FileNotFoundError:
        if fallback_config is None:
            raise
        logger.warning("No config.yaml in %s; using global config as fallback", run_dir)
        cfg = fallback_config
    ckpt_path = _find_checkpoint(run_dir)
    logger.info("Run %s — checkpoint: %s", run_dir.name, ckpt_path)

    scoring = _extract_final_scoring(metrics) if not force_rescore else None
    val_ppl = metrics.get("final_val_perplexity") if not force_reppl else None
    test_ppl = metrics.get("final_test_perplexity") if not force_reppl else None
    training_history = metrics.get("training_history", [])

    model = ESM2Model(build_model_config(cfg, device=str(device)))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    if scoring is None:
        logger.info("  re-scoring on D2 datasets")
        scoring = _score_model(model, tokenizer, datasets, device,
                               scoring_batch_size, seed, flank_ks=flank_ks)

    if val_ppl is None or test_ppl is None:
        fasta = cfg.data.fasta_path if "data" in cfg else None
        if fasta and Path(fasta).exists():
            logger.info("  recomputing val/test perplexity on %s", fasta)
            val_loader, test_loader = _make_eval_loaders(fasta, cfg)
            if val_ppl is None and len(val_loader.dataset) > 0:
                val_ppl, _ = compute_perplexity(
                    model, val_loader, device, max_batches=max(len(val_loader), 1),
                )
            if test_ppl is None and len(test_loader.dataset) > 0:
                test_ppl, _ = compute_perplexity(
                    model, test_loader, device, max_batches=max(len(test_loader), 1),
                )
        else:
            logger.warning("  fasta path missing, skipping perplexity recomputation")

    cdr_ppl = metrics.get("final_cdr_pseudo_perplexity")
    if cdr_ppl is None:
        cdr_ppl = compute_cdr_pseudo_perplexity(model, tokenizer, device)

    del model
    torch.cuda.empty_cache()

    return scoring, val_ppl, test_ppl, cdr_ppl, training_history


def evaluate_base(scoring_cfg: DictConfig, datasets, tokenizer, device, seed,
                  flank_ks: Sequence[int] = ()):
    """Score base ESM2 and compute val/test perplexity on the OAS split."""
    logger.info("Evaluating base ESM2")
    model = ESM2Model(build_model_config(scoring_cfg, device=str(device)))
    model.to(device).eval()
    batch_size = int(scoring_cfg.scoring.batch_size)
    scoring = _score_model(model, tokenizer, datasets, device, batch_size, seed,
                           flank_ks=flank_ks)

    val_ppl, test_ppl = None, None
    fasta = scoring_cfg.data.fasta_path if "data" in scoring_cfg else None
    if fasta and Path(fasta).exists():
        val_loader, test_loader = _make_eval_loaders(fasta, scoring_cfg)
        if len(val_loader.dataset) > 0:
            val_ppl, _ = compute_perplexity(
                model, val_loader, device, max_batches=max(len(val_loader), 1),
            )
        if len(test_loader.dataset) > 0:
            test_ppl, _ = compute_perplexity(
                model, test_loader, device, max_batches=max(len(test_loader), 1),
            )

    cdr_ppl = compute_cdr_pseudo_perplexity(model, tokenizer, device)

    del model
    torch.cuda.empty_cache()
    return scoring, val_ppl, test_ppl, cdr_ppl


def plot_spearman_bar(rows_df: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart: x = dataset, groups = model. Annotate p-value significance."""
    datasets = sorted(rows_df["dataset"].unique())
    models = list(rows_df["model"].drop_duplicates())
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(max(8, 2 * len(datasets) * n_models / 3), 5))
    width = 0.8 / max(n_models, 1)
    x = np.arange(len(datasets))

    for i, model in enumerate(models):
        sub = rows_df[rows_df["model"] == model].set_index("dataset").reindex(datasets)
        rhos = sub["rho"].values
        pvals = sub["pval"].values
        positions = x + (i - (n_models - 1) / 2) * width
        bars = ax.bar(positions, rhos, width=width, label=model)
        for bar, rho, p in zip(bars, rhos, pvals):
            if np.isnan(rho):
                continue
            stars = ("***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns")
            y = rho + (0.01 if rho >= 0 else -0.03)
            ax.text(bar.get_x() + bar.get_width() / 2, y, stars,
                    ha="center", va="bottom" if rho >= 0 else "top", fontsize=8)

    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel(f"Spearman ρ  (strategy: {STRATEGY})")
    ax.set_title("Mutational-path ranking correlation")
    ax.legend(loc="best", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def plot_perplexity_bar(
    ppl_series: dict,
    out_path: Path,
    ylabel: str,
    title: str,
) -> None:
    labels = [k for k, v in ppl_series.items() if v is not None]
    values = [ppl_series[k] for k in labels]
    if not labels:
        logger.warning("No perplexity values to plot for %s; skipping", out_path.name)
        return

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(labels)), 5))
    bars = ax.bar(labels, values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def plot_loss_curves(history_by_label: dict[str, list], out_path: Path) -> None:
    """One subplot per run: train-loss dots + val-perplexity (log y on right axis)."""
    labels = [l for l, h in history_by_label.items() if h]
    if not labels:
        logger.warning("No training histories available; skipping loss curves")
        return

    n = len(labels)
    fig, axes = plt.subplots(n, 1, figsize=(9, 3.2 * n), sharex=False, squeeze=False)
    for ax, label in zip(axes.flatten(), labels):
        history = history_by_label[label]
        train_steps = [e["step"] for e in history if "train_loss" in e]
        train_losses = [e["train_loss"] for e in history if "train_loss" in e]
        val_steps = [e["step"] for e in history if "val_loss" in e]
        val_losses = [e["val_loss"] for e in history if "val_loss" in e]

        if train_steps:
            ax.plot(train_steps, train_losses, label="train", color="tab:blue", linewidth=1)
        if val_steps:
            ax.plot(val_steps, val_losses, label="val", color="tab:orange",
                    marker="o", linestyle="--", linewidth=1)
        ax.set_title(label)
        ax.set_xlabel("step")
        ax.set_ylabel("MLM loss")
        ax.legend(loc="best", fontsize=8, frameon=False)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def build_summary_rows(label: str, scoring: dict) -> list[dict]:
    """Parse all rho keys into rows with (model, dataset, strategy, slice, rho, pval).

    `slice == "all"` for whole-dataset Spearman; otherwise e.g. "left1", "right5".
    """
    rows = []
    if not scoring:
        return rows
    for key, val in scoring.items():
        if "_pval_" in key:
            continue
        m = _SCORING_KEY_RE.match(key)
        if m is None:
            continue
        strategy = m.group("strategy")
        dataset = m.group("dataset")
        side, k = m.group("side"), m.group("k")
        slice_name = f"{side}{k}" if side else "all"
        pval_key = (
            f"spearman_{strategy}_pval_{side}{k}_{dataset}"
            if side else f"spearman_{strategy}_pval_{dataset}"
        )
        rows.append({
            "model": label,
            "dataset": dataset,
            "strategy": strategy,
            "slice": slice_name,
            "rho": float(val) if val is not None else float("nan"),
            "pval": float(scoring.get(pval_key, float("nan"))),
        })
    return rows


def plot_flank_spearman(rows_df: pd.DataFrame, out_path: Path,
                        flank_ks: Sequence[int]) -> None:
    """One subplot per dataset; x-axis = slice (all, left_k..., right_k...), groups = model."""
    if rows_df.empty:
        logger.warning("No flank rows to plot; skipping %s", out_path.name)
        return

    datasets = sorted(rows_df["dataset"].unique())
    models = list(rows_df["model"].drop_duplicates())
    n_models = len(models)
    slice_order = ["all"] + [f"left{k}" for k in flank_ks] + [f"right{k}" for k in flank_ks]

    fig, axes = plt.subplots(
        1, len(datasets),
        figsize=(max(5, 1.6 * len(slice_order) * n_models / 3) * len(datasets), 5),
        sharey=True, squeeze=False,
    )
    width = 0.8 / max(n_models, 1)
    x = np.arange(len(slice_order))

    for ax, ds in zip(axes.flatten(), datasets):
        ds_df = rows_df[rows_df["dataset"] == ds]
        for i, model in enumerate(models):
            sub = (
                ds_df[ds_df["model"] == model]
                .set_index("slice")
                .reindex(slice_order)
            )
            rhos = sub["rho"].values
            pvals = sub["pval"].values
            positions = x + (i - (n_models - 1) / 2) * width
            bars = ax.bar(positions, rhos, width=width, label=model)
            for bar, rho, p in zip(bars, rhos, pvals):
                if np.isnan(rho):
                    continue
                stars = ("***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns")
                y = rho + (0.01 if rho >= 0 else -0.03)
                ax.text(bar.get_x() + bar.get_width() / 2, y, stars,
                        ha="center", va="bottom" if rho >= 0 else "top", fontsize=7)

        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(slice_order, rotation=30, ha="right")
        ax.set_title(ds)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0, 0].set_ylabel(f"Spearman ρ  (strategy: {STRATEGY})")
    axes[0, -1].legend(loc="best", fontsize=8, frameon=False)
    fig.suptitle("Spearman by CDR-H3 flank slice (mutation must hit left_k or right_k window)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    runs = list(OmegaConf.to_container(cfg.runs, resolve=True))
    labels = list(OmegaConf.to_container(cfg.labels, resolve=True))
    out_dir_str = cfg.out_dir
    force_rescore = cfg.get("force_rescore", False)
    force_reppl = cfg.get("force_reppl", False)
    skip_base = cfg.get("skip_base", False)

    if len(runs) != len(labels):
        raise ValueError("runs and labels must have the same length")

    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    seed = cfg.seed
    n_samples = cfg.scoring.n_samples
    scoring_batch_size = cfg.scoring.batch_size
    flank_ks = [int(k) for k in cfg.scoring.get("flank_ks", [])]
    datasets = load_scoring_datasets(
        OmegaConf.to_container(cfg.scoring.datasets, resolve=True), n_samples, seed
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

    rows: list[dict] = []
    val_ppl_series: dict[str, float | None] = {}
    test_ppl_series: dict[str, float | None] = {}
    cdr_ppl_series: dict[str, float | None] = {}
    history_by_label: dict[str, list] = {}

    if not skip_base:
        base_scoring, base_val, base_test, base_cdr = evaluate_base(
            cfg, datasets, tokenizer, device, seed, flank_ks=flank_ks,
        )
        rows.extend(build_summary_rows(BASE_LABEL, base_scoring))
        val_ppl_series[BASE_LABEL] = base_val
        test_ppl_series[BASE_LABEL] = base_test
        cdr_ppl_series[BASE_LABEL] = base_cdr

    for run_dir, label in zip(runs, labels):
        scoring, val_ppl, test_ppl, cdr_ppl, history = evaluate_run(
            Path(run_dir), datasets, tokenizer, device,
            scoring_batch_size, seed,
            force_rescore=force_rescore, force_reppl=force_reppl,
            fallback_config=cfg, flank_ks=flank_ks,
        )
        rows.extend(build_summary_rows(label, scoring))
        val_ppl_series[label] = val_ppl
        test_ppl_series[label] = test_ppl
        cdr_ppl_series[label] = cdr_ppl
        history_by_label[label] = history

    df = pd.DataFrame(rows)
    if df.empty:
        logger.error("No scoring rows collected; aborting plots")
        return

    headline = df[(df["strategy"] == STRATEGY) & (df["slice"] == "all")].copy()
    plot_spearman_bar(headline, out_dir / "spearman_bar.png")
    if flank_ks:
        flank_df = df[(df["strategy"] == STRATEGY)].copy()
        plot_flank_spearman(flank_df, out_dir / "spearman_flank.png", flank_ks)
    plot_perplexity_bar(
        test_ppl_series, out_dir / "test_perplexity_comparison.png",
        ylabel="Test perplexity  (↓ better)",
        title="Masked-LM perplexity on held-out test split",
    )
    plot_perplexity_bar(
        val_ppl_series, out_dir / "val_perplexity_comparison.png",
        ylabel="Val perplexity  (↓ better)",
        title="Masked-LM perplexity on validation split (diagnostic)",
    )
    plot_perplexity_bar(
        cdr_ppl_series, out_dir / "cdr_perplexity_comparison.png",
        ylabel="CDR-H3 pseudo-perplexity  (↓ better)",
        title="CDR-H3 pseudo-perplexity (full VH context, single-position masking)",
    )
    plot_loss_curves(history_by_label, out_dir / "train_val_loss_curves.png")

    csv_path = out_dir / "metrics_summary.csv"
    df_out = df.copy()
    df_out["final_val_perplexity"] = df_out["model"].map(val_ppl_series)
    df_out["final_test_perplexity"] = df_out["model"].map(test_ppl_series)
    df_out["cdr_pseudo_perplexity"] = df_out["model"].map(cdr_ppl_series)
    df_out.to_csv(csv_path, index=False)
    logger.info("Saved %s", csv_path)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    (out_dir / "manifest.txt").write_text(
        f"Generated {ts}\n"
        f"Scoring config: Hydra composable (model={cfg.model.name})\n\n"
        + "\n".join(f"{l}\t{r}" for l, r in zip(labels, runs))
    )


if __name__ == "__main__":
    main()
