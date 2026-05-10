#!/usr/bin/env python
"""Generate comparison plots across evotuning / C05 / TTT runs.

Produces:
    spearman_bar.png                 Grouped bar chart of Spearman rho per
                                     {model} x {dataset}, with p-value significance.
    test_perplexity_comparison.png   Final test-set MLM perplexity per model,
                                     measured on the top-level cfg.data fasta+split.
    val_perplexity_comparison.png    Final val-set MLM perplexity per model,
                                     measured on the top-level cfg.data fasta+split.
    cdr_perplexity_comparison.png    CDR-H3 pseudo-perplexity on the C05 reference VH.
    m22_cdr_perplexity_comparison.png  Corpus-level CDR-H3 pseudo-perplexity over the
                                     M22 D2 variant library.
    train_val_loss_curves.png        One subplot per run: train loss vs. val loss.
    metrics_summary.csv              One row per (model, dataset, strategy): rho, pval, ppls.

For each run dir passed in, the script:
  1. Reads metrics.json if present (for cached scoring/CDR-H3 metrics).
  2. Re-scores best.pt (or final.pt) on the D2 datasets if scoring is missing
     or --force-rescore is set.
  3. ALWAYS recomputes val/test MLM perplexity on the shared loaders built from
     the top-level cfg.data.fasta_path + SplitConfig (so numbers are comparable
     across runs trained on different corpora). Any cached
     final_val/test_perplexity in metrics.json is ignored.

Usage (repeat run/label pairs via Hydra list overrides):
    python scripts/analysis/plot_results.py scoring=d2 data=oas_full \\
        '+runs=[/path/to/run1,/path/to/run2]' \\
        '+labels=[evotuned,+C05]' \\
        +out_dir=${HOME}/protein-design/plots/meeting
"""

import functools
import hashlib
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Sequence

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from protein_design.eval import (
    compute_m22_cdr_pseudo_perplexity,
    compute_perplexity,
    corpus_perplexity,
    load_scoring_datasets,
    run_multi_scoring_evaluation,
)
from protein_design.evotuning.data import make_dataloaders
from protein_design.evotuning.splits import SplitConfig
from protein_design.model import ESM2Model
from protein_design.config import build_model_config
from protein_design.constants import C05_CDRH3

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_LABEL = "base ESM2"
STRATEGY = "avg"  # headline strategy for the Spearman bar plot
STRATEGIES = ("avg", "random")  # emitted by run_scoring_evaluation (eval.py)


def _split_cfg_from(cfg: DictConfig) -> SplitConfig:
    """Read the hash-split policy from a run's config, deferring missing keys to SplitConfig defaults."""
    split_node = cfg.data.get("split") if "split" in cfg.data else None
    if split_node is None:
        return SplitConfig()
    kwargs: dict = {}
    for key in ("salt", "train_pct", "val_pct", "test_pct"):
        if key in split_node:
            kwargs[key] = split_node[key]
    return SplitConfig(**kwargs)


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
    _train, val_loader, test_loader, *_ = make_dataloaders(
        fasta_path=fasta_path,
        tokenizer_name=cfg.model.name,
        max_seq_len=int(cfg.data.max_seq_len),
        mlm_probability=float(cfg.data.mlm_probability),
        batch_size=_batch_size_from(cfg),
        split_cfg=split_cfg,
    )
    return val_loader, test_loader


@functools.lru_cache(maxsize=32)
def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head_sha() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}"
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return "<unknown>"


def _find_checkpoint(run_dir: Path) -> tuple[Path, bool]:
    """Prefer best.pt; fall back to final.pt (in root or checkpoints/).

    Returns (ckpt_path, stale_final_metrics). `stale_final_metrics` is True when we picked
    best.pt AND final.pt also exists AND they differ — in which case the cached
    `final_*_perplexity` and last `scoring_history` in metrics.json were computed on
    final.pt and must NOT be trusted for best.pt (train.py:257-282 computes final metrics
    on the final model state before the best-copy step at line 310).
    """
    best = run_dir / "best.pt"
    finals = [run_dir / "final.pt", run_dir / "checkpoints" / "final.pt"]
    final = next((p for p in finals if p.exists()), None)

    if best.exists():
        stale = False
        if final is not None:
            if best.stat().st_size != final.stat().st_size:
                stale = True
            elif _sha256_file(best) != _sha256_file(final):
                stale = True
        if stale:
            logger.warning(
                "[%s] best.pt differs from final.pt — invalidating cached final_* metrics "
                "(they were computed on final.pt)",
                run_dir.name,
            )
        return best, stale
    if final is not None:
        return final, False
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


def _load_training_history(metrics: dict, run_dir: Path) -> list:
    """Per-step history. Prefer history.csv (new layout); fall back to
    metrics['training_history'] for legacy runs.
    """
    history_path = run_dir / "history.csv"
    if history_path.exists():
        try:
            df = pd.read_csv(history_path)
        except Exception as exc:
            logger.warning("Could not read %s (%s); falling back to legacy.", history_path, exc)
        else:
            return df.to_dict("records")
    return metrics.get("training_history", [])


def _extract_final_scoring(metrics: dict, run_dir: Path | None = None) -> dict | None:
    """Final scoring entry. Reads new layout (metrics['final'] + history.csv)
    first, falls back to the old `scoring_history` list-of-dicts shape.
    """
    final_block = metrics.get("final") or {}
    if final_block:
        # New shape: pull all `test_spearman_*` keys back out as a flat dict
        # using the same key names the old scoring_history entries used
        # (i.e. without the `test_` prefix).
        scoring = {}
        for key, value in final_block.items():
            if key.startswith("test_spearman_") or key.startswith("test_n_"):
                scoring[key.removeprefix("test_")] = value
        if scoring:
            return scoring
    # Legacy shape: list-of-dicts under `scoring_history`.
    history = metrics.get("scoring_history", [])
    return history[-1] if history else None


def _scoring_has_flank_keys(
    scoring: dict | None,
    dataset_names: Sequence[str],
    flank_ks: Sequence[int],
) -> bool:
    """True when `scoring` contains at least one flank key for every requested k × side × dataset."""
    if not scoring or not flank_ks:
        return True
    for k in flank_ks:
        for side in ("left", "right"):
            slice_name = f"{side}{k}"
            for dataset in dataset_names:
                rho_key, _ = _slice_spec(STRATEGY, slice_name, dataset)
                if rho_key not in scoring:
                    return False
    return True


def _find_m22_df(datasets):
    """Return the M22 dataframe from a loaded scoring datasets list, or None."""
    for name, df, _col in datasets:
        if name == "M22":
            return df
    return None


def evaluate_run(run_dir: Path, datasets, tokenizer, device, scoring_batch_size,
                 seed, force_rescore: bool, force_cdr: bool, force_m22_cdr: bool,
                 dataset_names: Sequence[str],
                 shared_val_loader, shared_test_loader,
                 shared_ppl_fasta: str,
                 fallback_config: DictConfig | None = None,
                 flank_ks: Sequence[int] = (),
                 max_ppl_batches: int = 500):
    """Return (scoring, val_ppl, test_ppl, cdr_ppl, m22_cdr_ppl, training_history, ckpt_path).

    Whole-sequence val/test perplexity is always recomputed on the *shared* OAS
    val/test loaders (built from the top-level cfg), ignoring any cached
    `final_val_perplexity` / `final_test_perplexity` in metrics.json — those were
    measured on the run's own corpus and are not comparable across runs.
    """
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
    ckpt_path, stale_final_metrics = _find_checkpoint(run_dir)
    logger.info("Run %s — checkpoint: %s", run_dir.name, ckpt_path)

    # Cache lookup with three invalidation paths:
    #   1. explicit force flags
    #   2. checkpoint mismatch (best.pt picked but final_* were computed on final.pt)
    #   3. cached scoring lacks flank keys for the requested flank_ks
    scoring = None if force_rescore else _extract_final_scoring(metrics, run_dir)
    if scoring is not None and stale_final_metrics:
        scoring = None  # last scoring_history entry was recorded on final.pt
    if scoring is not None and not _scoring_has_flank_keys(scoring, dataset_names, flank_ks):
        logger.warning(
            "[%s] cached scoring lacks flank keys for flank_ks=%s — rescoring",
            run_dir.name, list(flank_ks),
        )
        scoring = None

    final_block = metrics.get("final") or {}
    legacy_val = metrics.get("final_val_perplexity")
    legacy_test = metrics.get("final_test_perplexity")
    if legacy_val is not None or legacy_test is not None or final_block.get("test_perplexity") is not None:
        logger.info(
            "  ignoring cached final_val/test_perplexity (they were computed on the run's own corpus)"
        )
    val_ppl = None
    test_ppl = None
    cached_cdr = final_block.get("val_cdr_ppl") or metrics.get("final_cdr_pseudo_perplexity")
    cdr_ppl = None if (force_cdr or stale_final_metrics) else cached_cdr
    m22_cdr_ppl = (
        None if (force_m22_cdr or stale_final_metrics)
        else metrics.get("final_m22_cdr_pseudo_perplexity")
    )
    # New layout writes step-level history to history.csv; fall back to the
    # legacy list-of-dicts shape if that doesn't exist.
    training_history = _load_training_history(metrics, run_dir)

    model = ESM2Model(build_model_config(cfg, device=str(device)))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    if scoring is None:
        logger.info("  re-scoring on D2 datasets")
        scoring = run_multi_scoring_evaluation(
            model, tokenizer, datasets, device=device,
            batch_size=scoring_batch_size, seed=seed, flank_ks=flank_ks,
        )

    logger.info(
        "  recomputing val/test perplexity on %s (max %d batches)",
        shared_ppl_fasta, max_ppl_batches,
    )
    if len(shared_val_loader.dataset) > 0:
        val_ppl, _ = compute_perplexity(
            model, shared_val_loader, device, max_batches=max_ppl_batches,
        )
    if len(shared_test_loader.dataset) > 0:
        test_ppl, _ = compute_perplexity(
            model, shared_test_loader, device, max_batches=max_ppl_batches,
        )

    if cdr_ppl is None:
        cdr_ppl = corpus_perplexity([C05_CDRH3], scorer=model, cdr_only=True)

    if m22_cdr_ppl is None:
        m22_df = _find_m22_df(datasets)
        if m22_df is None:
            logger.warning("  no M22 dataset loaded — skipping M22 CDR pseudo-perplexity")
        else:
            m22_cdr_ppl = compute_m22_cdr_pseudo_perplexity(
                model, tokenizer, device, m22_df, batch_size=scoring_batch_size,
            )

    del model
    torch.cuda.empty_cache()

    return scoring, val_ppl, test_ppl, cdr_ppl, m22_cdr_ppl, training_history, ckpt_path


def evaluate_base(scoring_cfg: DictConfig, datasets, tokenizer, device, seed,
                  shared_val_loader, shared_test_loader, shared_ppl_fasta: str,
                  flank_ks: Sequence[int] = (), max_ppl_batches: int = 500):
    """Score base ESM2 and compute val/test perplexity on the shared OAS split."""
    logger.info("Evaluating base ESM2")
    model = ESM2Model(build_model_config(scoring_cfg, device=str(device)))
    model.to(device).eval()
    batch_size = int(scoring_cfg.scoring.batch_size)
    scoring = _score_model(model, tokenizer, datasets, device, batch_size, seed,
                           flank_ks=flank_ks)

    val_ppl, test_ppl = None, None
    logger.info(
        "  computing val/test perplexity on %s (max %d batches)",
        shared_ppl_fasta, max_ppl_batches,
    )
    if len(shared_val_loader.dataset) > 0:
        val_ppl, _ = compute_perplexity(
            model, shared_val_loader, device, max_batches=max_ppl_batches,
        )
    if len(shared_test_loader.dataset) > 0:
        test_ppl, _ = compute_perplexity(
            model, shared_test_loader, device, max_batches=max_ppl_batches,
        )

    cdr_ppl = _compute_cdr_perplexity(model)

    m22_cdr_ppl = None
    m22_df = _find_m22_df(datasets)
    if m22_df is None:
        logger.warning("  no M22 dataset loaded — skipping M22 CDR pseudo-perplexity")
    else:
        m22_cdr_ppl = compute_m22_cdr_pseudo_perplexity(
            model, tokenizer, device, m22_df, batch_size=batch_size,
        )

    del model
    torch.cuda.empty_cache()
    return scoring, val_ppl, test_ppl, cdr_ppl, m22_cdr_ppl


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
    """One subplot per run: train-loss line + val-loss markers on a shared linear axis."""
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


def _slice_spec(strategy: str, slice_name: str, dataset: str) -> tuple[str, str]:
    """Return the (rho_key, pval_key) pair expected in a scoring dict."""
    if slice_name == "all":
        return f"spearman_{strategy}_{dataset}", f"spearman_{strategy}_pval_{dataset}"
    return (
        f"spearman_{strategy}_{slice_name}_{dataset}",
        f"spearman_{strategy}_pval_{slice_name}_{dataset}",
    )


def _expected_slices(flank_ks: Sequence[int]) -> list[str]:
    return ["all"] + [f"left{k}" for k in flank_ks] + [f"right{k}" for k in flank_ks]


def build_summary_rows(
    label: str,
    scoring: dict,
    dataset_names: Sequence[str],
    flank_ks: Sequence[int],
) -> list[dict]:
    """Emit (model, dataset, strategy, slice, rho, pval) rows by *constructing* expected keys
    from the known dataset vocabulary and flank_ks — never by parsing. Dataset names
    containing underscores are handled correctly.
    """
    rows: list[dict] = []
    if not scoring:
        return rows
    for dataset in dataset_names:
        for strategy in STRATEGIES:
            for slice_name in _expected_slices(flank_ks):
                rho_key, pval_key = _slice_spec(strategy, slice_name, dataset)
                if rho_key not in scoring:
                    continue
                rho_val = scoring.get(rho_key)
                pval_val = scoring.get(pval_key)
                rows.append({
                    "model": label,
                    "dataset": dataset,
                    "strategy": strategy,
                    "slice": slice_name,
                    "rho": float(rho_val) if rho_val is not None else float("nan"),
                    "pval": float(pval_val) if pval_val is not None else float("nan"),
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
    runs_raw = OmegaConf.to_container(cfg.runs, resolve=True)
    labels_raw = OmegaConf.to_container(cfg.labels, resolve=True)
    if not isinstance(runs_raw, list) or not isinstance(labels_raw, list):
        raise TypeError(
            "runs/labels must be Hydra lists, e.g. '+runs=[/path/to/run1,/path/to/run2]'. "
            f"Got runs={type(runs_raw).__name__}, labels={type(labels_raw).__name__}."
        )
    runs = list(runs_raw)
    labels = list(labels_raw)
    out_dir_str = cfg.out_dir
    force_rescore = cfg.get("force_rescore", False)
    force_cdr = cfg.get("force_cdr", False)
    force_m22_cdr = cfg.get("force_m22_cdr", False)
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
    max_ppl_batches = int(cfg.scoring.get("max_ppl_batches", 500))
    flank_ks = [int(k) for k in cfg.scoring.get("flank_ks", [])]
    datasets_cfg = OmegaConf.to_container(cfg.scoring.datasets, resolve=True)
    dataset_names = [d["name"] for d in datasets_cfg]
    datasets = load_scoring_datasets(datasets_cfg, n_samples, seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

    # Build the shared val/test loaders ONCE from the top-level cfg. Every run
    # (base and finetunes) is evaluated on this same split so whole-sequence
    # MLM perplexities are comparable across checkpoints trained on different
    # corpora (OAS vs. C05 FASTAs).
    shared_ppl_fasta = cfg.data.get("fasta_path")
    if not shared_ppl_fasta or not Path(shared_ppl_fasta).exists():
        raise FileNotFoundError(
            f"Top-level cfg.data.fasta_path is required for shared val/test perplexity: "
            f"got {shared_ppl_fasta!r}"
        )
    logger.info(
        "Shared val/test loaders: fasta=%s, split=%s",
        shared_ppl_fasta, _split_cfg_from(cfg),
    )
    shared_val_loader, shared_test_loader = _make_eval_loaders(shared_ppl_fasta, cfg)

    rows: list[dict] = []
    val_ppl_series: dict[str, float | None] = {}
    test_ppl_series: dict[str, float | None] = {}
    cdr_ppl_series: dict[str, float | None] = {}
    m22_cdr_ppl_series: dict[str, float | None] = {}
    history_by_label: dict[str, list] = {}

    if not skip_base:
        base_scoring, base_val, base_test, base_cdr, base_m22_cdr = evaluate_base(
            cfg, datasets, tokenizer, device, seed,
            shared_val_loader, shared_test_loader, shared_ppl_fasta,
            flank_ks=flank_ks, max_ppl_batches=max_ppl_batches,
        )
        rows.extend(build_summary_rows(BASE_LABEL, base_scoring, dataset_names, flank_ks))
        val_ppl_series[BASE_LABEL] = base_val
        test_ppl_series[BASE_LABEL] = base_test
        cdr_ppl_series[BASE_LABEL] = base_cdr
        m22_cdr_ppl_series[BASE_LABEL] = base_m22_cdr

    ckpt_by_label: dict[str, Path] = {}
    fasta_by_label: dict[str, str | None] = {BASE_LABEL: cfg.data.get("fasta_path")} if not skip_base else {}
    for run_dir, label in zip(runs, labels):
        scoring, val_ppl, test_ppl, cdr_ppl, m22_cdr_ppl, history, ckpt_path = evaluate_run(
            Path(run_dir), datasets, tokenizer, device,
            scoring_batch_size, seed,
            force_rescore=force_rescore, force_cdr=force_cdr, force_m22_cdr=force_m22_cdr,
            dataset_names=dataset_names,
            shared_val_loader=shared_val_loader,
            shared_test_loader=shared_test_loader,
            shared_ppl_fasta=shared_ppl_fasta,
            fallback_config=cfg, flank_ks=flank_ks,
            max_ppl_batches=max_ppl_batches,
        )
        rows.extend(build_summary_rows(label, scoring, dataset_names, flank_ks))
        val_ppl_series[label] = val_ppl
        test_ppl_series[label] = test_ppl
        cdr_ppl_series[label] = cdr_ppl
        m22_cdr_ppl_series[label] = m22_cdr_ppl
        history_by_label[label] = history
        ckpt_by_label[label] = ckpt_path
        run_cfg = _load_run_config(Path(run_dir)) if (Path(run_dir) / "config.yaml").exists() else cfg
        fasta_by_label[label] = run_cfg.data.get("fasta_path") if "data" in run_cfg else None

    df = pd.DataFrame(rows)
    if df.empty:
        logger.error("No scoring rows collected; aborting plots")
        return

    unique_fastas = {fasta_by_label.get(lbl) for lbl in fasta_by_label}
    if len(unique_fastas) > 1:
        logger.info(
            "Runs trained on %d distinct fastas: %s. All val/test perplexities are "
            "measured on the shared corpus %s — plots are directly comparable.",
            len(unique_fastas), sorted(str(f) for f in unique_fastas), shared_ppl_fasta,
        )

    corpus_stem = Path(shared_ppl_fasta).stem
    headline = df[(df["strategy"] == STRATEGY) & (df["slice"] == "all")].copy()
    plot_spearman_bar(headline, out_dir / "spearman_bar.png")
    if flank_ks:
        flank_df = df[(df["strategy"] == STRATEGY)].copy()
        plot_flank_spearman(flank_df, out_dir / "spearman_flank.png", flank_ks)
    plot_perplexity_bar(
        test_ppl_series, out_dir / "test_perplexity_comparison.png",
        ylabel="Test perplexity  (↓ better)",
        title=f"Masked-LM perplexity on shared test split\ncorpus: {corpus_stem}",
    )
    plot_perplexity_bar(
        val_ppl_series, out_dir / "val_perplexity_comparison.png",
        ylabel="Val perplexity  (↓ better)",
        title=f"Masked-LM perplexity on shared validation split\ncorpus: {corpus_stem}",
    )
    # CDR-H3 pseudo-perplexity is measured on a fixed C05 VH reference (eval.py:C05_VH),
    # independent of the training fasta — do not split by corpus.
    plot_perplexity_bar(
        cdr_ppl_series, out_dir / "cdr_perplexity_comparison.png",
        ylabel="CDR-H3 pseudo-perplexity  (↓ better)",
        title="CDR-H3 pseudo-perplexity on C05 VH (single-position masking)",
    )
    plot_perplexity_bar(
        m22_cdr_ppl_series, out_dir / "m22_cdr_perplexity_comparison.png",
        ylabel="M22 CDR-H3 pseudo-perplexity  (↓ better)",
        title="M22 CDR-H3 pseudo-perplexity (corpus-level over D2 variant library)",
    )
    plot_loss_curves(history_by_label, out_dir / "train_val_loss_curves.png")

    csv_path = out_dir / "metrics_summary.csv"
    df_out = df.copy()
    df_out["final_val_perplexity"] = df_out["model"].map(val_ppl_series)
    df_out["final_test_perplexity"] = df_out["model"].map(test_ppl_series)
    df_out["cdr_pseudo_perplexity"] = df_out["model"].map(cdr_ppl_series)
    df_out["m22_cdr_pseudo_perplexity"] = df_out["model"].map(m22_cdr_ppl_series)
    df_out.to_csv(csv_path, index=False)
    logger.info("Saved %s", csv_path)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    git_sha = _git_head_sha()
    run_lines = []
    for label, run_dir in zip(labels, runs):
        ckpt = ckpt_by_label.get(label)
        if ckpt is not None and Path(ckpt).exists():
            ckpt_hash = _sha256_file(Path(ckpt))[:16]
            mtime = datetime.fromtimestamp(Path(ckpt).stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            run_lines.append(
                f"{label}\t{run_dir}\tckpt={Path(ckpt).name}\tsha256={ckpt_hash}\tmtime={mtime}"
            )
        else:
            run_lines.append(f"{label}\t{run_dir}\t<ckpt missing>")

    manifest = (
        f"Generated: {ts}\n"
        f"Git HEAD:  {git_sha}\n"
        f"Model:     {cfg.model.name}\n"
        f"Base fasta: {cfg.data.get('fasta_path', '<none>')}\n"
        f"\n# Runs (label, run_dir, checkpoint provenance)\n"
        + "\n".join(run_lines)
        + "\n\n# Resolved Hydra config\n"
        + OmegaConf.to_yaml(cfg, resolve=True)
    )
    (out_dir / "manifest.txt").write_text(manifest)
    logger.info("Saved %s", out_dir / "manifest.txt")


if __name__ == "__main__":
    main()
