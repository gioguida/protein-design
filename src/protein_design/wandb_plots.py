"""Publication-quality matplotlib helpers for W&B logging.

All plot builders take plain Python/numpy data (no W&B dependency) so the
same functions can regenerate paper figures offline from `metrics.json`
and `scores/*.csv`. The only W&B coupling is the `log_figure` and
`build_summary_table` helpers below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Colorblind-safe palette (IBM 5-color).
_PALETTE = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]


def set_publication_style() -> None:
    """Apply rcParams for publication-quality figures. Idempotent."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "lines.linewidth": 1.5,
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def _color(i: int) -> str:
    return _PALETTE[i % len(_PALETTE)]


# ---------------------------------------------------------------------------
# Training curves: train/val loss + val perplexity vs step
# ---------------------------------------------------------------------------


def plot_training_curves(
    training_history: Sequence[Mapping[str, Any]],
    best_step: Optional[int] = None,
) -> plt.Figure:
    """Two-panel training curves figure.

    `training_history` is a list of dicts. Each dict has either a
    `train_loss` field (training step record) or a `val_loss` field
    (validation/checkpoint record). Both share a `step` field.
    """
    train_steps = [r["step"] for r in training_history if "train_loss" in r]
    train_losses = [r["train_loss"] for r in training_history if "train_loss" in r]

    val_steps = [r["step"] for r in training_history if "val_loss" in r]
    val_losses = [r["val_loss"] for r in training_history if "val_loss" in r]
    val_ppls = [r.get("val_perplexity") for r in training_history if "val_loss" in r]
    val_cdr_ppls = [r.get("val_cdr_ppl") for r in training_history if "val_loss" in r]

    fig, (ax_loss, ax_ppl) = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)

    if train_steps:
        ax_loss.plot(train_steps, train_losses, color=_color(0), label="train", linewidth=1.2, alpha=0.8)
    if val_steps:
        ax_loss.plot(val_steps, val_losses, color=_color(2), label="val", marker="o", markersize=4)
    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.set_title("Loss")
    ax_loss.legend(loc="upper right")

    if val_steps:
        ppls_clean = [p for p in val_ppls if p is not None and np.isfinite(p)]
        cdr_clean = [c for c in val_cdr_ppls if c is not None and np.isfinite(c)]
        if ppls_clean:
            ax_ppl.plot(val_steps[: len(ppls_clean)], ppls_clean, color=_color(0), marker="o", markersize=4, label="val PPL")
        if cdr_clean:
            ax_ppl.plot(val_steps[: len(cdr_clean)], cdr_clean, color=_color(3), marker="s", markersize=4, label="CDR-H3 PPL")
        ax_ppl.set_yscale("log")
        ax_ppl.legend(loc="upper right")
    ax_ppl.set_xlabel("step")
    ax_ppl.set_ylabel("perplexity (log scale)")
    ax_ppl.set_title("Perplexity")

    if best_step is not None:
        for ax in (ax_loss, ax_ppl):
            ax.axvline(best_step, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

    return fig


# ---------------------------------------------------------------------------
# Spearman evolution: ρ over steps, one line per dataset
# ---------------------------------------------------------------------------


def plot_spearman_evolution(
    scoring_history: Sequence[Mapping[str, Any]],
    dataset_names: Sequence[str],
    pretrained_baseline: Optional[Mapping[str, float]] = None,
) -> plt.Figure:
    """Line plot of Spearman ρ vs step for each scoring dataset.

    Each scoring_history record looks like {"step": ..., "spearman_avg_<name>": ρ, ...}
    """
    fig, ax = plt.subplots(figsize=(6.5, 3.5), constrained_layout=True)

    steps = [r["step"] for r in scoring_history]
    if not steps:
        ax.text(0.5, 0.5, "No scoring data yet", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    for i, name in enumerate(dataset_names):
        key = f"spearman_avg_{name}"
        ys = [r.get(key) for r in scoring_history]
        valid = [(s, y) for s, y in zip(steps, ys) if y is not None and np.isfinite(y)]
        if not valid:
            continue
        xs, ys2 = zip(*valid)
        ax.plot(xs, ys2, color=_color(i), marker="o", markersize=4, label=name)
        if pretrained_baseline and name in pretrained_baseline:
            base = pretrained_baseline[name]
            if np.isfinite(base):
                ax.axhline(base, color=_color(i), linestyle=":", linewidth=0.8, alpha=0.6)

    ax.axhline(0.0, color="grey", linestyle="-", linewidth=0.5, alpha=0.4)
    ax.set_xlabel("step")
    ax.set_ylabel(r"Spearman $\rho$")
    ax.set_title("Fitness-prediction correlation")
    ax.legend(loc="best", title="dataset")
    return fig


# ---------------------------------------------------------------------------
# Flank breakdown: grouped bar chart, datasets × {ALL, left-k, right-k}
# ---------------------------------------------------------------------------


def plot_flank_breakdown(
    final_results: Mapping[str, float],
    dataset_names: Sequence[str],
    flank_ks: Sequence[int],
) -> plt.Figure:
    """Grouped bar chart of Spearman ρ by dataset and flank window.

    `final_results` is the flat dict returned by run_multi_scoring_evaluation
    with keys like `spearman_avg_<dataset>` and `spearman_avg_<side><k>_<dataset>`.
    """
    bar_specs = [("ALL", None, None)]
    for k in flank_ks:
        for side in ("left", "right"):
            bar_specs.append((f"{side}-{k}", side, k))

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(dataset_names) * len(bar_specs) / 4), 3.5),
                           constrained_layout=True)

    n_datasets = len(dataset_names)
    n_bars = len(bar_specs)
    width = 0.8 / n_bars
    x_positions = np.arange(n_datasets)

    for i, (label, side, k) in enumerate(bar_specs):
        ys = []
        for name in dataset_names:
            if side is None:
                key = f"spearman_avg_{name}"
            else:
                key = f"spearman_avg_{side}{k}_{name}"
            val = final_results.get(key)
            ys.append(val if val is not None and np.isfinite(val) else np.nan)
        offsets = x_positions + (i - (n_bars - 1) / 2) * width
        ax.bar(offsets, ys, width=width, color=_color(i), label=label, edgecolor="white", linewidth=0.5)

    ax.axhline(0.0, color="grey", linewidth=0.5)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(dataset_names)
    ax.set_ylabel(r"Spearman $\rho$")
    ax.set_title("Flank-window breakdown")
    ax.legend(loc="best", ncol=min(n_bars, 4), fontsize=8)
    return fig


# ---------------------------------------------------------------------------
# PLL pos/neg/wt comparison
# ---------------------------------------------------------------------------


def plot_pll_comparison(
    ppl_metrics: Mapping[str, float],
    label: str = "test",
) -> plt.Figure:
    """3-bar chart of pos / neg / wt PLL.

    Reads keys `ppl/{label}_pos`, `ppl/{label}_neg`, `ppl/{label}_wt` (the same
    naming used by evaluate_pll_eval_sets).
    """
    categories = ["pos", "neg", "wt"]
    values = [ppl_metrics.get(f"ppl/{label}_{c}", np.nan) for c in categories]
    colors = [_color(0), _color(2), _color(4)]

    fig, ax = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)
    bars = ax.bar(categories, values, color=colors, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, values):
        if np.isfinite(v):
            ax.annotate(f"{v:.2f}", (bar.get_x() + bar.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("perplexity (lower is better)")
    ax.set_title(f"{label} PLL by class")
    return fig


# ---------------------------------------------------------------------------
# PLL vs log-enrichment scatter
# ---------------------------------------------------------------------------


def _annotate_rho(ax: plt.Axes, rho: float, pval: float, n: int) -> None:
    txt = rf"$\rho$ = {rho:.3f}  ($p$ = {pval:.1e}, $n$ = {n})"
    ax.text(0.03, 0.97, txt, ha="left", va="top",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"))


def plot_pll_vs_enrichment(
    scores: np.ndarray,
    enrichment: np.ndarray,
    dataset_name: str,
    rho: float,
    pval: float,
    num_mut: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Scatter of PLL (y) vs log-enrichment (x) for one dataset.

    Hexbin if N > 5000, alpha-blended scatter otherwise.
    """
    scores = np.asarray(scores, dtype=float)
    enrichment = np.asarray(enrichment, dtype=float)
    mask = np.isfinite(scores) & np.isfinite(enrichment)
    scores = scores[mask]
    enrichment = enrichment[mask]
    n = len(scores)

    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 4), constrained_layout=True)
    else:
        fig = ax.figure

    if n == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    if n > 5000:
        ax.hexbin(enrichment, scores, gridsize=40, cmap="Blues", mincnt=1)
    else:
        if num_mut is not None:
            num_mut_arr = np.asarray(num_mut)[mask]
            unique_muts = sorted(set(int(m) for m in num_mut_arr if np.isfinite(m)))
            for i, m in enumerate(unique_muts):
                sel = num_mut_arr == m
                ax.scatter(enrichment[sel], scores[sel], s=8, alpha=0.35,
                           color=_color(i), label=f"{m} mut", linewidths=0)
            if len(unique_muts) > 1:
                ax.legend(loc="lower right", fontsize=8, markerscale=2)
        else:
            ax.scatter(enrichment, scores, s=8, alpha=0.25, color=_color(0), linewidths=0)

    _annotate_rho(ax, rho, pval, n)
    ax.set_xlabel("log-enrichment")
    ax.set_ylabel("PLL")
    ax.set_title(dataset_name)
    return fig


def plot_pll_vs_enrichment_grid(
    per_dataset: Mapping[str, Mapping[str, Any]],
) -> plt.Figure:
    """1×N grid of PLL-vs-enrichment scatter panels.

    `per_dataset[name]` carries {"scores", "enrichment", "rho", "pval",
    optionally "num_mut"}.
    """
    names = list(per_dataset.keys())
    n = len(names)
    if n == 0:
        fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
        ax.text(0.5, 0.5, "No scoring data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), constrained_layout=True, squeeze=False)
    for i, name in enumerate(names):
        d = per_dataset[name]
        plot_pll_vs_enrichment(
            scores=d["scores"],
            enrichment=d["enrichment"],
            dataset_name=name,
            rho=d.get("rho", float("nan")),
            pval=d.get("pval", float("nan")),
            num_mut=d.get("num_mut"),
            ax=axes[0, i],
        )
    return fig


# ---------------------------------------------------------------------------
# W&B convenience wrappers
# ---------------------------------------------------------------------------


def log_figure(
    fig: plt.Figure,
    key: str,
    step: Optional[int],
    wandb_mod: Any,
    save_dir: Optional[Path] = None,
    save_name: Optional[str] = None,
) -> None:
    """Log a figure to W&B and optionally save a local PNG mirror.

    Always closes the figure to free memory.
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        name = save_name or (key.replace("/", "_") + (f"_step_{step}" if step is not None else "") + ".png")
        fig.savefig(save_dir / name)
    if wandb_mod is not None:
        if step is not None:
            wandb_mod.log({key: wandb_mod.Image(fig)}, step=step)
        else:
            wandb_mod.log({key: wandb_mod.Image(fig)})
    plt.close(fig)


def build_summary_table(summary: Mapping[str, Any], wandb_mod: Any) -> Any:
    """Build a 2-column wandb.Table (metric, value) for the final results panel."""
    rows = []
    for k, v in summary.items():
        if isinstance(v, float):
            v_str = f"{v:.4f}" if abs(v) < 1e4 else f"{v:.4e}"
        else:
            v_str = str(v)
        rows.append([k, v_str])
    table = wandb_mod.Table(columns=["metric", "value"], data=rows)
    return table


# ---------------------------------------------------------------------------
# RunArtifacts: mirror every wandb.log call to history.csv on disk
# ---------------------------------------------------------------------------


class RunArtifacts:
    """Owns the on-disk run dir layout and mirrors wandb.log to history.csv.

    Layout:
        run_dir/
            history.csv     # one row per logged step
            scores/         # per-dataset (PLL, enrichment) CSVs
            figures/        # PNG mirrors of every logged figure

    The history is held in memory as {step: {key: value}} and flushed to CSV
    on every call to `flush()` (so a crashed run still has data). Pandas is
    only imported lazily.
    """

    def __init__(self, run_dir: Path, wandb_mod: Optional[Any] = None) -> None:
        self.run_dir = Path(run_dir)
        self.wandb_mod = wandb_mod
        self.scores_dir = self.run_dir / "scores"
        self.figures_dir = self.run_dir / "figures"
        self.history_path = self.run_dir / "history.csv"
        # Use a dict-of-dicts so each row is keyed by step.
        self._history: Dict[int, Dict[str, Any]] = {}
        # On resume: seed the in-memory history from the existing CSV so that
        # flush() appends rather than overwrites the previous run's data.
        if self.history_path.exists():
            try:
                import pandas as pd
                df = pd.read_csv(self.history_path)
                for _, row in df.iterrows():
                    step = int(row["step"])
                    self._history[step] = {
                        k: v for k, v in row.items()
                        if k != "step" and pd.notna(v)
                    }
            except Exception:
                pass

    def log(self, payload: Mapping[str, Any], step: Optional[int] = None) -> None:
        """Mirror payload to W&B (if enabled) and to the in-memory history dict.

        Skips non-scalar values (e.g. wandb.Image) for the on-disk history but
        still forwards them to W&B.
        """
        if self.wandb_mod is not None:
            if step is not None:
                self.wandb_mod.log(dict(payload), step=step)
            else:
                self.wandb_mod.log(dict(payload))

        if step is None:
            return
        row = self._history.setdefault(int(step), {})
        for k, v in payload.items():
            if isinstance(v, (int, float, bool)) or v is None:
                row[k] = v
            # Non-scalar (Image, Table, np.ndarray) values are W&B-only.

    def log_figure(
        self,
        fig: "plt.Figure",
        key: str,
        step: Optional[int] = None,
        save_name: Optional[str] = None,
    ) -> None:
        """Render `fig` to PNG in figures/ and log it to W&B."""
        log_figure(fig, key, step, self.wandb_mod, save_dir=self.figures_dir, save_name=save_name)

    def write_scores_csv(self, dataset_name: str, df_or_records: Any, suffix: str = "test") -> Path:
        """Persist per-sequence (PLL, enrichment, ...) for one dataset.

        Accepts a pandas DataFrame; passes it through to_csv. Caller is
        responsible for column shape (typically aa, mut, num_mut, pll, enrichment).
        """
        self.scores_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.scores_dir / f"{dataset_name}_{suffix}.csv"
        df_or_records.to_csv(out_path, index=False)
        return out_path

    def flush(self) -> None:
        """Write the in-memory history dict to history.csv (atomic-ish)."""
        if not self._history:
            return
        # Lazy import: only needed at flush time.
        import pandas as pd

        steps = sorted(self._history.keys())
        rows = [{"step": s, **self._history[s]} for s in steps]
        df = pd.DataFrame(rows)
        cols = ["step"] + sorted(c for c in df.columns if c != "step")
        df = df[cols]
        tmp_path = self.history_path.with_suffix(".csv.tmp")
        df.to_csv(tmp_path, index=False)
        tmp_path.replace(self.history_path)
