"""Notebook-facing figure functions for the paper.

Each `plot_*` takes a list of model keys (order = plot order; add/remove freely)
and returns a matplotlib Figure, so the notebook can display it inline and
`save_fig()` it to report/figures/<name>.pdf.

Reads only the cached artifacts via `registry`; run `preflight()` first to see
what still needs extracting.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import registry

log = logging.getLogger(__name__)

FIGURES_DIR = registry.REPO_ROOT / "report" / "figures"


# ── helpers ─────────────────────────────────────────────────────────────────

def _label(model_key: str) -> str:
    return registry.load_models().get(model_key, {}).get("label", model_key)


def _color(model_key: str, fallback: str) -> str:
    return registry.load_models().get(model_key, {}).get("color", fallback)


def pll_truth_spearman(model_key: str, dataset_key: str) -> tuple[float, int]:
    """Spearman rho between cached PLL and experimental truth, joined on sequence.

    Returns (rho, n). Raises FileNotFoundError if the PLL artifact is missing.
    """
    ds = registry.load_datasets_cfg()["datasets"][dataset_key]
    seq_col = ds["seq_col"]
    pll = registry.load_pll(model_key, dataset_key)
    truth = registry.load_truth(dataset_key)
    joined = pll.merge(truth, on=seq_col, how="inner")
    x = joined["pll"].to_numpy(np.float64)
    y = joined["enrichment"].to_numpy(np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan"), int(mask.sum())
    rho, _ = spearmanr(x[mask], y[mask])
    return float(rho), int(mask.sum())


def save_fig(fig: plt.Figure, name: str) -> Path:
    """Save `fig` to report/figures/<name>.pdf (vector, embedded fonts)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / (name if name.endswith(".pdf") else f"{name}.pdf")
    plt.rcParams["pdf.fonttype"] = 42  # embed TrueType, not Type 3
    fig.savefig(out, format="pdf", bbox_inches="tight")
    print(f"Saved: {out}")
    return out


# ── preflight ────────────────────────────────────────────────────────────────

def preflight(model_keys: list[str], datasets: str | list[str] = "all",
              kind: str = "pll") -> list[tuple[str, str]]:
    """Report which (model, dataset) artifacts are missing and how to make them.

    Prints copy-pasteable sbatch commands and returns the missing pairs so a
    notebook cell can optionally submit them.
    """
    ds_keys = registry.dataset_keys("all") if datasets == "all" else (
        [datasets] if isinstance(datasets, str) else datasets)
    missing = []
    for m in model_keys:
        for d in ds_keys:
            if not registry.artifact_path(m, kind, f"{d}.csv").exists():
                missing.append((m, d))
    if not missing:
        print(f"All {kind} artifacts present for {len(model_keys)} model(s) "
              f"x {len(ds_keys)} dataset(s).")
        return []
    print(f"Missing {len(missing)} {kind} artifact(s). To extract:\n")
    for m in sorted({m for m, _ in missing}):
        ds = ",".join(d for mm, d in missing if mm == m)
        print(f"  sbatch bash_scripts/extract.sbatch --what {kind} "
              f"--model {m} --dataset {ds}")
    return missing


# ── figures ──────────────────────────────────────────────────────────────────

def plot_pll_spearman(model_keys: list[str], datasets: str | list[str] = "all",
                      ax: plt.Axes | None = None) -> plt.Figure:
    """Grouped bars: Spearman(PLL, truth) per dataset, one bar per model.

    The canonical "which model best ranks binders" comparison. Reorder/extend
    `model_keys` to change the bars shown. Missing artifacts are skipped with a
    warning (run `preflight` to extract them first).
    """
    ds_keys = registry.dataset_keys("all") if datasets == "all" else (
        [datasets] if isinstance(datasets, str) else datasets)

    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])
    rhos: dict[str, list[float]] = {}
    for i, m in enumerate(model_keys):
        row = []
        for d in ds_keys:
            try:
                rho, _ = pll_truth_spearman(m, d)
            except FileNotFoundError:
                log.warning("[%s/%s] PLL artifact missing — bar omitted", m, d)
                rho = np.nan
            row.append(rho)
        rhos[m] = row

    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(
        figsize=(1.6 + 1.1 * len(ds_keys), 4.2), constrained_layout=True)
    n_models = len(model_keys)
    width = 0.8 / max(n_models, 1)
    x = np.arange(len(ds_keys))
    for i, m in enumerate(model_keys):
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, rhos[m], width,
               label=_label(m), color=_color(m, palette[i % len(palette)]))

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(ds_keys, rotation=30, ha="right")
    ax.set_ylabel(r"Spearman $\rho$ (PLL vs truth)")
    ax.set_title("CDR-H3 PLL ranks experimental enrichment")
    ax.legend(fontsize=8, ncol=1, frameon=False)
    return fig


def preflight_scorer(datasets: str | list[str] = "all") -> list[str]:
    """Report which scorer artifacts are missing and how to produce them.

    Scorer predictions are computed by score_dms_with_esme.py (flu conda env)
    and stored at $cache_root/scorer_preds/<dataset>/<scorer>.csv.
    """
    ds_keys = registry.dataset_keys("all") if datasets == "all" else (
        [datasets] if isinstance(datasets, str) else datasets)
    missing = [d for d in ds_keys
               if registry.scorer_artifact_path(d) is not None
               and not registry.scorer_artifact_path(d).exists()]
    if not missing:
        print("All scorer artifacts present.")
        return []
    ds_arg = ",".join(missing)
    print(f"Missing scorer predictions for: {', '.join(missing)}\n")
    print(f"  sbatch bash_scripts/utils/score_dms.sbatch --dataset {ds_arg}")
    return missing


def _scorer_truth_spearman(dataset_key: str) -> tuple[float, int]:
    """Spearman rho between scorer predictions and experimental truth."""
    cfg = registry.load_datasets_cfg()
    seq_col = cfg["datasets"][dataset_key]["seq_col"]
    score_df = registry.load_scorer(dataset_key)
    if score_df is None:
        return float("nan"), 0
    truth = registry.load_truth(dataset_key)
    joined = score_df.merge(truth, on=seq_col, how="inner")
    x = joined["score"].to_numpy(np.float64)
    y = joined["enrichment"].to_numpy(np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan"), int(mask.sum())
    rho, _ = spearmanr(x[mask], y[mask])
    return float(rho), int(mask.sum())


def plot_pll_vs_scorer_spearman(
    model_keys: list[str],
    datasets: str | list[str] = "all",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Grouped bars: Spearman(PLL, truth) per model, with scorer-vs-truth baseline.

    The dashed horizontal segment per dataset shows the fixed binder scorer's
    Spearman with experimental truth — a supervised upper bound that doesn't
    vary per model. Missing PLL artifacts are skipped; missing scorer predictions
    produce a gap in the baseline (run preflight_scorer to see what's needed).
    """
    ds_keys = registry.dataset_keys("all") if datasets == "all" else (
        [datasets] if isinstance(datasets, str) else datasets)

    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])
    rhos: dict[str, list[float]] = {}
    for m in model_keys:
        row = []
        for d in ds_keys:
            try:
                rho, _ = pll_truth_spearman(m, d)
            except FileNotFoundError:
                log.warning("[%s/%s] PLL artifact missing — bar omitted", m, d)
                rho = np.nan
            row.append(rho)
        rhos[m] = row

    scorer_rhos = [_scorer_truth_spearman(d)[0] for d in ds_keys]

    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(
        figsize=(1.6 + 1.1 * len(ds_keys), 4.2), constrained_layout=True)
    n_models = len(model_keys)
    width = 0.8 / max(n_models, 1)
    x = np.arange(len(ds_keys))

    for i, m in enumerate(model_keys):
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, rhos[m], width,
               label=_label(m), color=_color(m, palette[i % len(palette)]))

    half_group = 0.42
    for j, (d, srho) in enumerate(zip(ds_keys, scorer_rhos)):
        if np.isfinite(srho):
            ax.plot([x[j] - half_group, x[j] + half_group], [srho, srho],
                    color="black", linewidth=2, linestyle="--",
                    label="Scorer baseline" if j == 0 else "_nolegend_")

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(ds_keys, rotation=30, ha="right")
    ax.set_ylabel(r"Spearman $\rho$ (vs experimental truth)")
    ax.set_title("PLL vs supervised scorer: ranking experimental enrichment")
    ax.legend(fontsize=8, ncol=1, frameon=False)
    return fig


def plot_pseudo_perplexity(
    model_keys: list[str],
    datasets: str | list[str] = "all",
) -> plt.Figure:
    """Violin plots of per-sequence CDR-H3 pseudo-perplexity for each model.

    Pseudo-perplexity PP = exp(-PLL / L) where L is the CDR-H3 sequence length.
    Lower PP means the model assigns higher average probability to each residue.
    Sequences are pooled across the selected datasets (deduped by sequence string).
    Missing artifacts for a model are skipped with a warning.
    """
    ds_keys = registry.dataset_keys("all") if datasets == "all" else (
        [datasets] if isinstance(datasets, str) else datasets)

    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])

    # Collect perplexity values per model (pool + dedup sequences across datasets)
    labels_ordered: list[str] = []
    ppl_arrays: list[np.ndarray] = []
    colors: list[str] = []

    for i, m in enumerate(model_keys):
        frames = []
        for d in ds_keys:
            try:
                df = registry.load_pll(m, d)
                seq_col = registry.load_datasets_cfg()["datasets"][d]["seq_col"]
                tmp = df[[seq_col, "pll"]].copy()
                tmp["seq_len"] = tmp[seq_col].str.len()
                frames.append(tmp[[seq_col, "pll", "seq_len"]])
            except FileNotFoundError:
                log.warning("[%s/%s] PLL artifact missing — skipped", m, d)
        if not frames:
            continue
        pooled = pd.concat(frames).drop_duplicates(subset=[seq_col], keep="first")
        pll_vals = pooled["pll"].to_numpy(np.float64)
        seq_lens = pooled["seq_len"].to_numpy(np.float64)
        ppl = np.exp(-pll_vals / seq_lens)
        ppl = ppl[np.isfinite(ppl)]
        if ppl.size == 0:
            continue
        labels_ordered.append(_label(m))
        ppl_arrays.append(ppl)
        colors.append(_color(m, palette[i % len(palette)]))

    if not ppl_arrays:
        raise RuntimeError("No perplexity data — check that PLL artifacts exist (run preflight).")

    fig, ax = plt.subplots(figsize=(max(3.5, 1.2 * len(ppl_arrays)), 4.0),
                           constrained_layout=True)
    positions = np.arange(1, len(ppl_arrays) + 1)
    parts = ax.violinplot(ppl_arrays, positions=positions, showmedians=True,
                          showextrema=False)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c)
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels_ordered, rotation=30, ha="right")
    ax.set_ylabel("CDR-H3 pseudo-perplexity")
    ax.set_title("Per-sequence pseudo-perplexity across models\n"
                 f"(pooled from: {', '.join(ds_keys)})")
    return fig


# ── embedding probe: is the binding signal linearly decodable? ────────────────
#
# A frozen-backbone linear probe measures how accessible the binding signal is in
# a model's CDR-H3 representation: standardize, fit Ridge + kNN on the DMS *train*
# split, score Spearman/R² on the *test* split. Higher = the embedding space
# already organizes sequences by binding — a better DPO starting base. Single-
# split datasets (no train sibling, e.g. `exp`) fall back to K-fold CV.

def _load_emb(model_key: str, dataset_key: str) -> dict:
    """Load the cached embedding artifact; returns {split: {'X','y','seq'}}.

    Raises FileNotFoundError if the artifact is missing (run preflight_emb).
    """
    path = registry.artifact_path(model_key, "emb", f"{dataset_key}.npz")
    if not path.exists():
        raise FileNotFoundError(
            f"Embedding artifact missing: {path}\n"
            f"Extract it: sbatch bash_scripts/extract.sbatch --what emb "
            f"--model {model_key} --dataset {dataset_key}"
        )
    with np.load(path, allow_pickle=True) as z:
        splits = [str(s) for s in z["splits"]]
        return {s: {"X": z[f"{s}_emb"], "y": z[f"{s}_y"], "seq": z[f"{s}_seq"]}
                for s in splits}


def preflight_emb(model_keys: list[str], datasets: str | list[str] = "all"
                  ) -> list[tuple[str, str]]:
    """Report which (model, dataset) embedding artifacts are missing, with sbatch cmds."""
    ds_keys = registry.dataset_keys("all") if datasets == "all" else (
        [datasets] if isinstance(datasets, str) else datasets)
    missing = [(m, d) for m in model_keys for d in ds_keys
               if not registry.artifact_path(m, "emb", f"{d}.npz").exists()]
    if not missing:
        print(f"All emb artifacts present for {len(model_keys)} model(s) "
              f"x {len(ds_keys)} dataset(s).")
        return []
    print(f"Missing {len(missing)} emb artifact(s). To extract:\n")
    for m in sorted({m for m, _ in missing}):
        ds = ",".join(d for mm, d in missing if mm == m)
        print(f"  sbatch bash_scripts/extract.sbatch --what emb "
              f"--model {m} --dataset {ds}")
    return missing


def _probe_one(emb: dict, *, ridge_alpha: float, n_neighbors: int, cv_folds: int,
               seed: int) -> dict:
    """Fit Ridge + kNN on one model/dataset's embeddings; return probe metrics.

    Uses train→test when both splits exist (no leakage), else K-fold CV on the
    single split. Features are standardized on the fit fold only.
    """
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict, KFold

    def _ridge():
        return make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha))

    def _knn(n):
        return make_pipeline(StandardScaler(),
                             KNeighborsRegressor(n_neighbors=n, weights="distance"))

    def _spear(y, yhat):
        m = np.isfinite(y) & np.isfinite(yhat)
        if m.sum() < 2:
            return float("nan")
        return float(spearmanr(y[m], yhat[m])[0])

    if "train" in emb and "test" in emb:
        Xtr, ytr = emb["train"]["X"], emb["train"]["y"]
        Xte, yte = emb["test"]["X"], emb["test"]["y"]
        n_tr, n_te = len(ytr), len(yte)
        k = max(1, min(n_neighbors, n_tr))
        rid = _ridge().fit(Xtr, ytr)
        knn = _knn(k).fit(Xtr, ytr)
        return {"ridge_spearman": _spear(yte, rid.predict(Xte)),
                "ridge_r2": float(rid.score(Xte, yte)),
                "knn_spearman": _spear(yte, knn.predict(Xte)),
                "n_train": n_tr, "n_test": n_te, "eval": "train->test"}

    split = "all" if "all" in emb else next(iter(emb))
    X, y = emb[split]["X"], emb[split]["y"]
    n = len(y)
    folds = min(cv_folds, n)
    cv = KFold(n_splits=folds, shuffle=True, random_state=seed)
    k = max(1, min(n_neighbors, n - n // folds))
    yr = cross_val_predict(_ridge(), X, y, cv=cv)
    yk = cross_val_predict(_knn(k), X, y, cv=cv)
    ss_res = np.nansum((y - yr) ** 2)
    ss_tot = np.nansum((y - np.nanmean(y)) ** 2)
    return {"ridge_spearman": _spear(y, yr),
            "ridge_r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "knn_spearman": _spear(y, yk),
            "n_train": n, "n_test": n, "eval": f"{folds}-fold cv"}


def probe_scores(model_keys: list[str], datasets: str | list[str] = "all", *,
                 ridge_alpha: float = 10.0, n_neighbors: int = 15,
                 cv_folds: int = 5, seed: int = 0) -> pd.DataFrame:
    """Linear-probe comparison table: ridge/kNN Spearman + ridge R² per model×dataset.

    Reads the cached `emb` artifacts (run `preflight_emb` first). Returns a tidy
    DataFrame; missing artifacts are skipped with a warning.
    """
    ds_keys = registry.dataset_keys("all") if datasets == "all" else (
        [datasets] if isinstance(datasets, str) else datasets)
    rows = []
    for m in model_keys:
        for d in ds_keys:
            try:
                emb = _load_emb(m, d)
            except FileNotFoundError:
                log.warning("[%s/%s] emb artifact missing — skipped", m, d)
                continue
            rows.append({"model": m, "label": _label(m), "dataset": d,
                         **_probe_one(emb, ridge_alpha=ridge_alpha,
                                      n_neighbors=n_neighbors, cv_folds=cv_folds,
                                      seed=seed)})
    return pd.DataFrame(rows)


def plot_probe_spearman(model_keys: list[str], datasets: str | list[str] = "all",
                        *, metric: str = "ridge_spearman",
                        df: pd.DataFrame | None = None,
                        ax: plt.Axes | None = None, **probe_kwargs) -> plt.Figure:
    """Grouped bars: linear-probe Spearman per dataset, one bar per model.

    `metric` is a column of `probe_scores` ('ridge_spearman' default, or
    'knn_spearman'). Higher = binding signal more linearly decodable from the
    frozen embedding → better DPO starting base. Pass a precomputed `df` (from
    `probe_scores`) to avoid recomputing the probe (expensive on large splits).
    """
    ds_keys = registry.dataset_keys("all") if datasets == "all" else (
        [datasets] if isinstance(datasets, str) else datasets)
    df = probe_scores(model_keys, ds_keys, **probe_kwargs) if df is None else df
    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])

    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(
        figsize=(1.6 + 1.1 * len(ds_keys), 4.2), constrained_layout=True)
    n_models = len(model_keys)
    width = 0.8 / max(n_models, 1)
    x = np.arange(len(ds_keys))
    for i, m in enumerate(model_keys):
        vals = [df.loc[(df.model == m) & (df.dataset == d), metric].squeeze()
                if not df.loc[(df.model == m) & (df.dataset == d)].empty else np.nan
                for d in ds_keys]
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, vals, width,
               label=_label(m), color=_color(m, palette[i % len(palette)]))

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(ds_keys, rotation=30, ha="right")
    pretty = {"ridge_spearman": "ridge", "knn_spearman": "kNN"}.get(metric, metric)
    ax.set_ylabel(rf"Probe Spearman $\rho$ ({pretty})")
    ax.set_title("Linear probe on frozen CDR-H3 embeddings\n"
                 "(fit on DMS train, scored on test)")
    ax.legend(fontsize=8, ncol=1, frameon=False)
    return fig


def plot_embedding_pca(model_keys: list[str], datasets: str | list[str] = "all",
                       *, ncols: int = 3, max_points_per_panel: int = 50_000,
                       dpi: int = 200) -> plt.Figure:
    """PCA(2) of frozen CDR-H3 embeddings per model, colored by enrichment.

    Illustrative figure: one panel per model, PCA fit on that model's pooled
    embeddings (all splits of the selected datasets), points colored by the
    experimental enrichment. The quantitative claim lives in `probe_scores` /
    `plot_probe_spearman`; this just visualizes the organization.

    Each panel can hold ~1.5M embeddings; drawing those as vector points yields
    a multi-tens-of-MB PDF that hangs viewers. To stay small and faithful we
    subsample to `max_points_per_panel` points (fixed seed) BEFORE both the PCA
    fit and the scatter, and rasterize the scatter layer (axes/labels/colorbar
    stay vector). `dpi` controls the rasterized layer's resolution.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    ds_keys = registry.dataset_keys("all") if datasets == "all" else (
        [datasets] if isinstance(datasets, str) else datasets)

    rng = np.random.default_rng(0)
    panels = []
    for m in model_keys:
        Xs, ys = [], []
        for d in ds_keys:
            try:
                emb = _load_emb(m, d)
            except FileNotFoundError:
                log.warning("[%s/%s] emb artifact missing — skipped", m, d)
                continue
            for s in emb.values():
                Xs.append(s["X"])
                ys.append(s["y"])
        if not Xs:
            continue
        X = np.concatenate(Xs)
        y = np.concatenate(ys)
        if len(y) > max_points_per_panel:
            idx = rng.choice(len(y), size=max_points_per_panel, replace=False)
            X, y = X[idx], y[idx]
        coords = PCA(n_components=2, random_state=0).fit_transform(
            StandardScaler().fit_transform(X))
        panels.append((m, coords, y))

    if not panels:
        raise RuntimeError("No embedding data — run preflight_emb / extract first.")

    finite_y = np.concatenate([y[np.isfinite(y)] for _, _, y in panels])
    vmin, vmax = np.percentile(finite_y, [2, 98])

    n = len(panels)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.6 * nrows),
                             constrained_layout=True, squeeze=False)
    fig.set_dpi(dpi)
    sc = None
    for ax, (m, coords, y) in zip(axes.flat, panels):
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=y, cmap="viridis",
                        vmin=vmin, vmax=vmax, s=8, alpha=0.7, rasterized=True)
        ax.set_title(_label(m), fontsize=10)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    if sc is not None:
        fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.8, label="enrichment")
    fig.suptitle("CDR-H3 embedding PCA, colored by binding enrichment")
    return fig


# ── learning curve: probe quality vs amount of supervision ───────────────────
#
# The full-data probe (probe_scores) saturates — with 200k-340k labelled train
# examples a linear map extracts the binding signal regardless of how the backbone
# was tuned, so representation differences wash out. The regime that matters for a
# DPO *starting base* is small-data: how few labels each model needs to reach a
# given probe quality. The learning curve subsamples the train split to increasing
# sizes (fixed test split, repeated seeds) and is where evo-vs-vanilla and the
# best.pt-vs-best_spearman drift signature can actually separate.

DEFAULT_TRAIN_SIZES = (25, 50, 100, 200, 500, 1000, 2000, 5000)


def _train_test_xy(emb: dict, *, holdout_frac: float, seed: int):
    """Return (Xtr, ytr, Xte, yte) with finite y; carve a holdout for single-split."""
    if "train" in emb and "test" in emb:
        Xtr, ytr = emb["train"]["X"], emb["train"]["y"]
        Xte, yte = emb["test"]["X"], emb["test"]["y"]
    else:
        split = "all" if "all" in emb else next(iter(emb))
        X, y = emb[split]["X"], emb[split]["y"]
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(y))
        n_te = int(round(holdout_frac * len(y)))
        te, tr = perm[:n_te], perm[n_te:]
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

    def _finite(X, y):
        m = np.isfinite(y)
        return X[m], y[m]

    Xtr, ytr = _finite(Xtr, ytr)
    Xte, yte = _finite(Xte, yte)
    return Xtr, ytr, Xte, yte


def probe_learning_curve(model_keys: list[str], datasets: str | list[str] = "all", *,
                         train_sizes=DEFAULT_TRAIN_SIZES, n_repeats: int = 5,
                         ridge_alpha: float = 10.0, seed: int = 0,
                         holdout_frac: float = 0.2) -> pd.DataFrame:
    """Probe Spearman vs train-set size: ridge fit on a random train subset, scored
    on the fixed test split, averaged over `n_repeats` seeds.

    Returns a tidy DataFrame (model, dataset, n_train, ridge_spearman_mean/std,
    n_repeats). Reads cached `emb` artifacts; missing ones are skipped.
    """
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ds_keys = registry.dataset_keys("all") if datasets == "all" else (
        [datasets] if isinstance(datasets, str) else datasets)

    def _spear(y, yhat):
        m = np.isfinite(yhat)
        return float(spearmanr(y[m], yhat[m])[0]) if m.sum() >= 2 else float("nan")

    rows = []
    for m in model_keys:
        for d in ds_keys:
            try:
                emb = _load_emb(m, d)
            except FileNotFoundError:
                log.warning("[%s/%s] emb artifact missing — skipped", m, d)
                continue
            Xtr, ytr, Xte, yte = _train_test_xy(emb, holdout_frac=holdout_frac,
                                                seed=seed)
            n_avail = len(ytr)
            for s in train_sizes:
                if s > n_avail:
                    continue
                scores = []
                for r in range(n_repeats):
                    rng = np.random.default_rng(seed + r)
                    idx = rng.choice(n_avail, size=s, replace=False)
                    pipe = make_pipeline(StandardScaler(), Ridge(alpha=ridge_alpha))
                    pipe.fit(Xtr[idx], ytr[idx])
                    scores.append(_spear(yte, pipe.predict(Xte)))
                scores = np.array(scores, dtype=float)
                rows.append({"model": m, "label": _label(m), "dataset": d,
                             "n_train": s,
                             "ridge_spearman_mean": float(np.nanmean(scores)),
                             "ridge_spearman_std": float(np.nanstd(scores)),
                             "n_repeats": n_repeats})
    return pd.DataFrame(rows)


def plot_learning_curve(model_keys: list[str], datasets: str | list[str] = "all",
                        *, df: pd.DataFrame | None = None, ncols: int = 3,
                        **curve_kwargs) -> plt.Figure:
    """Probe Spearman vs train-set size, one panel per dataset, one line per model.

    Pass a precomputed `df` (from `probe_learning_curve`) to avoid recomputing.
    A model whose line sits higher at small n_train organizes binding more
    accessibly = better small-data / DPO starting base.
    """
    ds_keys = registry.dataset_keys("all") if datasets == "all" else (
        [datasets] if isinstance(datasets, str) else datasets)
    if df is None:
        df = probe_learning_curve(model_keys, ds_keys, **curve_kwargs)
    present = [d for d in ds_keys if d in set(df.dataset)]
    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])

    n = len(present)
    ncols = min(ncols, max(n, 1))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows),
                             constrained_layout=True, squeeze=False, sharex=True)
    for ax, d in zip(axes.flat, present):
        for i, mk in enumerate(model_keys):
            sub = df[(df.model == mk) & (df.dataset == d)].sort_values("n_train")
            if sub.empty:
                continue
            c = _color(mk, palette[i % len(palette)])
            ax.plot(sub.n_train, sub.ridge_spearman_mean, marker="o", ms=4,
                    color=c, label=_label(mk))
            ax.fill_between(sub.n_train,
                            sub.ridge_spearman_mean - sub.ridge_spearman_std,
                            sub.ridge_spearman_mean + sub.ridge_spearman_std,
                            color=c, alpha=0.15)
        ax.set_xscale("log")
        ax.set_title(d, fontsize=10)
        ax.set_xlabel("train size")
        ax.set_ylabel(r"probe Spearman $\rho$")
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Probe learning curve — Spearman vs amount of supervision\n"
                 "(ridge on random train subset, scored on fixed test)")
    return fig
