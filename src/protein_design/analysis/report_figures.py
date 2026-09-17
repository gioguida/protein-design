"""Publication-style, notebook-facing figures for the final report."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter

from . import registry
from . import report_data

SINGLE_COL_WIDTH = 3.35
DOUBLE_COL_WIDTH = 7.0
AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA_COLORS = {aa: color for aa, color in zip(AA20, plt.get_cmap("tab20").colors)}
SAMPLER_MARKERS = {"random": "X", "pssm": "s", "gibbs": "o", "stochastic_beam": "^"}
SAMPLER_LABELS = {"random": "Random", "pssm": "PSSM", "gibbs": "Gibbs", "stochastic_beam": "Stochastic beam"}


def apply_report_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Computer Modern Roman", "Latin Modern Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm", "text.usetex": False, "font.size": 10.5,
        "axes.titlesize": 12.5, "axes.labelsize": 11.5, "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5, "legend.fontsize": 9.0, "figure.titlesize": 13.0,
        "axes.linewidth": 0.8, "axes.spines.top": True, "axes.spines.right": True,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 3.5,
        "ytick.major.size": 3.5, "xtick.direction": "out", "ytick.direction": "out",
        "lines.linewidth": 1.8, "lines.markersize": 5.0, "legend.frameon": True,
        "legend.framealpha": 0.92, "legend.fancybox": False, "legend.edgecolor": "0.80",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    })


def style_axes(ax: plt.Axes, *, grid: bool = True) -> None:
    ax.tick_params(axis="both", which="major", width=0.8, length=3.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis="y", color="0.90", linewidth=0.6, zorder=0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top", ha="right")


def save_figure(fig: plt.Figure, output_dir: str | Path, stem: str, *, dpi: int = 500) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pdf, png = output / f"{stem}.pdf", output / f"{stem}.png"
    mpl.rcParams["pdf.fonttype"] = 42
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    return pdf, png


def model_label(model_key: str) -> str:
    return registry.load_models().get(model_key, {}).get("label", model_key)


def model_color(model_key: str) -> str:
    return registry.load_models().get(model_key, {}).get("color", "#444444")


def preflight(config: dict[str, Any], sections: Iterable[str] | None = None) -> list[Path]:
    missing = report_data.missing_artifacts(config, sections)
    if not missing:
        print("All report JSON artifacts are available.")
        return []
    print("Missing report artifacts:")
    for path in missing:
        print("  ", path)
    print("\nRun on the cluster (the notebook never submits jobs):")
    for heading, commands in report_data.collection_commands(config, sections):
        print(f"\n{heading}:")
        for command in commands:
            print(f"  {command}")
    return missing


def _functional(config: dict[str, Any]) -> dict[str, Any]:
    return report_data.load_artifact(report_data.artifact_path(config, "functional_metrics.json"))


def _preference(config: dict[str, Any]) -> dict[str, Any]:
    return report_data.load_artifact(report_data.artifact_path(config, "preference_test_metrics.json"))


def _reference(config: dict[str, Any]) -> dict[str, Any]:
    return report_data.load_artifact(report_data.artifact_path(config, "generation_reference_ed2_m22_test.json"))


def _library(config: dict[str, Any], model_key: str) -> dict[str, Any]:
    return report_data.load_artifact(report_data.artifact_path(config, f"generation_library_{model_key}.json"))


def plot_evotune_functional(config: dict[str, Any]) -> plt.Figure:
    data = _functional(config)["models"]
    datasets = config["datasets"]["functional"]
    models = ["vanilla_650m", "evo_650m"]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_WIDTH, 2.7), sharey=True, constrained_layout=True)
    for i, (ax, dataset) in enumerate(zip(axes, datasets)):
        values = [data[model]["datasets"][dataset]["spearman_pll_enrichment"] for model in models]
        ax.plot(range(2), values, color="0.55", lw=1.2, zorder=2)
        for x, model, value in zip(range(2), models, values):
            ax.scatter(x, value, s=36, color=model_color(model), zorder=3)
        ax.axhline(0, color="0.45", lw=0.8, ls="--")
        ax.set_xticks(range(2), [model_label(model) for model in models], rotation=20, ha="right")
        ax.set_title(dataset.replace("_m22", "").upper())
        style_axes(ax)
        add_panel_label(ax, "abc"[i])
    axes[0].set_ylabel(r"Spearman $\rho$ (PLL vs enrichment)")
    return fig


def plot_cdr_pseudo_perplexity(config: dict[str, Any]) -> plt.Figure:
    data = _functional(config)["models"]
    models = ["vanilla_650m", "evo_650m"]
    values = [data[model]["datasets"]["ed2_m22"]["cdr_pseudo_perplexity"] for model in models]
    fig, ax = plt.subplots(figsize=(SINGLE_COL_WIDTH, 2.6), constrained_layout=True)
    ax.plot(range(2), values, color="0.55", lw=1.2, zorder=2)
    for x, model, value in zip(range(2), models, values):
        ax.scatter(x, value, s=42, color=model_color(model), zorder=3)
    ax.set_xticks(range(2), [model_label(model) for model in models], rotation=20, ha="right")
    ax.set_ylabel("CDR-H3 pseudo-perplexity")
    ax.set_title("Held-out ED2 naturalness")
    style_axes(ax)
    return fig


def plot_preference_metrics(config: dict[str, Any]) -> plt.Figure:
    data = _preference(config)["models"]
    models = config["models"]["preference_models"]
    metrics = [("test_reward_accuracy", "Reward accuracy", 0.5), ("test_reward_margin", "Reward margin", None), ("test_loss", "DPO test loss", None)]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_WIDTH, 2.8), constrained_layout=True)
    for i, (ax, (key, label, baseline)) in enumerate(zip(axes, metrics)):
        values = [data[model][key] for model in models]
        for x, model, value in zip(range(len(models)), models, values):
            ax.scatter(x, value, color=model_color(model), s=42, zorder=3)
        if baseline is not None:
            ax.axhline(baseline, color="0.45", lw=1.0, ls="--", label="chance")
        ax.set_xticks(range(len(models)), [model_label(model) for model in models], rotation=35, ha="right")
        ax.set_ylabel(label)
        style_axes(ax)
        add_panel_label(ax, "abc"[i])
    return fig


def plot_functional_groups(config: dict[str, Any], groups: list[list[str]], *, title: str) -> plt.Figure:
    data = _functional(config)["models"]
    datasets = config["datasets"]["functional"]
    fig, axes = plt.subplots(1, len(groups), figsize=(DOUBLE_COL_WIDTH, 2.9), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for i, (ax, group) in enumerate(zip(axes, groups)):
        x = np.arange(len(datasets))
        width = 0.75 / len(group)
        for j, model in enumerate(group):
            values = [data[model]["datasets"][dataset]["spearman_pll_enrichment"] for dataset in datasets]
            ax.bar(x + (j - (len(group) - 1) / 2) * width, values, width, label=model_label(model), color=model_color(model), edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="0.45", lw=0.8, ls="--")
        ax.set_xticks(x, [dataset.replace("_m22", "").upper() for dataset in datasets])
        ax.set_title(title if len(groups) == 1 else ("Base ESM2" if i == 0 else "Evo-tuned base"))
        style_axes(ax)
        add_panel_label(ax, "ab"[i])
    axes[0].set_ylabel(r"Spearman $\rho$ (PLL vs enrichment)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=len(handles), frameon=False)
    return fig


def plot_all_models_functional(config: dict[str, Any]) -> plt.Figure:
    return plot_functional_groups(config, [config["models"]["order"]], title="All 650M models")


def _generation_summaries(config: dict[str, Any], evaluator: str) -> list[dict[str, Any]]:
    rows = []
    key = "native" if evaluator == "native" else "common_esm2"
    for model in config["models"]["generation_models"]:
        library = _library(config, model)
        for sampler, payload in library["samplers"].items():
            rows.append({"model": model, "sampler": sampler, **payload[key]})
    return rows


def plot_generation_quality_diversity(config: dict[str, Any]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_WIDTH, 3.0), constrained_layout=True)
    for i, (ax, evaluator, title) in enumerate(zip(axes, ["native", "common"], ["Native-model PLL", "Common ESM2-650M PLL"])):
        for row in _generation_summaries(config, evaluator):
            ax.scatter(row["mean_pairwise_hamming"], row["mean_top_k_pll"], s=50,
                       marker=SAMPLER_MARKERS[row["sampler"]], color=model_color(row["model"]),
                       edgecolor="white", linewidth=0.5, zorder=3)
        ax.set_xlabel("Mean pairwise Hamming distance")
        ax.set_ylabel(r"Mean top-$50$ PLL")
        ax.set_title(title)
        style_axes(ax)
        add_panel_label(ax, "ab"[i])
    model_handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=model_color(model), label=model_label(model), markersize=6) for model in config["models"]["generation_models"]]
    sampler_handles = [Line2D([0], [0], marker=SAMPLER_MARKERS[sampler], color="0.25", lw=0, label=SAMPLER_LABELS[sampler]) for sampler in config["generation"]["samplers"]]
    fig.legend(model_handles + sampler_handles, [handle.get_label() for handle in model_handles + sampler_handles], loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=4, frameon=False)
    return fig


def plot_generation_mutation_distance(config: dict[str, Any]) -> plt.Figure:
    models = config["models"]["generation_models"]
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_WIDTH, 5.5), sharey=True, constrained_layout=True)
    for i, (ax, model) in enumerate(zip(axes.flat, models)):
        library = _library(config, model)
        arrays = [np.asarray([row["n_mutations"] for row in library["samplers"][sampler]["rows"]]) for sampler in config["generation"]["samplers"]]
        violin = ax.violinplot(arrays, showmedians=True, showextrema=False)
        for body, sampler in zip(violin["bodies"], config["generation"]["samplers"]):
            body.set_facecolor("#B0B0B0")
            body.set_edgecolor("white")
            body.set_alpha(0.8)
        ax.set_xticks(range(1, len(arrays) + 1), [SAMPLER_LABELS[s] for s in config["generation"]["samplers"]], rotation=25, ha="right")
        ax.set_title(model_label(model))
        ax.set_xlabel("Sampler")
        style_axes(ax)
        add_panel_label(ax, "abcd"[i])
    axes[0, 0].set_ylabel("Mutation count from WT")
    axes[1, 0].set_ylabel("Mutation count from WT")
    return fig


def _summary_table_rows(config: dict[str, Any], evaluator: str) -> list[list[str]]:
    rows = []
    for row in _generation_summaries(config, evaluator):
        rows.append([model_label(row["model"]), SAMPLER_LABELS[row["sampler"]],
                     f"{100 * row['fraction_above_wt']:.1f}\\%", f"{100 * row['novelty']:.1f}\\%",
                     f"{row['mean_pairwise_hamming']:.2f}", f"{row['median_mutation_count']:.1f}",
                     f"{row['top_k_nearest_dms_enrichment']:.2f}"])
    return rows


def save_latex_table(output_dir: str | Path, stem: str, source: str) -> Path:
    """Write a complete, report-ready LaTeX table as a text artifact."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{stem}.txt"
    path.write_text(source.rstrip() + "\n", encoding="utf-8")
    return path


def _latex_table(headers: list[str], rows: list[list[str]], *, alignment: str, caption: str, label: str) -> str:
    lines = [r"\begin{table}[t]", r"\centering", r"\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{alignment}}}", r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def generation_summary_table_latex(config: dict[str, Any], evaluator: str) -> str:
    evaluator_name = "native-model PLL" if evaluator == "native" else "common ESM2-650M PLL"
    return _latex_table(
        ["Model", "Sampler", "Above WT", "Novel", "Pairwise HD", "Median mutations", "Nearest-DMS enrichment"],
        _summary_table_rows(config, evaluator), alignment="llrrrrr",
        caption=f"Generated-library summary ranked by {evaluator_name}.",
        label=f"tab:generation-summary-{evaluator}",
    )


def write_generation_summary_table(config: dict[str, Any], evaluator: str, output_dir: str | Path) -> Path:
    suffix = "native_pll" if evaluator == "native" else "common_esm2_pll"
    return save_latex_table(output_dir, f"generation_library_summary_{suffix}_table",
                            generation_summary_table_latex(config, evaluator))


def _frequencies(sequences: list[str]) -> np.ndarray:
    array = np.asarray([list(sequence) for sequence in sequences])
    out = np.zeros((len(AA20), len(sequences[0])), dtype=float)
    for i, aa in enumerate(AA20):
        out[i] = (array == aa).mean(axis=0)
    return out


def _draw_logo(ax: plt.Axes, sequences: list[str], title: str) -> None:
    freq = _frequencies(sequences)
    bottom = np.zeros(freq.shape[1])
    for i, aa in enumerate(AA20):
        if np.any(freq[i]):
            ax.bar(np.arange(freq.shape[1]), freq[i], bottom=bottom, width=0.95, color=AA_COLORS[aa], linewidth=0)
            bottom += freq[i]
    ax.set_xlim(-0.5, freq.shape[1] - 0.5)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([0, 1], ["0", "1"], fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _winner(config: dict[str, Any], model: str, evaluator: str) -> tuple[str, list[str]]:
    key = "native" if evaluator == "native" else "common_esm2"
    library = _library(config, model)["samplers"]
    sampler, payload = max(library.items(), key=lambda item: float(item[1][key]["mean_top_k_pll"]))
    return sampler, list(payload[key]["top_sequences"])


def plot_generation_logos(config: dict[str, Any], evaluator: str) -> plt.Figure:
    reference = _reference(config)
    positives = [row["sequence"] for row in reference["sequences"] if float(row["enrichment"]) > float(reference["wild_type_enrichment"])]
    models = config["models"]["generation_models"]
    fig = plt.figure(figsize=(DOUBLE_COL_WIDTH, 6.4), constrained_layout=True)
    fig.suptitle("Three-way CDR-H3 composition", fontsize=12)
    outer = fig.add_gridspec(2, 2)
    for i, model in enumerate(models):
        sub = outer[i // 2, i % 2].subgridspec(3, 1, hspace=0.15)
        sampler, sequences = _winner(config, model, evaluator)
        for j, (source, seqs) in enumerate((("WT", [reference["wild_type"]]), ("ED2 positives", positives), (f"{SAMPLER_LABELS[sampler]} top-50", sequences))):
            ax = fig.add_subplot(sub[j])
            _draw_logo(ax, seqs, source)
            if j == 0:
                ax.text(-0.12, 1.07, "abcd"[i], transform=ax.transAxes, fontweight="bold", fontsize=14)
                ax.text(0.02, 1.07, model_label(model), transform=ax.transAxes, fontsize=10, fontweight="bold")
    return fig


def _jsd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    eps = 1e-12
    p, q = np.clip(p, eps, 1), np.clip(q, eps, 1)
    mid = 0.5 * (p + q)
    return 0.5 * ((p * np.log2(p / mid)).sum(axis=0) + (q * np.log2(q / mid)).sum(axis=0))


def plot_generation_jsd(config: dict[str, Any], evaluator: str) -> plt.Figure:
    reference = _reference(config)
    positives = [row["sequence"] for row in reference["sequences"] if float(row["enrichment"]) > float(reference["wild_type_enrichment"])]
    ref_freq = _frequencies(positives)
    models = config["models"]["generation_models"]
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_WIDTH, 4.8), sharex=True, sharey=True, constrained_layout=True)
    for i, (ax, model) in enumerate(zip(axes.flat, models)):
        sampler, sequences = _winner(config, model, evaluator)
        ax.bar(np.arange(len(sequences[0])), _jsd(_frequencies(sequences), ref_freq), color=model_color(model), width=0.8)
        ax.set_title(f"{model_label(model)} — {SAMPLER_LABELS[sampler]}")
        ax.set_xlabel("CDR-H3 position")
        ax.set_ylabel("JSD")
        style_axes(ax)
        add_panel_label(ax, "abcd"[i])
    return fig


def all_model_table_latex(config: dict[str, Any]) -> str:
    functional = _functional(config)["models"]
    preference = _preference(config)["models"]
    rows = []
    for model in config["models"]["order"]:
        pref = preference.get(model, {})
        values = [model_label(model),
                  r"\textemdash{}" if not pref else f"{pref['test_reward_accuracy']:.3f}",
                  r"\textemdash{}" if not pref else f"{pref['test_reward_margin']:.3f}",
                  f"{functional[model]['datasets']['ed2_m22']['cdr_pseudo_perplexity']:.2f}"]
        values += [f"{functional[model]['datasets'][dataset]['spearman_pll_enrichment']:.3f}" for dataset in config["datasets"]["functional"]]
        rows.append(values)
    return _latex_table(
        ["Model", "Reward acc.", "Reward margin", "CDR PPL", "ED2 $\\rho$", "ED5 $\\rho$", "ED8--11 $\\rho$"],
        rows, alignment="lrrrrrr", caption="Model summary on held-out evaluation sets.",
        label="tab:all-model-summary",
    )


def write_all_model_table(config: dict[str, Any], output_dir: str | Path) -> Path:
    return save_latex_table(output_dir, "all_models_compact_summary_table", all_model_table_latex(config))
