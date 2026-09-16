from unittest.mock import Mock
from pathlib import Path

import matplotlib.pyplot as plt

from protein_design.analysis import report_figures


MODELS = ["vanilla_650m", "evo_650m", "just_dpo_650m", "evo_dpo_650m"]
SAMPLERS = ["random", "pssm", "gibbs", "stochastic_beam"]


def _config():
    return {
        "datasets": {"functional": ["ed2_m22", "ed5_m22", "ed811_m22"]},
        "models": {"order": [*MODELS, "just_lora_dpo_650m", "evo_lora_dpo_650m"],
                   "preference_models": ["just_dpo_650m", "evo_dpo_650m", "just_lora_dpo_650m", "evo_lora_dpo_650m"],
                   "generation_models": MODELS},
        "generation": {"samplers": SAMPLERS},
    }


def _functional():
    datasets = {dataset: {"spearman_pll_enrichment": 0.1, "cdr_pseudo_perplexity": 3.0} for dataset in _config()["datasets"]["functional"]}
    return {"models": {model: {"datasets": datasets} for model in _config()["models"]["order"]}}


def _library(model):
    sequences = ["HMSMQQVVSAGWERADLVGDAFDV", "HMSMQQVVSAGWERADLVGDAFDA"]
    rows = [{"sequence": sequence, "n_mutations": index, "native_pll": -2.0 + index,
             "common_esm2_pll": -3.0 + index, "wt_native_pll": -2.0,
             "wt_common_esm2_pll": -3.0, "present_in_training_reference": False}
            for index, sequence in enumerate(sequences)]
    summary = {"fraction_above_wt": 0.5, "novelty": 1.0, "mean_pairwise_hamming": 1.0,
               "median_mutation_count": 0.5, "top_k_nearest_dms_enrichment": 2.0,
               "mean_top_k_pll": -1.0, "top_sequences": sequences}
    return {"model": model, "samplers": {sampler: {"rows": rows, "native": {**summary, "evaluator": "native", "top_k": 2},
                                                        "common_esm2": {**summary, "evaluator": "common", "top_k": 2}}
                                           for sampler in SAMPLERS}}


def test_report_figures_render_from_artifact_fixtures(monkeypatch) -> None:
    config = _config()
    monkeypatch.setattr(report_figures, "_functional", lambda _: _functional())
    monkeypatch.setattr(report_figures, "_preference", lambda _: {"models": {model: {"test_loss": 0.2, "test_reward_accuracy": 0.7, "test_reward_margin": 0.1} for model in config["models"]["order"][2:]}})
    monkeypatch.setattr(report_figures, "_reference", lambda _: {"wild_type": "HMSMQQVVSAGWERADLVGDAFDV", "wild_type_enrichment": 0.0,
                                                                     "sequences": [{"sequence": "HMSMQQVVSAGWERADLVGDAFDV", "enrichment": 1.0}, {"sequence": "HMSMQQVVSAGWERADLVGDAFDA", "enrichment": 2.0}]})
    monkeypatch.setattr(report_figures, "_library", lambda _, model: _library(model))

    figures = [
        report_figures.plot_evotune_functional(config),
        report_figures.plot_preference_metrics(config),
        report_figures.plot_generation_quality_diversity(config),
        report_figures.plot_generation_mutation_distance(config),
        report_figures.plot_generation_logos(config, "native"),
        report_figures.plot_generation_jsd(config, "common"),
    ]
    assert all(figure.axes for figure in figures)
    for figure in figures:
        plt.close(figure)


def test_save_figure_exports_matching_pdf_and_png(monkeypatch) -> None:
    fig, _ = plt.subplots()
    savefig = Mock()
    monkeypatch.setattr(fig, "savefig", savefig)
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: None)
    pdf, png = report_figures.save_figure(fig, Path("fixture-output"), "fixture")
    assert pdf.name == "fixture.pdf"
    assert png.name == "fixture.png"
    assert savefig.call_count == 2
    plt.close(fig)
