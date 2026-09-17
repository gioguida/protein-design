import pytest
from pathlib import Path

from protein_design.analysis import report_data


def test_preference_summary_requires_all_three_metrics() -> None:
    complete = {
        "test_loss": 0.23,
        "test_reward_accuracy": 0.71,
        "test_reward_margin": 0.18,
    }
    assert report_data.complete_preference_summary(complete)
    assert not report_data.complete_preference_summary({**complete, "test_reward_margin": None})
    assert not report_data.complete_preference_summary({"test_loss": 0.23})


def test_nearest_dms_proxy_means_tied_neighbours() -> None:
    reference = [
        {"sequence": "AAA", "enrichment": 1.0},
        {"sequence": "ABB", "enrichment": 5.0},
        {"sequence": "CCC", "enrichment": -1.0},
    ]
    # AAB is Hamming distance one from AAA and ABB, so the proxy is their mean.
    assert report_data.nearest_dms_proxy(["AAB"], reference) == [pytest.approx(3.0)]


def test_library_statistics_uses_requested_evaluator() -> None:
    rows = [
        {"sequence": "AAA", "n_mutations": 1, "native_pll": -1.0, "common_esm2_pll": -3.0,
         "wt_native_pll": -2.0, "wt_common_esm2_pll": -2.0, "present_in_training_reference": False},
        {"sequence": "AAB", "n_mutations": 2, "native_pll": -4.0, "common_esm2_pll": -1.0,
         "wt_native_pll": -2.0, "wt_common_esm2_pll": -2.0, "present_in_training_reference": True},
    ]
    reference = [{"sequence": "AAA", "enrichment": 1.0}, {"sequence": "AAB", "enrichment": 3.0}]
    native = report_data.library_statistics(rows, evaluator="native", reference=reference, top_k=1)
    common = report_data.library_statistics(rows, evaluator="common", reference=reference, top_k=1)
    assert native["top_sequences"] == ["AAA"]
    assert common["top_sequences"] == ["AAB"]
    assert native["fraction_above_wt"] == pytest.approx(0.5)


def test_json_conversion_normalizes_numpy_values() -> None:
    payload = report_data._json_value({"integer": __import__("numpy").int64(3), "float": __import__("numpy").float64(1.5)})
    assert payload == {"integer": 3, "float": 1.5}


def _work_unit_config():
    return {
        "output": {"data_dir": str(Path("__nonexistent_report_plot_data__"))},
        "models": {
            "order": ["vanilla_650m", "evo_650m"],
            "preference_models": ["just_dpo_650m"],
            "generation_models": ["vanilla_650m"],
        },
        "execution": {"profiles": {"rtx_4090": {}, "a100_80gb": {}}},
    }


def test_collection_commands_fan_out_models_and_gpu_classes() -> None:
    config = _work_unit_config()
    commands = report_data.collection_commands(config)
    text = "\n".join(command for _, group in commands for command in group)
    assert "functional-model --model vanilla_650m" in text
    assert "preference-model --model just_dpo_650m" in text
    assert "report_plot_data_4090.sbatch" in text
    assert "generation-sample --model vanilla_650m --sampler gibbs" in text
    assert "report_plot_data_a100_80gb.sbatch" in text
    assert "generation-baselines" in text


def test_preference_reducer_requires_every_model_part() -> None:
    config = _work_unit_config()
    with pytest.raises(FileNotFoundError, match="preference_just_dpo_650m"):
        report_data.reduce_preference_metrics(config)


def test_execution_profile_rejects_unknown_name() -> None:
    config = _work_unit_config()
    with pytest.raises(ValueError, match="Unknown report execution profile"):
        report_data.execution_profile(config, "unsupported")
