from pathlib import Path

import pandas as pd

from scripts.plot_dpo_metrics import _resolve_metric_series, discover_run_artifacts


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_discover_run_artifacts_prefers_live_run_directory(tmp_path: Path) -> None:
    train_root = tmp_path / "train"
    archive_root = tmp_path / "checkpoints"
    run_name = "esm2-35M__delta_based-c__weighted_dpo__loss-1__ep-15__bs-16__lr-1e-05__beta-0.04_20260420_231634"

    live_run_dir = train_root / run_name
    archive_run_dir = archive_root / run_name
    live_run_dir.mkdir(parents=True)
    archive_run_dir.mkdir(parents=True)

    _write_csv(
        live_run_dir / "history.csv",
        [
            {"epoch": 1, "train_loss": 1.0, "val_loss": 2.0},
            {"epoch": 2, "train_loss": 0.8, "val_loss": 1.6},
        ],
    )
    (live_run_dir / "summary.json").write_text('{"run_name": "' + run_name + '", "test_loss": 0.5}', encoding="utf-8")
    (live_run_dir / "metrics.json").write_text('{"run_name": "' + run_name + '", "val_ppl": 12.3}', encoding="utf-8")
    (archive_run_dir / "metrics.json").write_text('{"run_name": "' + run_name + '", "val_ppl": 99.9}', encoding="utf-8")

    artifacts = discover_run_artifacts(["20260420_231634"], train_root=train_root, archive_root=archive_root)

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.run_dir == live_run_dir
    assert artifact.archive_dir == archive_run_dir
    assert artifact.history_path == live_run_dir / "history.csv"
    assert artifact.summary_path == live_run_dir / "summary.json"
    assert artifact.metrics_path == live_run_dir / "metrics.json"
    assert artifact.metrics["val_ppl"] == 12.3
    assert artifact.summary["test_loss"] == 0.5


def test_discover_run_artifacts_falls_back_to_archive_metrics(tmp_path: Path) -> None:
    train_root = tmp_path / "train"
    archive_root = tmp_path / "checkpoints"
    run_name = "esm2-35M__delta_based-c__weighted_dpo__loss-1__ep-15__bs-16__lr-1e-05__beta-0.04_20260421_080923"

    live_run_dir = train_root / run_name
    archive_run_dir = archive_root / run_name
    live_run_dir.mkdir(parents=True)
    archive_run_dir.mkdir(parents=True)

    _write_csv(live_run_dir / "history.csv", [{"epoch": 1, "train_loss": 1.0}])
    (live_run_dir / "summary.json").write_text('{"run_name": "' + run_name + '", "test_loss": 0.4}', encoding="utf-8")
    (archive_run_dir / "metrics.json").write_text('{"run_name": "' + run_name + '", "val_ppl": 11.0}', encoding="utf-8")

    artifacts = discover_run_artifacts(["20260421_080923"], train_root=train_root, archive_root=archive_root)

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.metrics_path == archive_run_dir / "metrics.json"
    assert artifact.metrics["val_ppl"] == 11.0


def test_discover_run_artifacts_uses_custom_run_labels(tmp_path: Path) -> None:
    train_root = tmp_path / "train"
    run_name = "esm2-35M__delta_based-c__weighted_dpo__loss-1__ep-15__bs-16__lr-1e-05__beta-0.04_20260421_111111"

    run_dir = train_root / run_name
    run_dir.mkdir(parents=True)
    _write_csv(run_dir / "history.csv", [{"epoch": 1, "train_loss": 1.0}])
    (run_dir / "summary.json").write_text('{"run_name": "' + run_name + '"}', encoding="utf-8")

    artifacts = discover_run_artifacts(
        ["20260421_111111"],
        train_root=train_root,
        run_labels=["baseline-dpo"],
    )

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.display_name == "baseline-dpo"


def test_resolve_metric_series_supports_slash_validation_metrics() -> None:
    history = pd.DataFrame(
        {
            "epoch": [1, 2],
            "ppl/val_wt": [20.0, 19.5],
        }
    )

    resolved = _resolve_metric_series(history, value_prefix="val", metric="ppl/val_wt")

    assert resolved is not None
    column, epochs, values = resolved
    assert column == "ppl/val_wt"
    assert epochs.tolist() == [1, 2]
    assert values.tolist() == [20.0, 19.5]


def test_resolve_metric_series_prefers_prefixed_standard_metrics() -> None:
    history = pd.DataFrame(
        {
            "epoch": [1, 2],
            "val_loss": [0.9, 0.8],
            "loss": [9.0, 8.0],
        }
    )

    resolved = _resolve_metric_series(history, value_prefix="val", metric="loss")

    assert resolved is not None
    column, _epochs, values = resolved
    assert column == "val_loss"
    assert values.tolist() == [0.9, 0.8]