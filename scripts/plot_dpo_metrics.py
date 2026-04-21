"""Plot training, validation, and test metrics for DPO runs.

The script expects each run to live under the configured training root as
``<run_name>/`` where ``run_name`` ends with ``YYYYMMDD_HHmmss``.

It consumes the files written by ``src.train_dpo``:
- ``history.csv`` for epoch curves
- ``summary.json`` for final test metrics
- ``metrics.json`` for final validation summary metrics

Test metrics are rendered as run-comparison bar charts, which are a better fit
than histograms here because each run contributes a single scalar per metric.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:  # pragma: no cover - dependency should be installed
    raise ModuleNotFoundError(
        "matplotlib is required for plotting. Install it with `pip install matplotlib`."
    ) from exc


RUN_TIMESTAMP_RE = re.compile(r"(\d{8}_\d{6})$")

DEFAULT_TRAIN_CURVE_METRICS = ["loss", "reward_accuracy", "reward_margin", "implicit_kl"]
DEFAULT_VAL_CURVE_METRICS = [
    "loss",
    "reward_accuracy",
    "reward_margin",
    "implicit_kl",
    "perplexity",
    "spearman_avg",
    "spearman_random",
    "ppl/val_pos",
    "ppl/val_neg",
    "ppl/val_wt",
]
DEFAULT_VAL_SUMMARY_METRICS = ["val_ppl", "spearman_M22", "spearman_SI06", "spearman_exp"]
DEFAULT_TEST_SUMMARY_METRICS = [
    "test_loss",
    "test_reward_accuracy",
    "test_reward_margin",
    "test_implicit_kl",
    "test_perplexity",
]


@dataclass
class RunArtifacts:
    run_name: str
    display_name: str
    timestamp: str
    run_dir: Path
    archive_dir: Path | None
    history_path: Path | None
    summary_path: Path | None
    metrics_path: Path | None
    history: pd.DataFrame | None
    summary: dict[str, Any]
    metrics: dict[str, Any]


def _default_scratch_root() -> Path:
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"
    return Path(os.environ.get("SCRATCH_DIR", f"/cluster/scratch/{user}/protein-design"))


def _default_train_root() -> Path:
    scratch_root = _default_scratch_root()
    return Path(os.environ.get("TRAIN_DIR", str(scratch_root / "train")))


def _default_archive_root() -> Path:
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"
    project_root = Path(
        os.environ.get("PROJECT_DIR", f"/cluster/project/infk/krause/{user}/protein-design")
    )
    return project_root / "checkpoints"


def _default_output_root() -> Path:
    return _default_scratch_root() / "plots" / "dpo"


def _sanitize_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "plot"


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, found {type(payload).__name__}.")
    return payload


def _normalize_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive
            return value
    return value


def _extract_timestamp(text: str) -> str:
    match = RUN_TIMESTAMP_RE.search(text)
    if match is None:
        raise ValueError(f"Could not extract a YYYYMMDD_HHmmss timestamp from {text!r}.")
    return match.group(1)


def _candidate_run_dirs(selector: str, roots: Sequence[Path]) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    for root in roots:
        if not root.exists():
            continue

        exact = root / selector
        if exact.is_dir():
            resolved = exact.resolve()
            if resolved not in seen:
                candidates.append(exact)
                seen.add(resolved)

        for match in root.rglob(f"*{selector}"):
            if not match.is_dir():
                continue
            resolved = match.resolve()
            if resolved in seen:
                continue
            candidates.append(match)
            seen.add(resolved)

    return candidates


def discover_run_artifacts(
    run_ids: Sequence[str],
    train_root: Path,
    archive_root: Path | None = None,
    run_labels: Sequence[str] | None = None,
) -> list[RunArtifacts]:
    if run_labels is not None and len(run_labels) != len(run_ids):
        raise ValueError("run_labels must have the same length as run_ids.")

    roots: list[Path] = [train_root]
    if archive_root is not None:
        roots.append(archive_root)

    artifacts: list[RunArtifacts] = []
    for index, run_id in enumerate(run_ids):
        selector = _extract_timestamp(str(run_id))
        candidate_dirs = _candidate_run_dirs(selector, roots)
        if not candidate_dirs:
            raise FileNotFoundError(
                f"Could not find a run directory for {run_id!r} under {train_root}"
                + (f" or {archive_root}" if archive_root is not None else "")
            )

        run_dir = candidate_dirs[0]
        run_name = run_dir.name
        history_path = run_dir / "history.csv"
        summary_path = run_dir / "summary.json"
        metrics_path = run_dir / "metrics.json"

        archive_dir = None
        if archive_root is not None:
            archive_candidate = archive_root / run_name
            if archive_candidate.exists():
                archive_dir = archive_candidate

        history_df = pd.read_csv(history_path) if history_path.exists() else None

        summary = _load_json(summary_path)
        metrics = _load_json(metrics_path)
        if not metrics and archive_dir is not None:
            metrics = _load_json(archive_dir / "metrics.json")

        resolved_name = str(summary.get("run_name") or metrics.get("run_name") or run_name)
        timestamp = _extract_timestamp(resolved_name)
        display_name = run_labels[index] if run_labels is not None else timestamp

        artifacts.append(
            RunArtifacts(
                run_name=resolved_name,
                display_name=display_name,
                timestamp=timestamp,
                run_dir=run_dir,
                archive_dir=archive_dir,
                history_path=history_path if history_path.exists() else None,
                summary_path=summary_path if summary_path.exists() else None,
                metrics_path=(metrics_path if metrics_path.exists() else None)
                or (archive_dir / "metrics.json" if archive_dir and (archive_dir / "metrics.json").exists() else None),
                history=history_df,
                summary=summary,
                metrics=metrics,
            )
        )

    return artifacts


def _prepare_epoch_series(history: pd.DataFrame, value_column: str) -> tuple[pd.Series, pd.Series] | None:
    if value_column not in history.columns:
        return None

    values = pd.to_numeric(history[value_column], errors="coerce")
    if "epoch" in history.columns:
        epochs = pd.to_numeric(history["epoch"], errors="coerce")
    else:
        epochs = pd.Series(range(1, len(history) + 1), index=history.index, dtype="float64")

    valid = values.notna() & epochs.notna()
    if not valid.any():
        return None

    return epochs.loc[valid], values.loc[valid]


def _plot_curve_group(
    artifacts: Sequence[RunArtifacts],
    *,
    metrics: Sequence[str],
    value_prefix: str,
    origin_label: str,
    output_dir: Path,
    dpi: int,
    formats: Sequence[str],
) -> list[Path]:
    if not metrics:
        return []

    selected_metrics = list(metrics)
    figure, axes = plt.subplots(
        len(selected_metrics),
        1,
        figsize=(12, max(3.5, 3.25 * len(selected_metrics))),
        sharex=True,
        constrained_layout=True,
    )
    if len(selected_metrics) == 1:
        axes = [axes]

    for axis, metric in zip(axes, selected_metrics):
        plotted_any = False
        value_column = f"{value_prefix}_{metric}"
        for artifact in artifacts:
            if artifact.history is None:
                continue
            series = _prepare_epoch_series(artifact.history, value_column)
            if series is None:
                continue
            epochs, values = series
            axis.plot(
                epochs,
                values,
                linewidth=1.8,
                marker="o",
                markersize=3,
                label=artifact.display_name,
            )
            plotted_any = True

        axis.set_title(metric)
        axis.set_ylabel(metric)
        axis.grid(True, alpha=0.25)
        if not plotted_any:
            axis.text(0.5, 0.5, f"No data for {value_column}", ha="center", va="center", transform=axis.transAxes)
        else:
            axis.legend(fontsize=8, loc="best")

    axes[-1].set_xlabel("Epoch")
    figure.suptitle(f"{origin_label} metrics")

    written_paths: list[Path] = []
    safe_name = _sanitize_name(f"{value_prefix}_curves")
    for file_format in formats:
        output_path = output_dir / f"{safe_name}.{file_format}"
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        written_paths.append(output_path)
    plt.close(figure)
    return written_paths


def _plot_summary_group(
    artifacts: Sequence[RunArtifacts],
    *,
    metrics: Sequence[str],
    origin_label: str,
    output_dir: Path,
    dpi: int,
    formats: Sequence[str],
) -> list[Path]:
    if not metrics:
        return []

    selected_metrics = list(metrics)
    figure, axes = plt.subplots(
        len(selected_metrics),
        1,
        figsize=(12, max(3.2, 3.0 * len(selected_metrics))),
        constrained_layout=True,
    )
    if len(selected_metrics) == 1:
        axes = [axes]

    run_labels = [artifact.display_name for artifact in artifacts]
    x_positions = list(range(len(run_labels)))

    for axis, metric in zip(axes, selected_metrics):
        values = []
        for artifact in artifacts:
            summary_value = artifact.metrics.get(metric)
            if summary_value is None:
                summary_value = artifact.summary.get(metric)
            values.append(_normalize_scalar(summary_value))

        numeric_values = pd.to_numeric(pd.Series(values), errors="coerce")
        valid_mask = numeric_values.notna()
        if valid_mask.any():
            valid_positions = [xpos for xpos, is_valid in zip(x_positions, valid_mask) if bool(is_valid)]
            valid_values = numeric_values.loc[valid_mask].tolist()
            axis.bar(valid_positions, valid_values, color="#4063D8", alpha=0.85)
            for xpos, value in zip(valid_positions, valid_values):
                if pd.notna(value):
                    axis.text(xpos, float(value), f"{float(value):.4g}", ha="center", va="bottom", fontsize=8)
        else:
            axis.text(0.5, 0.5, f"No data for {metric}", ha="center", va="center", transform=axis.transAxes)

        axis.set_title(metric)
        axis.set_ylabel(metric)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(run_labels, rotation=30, ha="right")
        axis.grid(True, axis="y", alpha=0.25)

    figure.suptitle(f"{origin_label} metrics across runs")

    written_paths: list[Path] = []
    safe_name = _sanitize_name(f"{origin_label}_summary")
    for file_format in formats:
        output_path = output_dir / f"{safe_name}.{file_format}"
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        written_paths.append(output_path)
    plt.close(figure)
    return written_paths


def _build_summary_table(artifacts: Sequence[RunArtifacts]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        row: dict[str, Any] = {
            "run_name": artifact.run_name,
            "timestamp": artifact.timestamp,
            "run_dir": str(artifact.run_dir),
            "archive_dir": None if artifact.archive_dir is None else str(artifact.archive_dir),
            "history_path": None if artifact.history_path is None else str(artifact.history_path),
            "summary_path": None if artifact.summary_path is None else str(artifact.summary_path),
            "metrics_path": None if artifact.metrics_path is None else str(artifact.metrics_path),
        }
        for key, value in artifact.summary.items():
            row[f"summary.{key}"] = _normalize_scalar(value)
        for key, value in artifact.metrics.items():
            row[f"metrics.{key}"] = _normalize_scalar(value)
        rows.append(row)
    return pd.DataFrame(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        action="append",
        required=True,
        help="Run identifier or timestamp to plot. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--run-label",
        action="append",
        default=None,
        help="Display name for the corresponding --run-id. Repeat in the same order as --run-id.",
    )
    parser.add_argument("--train-root", type=Path, default=_default_train_root())
    parser.add_argument("--archive-root", type=Path, default=_default_archive_root())
    parser.add_argument("--output-dir", type=Path, default=_default_output_root())
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--formats", nargs="+", default=["png"], help="Figure formats to write.")
    parser.add_argument("--plot-training-curves", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-validation-curves", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-validation-summary", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-test-summary", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--training-metrics", nargs="+", default=DEFAULT_TRAIN_CURVE_METRICS)
    parser.add_argument("--validation-metrics", nargs="+", default=DEFAULT_VAL_CURVE_METRICS)
    parser.add_argument("--validation-summary-metrics", nargs="+", default=DEFAULT_VAL_SUMMARY_METRICS)
    parser.add_argument("--test-summary-metrics", nargs="+", default=DEFAULT_TEST_SUMMARY_METRICS)
    parser.add_argument(
        "--summary-table-name",
        default="run_metrics_summary.csv",
        help="CSV file containing all discovered run summaries and metrics.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = discover_run_artifacts(
        run_ids=args.run_id,
        train_root=args.train_root,
        archive_root=args.archive_root,
        run_labels=args.run_label,
    )

    written_files: list[Path] = []
    if args.plot_training_curves:
        written_files.extend(
            _plot_curve_group(
                artifacts,
                metrics=args.training_metrics,
                value_prefix="train",
                origin_label="Training",
                output_dir=output_dir,
                dpi=args.dpi,
                formats=args.formats,
            )
        )

    if args.plot_validation_curves:
        written_files.extend(
            _plot_curve_group(
                artifacts,
                metrics=args.validation_metrics,
                value_prefix="val",
                origin_label="Validation",
                output_dir=output_dir,
                dpi=args.dpi,
                formats=args.formats,
            )
        )

    if args.plot_validation_summary:
        written_files.extend(
            _plot_summary_group(
                artifacts,
                metrics=args.validation_summary_metrics,
                origin_label="Validation",
                output_dir=output_dir,
                dpi=args.dpi,
                formats=args.formats,
            )
        )

    if args.plot_test_summary:
        written_files.extend(
            _plot_summary_group(
                artifacts,
                metrics=args.test_summary_metrics,
                origin_label="Test",
                output_dir=output_dir,
                dpi=args.dpi,
                formats=args.formats,
            )
        )

    summary_table = _build_summary_table(artifacts)
    summary_table_path = output_dir / args.summary_table_name
    summary_table.to_csv(summary_table_path, index=False)
    written_files.append(summary_table_path)

    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "run_name": artifact.run_name,
                    "timestamp": artifact.timestamp,
                    "run_dir": str(artifact.run_dir),
                    "archive_dir": None if artifact.archive_dir is None else str(artifact.archive_dir),
                    "history_path": None if artifact.history_path is None else str(artifact.history_path),
                    "summary_path": None if artifact.summary_path is None else str(artifact.summary_path),
                    "metrics_path": None if artifact.metrics_path is None else str(artifact.metrics_path),
                }
                for artifact in artifacts
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    written_files.append(manifest_path)

    print(f"Wrote {len(written_files)} files to {output_dir}")
    for path in written_files:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())