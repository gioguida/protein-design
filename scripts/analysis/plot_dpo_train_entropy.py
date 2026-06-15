"""Plot per-position entropy for chosen and rejected sequences in the DPO train split."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from protein_design.analysis.entropy import position_entropy
from protein_design.constants import C05_CDRH3
from protein_design.dpo.dataset import build_split_pair_dataframes_from_cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("plot_dpo_train_entropy")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis" / "dpo_train_entropy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for plots and CSV summaries.",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=REPO_ROOT / "conf" / "config.yaml",
        help="Base config providing the default seed.",
    )
    parser.add_argument(
        "--task-config",
        type=Path,
        default=REPO_ROOT / "conf" / "task" / "dpo.yaml",
        help="Task config used to resolve the DPO data config.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional dotted override such as data.force_rebuild=true. Repeatable.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected mapping at {path}, got {type(loaded)}.")
    return loaded


def _extract_default_override(defaults: list[Any], prefix: str) -> str:
    for entry in defaults:
        if not isinstance(entry, dict):
            continue
        key = f"override {prefix}"
        if key in entry:
            return str(entry[key])
        if prefix in entry:
            return str(entry[prefix])
    raise KeyError(f"Could not find {prefix} override in task defaults.")


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _to_namespace(val) for key, val in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def _apply_override(config: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Override must be KEY=VALUE, got {override!r}.")
    dotted_key, raw_value = override.split("=", 1)
    keys = [part for part in dotted_key.strip().split(".") if part]
    if not keys:
        raise ValueError(f"Invalid override key in {override!r}.")
    value = yaml.safe_load(raw_value)
    cursor = config
    for key in keys[:-1]:
        next_cursor = cursor.get(key)
        if not isinstance(next_cursor, dict):
            next_cursor = {}
            cursor[key] = next_cursor
        cursor = next_cursor
    cursor[keys[-1]] = value


def _compose_cfg(base_config_path: Path, task_config_path: Path, overrides: list[str]) -> Any:
    base_payload = _load_yaml(base_config_path)
    task_payload = _load_yaml(task_config_path)
    data_override = _extract_default_override(list(task_payload.get("defaults", [])), "/data")
    data_config_path = REPO_ROOT / "conf" / "data" / f"{data_override}.yaml"
    data_payload = _load_yaml(data_config_path)

    cfg_dict: dict[str, Any] = {
        "seed": int(base_payload.get("seed", 42)),
        "data": dict(data_payload.get("data", {})),
    }
    for override in overrides:
        _apply_override(cfg_dict, override)
    return _to_namespace(cfg_dict)


def _materialize_local_dms_config(cfg: Any, output_dir: Path) -> None:
    dms_config_path = Path(str(cfg.data.dms_config))
    if not dms_config_path.is_absolute():
        dms_config_path = REPO_ROOT / dms_config_path
    dms_payload = _load_yaml(dms_config_path)

    datasets = dms_payload.get("datasets", {})
    for dataset_name, dataset_cfg in datasets.items():
        if not isinstance(dataset_cfg, dict):
            continue
        raw_path = Path(str(dataset_cfg.get("path", "")))
        if raw_path.exists():
            continue
        local_candidate = REPO_ROOT / "data" / "raw" / raw_path.name
        if local_candidate.exists():
            dataset_cfg["path"] = str(local_candidate)
            log.info("Using local DMS CSV for %s: %s", dataset_name, local_candidate)

    split_cfg = dms_payload.get("split")
    if isinstance(split_cfg, dict):
        split_cfg["output_dir"] = str(REPO_ROOT / "data" / "dms_splits")

    resolved_dms_path = output_dir / "resolved_dms_config.yaml"
    resolved_dms_path.write_text(yaml.safe_dump(dms_payload, sort_keys=False), encoding="utf-8")
    cfg.data.dms_config = str(resolved_dms_path)


def _prepare_sequences(values: list[str], expected_length: int) -> tuple[list[str], int]:
    valid = [str(value) for value in values if len(str(value)) == expected_length]
    return valid, len(values) - len(valid)


def _write_entropy_csv(
    out_path: Path,
    entropies: np.ndarray,
    *,
    sequence_count: int,
) -> None:
    positions = np.arange(1, len(entropies) + 1)
    residues = list(C05_CDRH3)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("position,wt_residue,entropy_bits,num_sequences\n")
        for position, residue, entropy in zip(positions, residues, entropies):
            handle.write(f"{position},{residue},{float(entropy):.8f},{sequence_count}\n")


def _plot_entropy_heatmap(
    entropies: np.ndarray,
    *,
    row_label: str,
    title: str,
    out_path: Path,
) -> None:
    positions = np.arange(1, len(entropies) + 1)
    fig, ax = plt.subplots(figsize=(11.5, 2.6))
    image = ax.imshow(entropies[np.newaxis, :], aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xticks(np.arange(len(entropies)))
    ax.set_xticklabels([f"{idx}\n{aa}" for idx, aa in zip(positions, C05_CDRH3)], fontsize=8)
    ax.set_yticks([0])
    ax.set_yticklabels([row_label])
    ax.set_xlabel("CDR-H3 position (WT residue)")
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Shannon entropy (bits)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _compose_cfg(args.base_config.resolve(), args.task_config.resolve(), list(args.override))
    _materialize_local_dms_config(cfg, output_dir)
    train_df, _, _ = build_split_pair_dataframes_from_cfg(cfg)
    if train_df.empty:
        raise ValueError("No DPO train pairs were generated from the current config.")

    expected_length = len(C05_CDRH3)
    chosen, chosen_dropped = _prepare_sequences(train_df["chosen_sequence"].astype(str).tolist(), expected_length)
    rejected, rejected_dropped = _prepare_sequences(
        train_df["rejected_sequence"].astype(str).tolist(),
        expected_length,
    )

    chosen_entropy = position_entropy(chosen, expected_length=expected_length)
    rejected_entropy = position_entropy(rejected, expected_length=expected_length)

    _plot_entropy_heatmap(
        chosen_entropy,
        row_label="chosen",
        title="DPO train split: chosen-sequence position-wise entropy",
        out_path=output_dir / "chosen_temp_entropy_heatmap.png",
    )
    _plot_entropy_heatmap(
        rejected_entropy,
        row_label="rejected",
        title="DPO train split: rejected-sequence position-wise entropy",
        out_path=output_dir / "rejected_temp_entropy_heatmap.png",
    )
    _write_entropy_csv(
        output_dir / "chosen_position_entropy.csv",
        chosen_entropy,
        sequence_count=len(chosen),
    )
    _write_entropy_csv(
        output_dir / "rejected_position_entropy.csv",
        rejected_entropy,
        sequence_count=len(rejected),
    )

    summary = {
        "task_config": str(args.task_config.resolve()),
        "base_config": str(args.base_config.resolve()),
        "overrides": list(args.override),
        "train_pairs": int(len(train_df)),
        "chosen_sequences": int(len(chosen)),
        "rejected_sequences": int(len(rejected)),
        "chosen_dropped_wrong_length": int(chosen_dropped),
        "rejected_dropped_wrong_length": int(rejected_dropped),
        "dataset_key": str(getattr(cfg.data, "dpo_dataset_key", "")),
        "dms_config": str(getattr(cfg.data, "dms_config", "")),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    log.info("Wrote DPO train entropy plots to %s", output_dir)
    log.info(
        "Train pairs=%d | chosen=%d (dropped %d) | rejected=%d (dropped %d)",
        len(train_df),
        len(chosen),
        chosen_dropped,
        len(rejected),
        rejected_dropped,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
