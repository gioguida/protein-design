#!/usr/bin/env python3
"""Filter DPO pairs by chosen-sequence mutation positions and plot entropy.

Run directly with:
    python tests/test_remove_positions.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


# Set 1-based CDR-H3 positions to exclude here.
EXCLUDED_POSITIONS: list[int] = [15, 16] # [8, 9, 15, 16]


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protein_design.analysis.entropy import position_entropy  # noqa: E402
from protein_design.constants import C05_CDRH3  # noqa: E402
from protein_design.dms_splitting import dataset_spec, resolve_dataset_split  # noqa: E402
from protein_design.dpo.dataset import build_split_pair_dataframes_from_raw  # noqa: E402


DPO_DATA_CONFIG_PATH = REPO_ROOT / "conf" / "data" / "dpo" / "default.yaml"
LOCAL_RUNTIME_DIR = REPO_ROOT / "data" / "processed" / "remove_positions"
PLOTS_DIR = REPO_ROOT / "plots" / "remove_positions"
LOCAL_RAW_FILENAMES = {
    "ed2_m22": "ED2_M22_binding_enrichment.csv",
    "ed2_si06": "ED2_SI06_binding_enrichment.csv",
    "ed5_m22": "ED5_M22_binding_enrichment.csv",
    "ed5_si06": "ED5_SI06_binding_enrichment.csv",
    "ed811_m22": "ED811_M22_enrichment_full.csv",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _build_local_dms_config(official_dms_config_path: Path) -> Path:
    dms_cfg = _load_yaml(official_dms_config_path)
    local_raw_dir = REPO_ROOT / "data" / "raw"
    datasets = dms_cfg.get("datasets", {}) or {}
    for dataset_key, filename in LOCAL_RAW_FILENAMES.items():
        local_path = local_raw_dir / filename
        if dataset_key in datasets and local_path.exists():
            datasets[dataset_key]["path"] = str(local_path)

    LOCAL_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    local_config_path = LOCAL_RUNTIME_DIR / "dms.local.yaml"
    with local_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dms_cfg, handle, sort_keys=False)
    return local_config_path


def _normalize_positions(positions: Iterable[int]) -> list[int]:
    normalized = sorted({int(position) for position in positions})
    invalid = [position for position in normalized if position < 1 or position > len(C05_CDRH3)]
    if invalid:
        raise ValueError(
            f"Excluded positions must be within 1..{len(C05_CDRH3)}. Got invalid positions: {invalid}"
        )
    return normalized


def _series_summary(values: pd.Series) -> str:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return "count=0"
    return (
        f"count={len(clean)}, mean={clean.mean():.4f}, std={clean.std():.4f}, "
        f"min={clean.min():.4f}, p25={clean.quantile(0.25):.4f}, "
        f"p50={clean.quantile(0.50):.4f}, p75={clean.quantile(0.75):.4f}, "
        f"max={clean.max():.4f}"
    )


def _format_positions(positions: list[int]) -> str:
    if not positions:
        return "[]"
    return "[" + ", ".join(f"{position}{C05_CDRH3[position - 1]}" for position in positions) + "]"


def _sequence_has_mutation_at_positions(sequence: str, positions: list[int]) -> bool:
    if len(sequence) != len(C05_CDRH3):
        return False
    return any(sequence[position - 1] != C05_CDRH3[position - 1] for position in positions)


def _count_position_hits(sequences: pd.Series, positions: list[int]) -> dict[int, int]:
    counts = {position: 0 for position in positions}
    expected_length = len(C05_CDRH3)
    for sequence in sequences.astype(str):
        if len(sequence) != expected_length:
            continue
        for position in positions:
            if sequence[position - 1] != C05_CDRH3[position - 1]:
                counts[position] += 1
    return counts


def _filter_pairs_by_chosen_positions(pairs_df: pd.DataFrame, positions: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pairs_df.empty or not positions:
        empty = pairs_df.iloc[0:0].copy()
        return pairs_df.reset_index(drop=True), empty

    chosen = pairs_df["chosen_sequence"].astype(str)
    remove_mask = chosen.map(lambda sequence: _sequence_has_mutation_at_positions(sequence, positions))
    removed = pairs_df.loc[remove_mask].reset_index(drop=True)
    kept = pairs_df.loc[~remove_mask].reset_index(drop=True)
    return kept, removed


def _print_dms_split_stats(
    split_name: str,
    split_path: Path,
    *,
    sequence_col: str,
    key_metric_col: str,
) -> None:
    df = pd.read_csv(split_path)
    print(f"\n[{split_name}] DMS split")
    print(f"path: {split_path}")
    print(f"rows: {len(df)}")
    print(f"unique {sequence_col}: {df[sequence_col].nunique() if sequence_col in df.columns else 'missing'}")
    if "num_mut" in df.columns:
        print("num_mut counts:")
        print(df["num_mut"].value_counts(dropna=False).sort_index().to_string())
    print(f"{key_metric_col}: {_series_summary(df[key_metric_col])}")


def _print_pair_stats(split_name: str, pairs_df: pd.DataFrame, positions: list[int], *, label: str) -> None:
    print(f"\n[{split_name}] {label}")
    print(f"pairs: {len(pairs_df)}")
    if pairs_df.empty:
        return

    chosen = pairs_df["chosen_sequence"].astype(str)
    rejected = pairs_df["rejected_sequence"].astype(str)
    all_pair_sequences = pd.concat([chosen, rejected], ignore_index=True)
    print(f"unique chosen sequences: {chosen.nunique()}")
    print(f"unique rejected sequences: {rejected.nunique()}")
    print(f"unique sequences in pairs: {all_pair_sequences.nunique()}")
    print(f"self-pairs: {int((chosen == rejected).sum())}")
    print(f"invalid margins <= 0: {int((pairs_df['delta_margin'] <= 0).sum())}")
    print(f"chosen_delta: {_series_summary(pairs_df['chosen_delta'])}")
    print(f"rejected_delta: {_series_summary(pairs_df['rejected_delta'])}")
    print(f"delta_margin: {_series_summary(pairs_df['delta_margin'])}")
    if "delta_component" in pairs_df.columns:
        print("delta_component counts:")
        print(pairs_df["delta_component"].value_counts(dropna=False).to_string())
    if positions:
        hits = _count_position_hits(chosen, positions)
        print(
            "chosen mutations at excluded positions: "
            + ", ".join(f"{position}{C05_CDRH3[position - 1]}={count}" for position, count in hits.items())
        )


def _plot_entropy_heatmap(
    entropies: np.ndarray,
    *,
    row_label: str,
    title: str,
    output_path: Path,
) -> None:
    positions = np.arange(1, len(entropies) + 1)
    fig, ax = plt.subplots(figsize=(11.5, 2.6))
    image = ax.imshow(entropies[np.newaxis, :], aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xticks(np.arange(len(entropies)))
    ax.set_xticklabels([f"{idx}\n{aa}" for idx, aa in zip(positions, C05_CDRH3)], fontsize=8)
    ax.set_yticks([0])
    ax.set_yticklabels([row_label], rotation=90, va="center")
    ax.set_xlabel("CDR-H3 position (WT residue)")
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Shannon entropy (bits)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_entropy_csv(output_path: Path, entropies: np.ndarray, *, num_sequences: int) -> None:
    positions = np.arange(1, len(entropies) + 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("position,wt_residue,entropy_bits,num_sequences\n")
        for position, residue, entropy in zip(positions, C05_CDRH3, entropies):
            handle.write(f"{position},{residue},{float(entropy):.8f},{num_sequences}\n")


def _plot_chosen_entropy(split_name: str, pairs_df: pd.DataFrame, positions: list[int]) -> None:
    expected_length = len(C05_CDRH3)
    chosen_sequences = [
        sequence
        for sequence in pairs_df["chosen_sequence"].astype(str).tolist()
        if len(sequence) == expected_length
    ]
    entropies = position_entropy(chosen_sequences, expected_length=expected_length)

    suffix = "all_positions" if not positions else "exclude_" + "_".join(str(position) for position in positions)
    heatmap_path = PLOTS_DIR / f"dpo_{split_name}_chosen_{suffix}_temp_entropy_heatmap.png"
    csv_path = PLOTS_DIR / f"dpo_{split_name}_chosen_{suffix}_position_entropy.csv"

    _plot_entropy_heatmap(
        entropies,
        row_label="chosen",
        title=f"DPO {split_name} split: chosen-sequence position-wise entropy",
        output_path=heatmap_path,
    )
    _write_entropy_csv(csv_path, entropies, num_sequences=len(chosen_sequences))

    print(f"\n[{split_name}] chosen entropy")
    print(f"heatmap: {heatmap_path}")
    print(f"csv: {csv_path}")
    print(f"chosen sequences used: {len(chosen_sequences)}")


def main() -> None:
    positions = _normalize_positions(EXCLUDED_POSITIONS)
    dpo_cfg = _load_yaml(DPO_DATA_CONFIG_PATH)
    data_cfg = dpo_cfg.get("data", {}) or {}
    delta_cfg = data_cfg.get("delta_based", {}) or {}
    mix_cfg = delta_cfg.get("mix", {}) or {}
    mix_count_cfg = mix_cfg.get("count", {}) or {}
    mix_fraction_cfg = mix_cfg.get("fraction", {}) or {}
    pair_split_cfg = data_cfg.get("pair_split", {}) or {}

    official_dms_config_path = _resolve_repo_path(
        data_cfg.get("dms_config", "conf/data/dms/default.yaml")
    )
    dataset_key = str(data_cfg.get("dpo_dataset_key", "ed2_m22"))
    pairing_strategy = str(data_cfg.get("pairing_strategy", "delta_based"))
    force_rebuild = bool(data_cfg.get("force_rebuild", False))

    dms_config_path = _build_local_dms_config(official_dms_config_path)
    spec = dataset_spec(dataset_key, dms_config_path)

    print("DPO chosen-sequence position filter")
    print(f"dpo_data_config: {DPO_DATA_CONFIG_PATH}")
    print(f"official_dms_config: {official_dms_config_path}")
    print(f"local_dms_config: {dms_config_path}")
    print(f"dataset_key: {dataset_key}")
    print(f"sequence_col: {spec.sequence_col}")
    print(f"key_metric_col: {spec.key_metric_col}")
    print(f"pairing_strategy: {pairing_strategy}")
    print(f"excluded_positions: {_format_positions(positions)}")
    print(f"force_rebuild: {force_rebuild}")

    split_paths = {
        split_name: resolve_dataset_split(
            dataset_key,
            split_name,
            dms_config_path,
            force=force_rebuild,
        )
        for split_name in ("train", "val", "test")
    }
    for split_name, split_path in split_paths.items():
        _print_dms_split_stats(
            split_name,
            split_path,
            sequence_col=spec.sequence_col,
            key_metric_col=spec.key_metric_col,
        )

    train_pairs, val_pairs, test_pairs = build_split_pair_dataframes_from_raw(
        pairing_strategy=pairing_strategy,
        force_rebuild=force_rebuild,
        min_positive_delta=float(data_cfg.get("min_positive_delta", 0.0)),
        min_delta_margin=float(data_cfg.get("min_delta_margin", 0.0)),
        train_frac=float(pair_split_cfg.get("train_frac", 0.8)),
        val_frac=float(pair_split_cfg.get("val_frac", 0.1)),
        test_frac=float(pair_split_cfg.get("test_frac", 0.1)),
        enforce_train_controlled_split_sizes=bool(
            pair_split_cfg.get("enforce_train_controlled_sizes", False)
        ),
        delta_components=tuple(delta_cfg.get("components", [])),
        delta_mix_mode=str(mix_cfg.get("mode", "count")),
        delta_component_pair_counts={str(key): int(value) for key, value in mix_count_cfg.items()},
        delta_component_pair_fractions={
            str(key): float(value) for key, value in mix_fraction_cfg.items()
        },
        gap=float(delta_cfg.get("gap", 0.5)),
        wt_pairs_frac=float(delta_cfg.get("wt_pairs_frac", 0.1)),
        cross_pairs_frac=float(delta_cfg.get("cross_pairs_frac", 0.1)),
        strong_pos_threshold=float(delta_cfg.get("strong_pos_threshold", 1.0)),
        strong_neg_threshold=float(delta_cfg.get("strong_neg_threshold", -5.0)),
        min_score_margin=float(delta_cfg.get("min_score_margin", 0.1)),
        dms_config_path=dms_config_path,
        dataset_key=dataset_key,
    )

    for split_name, pairs_df in (
        ("train", train_pairs),
        ("val", val_pairs),
        ("test", test_pairs),
    ):
        _print_pair_stats(split_name, pairs_df, positions, label="original DPO pairs")
        filtered_pairs, removed_pairs = _filter_pairs_by_chosen_positions(pairs_df, positions)
        print(f"\n[{split_name}] filtering summary")
        print(f"removed pairs: {len(removed_pairs)}")
        print(f"kept pairs: {len(filtered_pairs)}")
        fraction_removed = (len(removed_pairs) / len(pairs_df)) if len(pairs_df) > 0 else 0.0
        print(f"removed fraction: {fraction_removed:.4f}")
        _print_pair_stats(split_name, filtered_pairs, positions, label="filtered DPO pairs")
        _plot_chosen_entropy(split_name, filtered_pairs, positions)


if __name__ == "__main__":
    main()
