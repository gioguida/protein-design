#!/usr/bin/env python3
"""Print DPO split and pair statistics from the official YAML configs.

Run from the tests directory:
    python dpo_pair_stats.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protein_design.dms_splitting import dataset_spec, resolve_dataset_split  # noqa: E402
from protein_design.dpo.dataset import build_split_pair_dataframes_from_raw  # noqa: E402


DPO_CONFIG_PATH = REPO_ROOT / "conf" / "data" / "dpo" / "default.yaml"
LOCAL_RUNTIME_DIR = REPO_ROOT / "data" / "processed" / "dpo_pair_stats"
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
    """Create a local-only DMS config overlay for this diagnostic script."""
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


def _print_split_stats(
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
        counts = df["num_mut"].value_counts(dropna=False).sort_index()
        print("num_mut counts:")
        print(counts.to_string())
    print(f"{key_metric_col}: {_series_summary(df[key_metric_col])}")


def _print_pair_stats(split_name: str, pairs_df: pd.DataFrame) -> None:
    print(f"\n[{split_name}] DPO pairs")
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


def main() -> None:
    dpo_cfg = _load_yaml(DPO_CONFIG_PATH)
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

    print("DPO pair statistics")
    print(f"dpo_config: {DPO_CONFIG_PATH}")
    print(f"official_dms_config: {official_dms_config_path}")
    print(f"local_dms_config: {dms_config_path}")
    print(f"local_raw_dir: {REPO_ROOT / 'data' / 'raw'}")
    print(f"dataset_key: {dataset_key}")
    print(f"raw_csv: {spec.path}")
    print(f"sequence_col: {spec.sequence_col}")
    print(f"key_metric_col: {spec.key_metric_col}")
    print(f"pairing_strategy: {pairing_strategy}")
    print(f"delta_components: {delta_cfg.get('components', [])}")
    print(f"delta_mix_mode: {mix_cfg.get('mode', 'count')}")
    print(
        "pair_split: "
        f"enforce_train_controlled_sizes={bool(pair_split_cfg.get('enforce_train_controlled_sizes', False))}, "
        f"train/val/test={float(pair_split_cfg.get('train_frac', 0.8))}/"
        f"{float(pair_split_cfg.get('val_frac', 0.1))}/{float(pair_split_cfg.get('test_frac', 0.1))}"
    )
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
        _print_split_stats(
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
        delta_component_pair_counts={str(k): int(v) for k, v in mix_count_cfg.items()},
        delta_component_pair_fractions={str(k): float(v) for k, v in mix_fraction_cfg.items()},
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
        _print_pair_stats(split_name, pairs_df)


if __name__ == "__main__":
    main()
