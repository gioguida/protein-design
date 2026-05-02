#!/usr/bin/env python3
"""Build unlikelihood train/val/test splits and print dataset statistics.

This script intentionally takes no CLI arguments.
Edit the constants below if you want to change paths or thresholds.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protein_design.dpo.data_processing import build_processed_views
from protein_design.dpo.dataset import default_data_paths
from protein_design.dpo.splitting import (
    build_or_load_cluster_split_membership,
    split_membership_keys,
)
from protein_design.unlikelihood.data import build_good_split_sequences

MUT_TOKEN_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")

_defaults = default_data_paths()

# ---------------------------------------------------------------------------
# Fixed configuration (edit here, no CLI flags)
# ---------------------------------------------------------------------------
RAW_CSV = Path(_defaults["raw_m22"])
PROCESSED_DIR = Path(_defaults["processed_dir"])
ENRICHMENT_THRESHOLD = 5.19
SEED = 42
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1
HAMMING_DISTANCE = 1
STRATIFY_BINS = 10
BATCH_SIZE = 128
FORCE_REBUILD = False
UNWANTED_SET_PATH = PROCESSED_DIR / "unwanted_set.json"
UNWANTED_SUMMARY_CSV = PROCESSED_DIR / "unwanted_substitution_enrichment.csv"
PREVIEW_VARIANTS = 10


def _to_abs(path_like: Any) -> str:
    return str(Path(path_like).resolve())


def _build_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        seed=int(SEED),
        data=SimpleNamespace(
            raw_csv=_to_abs(RAW_CSV),
            processed_dir=_to_abs(PROCESSED_DIR),
            force_rebuild=bool(FORCE_REBUILD),
            train_frac=float(TRAIN_FRAC),
            val_frac=float(VAL_FRAC),
            test_frac=float(TEST_FRAC),
            split=SimpleNamespace(
                hamming_distance=int(HAMMING_DISTANCE),
                stratify_bins=int(STRATIFY_BINS),
            ),
            enrichment_threshold=float(ENRICHMENT_THRESHOLD),
        ),
    )


def _series_stats(values: pd.Series) -> Dict[str, float | None]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "p25": None,
            "p75": None,
            "max": None,
        }
    desc = clean.describe(percentiles=[0.25, 0.75])
    return {
        "count": int(desc["count"]),
        "mean": float(desc["mean"]),
        "median": float(desc["50%"]),
        "std": float(desc["std"]) if pd.notna(desc["std"]) else 0.0,
        "min": float(desc["min"]),
        "p25": float(desc["25%"]),
        "p75": float(desc["75%"]),
        "max": float(desc["max"]),
    }


def _length_stats(seqs: pd.Series) -> Dict[str, float | None]:
    lengths = seqs.astype(str).str.len()
    return _series_stats(lengths)


def _split_summary(df: pd.DataFrame, batch_size: int, drop_last: bool) -> Dict[str, Any]:
    n_rows = int(len(df))
    n_unique = int(df["aa"].nunique()) if "aa" in df.columns else 0
    duplicate_rows = n_rows - n_unique
    if drop_last:
        n_batches = n_rows // max(1, int(batch_size))
    else:
        n_batches = int(math.ceil(n_rows / max(1, int(batch_size)))) if n_rows > 0 else 0
    return {
        "rows": n_rows,
        "unique_sequences": n_unique,
        "duplicate_rows": int(duplicate_rows),
        "duplicate_fraction": float(duplicate_rows / n_rows) if n_rows > 0 else 0.0,
        "estimated_batches": int(n_batches),
        "num_mut_counts": {
            str(int(k)): int(v)
            for k, v in df["num_mut"].value_counts().sort_index().items()
        } if "num_mut" in df.columns else {},
        "enrichment_stats": _series_stats(df["M22_binding_enrichment_adj"]),
        "length_stats": _length_stats(df["aa"]) if "aa" in df.columns else {},
    }


def _parse_mutation_tokens(mut_value: Any) -> list[tuple[int, str, str]]:
    if pd.isna(mut_value):
        return []
    text = str(mut_value).strip()
    if text == "" or text == "0":
        return []
    out: list[tuple[int, str, str]] = []
    for token in text.split(";"):
        token = token.strip()
        if token == "" or token == "0":
            continue
        m = MUT_TOKEN_RE.match(token)
        if m is None:
            continue
        wt_aa, pos_str, mut_aa = m.groups()
        out.append((int(pos_str), wt_aa, mut_aa))
    return out


def _mutation_level_summary(df: pd.DataFrame) -> Dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        tokens = _parse_mutation_tokens(row.get("mut"))
        for pos, wt_aa, mut_aa in tokens:
            rows.append(
                {
                    "position": int(pos),
                    "wt_aa": wt_aa,
                    "mut_aa": mut_aa,
                    "enrichment": float(row["M22_binding_enrichment_adj"]),
                    "num_mut": int(row["num_mut"]),
                }
            )

    if not rows:
        return {
            "num_substitution_events": 0,
            "position_event_counts": {},
            "top_mutated_substitutions": [],
        }

    sub_df = pd.DataFrame(rows)
    pos_counts = sub_df["position"].value_counts().sort_index()
    top_subs = (
        sub_df.groupby(["position", "wt_aa", "mut_aa"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
        .head(20)
    )
    return {
        "num_substitution_events": int(len(sub_df)),
        "position_event_counts": {str(int(k)): int(v) for k, v in pos_counts.items()},
        "top_mutated_substitutions": [
            {
                "position": int(row["position"]),
                "wt_aa": str(row["wt_aa"]),
                "mut_aa": str(row["mut_aa"]),
                "count": int(row["count"]),
            }
            for _, row in top_subs.iterrows()
        ],
    }


def _build_unwanted_stats(
    unwanted_set_path: Path,
    unwanted_summary_csv: Path,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "unwanted_set_path": str(unwanted_set_path.resolve()),
        "unwanted_summary_csv": str(unwanted_summary_csv.resolve()),
        "json_exists": unwanted_set_path.exists(),
        "summary_csv_exists": unwanted_summary_csv.exists(),
    }

    unwanted_map: Dict[str, list[str]] = {}
    if unwanted_set_path.exists():
        with unwanted_set_path.open("r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        unwanted_map = {str(k): [str(x) for x in v] for k, v in loaded.items()}

    payload["json_num_positions"] = int(len(unwanted_map))
    payload["json_num_unwanted_aas_total"] = int(sum(len(v) for v in unwanted_map.values()))
    payload["json_unwanted_counts_by_position"] = {
        str(k): int(len(v)) for k, v in sorted(unwanted_map.items(), key=lambda kv: int(kv[0]))
    }

    if unwanted_summary_csv.exists():
        df = pd.read_csv(unwanted_summary_csv)
        payload["summary_rows"] = int(len(df))
        payload["summary_unwanted_rows"] = (
            int(df["is_unwanted"].sum()) if "is_unwanted" in df.columns else None
        )
        if "mean_source" in df.columns:
            payload["summary_mean_source_counts"] = {
                str(k): int(v) for k, v in df["mean_source"].value_counts().items()
            }
        if "n_obs" in df.columns:
            payload["summary_n_obs_stats"] = _series_stats(df["n_obs"])
    else:
        payload["summary_rows"] = None
        payload["summary_unwanted_rows"] = None

    return payload


def _flag_unwanted_events(df: pd.DataFrame, unwanted_map: Dict[int, set[str]]) -> Dict[str, Any]:
    if df.empty:
        return {
            "variants_with_any_unwanted_mutation": 0,
            "fraction_variants_with_any_unwanted_mutation": 0.0,
            "total_unwanted_mutation_events": 0,
        }

    num_with_any = 0
    event_count = 0
    for mut in df["mut"]:
        tokens = _parse_mutation_tokens(mut)
        row_has_any = False
        for pos, _, mut_aa in tokens:
            bad = unwanted_map.get(int(pos))
            if bad is not None and mut_aa in bad:
                row_has_any = True
                event_count += 1
        if row_has_any:
            num_with_any += 1

    return {
        "variants_with_any_unwanted_mutation": int(num_with_any),
        "fraction_variants_with_any_unwanted_mutation": float(num_with_any / len(df)),
        "total_unwanted_mutation_events": int(event_count),
    }


def _build_row_level_good_splits() -> Dict[str, pd.DataFrame]:
    processed_paths = build_processed_views(
        raw_csv_path=Path(RAW_CSV),
        processed_dir=Path(PROCESSED_DIR),
        force=bool(FORCE_REBUILD),
        verbose=False,
    )
    base_csv_path = Path(processed_paths["ed2_all"])
    base_df = pd.read_csv(base_csv_path)

    split_membership = build_or_load_cluster_split_membership(
        base_df=base_df,
        base_csv_path=base_csv_path,
        processed_dir=Path(PROCESSED_DIR),
        train_frac=float(TRAIN_FRAC),
        val_frac=float(VAL_FRAC),
        test_frac=float(TEST_FRAC),
        seed=int(SEED),
        force_rebuild=bool(FORCE_REBUILD),
        positive_threshold=0.0,
        stratify_bins=int(STRATIFY_BINS),
        hamming_distance=int(HAMMING_DISTANCE),
    )

    working = base_df.copy()
    key_map = dict(
        zip(
            split_membership["split_key"].astype(str),
            split_membership["split"].astype(str),
        )
    )
    keys = split_membership_keys(working).astype(str)
    working["split"] = keys.map(key_map)
    working["M22_binding_enrichment_adj"] = pd.to_numeric(
        working["M22_binding_enrichment_adj"],
        errors="coerce",
    )
    working["aa"] = working["aa"].astype(str).str.strip()
    working = working.dropna(subset=["M22_binding_enrichment_adj", "split"]).copy()
    working = working[working["aa"] != ""].copy()
    working = working[working["M22_binding_enrichment_adj"] > float(ENRICHMENT_THRESHOLD)].copy()

    return {
        split: working[working["split"] == split].copy().reset_index(drop=True)
        for split in ("train", "val", "test")
    }


def main() -> None:
    if len(sys.argv) != 1:
        raise SystemExit("This script takes no CLI arguments. Edit constants in tests/unlikelihood_stats.py.")

    cfg = _build_cfg()
    sequences_by_split = build_good_split_sequences(cfg=cfg, to_absolute_path=_to_abs)
    row_splits = _build_row_level_good_splits()

    train_set = set(sequences_by_split["train"])
    val_set = set(sequences_by_split["val"])
    test_set = set(sequences_by_split["test"])

    unwanted_stats = _build_unwanted_stats(
        unwanted_set_path=Path(UNWANTED_SET_PATH),
        unwanted_summary_csv=Path(UNWANTED_SUMMARY_CSV),
    )
    unwanted_map_raw: Dict[int, set[str]] = {}
    if Path(UNWANTED_SET_PATH).exists():
        with Path(UNWANTED_SET_PATH).open("r", encoding="utf-8") as fh:
            _payload = json.load(fh)
        unwanted_map_raw = {int(k): {str(x) for x in v} for k, v in _payload.items()}

    payload: Dict[str, Any] = {
        "config": {
            "raw_csv": _to_abs(RAW_CSV),
            "processed_dir": _to_abs(PROCESSED_DIR),
            "enrichment_threshold": float(ENRICHMENT_THRESHOLD),
            "seed": int(SEED),
            "train_frac": float(TRAIN_FRAC),
            "val_frac": float(VAL_FRAC),
            "test_frac": float(TEST_FRAC),
            "hamming_distance": int(HAMMING_DISTANCE),
            "stratify_bins": int(STRATIFY_BINS),
            "batch_size": int(BATCH_SIZE),
            "unwanted_set_path": _to_abs(UNWANTED_SET_PATH),
            "unwanted_summary_csv": _to_abs(UNWANTED_SUMMARY_CSV),
            "preview_variants": int(PREVIEW_VARIANTS),
        },
        "overall": {
            "total_rows_train_val_test": int(sum(len(v) for v in row_splits.values())),
            "total_unique_sequences_train_val_test": int(len(train_set | val_set | test_set)),
            "split_sequence_counts": {
                split: int(len(sequences))
                for split, sequences in sequences_by_split.items()
            },
            "split_unique_sequence_counts": {
                "train": int(len(train_set)),
                "val": int(len(val_set)),
                "test": int(len(test_set)),
            },
        },
        "overlap_checks": {
            "train_val_overlap_unique_sequences": int(len(train_set & val_set)),
            "train_test_overlap_unique_sequences": int(len(train_set & test_set)),
            "val_test_overlap_unique_sequences": int(len(val_set & test_set)),
        },
        "splits": {
            "train": _split_summary(row_splits["train"], batch_size=int(BATCH_SIZE), drop_last=True),
            "val": _split_summary(row_splits["val"], batch_size=int(BATCH_SIZE), drop_last=False),
            "test": _split_summary(row_splits["test"], batch_size=int(BATCH_SIZE), drop_last=False),
        },
        "unwanted_set_stats": unwanted_stats,
        "ground_truth_variants": {
            split: {
                "selection_rule": f"M22_binding_enrichment_adj > {float(ENRICHMENT_THRESHOLD)} and split={split}",
                "mutation_level_summary": _mutation_level_summary(df),
                "unwanted_mutation_presence": _flag_unwanted_events(df, unwanted_map_raw),
                "top_enrichment_examples": (
                    df.sort_values("M22_binding_enrichment_adj", ascending=False)
                    .head(int(PREVIEW_VARIANTS))[["aa", "mut", "num_mut", "M22_binding_enrichment_adj"]]
                    .to_dict(orient="records")
                ),
                "low_enrichment_examples_within_good_set": (
                    df.sort_values("M22_binding_enrichment_adj", ascending=True)
                    .head(int(PREVIEW_VARIANTS))[["aa", "mut", "num_mut", "M22_binding_enrichment_adj"]]
                    .to_dict(orient="records")
                ),
            }
            for split, df in row_splits.items()
        },
    }

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
