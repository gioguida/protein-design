"""Preprocessing utilities for unlikelihood training artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

MUTATION_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def _parse_mutations(mut_str: str) -> List[Tuple[int, str, str]]:
    out: List[Tuple[int, str, str]] = []
    for token in str(mut_str).split(";"):
        token = token.strip()
        if token in {"", "0", "WT", "wt"}:
            continue
        match = MUTATION_RE.match(token)
        if match is None:
            raise ValueError(f"Could not parse mutation token: {token!r}")
        wt_aa, pos_str, mut_aa = match.groups()
        out.append((int(pos_str), wt_aa, mut_aa))
    return out


def _infer_total_reads(df: pd.DataFrame) -> pd.Series:
    if "total_reads" in df.columns:
        return pd.to_numeric(df["total_reads"], errors="coerce").astype(float)

    if {"M22_binding_count_adj", "M22_non_binding_count_adj"}.issubset(df.columns):
        return (
            pd.to_numeric(df["M22_binding_count_adj"], errors="coerce").astype(float)
            + pd.to_numeric(df["M22_non_binding_count_adj"], errors="coerce").astype(float)
        )

    if {"count_ED2M22pos", "count_ED2M22neg"}.issubset(df.columns):
        return (
            pd.to_numeric(df["count_ED2M22pos"], errors="coerce").astype(float)
            + pd.to_numeric(df["count_ED2M22neg"], errors="coerce").astype(float)
        )

    raise ValueError(
        "Could not infer total reads. Expected one of: "
        "'total_reads', ('M22_binding_count_adj' and 'M22_non_binding_count_adj'), "
        "or ('count_ED2M22pos' and 'count_ED2M22neg')."
    )


def _rows_to_substitutions(
    df: pd.DataFrame,
    enrichment_col: str,
    wt_seq: str,
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for mut_str, enrichment in zip(df["mut"], df[enrichment_col]):
        for pos, wt_aa, mut_aa in _parse_mutations(mut_str):
            if pos < 1 or pos > len(wt_seq):
                raise ValueError(
                    f"Mutation position {pos} out of range for WT length {len(wt_seq)}."
                )
            expected_wt = wt_seq[pos - 1]
            if wt_aa != expected_wt:
                raise ValueError(
                    f"WT mismatch at position {pos}: token={wt_aa!r}, expected={expected_wt!r}."
                )
            records.append(
                {
                    "position": pos,
                    "amino_acid": mut_aa,
                    "wt_amino_acid": expected_wt,
                    "enrichment": float(enrichment),
                }
            )
    return pd.DataFrame.from_records(records)


def _build_unwanted_lookup(table: pd.DataFrame, min_observations: int) -> Dict[int, List[str]]:
    filtered = table[
        (table["mean_enrichment"] < 0.0)
        & (table["n_obs"] >= int(min_observations))
    ].copy()
    if filtered.empty:
        return {}

    grouped = (
        filtered.sort_values(["position", "amino_acid"])
        .groupby("position", sort=True)["amino_acid"]
        .apply(list)
    )
    return {int(pos): [str(aa) for aa in aas] for pos, aas in grouped.items()}


def _validate_no_wt_in_unwanted(unwanted: Dict[int, List[str]], wt_seq: str) -> None:
    for pos, aas in unwanted.items():
        wt_aa = wt_seq[int(pos) - 1]
        if wt_aa in aas:
            raise ValueError(
                f"Unwanted set contains WT residue at position {pos}: {wt_aa!r}"
            )


def build_unwanted_set(
    raw_csv_path: Path,
    processed_dir: Path,
    enrichment_col: str,
    wt_seq: str,
    min_total_reads: int,
    min_observations: int,
    summary_csv_name: str,
    unwanted_json_name: str,
) -> Tuple[Path, Path]:
    df = pd.read_csv(raw_csv_path)

    required_cols = {"mut", "num_mut", enrichment_col}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {raw_csv_path}: {', '.join(sorted(missing))}"
        )

    df = df.copy()
    df[enrichment_col] = pd.to_numeric(df[enrichment_col], errors="coerce").astype(float)
    df["num_mut"] = pd.to_numeric(df["num_mut"], errors="coerce").astype(float)
    df["total_reads"] = _infer_total_reads(df)

    df = df.dropna(subset=[enrichment_col, "num_mut", "mut", "total_reads"]).copy()
    df = df[df["total_reads"] >= float(min_total_reads)].copy()

    singles = df[df["num_mut"] == 1].copy()
    use_singles = len(singles) > 100
    all_substitutions = _rows_to_substitutions(
        df=df,
        enrichment_col=enrichment_col,
        wt_seq=wt_seq,
    )
    if all_substitutions.empty:
        raise ValueError("No parsed substitutions found after filtering.")

    all_summary = (
        all_substitutions
        .groupby(["position", "amino_acid", "wt_amino_acid"], as_index=False)
        .agg(
            n_obs_all=("enrichment", "size"),
            mean_enrichment_all=("enrichment", "mean"),
            median_enrichment_all=("enrichment", "median"),
        )
    )

    if use_singles:
        single_substitutions = _rows_to_substitutions(
            df=singles,
            enrichment_col=enrichment_col,
            wt_seq=wt_seq,
        )
        single_summary = (
            single_substitutions
            .groupby(["position", "amino_acid", "wt_amino_acid"], as_index=False)
            .agg(
                n_obs_single=("enrichment", "size"),
                mean_enrichment_single=("enrichment", "mean"),
                median_enrichment_single=("enrichment", "median"),
            )
        )
        summary = all_summary.merge(
            single_summary,
            on=["position", "amino_acid", "wt_amino_acid"],
            how="left",
        )
        single_ok = summary["n_obs_single"].fillna(0).astype(float) >= float(min_observations)
        summary["n_obs"] = summary["n_obs_all"]
        summary["mean_enrichment"] = summary["mean_enrichment_all"]
        summary["median_enrichment"] = summary["median_enrichment_all"]
        summary.loc[single_ok, "n_obs"] = summary.loc[single_ok, "n_obs_single"]
        summary.loc[single_ok, "mean_enrichment"] = summary.loc[single_ok, "mean_enrichment_single"]
        summary.loc[single_ok, "median_enrichment"] = summary.loc[single_ok, "median_enrichment_single"]
        summary["source_subset"] = "single_mutants_with_all_variants_fallback"
        summary["mean_source"] = "all_variants"
        summary.loc[single_ok, "mean_source"] = "single_mutants"
    else:
        summary = all_summary.copy()
        summary["n_obs"] = summary["n_obs_all"]
        summary["mean_enrichment"] = summary["mean_enrichment_all"]
        summary["median_enrichment"] = summary["median_enrichment_all"]
        summary["source_subset"] = "all_variants"
        summary["mean_source"] = "all_variants"

    summary = summary.sort_values(["position", "amino_acid"]).reset_index(drop=True)
    summary["is_wt_residue"] = summary["amino_acid"] == summary["wt_amino_acid"]
    summary["is_unwanted"] = (
        (summary["mean_enrichment"] < 0.0)
        & (summary["n_obs"] >= int(min_observations))
    )

    unwanted_lookup = _build_unwanted_lookup(
        table=summary,
        min_observations=min_observations,
    )
    _validate_no_wt_in_unwanted(unwanted_lookup, wt_seq=wt_seq)

    processed_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = processed_dir / summary_csv_name
    unwanted_json_path = processed_dir / unwanted_json_name

    summary.to_csv(summary_csv_path, index=False)
    with unwanted_json_path.open("w", encoding="utf-8") as fh:
        json.dump(unwanted_lookup, fh, indent=2, sort_keys=True)

    return summary_csv_path, unwanted_json_path
