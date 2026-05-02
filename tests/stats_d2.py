#!/usr/bin/env python3
"""Simple stats + plots for ED2/ED5 M22 enrichment data."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MUT_TOKEN_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
DEFAULT_INPUT_CSV = Path("data/raw/ED2_M22_binding_enrichment.csv")
DEFAULT_FIGURES_DIR = Path("plots")
DEFAULT_LOW_COUNT_THRESHOLD = 10


def find_existing_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find a {label} column. Tried: {candidates}")


def resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    return {
        "aa": find_existing_column(df, ["aa"], "sequence"),
        "mut": find_existing_column(df, ["mut"], "mutation"),
        "num_mut": find_existing_column(df, ["num_mut"], "mutation count"),
        "enrichment": find_existing_column(
            df,
            ["M22_binding_enrichment_adj", "M22_enrichment_PosdivPre_adj"],
            "enrichment",
        ),
        "pos_count": find_existing_column(
            df,
            [
                "count_ED2M22pos",
                "count_ED5M22pos",
                "count_ED2M22pos_adj",
                "count_ED5M22pos_adj",
            ],
            "positive count",
        ),
        "neg_count": find_existing_column(
            df,
            [
                "count_ED2M22neg",
                "count_ED5M22neg",
                "count_ED2M22neg_adj",
                "count_ED5M22neg_adj",
            ],
            "negative count",
        ),
    }


def parse_mutation_tokens(mut_value: Any) -> list[tuple[int, str, str]]:
    if pd.isna(mut_value):
        return []
    text = str(mut_value).strip()
    if text == "" or text == "0":
        return []

    parsed: list[tuple[int, str, str]] = []
    for token in text.split(";"):
        token = token.strip()
        if token == "" or token == "0":
            continue
        match = MUT_TOKEN_RE.match(token)
        if match is None:
            continue
        wt_aa, pos_str, mut_aa = match.groups()
        parsed.append((int(pos_str), wt_aa, mut_aa))
    return parsed


def describe_series(name: str, values: pd.Series) -> None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        print(f"{name}: no valid values")
        return
    print(
        f"{name}: mean={clean.mean():.4f}, median={clean.median():.4f}, "
        f"std={clean.std():.4f}, min={clean.min():.4f}, max={clean.max():.4f}"
    )


def build_substitution_table(
    df: pd.DataFrame,
    mut_col: str,
    enrichment_col: str,
    pos_count_col: str,
    neg_count_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, (mut_text, enrichment, pos_count, neg_count) in enumerate(
        zip(
            df[mut_col].values,
            df[enrichment_col].values,
            df[pos_count_col].values,
            df[neg_count_col].values,
            strict=False,
        )
    ):
        pos_val = 0.0 if pd.isna(pos_count) else float(pos_count)
        neg_val = 0.0 if pd.isna(neg_count) else float(neg_count)
        total_reads = pos_val + neg_val
        for position, wt_aa, mut_aa in parse_mutation_tokens(mut_text):
            rows.append(
                {
                    "variant_index": int(df.index[idx]),
                    "position": int(position),
                    "wt_aa": wt_aa,
                    "mut_aa": mut_aa,
                    "enrichment": float(enrichment),
                    "pos_count": pos_val,
                    "neg_count": neg_val,
                    "total_reads": total_reads,
                }
            )
    return pd.DataFrame(rows)


def get_wild_type_enrichment(df: pd.DataFrame, num_mut_col: str, enrichment_col: str) -> float:
    wt_rows = df[df[num_mut_col] == 0]
    if wt_rows.empty:
        raise ValueError("Could not find wild-type row (num_mut == 0).")
    return float(wt_rows.iloc[0][enrichment_col])


def plot_enrichment_histogram(
    mutated_df: pd.DataFrame,
    enrichment_col: str,
    wt_enrichment: float,
    figures_dir: Path,
) -> Path:
    fig_path = figures_dir / "enrichment_histogram.png"
    plt.figure(figsize=(8, 5))
    plt.hist(mutated_df[enrichment_col], bins=50, alpha=0.8)
    plt.axvline(wt_enrichment, color="red", linestyle="--", linewidth=2, label="Wild type")
    plt.xlabel("Enrichment score")
    plt.ylabel("Number of variants")
    plt.title("Enrichment Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()
    return fig_path


def per_position_analysis(
    substitution_df: pd.DataFrame,
    wt_enrichment: float,
    figures_dir: Path,
) -> tuple[pd.DataFrame, Path]:
    per_pos = (
        substitution_df.groupby("position")["enrichment"]
        .agg(["mean", "std", "count"])
        .sort_index()
    )
    per_pos["frac_bad"] = (
        substitution_df.assign(is_bad=substitution_df["enrichment"] < wt_enrichment)
        .groupby("position")["is_bad"]
        .mean()
    )

    fig_path = figures_dir / "mean_enrichment_per_position.png"
    plt.figure(figsize=(10, 5))
    plt.errorbar(
        per_pos.index,
        per_pos["mean"],
        yerr=per_pos["std"].fillna(0.0),
        fmt="-o",
        capsize=3,
    )
    plt.axhline(wt_enrichment, color="red", linestyle="--", linewidth=1.5, label="Wild type")
    plt.xlabel("Position")
    plt.ylabel("Mean enrichment")
    plt.title("Mean Enrichment per Position (error bars = std)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()
    return per_pos, fig_path


def plot_substitution_heatmap(
    substitution_df: pd.DataFrame,
    wt_enrichment: float,
    figures_dir: Path,
) -> Path:
    pivot = substitution_df.pivot_table(
        index="position",
        columns="mut_aa",
        values="enrichment",
        aggfunc="mean",
    )
    extra_cols = [aa for aa in pivot.columns if aa not in AA_ORDER]
    pivot = pivot.reindex(columns=AA_ORDER + sorted(extra_cols))

    fig_path = figures_dir / "position_amino_acid_heatmap.png"
    fig, ax = plt.subplots(figsize=(12, 6))
    masked_values = np.ma.masked_invalid(pivot.values)
    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad(color="lightgray")
    image = ax.imshow(masked_values, aspect="auto", cmap=cmap)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Mean enrichment")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns), rotation=90)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(v) for v in pivot.index.tolist()])
    ax.set_xlabel("Mutated amino acid")
    ax.set_ylabel("Position")
    ax.set_title("Position x Amino Acid Enrichment Heatmap")

    below_wt = np.where((pivot.values < wt_enrichment) & np.isfinite(pivot.values))
    if len(below_wt[0]) > 0:
        ax.scatter(
            below_wt[1],
            below_wt[0],
            marker="x",
            c="black",
            s=10,
            linewidths=0.5,
            label="Below WT",
        )
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()
    return fig_path


def plot_count_quality(
    df: pd.DataFrame,
    enrichment_col: str,
    pos_count_col: str,
    neg_count_col: str,
    low_count_threshold: int,
    figures_dir: Path,
) -> tuple[pd.Series, Path]:
    pos_vals = pd.to_numeric(df[pos_count_col], errors="coerce").fillna(0.0)
    neg_vals = pd.to_numeric(df[neg_count_col], errors="coerce").fillna(0.0)
    total_reads = pos_vals + neg_vals
    low_mask = total_reads < low_count_threshold

    fig_path = figures_dir / "count_quality_scatter.png"
    plt.figure(figsize=(8, 6))
    x = np.log10(pos_vals + 1.0)
    y = np.log10(neg_vals + 1.0)
    enrich_vals = pd.to_numeric(df[enrichment_col], errors="coerce")
    sc = plt.scatter(x, y, c=enrich_vals, s=8, alpha=0.6, cmap="viridis")
    cbar = plt.colorbar(sc)
    cbar.set_label("Enrichment")

    if low_mask.any():
        plt.scatter(
            x[low_mask],
            y[low_mask],
            facecolors="none",
            edgecolors="red",
            s=30,
            linewidths=0.7,
            label=f"Total reads < {low_count_threshold}",
        )
        plt.legend()

    plt.xlabel("log10(positive count + 1)")
    plt.ylabel("log10(negative count + 1)")
    plt.title("Count Quality Check")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close()
    return total_reads, fig_path


def threshold_suggestions(
    mutated_df: pd.DataFrame,
    substitution_df: pd.DataFrame,
    enrichment_col: str,
    wt_enrichment: float,
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    enrich = pd.to_numeric(mutated_df[enrichment_col], errors="coerce")
    p25 = float(enrich.quantile(0.25))
    std = float(enrich.std())
    pair_mean = (
        substitution_df.groupby(["position", "mut_aa"], as_index=False)["enrichment"]
        .mean()
        .rename(columns={"enrichment": "pair_mean_enrichment"})
    )
    total_unique_pairs = int(len(pair_mean))

    threshold_defs = [
        ("bottom_25_percentile", p25, "<= value"),
        ("below_wild_type", wt_enrichment, "< value"),
        ("below_wild_type_minus_1std", wt_enrichment - std, "< value"),
        ("enrichment_below_zero", 0.0, "< value"),
    ]

    rows = []
    for name, value, mode in threshold_defs:
        if mode == "<= value":
            selected_variants = mutated_df[enrich <= value]
            selected_pairs = pair_mean[pair_mean["pair_mean_enrichment"] <= value]
        else:
            selected_variants = mutated_df[enrich < value]
            selected_pairs = pair_mean[pair_mean["pair_mean_enrichment"] < value]

        pair_count = int(len(selected_pairs))
        rows.append(
            {
                "threshold_name": name,
                "threshold_value": float(value),
                "num_variants": int(len(selected_variants)),
                "num_unique_position_aa_pairs": pair_count,
                "pair_fraction": 0.0 if total_unique_pairs == 0 else pair_count / total_unique_pairs,
            }
        )

    out_df = pd.DataFrame(rows)
    by_name = {row["threshold_name"]: row for row in rows}
    strict = by_name["below_wild_type_minus_1std"]
    below_wt = by_name["below_wild_type"]
    q25 = by_name["bottom_25_percentile"]

    if strict["pair_fraction"] >= 0.10:
        recommendation = strict
        reason = (
            "It focuses on clearly bad variants while keeping a usable number of negatives."
        )
    elif below_wt["pair_fraction"] <= 0.60:
        recommendation = below_wt
        reason = "It keeps a balanced negative set without being overly strict."
    else:
        recommendation = q25
        reason = (
            "Below-WT is very broad here, so bottom-25% is a cleaner cut for clearly bad examples."
        )

    return out_df, recommendation, reason


def main() -> None:
    figures_dir = DEFAULT_FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)
    low_count_threshold = DEFAULT_LOW_COUNT_THRESHOLD

    df = pd.read_csv(DEFAULT_INPUT_CSV)
    cols = resolve_columns(df)
    enrichment_col = cols["enrichment"]
    pos_count_col = cols["pos_count"]
    neg_count_col = cols["neg_count"]
    mut_col = cols["mut"]
    num_mut_col = cols["num_mut"]

    mutated_df = df[df[num_mut_col] > 0].copy()
    wt_enrichment = get_wild_type_enrichment(df, num_mut_col, enrichment_col)
    substitution_df = build_substitution_table(
        mutated_df,
        mut_col=mut_col,
        enrichment_col=enrichment_col,
        pos_count_col=pos_count_col,
        neg_count_col=neg_count_col,
    )

    print("\n1) Basic data overview")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nDtypes:")
    print(df.dtypes.to_string())
    print("\nSample rows:")
    print(df.head(5).to_string(index=False))

    print("\nSummary stats")
    describe_series("Enrichment", df[enrichment_col])
    describe_series("Positive counts", df[pos_count_col])
    describe_series("Negative counts", df[neg_count_col])

    pos_vals = pd.to_numeric(df[pos_count_col], errors="coerce").fillna(0.0)
    neg_vals = pd.to_numeric(df[neg_count_col], errors="coerce").fillna(0.0)
    total_reads = pos_vals + neg_vals
    low_count_mask = total_reads < low_count_threshold
    print(f"\nTotal variants (all rows): {len(df)}")
    print(f"Total mutated variants (num_mut > 0): {len(mutated_df)}")
    print(f"Unique mutated positions: {substitution_df['position'].nunique()}")
    print(
        f"Variants with very low counts (< {low_count_threshold} total reads): "
        f"{int(low_count_mask.sum())}"
    )

    print("\n2) Enrichment score distribution")
    hist_path = plot_enrichment_histogram(
        mutated_df=mutated_df,
        enrichment_col=enrichment_col,
        wt_enrichment=wt_enrichment,
        figures_dir=figures_dir,
    )
    enrich_mut = pd.to_numeric(mutated_df[enrichment_col], errors="coerce")
    frac_above = float((enrich_mut > wt_enrichment).mean())
    frac_below = float((enrich_mut < wt_enrichment).mean())
    frac_equal = float((enrich_mut == wt_enrichment).mean())
    print(f"Wild type enrichment: {wt_enrichment:.4f}")
    print(f"Fraction above WT: {frac_above:.3f}")
    print(f"Fraction below WT: {frac_below:.3f}")
    print(f"Fraction equal to WT: {frac_equal:.3f}")
    print(f"Saved: {hist_path}")

    print("\n3) Per-position analysis")
    per_pos, per_pos_fig_path = per_position_analysis(
        substitution_df=substitution_df,
        wt_enrichment=wt_enrichment,
        figures_dir=figures_dir,
    )
    print("Per-position mean/std enrichment:")
    print(per_pos.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"Saved: {per_pos_fig_path}")

    min_obs = 20
    eligible = per_pos[per_pos["count"] >= min_obs].copy()
    conserved = eligible.sort_values("frac_bad", ascending=False).head(5)
    tolerant = eligible.sort_values("frac_bad", ascending=True).head(5)
    print(f"\nHighly conserved positions (high bad-mutation fraction, min {min_obs} obs):")
    print(conserved[["count", "mean", "std", "frac_bad"]].to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nTolerant positions (low bad-mutation fraction, min {min_obs} obs):")
    print(tolerant[["count", "mean", "std", "frac_bad"]].to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n4) Per-substitution heatmap")
    heatmap_path = plot_substitution_heatmap(
        substitution_df=substitution_df,
        wt_enrichment=wt_enrichment,
        figures_dir=figures_dir,
    )
    print(f"Saved: {heatmap_path}")

    print("\n5) Count quality check")
    _, scatter_path = plot_count_quality(
        df=mutated_df,
        enrichment_col=enrichment_col,
        pos_count_col=pos_count_col,
        neg_count_col=neg_count_col,
        low_count_threshold=low_count_threshold,
        figures_dir=figures_dir,
    )
    mut_total_reads = (
        pd.to_numeric(mutated_df[pos_count_col], errors="coerce").fillna(0.0)
        + pd.to_numeric(mutated_df[neg_count_col], errors="coerce").fillna(0.0)
    )
    mut_low_count = int((mut_total_reads < low_count_threshold).sum())
    print(f"Low-count mutated variants (< {low_count_threshold} reads): {mut_low_count}")
    print(f"Saved: {scatter_path}")

    print("\n6) Candidate thresholds for unwanted set")
    threshold_df, recommendation, reason = threshold_suggestions(
        mutated_df=mutated_df,
        substitution_df=substitution_df,
        enrichment_col=enrichment_col,
        wt_enrichment=wt_enrichment,
    )
    print(
        threshold_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nShort recommendation")
    print(
        f"Use '{recommendation['threshold_name']}' (threshold={recommendation['threshold_value']:.4f}). "
        f"{reason}"
    )
    print(
        f"It yields {recommendation['num_unique_position_aa_pairs']} unique position-amino-acid pairs "
        f"from {recommendation['num_variants']} variants."
    )
    print(
        f"Also filter variants with total reads < {low_count_threshold} to reduce noisy negatives."
    )


if __name__ == "__main__":
    main()
