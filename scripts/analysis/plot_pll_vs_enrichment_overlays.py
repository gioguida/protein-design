"""PLL vs enrichment scatter plots across datasets, with Gibbs/beam overlays.

For each model variant and each requested DMS dataset, compute sequence-level
CDR-H3 PLL and render:
  - DMS background: PLL vs assay enrichment
  - overlay points: Gibbs/beam samples matched by CDR-H3 against DMS enrichment.

Backwards compatible defaults still support built-in dataset keys and
directory-based sampler CSV discovery.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer

from protein_design.constants import C05_CDRH3

from gibbs_diagnostics import (
    DMS_DATASETS,
    ESM2_MODEL_ID,
    load_esm_for_mlm,
    load_gibbs_csv,
    per_position_cdr_log_probs,
    sequence_pll,
)

SEED = 42
FITNESS = [
    ("M22_binding_enrichment_adj", "M22 binding enrichment", "M22"),
    ("SI06_binding_enrichment_adj", "SI06 binding enrichment", "SI06"),
]
SAMPLER_SPECS = [
    ("gibbs_dist", "gibbs distribution", "o", "#f39c12"),
    ("gibbs_fit", "gibbs fitness", "s", "#d35400"),
    ("beam_dist", "beam distribution", "^", "#2980b9"),
    ("beam_fit", "beam fitness", "D", "#16a085"),
]
SAMPLER_KEYS = {k for k, _, _, _ in SAMPLER_SPECS}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("plot_pll_vs_enrichment_overlays")


def parse_variant_spec(spec: str) -> Tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--variant must be LABEL=CHECKPOINT, got {spec!r}")
    label, ckpt = spec.split("=", 1)
    return label.strip(), ckpt.strip()


def parse_dataset_spec(spec: str) -> Tuple[str, dict[str, str]]:
    parts = spec.split("=")
    if len(parts) < 2 or len(parts) > 3:
        raise argparse.ArgumentTypeError(
            "--dataset-spec must be KEY=M22_CSV_PATH[=SI06_CSV_PATH]"
        )
    key = parts[0].strip()
    m22 = parts[1].strip()
    si06 = parts[2].strip() if len(parts) == 3 and parts[2].strip() else None
    if not key or not m22:
        raise argparse.ArgumentTypeError("dataset key and M22 path must be non-empty")
    out = {"m22": m22}
    if si06 is not None:
        out["si06"] = si06
    return key, out


def parse_overlay_csv_spec(spec: str) -> Tuple[str, str, Path]:
    parts = spec.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--overlay-csv must be SAMPLER=VARIANT=CSV_PATH")
    sampler, variant, csv_path = parts[0].strip(), parts[1].strip(), parts[2].strip()
    if sampler not in SAMPLER_KEYS:
        raise argparse.ArgumentTypeError(
            f"--overlay-csv sampler must be one of {sorted(SAMPLER_KEYS)}, got {sampler!r}"
        )
    if not variant or not csv_path:
        raise argparse.ArgumentTypeError("--overlay-csv variant and path must be non-empty")
    return sampler, variant, Path(csv_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--variant",
        action="append",
        required=True,
        type=parse_variant_spec,
        help="LABEL=CHECKPOINT; repeat once per model variant. CHECKPOINT may be empty.",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["ed2", "ed5", "ed811"],
        help="Dataset keys to plot. Defaults to built-in ed2/ed5/ed811.",
    )
    p.add_argument(
        "--dataset-spec",
        action="append",
        default=[],
        type=parse_dataset_spec,
        help="Override/add dataset paths with KEY=M22_CSV_PATH[=SI06_CSV_PATH]. Repeatable.",
    )
    p.add_argument("--gibbs-dist-dir", type=Path, default=Path("outputs/gibbs/distribution"))
    p.add_argument("--gibbs-fit-dir", type=Path, default=Path("outputs/gibbs/fitness"))
    p.add_argument("--beam-dist-dir", type=Path, default=Path("outputs/beam_search/distribution"))
    p.add_argument("--beam-fit-dir", type=Path, default=Path("outputs/beam_search/fitness"))
    p.add_argument(
        "--overlay-csv",
        action="append",
        default=[],
        type=parse_overlay_csv_spec,
        help="Explicit overlay CSV mapping: SAMPLER=VARIANT=CSV_PATH. Repeatable.",
    )
    p.add_argument(
        "--include-samplers",
        nargs="+",
        default=[k for k, _, _, _ in SAMPLER_SPECS],
        choices=sorted(SAMPLER_KEYS),
        help="Subset of sampler overlays to include.",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--max-overlay",
        type=int,
        default=4000,
        help="Per source cap before PLL inference (for very large CSVs).",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def _dataset_label(dataset_key: str) -> str:
    if dataset_key == "ed811":
        return "ed8"
    return dataset_key


def _build_dataset_map(args: argparse.Namespace) -> Dict[str, Dict[str, str]]:
    dataset_map: Dict[str, Dict[str, str]] = {
        key: dict(paths) for key, paths in DMS_DATASETS.items()
    }
    for key, paths in args.dataset_spec:
        dataset_map[key] = paths
    for ds in args.datasets:
        if ds not in dataset_map:
            raise ValueError(
                f"Dataset key {ds!r} not found. Use --dataset-spec to define it."
            )
    return dataset_map


def load_dms_full(paths: Dict[str, str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    if "m22" in paths and paths["m22"]:
        frames.append(pd.read_csv(paths["m22"])[["aa", "M22_binding_enrichment_adj"]])
    if "si06" in paths and paths["si06"]:
        frames.append(pd.read_csv(paths["si06"])[["aa", "SI06_binding_enrichment_adj"]])
    if not frames:
        return pd.DataFrame(columns=["aa", "M22_binding_enrichment_adj", "SI06_binding_enrichment_adj"])
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on="aa", how="outer")
    for col in ("M22_binding_enrichment_adj", "SI06_binding_enrichment_adj"):
        if col not in merged.columns:
            merged[col] = np.nan
    merged = merged[merged["aa"].astype(str).str.len() == len(C05_CDRH3)].copy()
    merged = merged.drop_duplicates(subset=["aa"], keep="first").reset_index(drop=True)
    return merged


def load_sampler_cdrh3(path: Path, max_n: int) -> List[str]:
    if not path.exists():
        return []
    df = load_gibbs_csv(path)
    cdr = df["cdrh3"].astype(str).tolist()
    if len(cdr) > max_n:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(cdr), size=max_n, replace=False)
        cdr = [cdr[i] for i in idx]
    return cdr


def compute_pll_map(
    model,
    tokenizer,
    device: torch.device,
    sequences: List[str],
    batch_size: int,
) -> Dict[str, float]:
    if not sequences:
        return {}
    per_pos = per_position_cdr_log_probs(model, tokenizer, sequences, device, batch_size)
    pll = sequence_pll(per_pos)
    return {seq: float(score) for seq, score in zip(sequences, pll)}


def make_plot(
    out_path: Path,
    variant: str,
    dataset_label: str,
    fitness_col: str,
    fitness_label: str,
    dms_df: pd.DataFrame,
    sampler_rows: List[dict],
) -> None:
    valid = dms_df[fitness_col].notna() & dms_df["pll"].notna()
    x = dms_df.loc[valid, "pll"].to_numpy(dtype=np.float32)
    y = dms_df.loc[valid, fitness_col].to_numpy(dtype=np.float32)
    if len(x) < 2:
        log.warning("[%s/%s/%s] not enough DMS points to plot", variant, dataset_label, fitness_col)
        return

    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    ax.scatter(x, y, s=8, c="lightgrey", alpha=0.45, label=f"DMS (n={len(x)})", zorder=1)

    slope, intercept = np.polyfit(x, y, 1)
    xr = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    ax.plot(xr, slope * xr + intercept, color="black", linewidth=1.2, zorder=2)
    rp, pp = pearsonr(x, y)
    rs, ps = spearmanr(x, y)

    legend_rows = []
    for row in sampler_rows:
        matched = row["matched"]
        total = row["total"]
        if matched.empty:
            legend_rows.append(f"{row['label']}: 0/{total} matched")
            continue
        ax.scatter(
            matched["pll"],
            matched[fitness_col],
            s=28,
            marker=row["marker"],
            c=row["color"],
            alpha=0.78,
            edgecolors="black",
            linewidths=0.25,
            zorder=3,
            label=f"{row['label']} ({len(matched)}/{total} matched)",
        )
        legend_rows.append(f"{row['label']}: {len(matched)}/{total} matched")

    text = (
        f"DMS Pearson r = {rp:.3f} (p={pp:.1e})\n"
        f"DMS Spearman rho = {rs:.3f} (p={ps:.1e})\n"
        + "\n".join(legend_rows)
    )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.88, edgecolor="grey"),
    )

    ax.set_xlabel("CDR-H3 PLL")
    ax.set_ylabel(fitness_label)
    ax.set_title(
        f"{variant} - {dataset_label.upper()} - PLL vs {fitness_label}\n"
        "DMS background + Gibbs/beam overlays"
    )
    ax.grid(alpha=0.22)
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_path)


def _overlay_lookup(args: argparse.Namespace) -> Dict[Tuple[str, str], Path]:
    return {(sampler, variant): path for sampler, variant, path in args.overlay_csv}


def _resolve_overlay_path(
    sampler_key: str,
    variant: str,
    args: argparse.Namespace,
    lookup: Dict[Tuple[str, str], Path],
) -> Path:
    explicit = lookup.get((sampler_key, variant))
    if explicit is not None:
        return explicit
    if sampler_key == "gibbs_dist":
        return args.gibbs_dist_dir / f"{variant}.csv"
    if sampler_key == "gibbs_fit":
        return args.gibbs_fit_dir / f"{variant}.csv"
    if sampler_key == "beam_dist":
        return args.beam_dist_dir / f"{variant}.csv"
    if sampler_key == "beam_fit":
        return args.beam_fit_dir / f"{variant}.csv"
    return Path("__missing__.csv")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    include = set(args.include_samplers)
    dataset_map = _build_dataset_map(args)
    overlay_lookup = _overlay_lookup(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_ID)

    dms_by_dataset: Dict[str, pd.DataFrame] = {}
    for ds in args.datasets:
        df = load_dms_full(dataset_map[ds])
        dms_by_dataset[ds] = df
        log.info("DMS %s: %d unique CDR-H3 rows", ds, len(df))

    for variant, checkpoint in args.variant:
        log.info("=== %s (checkpoint=%r) ===", variant, checkpoint)
        model = load_esm_for_mlm(checkpoint).eval().to(device)
        for p_ in model.parameters():
            p_.requires_grad = False
        if device.type == "cuda":
            model = model.half()

        for ds in args.datasets:
            ds_label = _dataset_label(ds)
            dms_df = dms_by_dataset[ds].copy()

            sampler_cdr: Dict[str, List[str]] = {}
            for key, _, _, _ in SAMPLER_SPECS:
                if key not in include:
                    sampler_cdr[key] = []
                    continue
                sampler_path = _resolve_overlay_path(key, variant, args, overlay_lookup)
                sampler_cdr[key] = load_sampler_cdrh3(sampler_path, args.max_overlay)

            union_sequences = list(
                dict.fromkeys(
                    dms_df["aa"].astype(str).tolist()
                    + sampler_cdr["gibbs_dist"]
                    + sampler_cdr["gibbs_fit"]
                    + sampler_cdr["beam_dist"]
                    + sampler_cdr["beam_fit"]
                )
            )
            if not union_sequences:
                log.warning("[%s/%s] no sequences to score - skipping", variant, ds)
                continue

            pll_map = compute_pll_map(model, tokenizer, device, union_sequences, args.batch_size)
            dms_df["pll"] = dms_df["aa"].map(pll_map)

            lookup_cols = ["aa", "M22_binding_enrichment_adj", "SI06_binding_enrichment_adj"]
            dms_lookup = dms_df[lookup_cols].rename(columns={"aa": "cdrh3"})
            sampler_rows: List[dict] = []
            for key, label, marker, color in SAMPLER_SPECS:
                cdrs = sampler_cdr[key]
                overlay_df = pd.DataFrame({"cdrh3": cdrs})
                if overlay_df.empty:
                    matched = overlay_df.assign(
                        pll=np.array([], dtype=np.float32),
                        M22_binding_enrichment_adj=np.array([], dtype=np.float32),
                        SI06_binding_enrichment_adj=np.array([], dtype=np.float32),
                    )
                else:
                    overlay_df["pll"] = overlay_df["cdrh3"].map(pll_map)
                    matched = overlay_df.merge(dms_lookup, on="cdrh3", how="inner")
                sampler_rows.append(
                    {
                        "key": key,
                        "label": label,
                        "marker": marker,
                        "color": color,
                        "total": len(cdrs),
                        "matched": matched,
                    }
                )

            out_ds_dir = args.output_dir / ds_label
            out_ds_dir.mkdir(parents=True, exist_ok=True)
            for fitness_col, fitness_label, short in FITNESS:
                if dms_df[fitness_col].notna().sum() < 2:
                    log.info("[%s/%s] skipping %s (no assay values)", variant, ds, short)
                    continue
                out_path = out_ds_dir / f"pll_vs_enrichment_{variant}_{ds_label}_{short}.png"
                make_plot(out_path, variant, ds_label, fitness_col, fitness_label, dms_df, sampler_rows)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
