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
from omegaconf import OmegaConf
from scipy.stats import pearsonr, rankdata, spearmanr
from transformers import AutoTokenizer

from protein_design.constants import C05_CDRH3, WT_M22_BINDING_ENRICHMENT, WILD_TYPE
from protein_design.dpo.data_processing import ensure_delta_m22_binding_enrichment
from protein_design.dpo.splitting import (
    build_or_load_cluster_split_membership,
    split_membership_keys,
)

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
ENRICHMENT_BIMODAL_THRESHOLD = 0.0

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
    p.add_argument("--dms-m22-col", default="M22_binding_enrichment_adj")
    p.add_argument("--dms-si06-col", default="SI06_binding_enrichment_adj")
    p.add_argument(
        "--split-mode",
        choices=("full", "train_dpo", "val_dpo", "test_dpo"),
        default="full",
        help="Subset DMS background to one DPO cluster split partition.",
    )
    p.add_argument(
        "--dpo-split-config",
        type=Path,
        default=Path("conf/data/dpo/default.yaml"),
        help="DPO config used to read split fractions and clustering knobs.",
    )
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument(
        "--split-cache-dir",
        type=Path,
        default=None,
        help="Cache root for split membership CSVs. Defaults to <output-dir>/split_cache.",
    )
    p.add_argument(
        "--wt-seq",
        default=WILD_TYPE,
        help="WT CDR-H3 sequence used for dedicated WT marker/PLL scoring.",
    )
    p.add_argument(
        "--wt-m22-enrichment",
        type=float,
        default=float(WT_M22_BINDING_ENRICHMENT),
        help="WT M22 enrichment value used for WT marker on M22 plots.",
    )
    p.add_argument(
        "--wt-si06-enrichment",
        type=float,
        default=None,
        help="Optional WT SI06 enrichment value used for WT marker on SI06 plots.",
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


def load_dms_full(
    paths: Dict[str, str],
    m22_col: str = "M22_binding_enrichment_adj",
    si06_col: str = "SI06_binding_enrichment_adj",
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    if "m22" in paths and paths["m22"]:
        m22_df = pd.read_csv(paths["m22"])
        if "aa" not in m22_df.columns or m22_col not in m22_df.columns:
            raise ValueError(
                f"M22 dataset at {paths['m22']} must contain columns 'aa' and {m22_col!r}."
            )
        if m22_col != "M22_binding_enrichment_adj":
            m22_df = m22_df.rename(columns={m22_col: "M22_binding_enrichment_adj"})
        frames.append(m22_df)
    if "si06" in paths and paths["si06"]:
        si06_df = pd.read_csv(paths["si06"])
        required = {"aa", si06_col}
        missing = required.difference(si06_df.columns)
        if missing:
            raise ValueError(
                f"SI06 dataset at {paths['si06']} missing required columns: {sorted(missing)}"
            )
        if si06_col != "SI06_binding_enrichment_adj":
            si06_df = si06_df.rename(columns={si06_col: "SI06_binding_enrichment_adj"})
        frames.append(si06_df[["aa", "SI06_binding_enrichment_adj"]])
    if not frames:
        return pd.DataFrame(columns=["aa", "M22_binding_enrichment_adj", "SI06_binding_enrichment_adj"])
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on="aa", how="outer")
    for col in ("M22_binding_enrichment_adj", "SI06_binding_enrichment_adj"):
        if col not in merged.columns:
            merged[col] = np.nan
    merged = merged[merged["aa"].astype(str).str.len() == len(C05_CDRH3)].copy()
    merged["aa"] = merged["aa"].astype(str).str.strip()
    merged = merged[merged["aa"] != ""].reset_index(drop=True)
    return merged


def _load_dpo_split_params(config_path: Path) -> dict[str, float | int]:
    if not config_path.exists():
        raise FileNotFoundError(f"DPO split config not found: {config_path}")
    cfg = OmegaConf.load(config_path)
    if "data" not in cfg:
        raise ValueError(f"Expected 'data' section in DPO split config: {config_path}")
    split_cfg = cfg.data.get("split") if "split" in cfg.data else None
    return {
        "train_frac": float(cfg.data.get("train_frac", 0.8)),
        "val_frac": float(cfg.data.get("val_frac", 0.1)),
        "test_frac": float(cfg.data.get("test_frac", 0.1)),
        "stratify_bins": int(getattr(split_cfg, "stratify_bins", 10)),
        "hamming_distance": int(getattr(split_cfg, "hamming_distance", 1)),
    }


def _filter_split_like_dpo(
    *,
    dataset_name: str,
    dataset_m22_path: Path,
    df: pd.DataFrame,
    split_name: str,
    split_params: dict[str, float | int],
    split_seed: int,
    cache_root: Path,
) -> pd.DataFrame:
    required = {"aa", "mut", "num_mut", "M22_binding_enrichment_adj"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{dataset_name} must contain {sorted(required)} for DPO-style split. Missing: {sorted(missing)}"
        )

    base_df = ensure_delta_m22_binding_enrichment(df.copy())
    cache_dir = cache_root / dataset_name.lower()
    cache_dir.mkdir(parents=True, exist_ok=True)
    membership = build_or_load_cluster_split_membership(
        base_df=base_df,
        base_csv_path=dataset_m22_path,
        processed_dir=cache_dir,
        train_frac=float(split_params["train_frac"]),
        val_frac=float(split_params["val_frac"]),
        test_frac=float(split_params["test_frac"]),
        seed=int(split_seed),
        force_rebuild=False,
        positive_threshold=0.0,
        stratify_bins=int(split_params["stratify_bins"]),
        hamming_distance=int(split_params["hamming_distance"]),
    )
    split_keys = set(
        membership.loc[membership["split"] == split_name, "split_key"].astype(str).tolist()
    )
    row_keys = split_membership_keys(base_df).astype(str)
    return base_df.loc[row_keys.isin(split_keys)].reset_index(drop=True)


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
    wt_pll: float,
    wt_seq: str,
    wt_fitness: float,
) -> None:
    valid = dms_df[fitness_col].notna() & dms_df["pll"].notna()
    x = dms_df.loc[valid, "pll"].to_numpy(dtype=np.float32)
    y = dms_df.loc[valid, fitness_col].to_numpy(dtype=np.float32)
    if len(x) < 2:
        log.warning("[%s/%s/%s] not enough DMS points to plot", variant, dataset_label, fitness_col)
        return

    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    point_colors = np.where(y > ENRICHMENT_BIMODAL_THRESHOLD, "#e15759", "#4e79a7")
    ax.scatter(x, y, s=8, c=point_colors, alpha=0.45, label=f"DMS (n={len(x)})", zorder=1)

    slope, intercept = np.polyfit(x, y, 1)
    xr = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    ax.plot(xr, slope * xr + intercept, color="black", linewidth=1.2, zorder=2)
    rp, pp = pearsonr(x, y)
    rs, ps = spearmanr(x, y)
    pos_mask = y > ENRICHMENT_BIMODAL_THRESHOLD
    neg_mask = ~pos_mask
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))
    if n_pos >= 3:
        pos_stats = spearmanr(x[pos_mask], y[pos_mask])
        rs_pos = float(pos_stats.statistic if hasattr(pos_stats, "statistic") else pos_stats[0])
    else:
        rs_pos = float("nan")
    if n_neg >= 3:
        neg_stats = spearmanr(x[neg_mask], y[neg_mask])
        rs_neg = float(neg_stats.statistic if hasattr(neg_stats, "statistic") else neg_stats[0])
    else:
        rs_neg = float("nan")
    if n_pos > 0 and n_neg > 0:
        labels = pos_mask.astype(int)
        ranks = rankdata(x)
        auroc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        auroc = float(auroc)
    else:
        auroc = float("nan")

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

    if np.isfinite(wt_pll) and np.isfinite(wt_fitness):
        ax.scatter(
            [wt_pll],
            [wt_fitness],
            s=280,
            marker="*",
            c="#e31a1c",
            edgecolors="black",
            linewidths=1.0,
            zorder=6,
            label=f"WT ({wt_seq})",
        )
        ax.annotate(
            "WT",
            (wt_pll, wt_fitness),
            textcoords="offset points",
            xytext=(9, 8),
            color="#a50000",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9, edgecolor="none"),
        )
    elif np.isfinite(wt_pll):
        ax.axvline(
            wt_pll,
            color="#e31a1c",
            linestyle="--",
            linewidth=1.4,
            zorder=2.5,
            label="WT PLL",
        )

    text = (
        f"DMS Pearson r = {rp:.3f} (p={pp:.1e})\n"
        f"DMS Spearman rho = {rs:.3f} (p={ps:.1e})\n"
        f"DMS Spearman pos/neg = {rs_pos:.3f}/{rs_neg:.3f} (n={n_pos}/{n_neg})\n"
        f"DMS AUROC (binder vs non-binder) = {auroc:.3f}\n"
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
    wt_seq = str(args.wt_seq).strip() or C05_CDRH3
    wt_fitness_by_col = {
        "M22_binding_enrichment_adj": float(args.wt_m22_enrichment),
        "SI06_binding_enrichment_adj": float(args.wt_si06_enrichment)
        if args.wt_si06_enrichment is not None
        else float("nan"),
    }
    split_cache_root = (
        args.split_cache_dir
        if args.split_cache_dir is not None
        else (args.output_dir / "split_cache")
    )
    split_params: dict[str, float | int] | None = None
    split_name = ""
    if args.split_mode != "full":
        repo_root = Path(__file__).resolve().parents[2]
        dpo_cfg_path = args.dpo_split_config
        if not dpo_cfg_path.is_absolute():
            dpo_cfg_path = repo_root / dpo_cfg_path
        split_params = _load_dpo_split_params(dpo_cfg_path)
        split_name = args.split_mode.replace("_dpo", "")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_ID)

    dms_by_dataset: Dict[str, pd.DataFrame] = {}
    for ds in args.datasets:
        ds_paths = dataset_map[ds]
        df_full = load_dms_full(ds_paths, m22_col=args.dms_m22_col, si06_col=args.dms_si06_col)
        df = df_full
        if args.split_mode != "full":
            assert split_params is not None
            m22_path = Path(ds_paths["m22"])
            df = _filter_split_like_dpo(
                dataset_name=ds,
                dataset_m22_path=m22_path,
                df=df_full,
                split_name=split_name,
                split_params=split_params,
                split_seed=int(args.split_seed),
                cache_root=split_cache_root,
            )
        # Keep a single background point per CDR-H3 for plotting and regression.
        df = df.drop_duplicates(subset=["aa"], keep="first").reset_index(drop=True)
        dms_by_dataset[ds] = df
        log.info(
            "DMS %s: %d rows after split_mode=%s (%d before split/filter)",
            ds,
            len(df),
            args.split_mode,
            len(df_full),
        )

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
                    + [wt_seq]
                )
            )
            if not union_sequences:
                log.warning("[%s/%s] no sequences to score - skipping", variant, ds)
                continue

            pll_map = compute_pll_map(model, tokenizer, device, union_sequences, args.batch_size)
            dms_df["pll"] = dms_df["aa"].map(pll_map)
            wt_pll = float(pll_map.get(wt_seq, float("nan")))

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
                make_plot(
                    out_path,
                    variant,
                    ds_label,
                    fitness_col,
                    fitness_label,
                    dms_df,
                    sampler_rows,
                    wt_pll,
                    wt_seq,
                    float(wt_fitness_by_col.get(fitness_col, float("nan"))),
                )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
