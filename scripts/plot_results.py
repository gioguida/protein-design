#!/usr/bin/env python
"""Generate comparison plots across evotuning / C05 / TTT runs.

Produces:
    spearman_bar.png          Grouped bar chart of Spearman rho per
                              {model} x {dataset}, with p-values annotated.
    perplexity_comparison.png Bar chart of final val perplexity per model
                              (base ESM2 computed once on the OAS val split
                              for reference).
    metrics_summary.csv       One row per (model, dataset, strategy):
                              columns rho, pval, n.

For each run dir passed in, the script:
  1. Reads metrics.json if present.
  2. If the run dir has no scoring entry (or --force-rescore), re-scores
     best.pt (or final.pt) on the three datasets.
  3. Recomputes validation perplexity by loading best.pt and the run's
     fasta_path.

Usage (repeat --run per model, same order as --label):
    python scripts/plot_results.py \
        --run ${PROJECT_DIR}/checkpoints/<evotuning_run> --label evotuned \
        --run ${PROJECT_DIR}/checkpoints/<c05_5k_run>    --label "+C05" \
        --run ${TRAIN_DIR}/<ttt_only_run>                --label "+TTT" \
        --run ${TRAIN_DIR}/<c05_ttt_run>                 --label "+C05+TTT" \
        --scoring-config configs/evotuning_base.yaml \
        --out-dir ${PROJECT_DIR}/plots/meeting
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoTokenizer

from protein_design.evaluate import compute_cdr_pseudo_perplexity, compute_perplexity
from protein_design.model import EvotuningModel
from protein_design.scoring import load_scoring_datasets, run_multi_scoring_evaluation
from protein_design.utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_LABEL = "base ESM2"
STRATEGY = "avg"  # "avg" or "random" — avg is the headline strategy
MAX_SEQUENCES_FOR_PPL = 10_000  # cap sequences loaded for perplexity eval


def _make_val_loader_lightweight(fasta_path: str, config: dict, max_seqs: int = MAX_SEQUENCES_FOR_PPL):
    """Build a small val DataLoader by streaming the FASTA and stopping early."""
    from Bio import SeqIO
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import DataCollatorForLanguageModeling

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    max_len = config.get("max_seq_len", 256)
    seed = config.get("seed", 42)

    # Stream sequences, take a reproducible subset for validation
    seqs = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        seqs.append(str(record.seq))
        if len(seqs) >= max_seqs:
            break

    # Use last 5% as val (matches make_dataloaders split convention for small n)
    rng = torch.Generator().manual_seed(seed)
    n_val = max(int(len(seqs) * 0.05), 1)
    indices = torch.randperm(len(seqs), generator=rng).tolist()
    val_seqs = [seqs[i] for i in indices[-n_val:]]

    class _SeqDataset(torch.utils.data.Dataset):
        def __init__(self, sequences):
            self.sequences = sequences
            self.tokenizer = tokenizer
            self.max_len = max_len
        def __len__(self):
            return len(self.sequences)
        def __getitem__(self, idx):
            enc = self.tokenizer(self.sequences[idx], truncation=True,
                                 max_length=self.max_len, padding=False, return_tensors=None)
            return {k: torch.tensor(v) for k, v in enc.items()}

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True,
        mlm_probability=config.get("mlm_probability", 0.15),
        pad_to_multiple_of=8,
    )
    return DataLoader(_SeqDataset(val_seqs), batch_size=config.get("batch_size", 128),
                      shuffle=False, num_workers=0, collate_fn=collator)


def _find_checkpoint(run_dir: Path) -> Path:
    """Prefer best.pt; fall back to final.pt (in root or checkpoints/)."""
    for candidate in [
        run_dir / "best.pt",
        run_dir / "final.pt",
        run_dir / "checkpoints" / "final.pt",
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No best.pt or final.pt in {run_dir}")


def _load_run_config(run_dir: Path) -> dict:
    """Load the run's resolved config.yaml snapshot."""
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config.yaml in {run_dir}")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _score_model(model, tokenizer, datasets, device, batch_size, seed) -> dict:
    return run_multi_scoring_evaluation(
        model, tokenizer, datasets, device=device, batch_size=batch_size, seed=seed,
    )


def _extract_final_ppl(metrics: dict) -> float | None:
    history = metrics.get("training_history", [])
    ppls = [e["val_perplexity"] for e in history if "val_perplexity" in e]
    return ppls[-1] if ppls else None


def _extract_final_scoring(metrics: dict) -> dict | None:
    history = metrics.get("scoring_history", [])
    return history[-1] if history else None


def evaluate_run(run_dir: Path, datasets, tokenizer, device, scoring_batch_size,
                 seed, force_rescore: bool, force_reppl: bool,
                 fallback_config: dict | None = None):
    """Return (scoring_dict, final_val_ppl, cdr_ppl) for this run."""
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    try:
        cfg = _load_run_config(run_dir)
    except FileNotFoundError:
        if fallback_config is None:
            raise
        logger.warning("No config.yaml in %s; using scoring config as fallback", run_dir)
        cfg = fallback_config
    ckpt_path = _find_checkpoint(run_dir)
    logger.info("Run %s — checkpoint: %s", run_dir.name, ckpt_path)

    scoring = _extract_final_scoring(metrics) if not force_rescore else None
    ppl = _extract_final_ppl(metrics) if not force_reppl else None

    need_model = scoring is None or ppl is None
    # Always load model for CDR pseudo-perplexity (cheap, 24 forward passes)
    model = EvotuningModel(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    if scoring is None:
        logger.info("  re-scoring on M22/SI06/exp")
        scoring = _score_model(model, tokenizer, datasets, device,
                               scoring_batch_size, seed)

    if ppl is None:
        fasta = cfg.get("fasta_path")
        if fasta and Path(fasta).exists():
            logger.info("  recomputing val perplexity on %s", fasta)
            val_loader = _make_val_loader_lightweight(fasta, cfg)
            ppl, _ = compute_perplexity(model, val_loader, device)
        else:
            logger.warning("  fasta path missing, skipping perplexity")

    cdr_ppl = compute_cdr_pseudo_perplexity(model, tokenizer, device)

    del model
    torch.cuda.empty_cache()

    return scoring, ppl, cdr_ppl


def evaluate_base(scoring_config: dict, datasets, tokenizer, device, seed):
    """Score base ESM2 (no checkpoint) and compute val perplexity on the OAS split."""
    logger.info("Evaluating base ESM2")
    model = EvotuningModel(scoring_config)
    model.to(device).eval()
    batch_size = scoring_config.get("scoring_batch_size", 512)
    scoring = _score_model(model, tokenizer, datasets, device, batch_size, seed)

    ppl = None
    fasta = scoring_config.get("fasta_path")
    if fasta and Path(fasta).exists():
        val_loader = _make_val_loader_lightweight(fasta, scoring_config)
        ppl, _ = compute_perplexity(model, val_loader, device)

    cdr_ppl = compute_cdr_pseudo_perplexity(model, tokenizer, device)

    del model
    torch.cuda.empty_cache()
    return scoring, ppl, cdr_ppl


def plot_spearman_bar(rows_df: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart: x = dataset, groups = model. Annotate p-value significance."""
    datasets = sorted(rows_df["dataset"].unique())
    models = list(rows_df["model"].drop_duplicates())  # preserves input order
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(max(8, 2 * len(datasets) * n_models / 3), 5))
    width = 0.8 / max(n_models, 1)
    x = np.arange(len(datasets))

    for i, model in enumerate(models):
        sub = rows_df[rows_df["model"] == model].set_index("dataset").reindex(datasets)
        rhos = sub["rho"].values
        pvals = sub["pval"].values
        positions = x + (i - (n_models - 1) / 2) * width
        bars = ax.bar(positions, rhos, width=width, label=model)
        for bar, rho, p in zip(bars, rhos, pvals):
            if np.isnan(rho):
                continue
            stars = ("***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns")
            y = rho + (0.01 if rho >= 0 else -0.03)
            ax.text(bar.get_x() + bar.get_width() / 2, y, stars,
                    ha="center", va="bottom" if rho >= 0 else "top", fontsize=8)

    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel(f"Spearman ρ  (strategy: {STRATEGY})")
    ax.set_title("Mutational-path ranking correlation")
    ax.legend(loc="best", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def plot_perplexity_bar(
    ppl_series: dict,
    out_path: Path,
    ylabel: str = "Validation perplexity  (↓ better)",
    title: str = "Masked-LM perplexity on the run's validation split",
) -> None:
    labels = [k for k, v in ppl_series.items() if v is not None]
    values = [ppl_series[k] for k in labels]
    if not labels:
        logger.warning("No perplexity values to plot; skipping")
        return

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(labels)), 5))
    bars = ax.bar(labels, values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out_path)


def build_summary_rows(label: str, scoring: dict) -> list[dict]:
    """Flatten a scoring dict into long-format rows for the plot + CSV."""
    rows = []
    if not scoring:
        return rows
    # Scoring keys: spearman_avg_<DS>, spearman_avg_pval_<DS>, spearman_random_<DS>, ...
    for key, val in scoring.items():
        if "pval" in key or not key.startswith("spearman_"):
            continue
        # spearman_avg_M22 -> strategy=avg, ds=M22
        parts = key.split("_")
        if len(parts) < 3:
            continue
        strategy = parts[1]  # "avg" or "random"
        ds = "_".join(parts[2:])
        pval_key = f"spearman_{strategy}_pval_{ds}"
        rows.append({
            "model": label,
            "dataset": ds,
            "strategy": strategy,
            "rho": float(val) if val is not None else float("nan"),
            "pval": float(scoring.get(pval_key, float("nan"))),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="Path to a run directory; repeat for each model")
    ap.add_argument("--label", action="append", required=True,
                    help="Label for each --run (same order, same count)")
    ap.add_argument("--scoring-config", required=True,
                    help="Config used to score base ESM2 (e.g. configs/evotuning_base.yaml)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--force-rescore", action="store_true")
    ap.add_argument("--force-reppl", action="store_true")
    ap.add_argument("--skip-base", action="store_true",
                    help="Don't evaluate base ESM2 (e.g. if already in the runs)")
    args = ap.parse_args()

    if len(args.run) != len(args.label):
        ap.error("--run and --label must be provided the same number of times")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    scoring_cfg = load_config(args.scoring_config)
    seed = scoring_cfg.get("seed", 42)
    n_samples = scoring_cfg.get("scoring_n_samples", 10000)
    scoring_batch_size = scoring_cfg.get("scoring_batch_size", 512)
    datasets = load_scoring_datasets(scoring_cfg["scoring_datasets"], n_samples, seed)
    tokenizer = AutoTokenizer.from_pretrained(scoring_cfg["model_name"])

    rows: list[dict] = []
    ppl_series: dict[str, float | None] = {}
    cdr_ppl_series: dict[str, float | None] = {}

    if not args.skip_base:
        base_scoring, base_ppl, base_cdr_ppl = evaluate_base(scoring_cfg, datasets, tokenizer, device, seed)
        rows.extend(build_summary_rows(BASE_LABEL, base_scoring))
        ppl_series[BASE_LABEL] = base_ppl
        cdr_ppl_series[BASE_LABEL] = base_cdr_ppl

    for run_dir, label in zip(args.run, args.label):
        scoring, ppl, cdr_ppl = evaluate_run(
            Path(run_dir), datasets, tokenizer, device,
            scoring_batch_size, seed,
            force_rescore=args.force_rescore, force_reppl=args.force_reppl,
            fallback_config=scoring_cfg,
        )
        rows.extend(build_summary_rows(label, scoring))
        ppl_series[label] = ppl
        cdr_ppl_series[label] = cdr_ppl

    df = pd.DataFrame(rows)
    if df.empty:
        logger.error("No scoring rows collected; aborting plots")
        return

    # Headline plot uses strategy = STRATEGY only (avg)
    headline = df[df["strategy"] == STRATEGY].copy()
    plot_spearman_bar(headline, out_dir / "spearman_bar.png")
    plot_perplexity_bar(ppl_series, out_dir / "perplexity_comparison.png")
    plot_perplexity_bar(
        cdr_ppl_series, out_dir / "cdr_perplexity_comparison.png",
        ylabel="CDR-H3 pseudo-perplexity  (↓ better)",
        title="CDR-H3 pseudo-perplexity (full VH context, single-position masking)",
    )

    csv_path = out_dir / "metrics_summary.csv"
    df_out = df.copy()
    df_out["final_val_perplexity"] = df_out["model"].map(ppl_series)
    df_out["cdr_pseudo_perplexity"] = df_out["model"].map(cdr_ppl_series)
    df_out.to_csv(csv_path, index=False)
    logger.info("Saved %s", csv_path)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    (out_dir / "manifest.txt").write_text(
        f"Generated {ts}\n"
        f"Scoring config: {args.scoring_config}\n\n"
        + "\n".join(f"{l}\t{r}" for l, r in zip(args.label, args.run))
    )


if __name__ == "__main__":
    main()
