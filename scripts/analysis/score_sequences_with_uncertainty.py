"""
Score an arbitrary CSV of CDR-H3 sequences with Lorenz Kaiser's uncertainty
scorer (``ModelArcEnsemble``, one-hot encoder) from the private repo
``NingNing-C/uncertainty_protein`` (branch ``Lorenz``).

This is the uncertainty-aware sibling of ``score_sequences_with_esme.py``: it
takes the same generated CSVs (one column of 24-residue CDR-H3 sequences) and
produces a per-sequence prediction *plus* an uncertainty decomposition.

Why a wrapper? The scorer's ``src/evaluate.py`` expects a CSV whose columns
match its config (``mut_loci_seq`` + ``binding_enrichment_adj``), writes its
output to ``results/<dataset>_<csvstem>/<ckpt>_eval_results.csv`` relative to the
scorer repo, and pulls in heavy deps (gpytorch/torchbnn/optuna/...) that live in
a dedicated venv. This script hides all of that:

  1. read --input-csv, take the unique sequences from --seq-col;
  2. write a temp CSV with ``mut_loci_seq`` (= the sequence) and a dummy
     ``binding_enrichment_adj`` label (the scorer only uses the label to log
     metrics; predictions do not depend on it);
  3. run the scorer's ``src/evaluate.py`` (in the scorer repo, scorer venv,
     ``WANDB_MODE=offline``);
  4. read its ``*_eval_results.csv`` and write a tidy
     ``<seq_col>,pred,epistd,alestd,totalstd`` CSV.

Columns written: <seq_col>, pred, epistd, alestd, totalstd  (one row / unique seq)
  - pred      : ensemble-mean predicted binding_enrichment_adj (the "score")
  - epistd    : epistemic std  (disagreement across the 6 ensemble members)
  - alestd    : aleatoric std  (mean predicted per-member noise)
  - totalstd  : sqrt(epistd**2 + alestd**2)

If the output file already exists it is reused unless --force is passed.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

# Defaults for the artifacts Lorenz handed over.
DEFAULT_SCORER_REPO = Path.home() / "uncertainty_protein"
DEFAULT_VENV_PY = DEFAULT_SCORER_REPO / ".venv" / "bin" / "python"
DEFAULT_CKPT = (
    "/cluster/project/infk/krause/mdenegri/protein-design/checkpoints/"
    "lorenz_scorer/ModelArcEnsemble_onehot_329760_seed42.pth"
)
DEFAULT_CONFIG = (
    "/cluster/project/infk/krause/mdenegri/protein-design/checkpoints/"
    "lorenz_scorer/config_onehot_updated.yaml"
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input-csv", required=True,
                   help="CSV with a column of CDR-H3 sequences to score.")
    p.add_argument("--seq-col", default="cdrh3",
                   help="Name of the sequence column in --input-csv (default: cdrh3).")
    p.add_argument("--output-csv", required=True,
                   help="Where to write the <seq_col>,pred,epistd,alestd,totalstd CSV.")
    p.add_argument("--method", default="ModelArcEnsemble",
                   help="Scorer method (default: ModelArcEnsemble).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scorer-repo", type=Path, default=DEFAULT_SCORER_REPO,
                   help="Path to the uncertainty_protein checkout (branch Lorenz).")
    p.add_argument("--venv-python", type=Path, default=DEFAULT_VENV_PY,
                   help="Python interpreter of the scorer's dedicated venv.")
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--force", action="store_true",
                   help="Re-score even if --output-csv already exists.")
    return p.parse_args()


def _dataset_name(config_path: str) -> str:
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["global"]["dataset"]


def main() -> None:
    args = parse_args()

    out_csv = Path(args.output_csv)
    if out_csv.exists() and not args.force:
        log.info("[skip] %s exists — pass --force to recompute", out_csv)
        return

    repo = args.scorer_repo.resolve()
    evaluate_py = repo / "src" / "evaluate.py"
    for path, what in [(repo, "scorer repo"), (evaluate_py, "evaluate.py"),
                       (args.venv_python, "venv python"),
                       (Path(args.checkpoint), "checkpoint"),
                       (Path(args.config), "config")]:
        if not Path(path).exists():
            raise SystemExit(f"{what} not found: {path}")

    # --- read input + unique sequences -------------------------------------
    in_csv = Path(args.input_csv)
    if not in_csv.exists():
        raise SystemExit(f"Input CSV not found: {in_csv}")
    df = pd.read_csv(in_csv)
    if args.seq_col not in df.columns:
        raise SystemExit(f"{in_csv} missing column {args.seq_col!r}. "
                         f"Columns: {list(df.columns)}")
    seqs = df[args.seq_col].astype(str).str.strip()
    seqs = sorted({s for s in seqs.tolist() if s})
    log.info("Scoring %d unique sequences from %s (col=%s)",
             len(seqs), in_csv, args.seq_col)

    # --- temp CSV in the scorer schema -------------------------------------
    # Name the temp file after the output stem so the scorer's results dir
    # (results/<dataset>_<tmpstem>/) is unique per call and re-runs are isolated.
    tmp_stem = out_csv.stem
    tmp_dir = Path(tempfile.mkdtemp(prefix="uq_scorer_"))
    tmp_csv = tmp_dir / f"{tmp_stem}.csv"
    pd.DataFrame({"mut_loci_seq": seqs, "binding_enrichment_adj": 0.0}).to_csv(
        tmp_csv, index=False
    )

    # --- run the scorer's evaluate.py --------------------------------------
    env = os.environ.copy()
    env["WANDB_MODE"] = "offline"
    env["PYTHONPATH"] = str(repo / "src") + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        str(args.venv_python), "src/evaluate.py",
        "--config", str(args.config),
        "--model_path", str(args.checkpoint),
        "--method", args.method,
        "--external_data_path", str(tmp_csv),
        "--uncertainty",
        "--seed", str(args.seed),
    ]
    log.info("Running scorer:\n  cd %s && %s", repo, " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=repo, env=env)

    # --- collect predictions -----------------------------------------------
    dataset = _dataset_name(args.config)
    results_dir = repo / "results" / f"{dataset}_{tmp_stem}"
    hits = sorted(results_dir.glob("*_eval_results.csv"),
                  key=lambda p: p.stat().st_mtime)
    if not hits:
        raise SystemExit(f"No *_eval_results.csv produced under {results_dir}")
    res = pd.read_csv(hits[-1])
    # evaluate.py renames mut_loci_seq -> 'sequence' (one-hot path keeps it).
    if "sequence" not in res.columns:
        raise SystemExit(f"{hits[-1]} missing 'sequence' column; cols={list(res.columns)}")
    keep = ["sequence", "pred", "epistd", "alestd", "totalstd"]
    missing = [c for c in keep if c not in res.columns]
    if missing:
        raise SystemExit(f"{hits[-1]} missing columns {missing}; cols={list(res.columns)}")
    out = res[keep].rename(columns={"sequence": args.seq_col})

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    log.info("Wrote %s  (n=%d)", out_csv, len(out))


if __name__ == "__main__":
    main()
