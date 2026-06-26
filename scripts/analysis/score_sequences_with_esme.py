"""
Score an arbitrary CSV of CDR-H3 sequences with the M22 (or SI06) binder scorer
using the reference flu repo loader (esme + flash_attn).

Unlike ``score_dms_with_esme.py`` (which is tied to the dataset registry in
``conf/analysis/dms_datasets.yaml``), this script scores any CSV: pass an input
CSV, a sequence column, and an output path. The scorer config is still read from
the ``scorers:`` section of ``conf/analysis/dms_datasets.yaml`` so it stays in
sync with the dataset-registry scorer.

Columns written: <seq_col>, score   (one row per unique sequence)
If the output file already exists it is reused unless --force is passed.

Run under the `flu` conda env (esme + flash_attn):

  /cluster/project/infk/krause/mdenegri/miniconda3/envs/flu/bin/python \
      scripts/analysis/score_sequences_with_esme.py \
      --input-csv path/to/generated.csv --output-csv path/to/scores.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "conf" / "analysis" / "dms_datasets.yaml"
DEFAULT_ESM = "/cluster/project/krause/flohmann/mgm/oracle_assets/esm2_8m.safetensors"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-csv", required=True,
                   help="CSV with a column of CDR-H3 sequences to score.")
    p.add_argument("--seq-col", default="cdrh3",
                   help="Name of the sequence column in --input-csv (default: cdrh3).")
    p.add_argument("--output-csv", required=True,
                   help="Where to write the <seq_col>,score CSV.")
    p.add_argument("--scorer", default="m22",
                   help="Scorer key under 'scorers:' in dms_datasets.yaml (default: m22).")
    p.add_argument("--esm-weights", default=DEFAULT_ESM)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--force", action="store_true",
                   help="Re-score even if --output-csv already exists.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_csv = Path(args.output_csv)
    if out_csv.exists() and not args.force:
        log.info("[skip] %s exists — pass --force to recompute", out_csv)
        return

    # Reuse the scorer build + scoring logic from the dataset-bound scorer so
    # there is a single implementation of the esme loader / batched forward
    # pass. Imported here (not at module top) so --help works without the flu
    # env, where `import model` (esme + flash_attn) is unavailable.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from score_dms_with_esme import build_scorer, score_sequences

    with CONFIG_PATH.open() as f:
        cfg = yaml.safe_load(f)
    if args.scorer not in cfg.get("scorers", {}):
        raise SystemExit(f"Unknown scorer {args.scorer!r}. "
                         f"Known: {list(cfg.get('scorers', {}))}")
    scorer_cfg = cfg["scorers"][args.scorer]

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise SystemExit(f"Input CSV not found: {input_csv}")
    df = pd.read_csv(input_csv)
    if args.seq_col not in df.columns:
        raise SystemExit(f"{input_csv} missing column {args.seq_col!r}. "
                         f"Columns: {list(df.columns)}")
    seqs = df[args.seq_col].astype(str).str.strip().drop_duplicates()
    seqs = [s for s in seqs.tolist() if s]
    log.info("Scoring %d unique sequences from %s (col=%s)",
             len(seqs), input_csv, args.seq_col)

    log.info("Building scorer %r from %s", args.scorer, scorer_cfg["checkpoint"])
    model = build_scorer(scorer_cfg, args.esm_weights, args.device)
    preds = score_sequences(model, seqs, args.batch_size)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({args.seq_col: seqs, "score": preds}).to_csv(out_csv, index=False)
    log.info("Wrote %s  (n=%d)", out_csv, len(seqs))


if __name__ == "__main__":
    main()
