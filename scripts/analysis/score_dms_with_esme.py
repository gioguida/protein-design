"""
Score CDR-H3 sequences from a DMS dataset with the M22 or SI06 binder scorer
using the reference flu repo loader (esme + flash_attn) and cache the result.

Cache location:
    $cache_root/scorer_preds/<dataset>/<scorer>.csv

Columns written: <seq_col>, score
If the cache file already exists it is reused unless --force is passed.

Run under the `flu` conda env (esme + flash_attn):

  /cluster/project/infk/krause/mdenegri/miniconda3/envs/flu/bin/python \
      scripts/analysis/score_dms_with_esme.py --dataset ed2_m22

Pass --dataset all to score every dataset in the registry that has a scorer.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

# Reference flu repo
sys.path.insert(0, "/cluster/home/mdenegri/flu/src")
from model import ESM2EnrichmentModel  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "conf" / "analysis" / "dms_datasets.yaml"
DEFAULT_ESM = "/cluster/project/krause/flohmann/mgm/oracle_assets/esm2_8m.safetensors"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True,
                   help="Dataset key from conf/analysis/dms_datasets.yaml, or 'all'.")
    p.add_argument("--scorer", default=None,
                   help="Override the scorer choice. By default uses the 'scorer' "
                        "field from the dataset entry. Must be a key under 'scorers:'.")
    p.add_argument("--esm-weights", default=DEFAULT_ESM)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--force", action="store_true",
                   help="Re-score even if the cache file already exists.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_scorer(scorer_cfg: dict, esm_weights: str, device: str) -> ESM2EnrichmentModel:
    model = ESM2EnrichmentModel(
        esm_model_path=esm_weights,
        hidden_dim=scorer_cfg["hidden_dim"],
        head_depth=scorer_cfg["head_depth"],
        embedding_dim=320,
        device=device,
        freeze_plm=False,
        prediction_task="log_enrichment",
        dropout=scorer_cfg["dropout"],
        aggregate=scorer_cfg["aggregate"],
        use_context=scorer_cfg["use_context"],
    )
    model.load_head(scorer_cfg["checkpoint"])
    model.eval()
    return model


def score_sequences(model: ESM2EnrichmentModel, seqs: list[str], batch_size: int) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in tqdm(range(0, len(seqs), batch_size), desc="score"):
            preds = model.predict(seqs[i:i + batch_size])
            out.append(preds.detach().cpu().float().numpy())
    return np.concatenate(out)


def main() -> None:
    args = parse_args()
    with CONFIG_PATH.open() as f:
        cfg = yaml.safe_load(f)
    cache_root = Path(cfg["paths"]["cache_root"]) / "scorer_preds"

    if args.dataset == "all":
        dataset_keys = list(cfg["datasets"].keys())
    else:
        if args.dataset not in cfg["datasets"]:
            raise SystemExit(f"Unknown dataset {args.dataset!r}. "
                             f"Known: {list(cfg['datasets'])}")
        dataset_keys = [args.dataset]

    # Group by scorer so we only build each scorer once.
    by_scorer: dict[str, list[tuple[str, Path]]] = {}
    for key in dataset_keys:
        ds = cfg["datasets"][key]
        scorer_name = args.scorer or ds.get("scorer")
        if scorer_name is None:
            log.info("[skip] %s has no scorer", key)
            continue
        out_csv = cache_root / key / f"{scorer_name}.csv"
        if out_csv.exists() and not args.force:
            log.info("[skip] %s exists — pass --force to recompute", out_csv)
            continue
        by_scorer.setdefault(scorer_name, []).append((key, out_csv))

    if not by_scorer:
        log.info("Nothing to do.")
        return

    for scorer_name, work in by_scorer.items():
        scorer_cfg = cfg["scorers"][scorer_name]
        log.info("Building scorer %r from %s", scorer_name, scorer_cfg["checkpoint"])
        model = build_scorer(scorer_cfg, args.esm_weights, args.device)

        for key, out_csv in work:
            ds = cfg["datasets"][key]
            log.info("[%s/%s] reading %s", key, scorer_name, ds["path"])
            df = pd.read_csv(ds["path"])
            seq_col = ds["seq_col"]
            if seq_col not in df.columns:
                raise ValueError(f"{ds['path']} missing column {seq_col!r}")
            seqs = df[seq_col].astype(str).drop_duplicates().tolist()
            log.info("[%s/%s] scoring %d unique sequences", key, scorer_name, len(seqs))
            preds = score_sequences(model, seqs, args.batch_size)
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({seq_col: seqs, "score": preds}).to_csv(out_csv, index=False)
            log.info("[%s/%s] wrote %s  (n=%d)", key, scorer_name, out_csv, len(seqs))

        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
