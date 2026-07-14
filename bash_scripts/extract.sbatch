#!/bin/bash
#SBATCH --job-name=extract
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=4:00:00
#SBATCH --output=bash_scripts/logs/extract_%j.log

# Unified, idempotent data-extraction dispatcher for the analysis pipeline.
# Writes per-model artifacts under $ANALYSIS_DIR/<model>/<kind>/ and skips
# anything already up to date (provenance-checked; pass --force to redo).
#
# Usage:
#   sbatch bash_scripts/extract.sbatch --what pll --model evo_35m --dataset all
#   sbatch bash_scripts/extract.sbatch --what pll --model evo_35m --dataset ed2_m22 --force
#   sbatch bash_scripts/extract.sbatch --what emb --model vanilla_650m --dataset all
#
# --what selects the extractor; remaining flags are forwarded to it unchanged.
# Models come from conf/analysis/models.yaml, datasets from dms_datasets.yaml.

set -euo pipefail
source bash_scripts/common_setup.sh

WHAT=""
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --what) WHAT="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

case "$WHAT" in
  pll)
    python scripts/analysis/compute_pll.py "${ARGS[@]}"
    ;;
  emb)
    python scripts/analysis/compute_dms_embeddings.py "${ARGS[@]}"
    ;;
  zeroshot_full)
    python scripts/analysis/score_zeroshot_full_dataset.py "${ARGS[@]}"
    ;;
  "")
    echo "error: --what is required (e.g. --what pll)" >&2; exit 2 ;;
  *)
    echo "error: unknown --what '$WHAT' (known: pll, emb, zeroshot_full)" >&2; exit 2 ;;
esac
