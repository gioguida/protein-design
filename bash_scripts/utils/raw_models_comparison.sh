#!/bin/bash
#SBATCH --job-name=raw_esm2_compare
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40g
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=bash_scripts/logs/raw_models_comparison_%j.log

set -euo pipefail
source bash_scripts/common_setup.sh

# ----------------------------- hardcoded switches -----------------------------
# Models from conf/model/
USE_ESM2_8M=1
USE_ESM2_35M=1
USE_ESM2_150M=1
USE_ESM2_650M=1

# Datasets
USE_ED2=1
USE_ED5=1
USE_ED811=1

# Metrics / artifacts
RUN_C05_PPL=1
RUN_DATASET_PPL=1
RUN_SPEARMAN=1
RUN_PLOTS=1

# Dataset selection mode
# 1 -> use DPO-style cluster split and keep only validation rows
# 0 -> use full raw datasets
USE_DPO_VAL_SPLIT=1
DPO_SPLIT_CONFIG="conf/data/dpo/default.yaml"
SPLIT_SEED=42

# Runtime knobs
BATCH_SIZE=64

# Cluster scoring datasets (as requested)
ED2_PATH="/cluster/project/infk/krause/mdenegri/protein-design/datasets/scoring/D2_M22.csv"
ED5_PATH="/cluster/project/infk/krause/mdenegri/protein-design/datasets/scoring/ED5_M22_binding_enrichment.csv"
ED811_PATH="/cluster/project/infk/krause/mdenegri/protein-design/datasets/scoring/ED811_M22_enrichment_full.csv"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="/cluster/project/infk/krause/${USER}/protein-design/reports/raw_models_comparison/${STAMP}"
mkdir -p "${OUT_DIR}"

ARGS=(
  --output-dir "${OUT_DIR}"
  --ed2-path "${ED2_PATH}"
  --ed5-path "${ED5_PATH}"
  --ed811-path "${ED811_PATH}"
  --batch-size "${BATCH_SIZE}"
)

if [[ "${USE_ESM2_8M}" == "1" ]]; then ARGS+=(--include-model "esm2_8m"); fi
if [[ "${USE_ESM2_35M}" == "1" ]]; then ARGS+=(--include-model "esm2_35m"); fi
if [[ "${USE_ESM2_150M}" == "1" ]]; then ARGS+=(--include-model "esm2_150m"); fi
if [[ "${USE_ESM2_650M}" == "1" ]]; then ARGS+=(--include-model "esm2_650m"); fi

if [[ "${USE_ED2}" == "1" ]]; then ARGS+=(--include-dataset "ED2"); fi
if [[ "${USE_ED5}" == "1" ]]; then ARGS+=(--include-dataset "ED5"); fi
if [[ "${USE_ED811}" == "1" ]]; then ARGS+=(--include-dataset "ED811"); fi

if [[ "${RUN_C05_PPL}" == "1" ]]; then ARGS+=(--run-c05-ppl); fi
if [[ "${RUN_DATASET_PPL}" == "1" ]]; then ARGS+=(--run-dataset-ppl); fi
if [[ "${RUN_SPEARMAN}" == "1" ]]; then ARGS+=(--run-spearman); fi
if [[ "${RUN_PLOTS}" == "1" ]]; then ARGS+=(--run-plots); fi

if [[ "${USE_DPO_VAL_SPLIT}" == "1" ]]; then
  ARGS+=(--split-mode "val_dpo")
  ARGS+=(--dpo-split-config "${DPO_SPLIT_CONFIG}")
  ARGS+=(--split-seed "${SPLIT_SEED}")
else
  ARGS+=(--split-mode "full")
fi

echo "[run] output dir: ${OUT_DIR}"
uv run python scripts/analysis/raw_models_comparison.py "${ARGS[@]}"
