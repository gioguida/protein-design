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

# Dataset selection mode: use shared DMS split resolver.
DMS_CONFIG="conf/data/dms/default.yaml"
SPLIT_NAME="test"
ED2_DATASET_KEY="ed2_m22"
ED5_DATASET_KEY="ed5_m22"
ED811_DATASET_KEY="ed811_m22"
FORCE_SPLIT_REBUILD=1
MAX_DATASET_ROWS=5000
SUBSAMPLE_SEED=42
SUBSAMPLE_STRATIFY_BINS=10
SUBSAMPLE_EQUAL_WEIGHT=1

# Runtime knobs
BATCH_SIZE=64

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="/cluster/project/infk/krause/${USER}/protein-design/reports/raw_models_comparison/${STAMP}"
mkdir -p "${OUT_DIR}"

ARGS=(
  --output-dir "${OUT_DIR}"
  --split-mode "dms_split"
  --dms-config "${DMS_CONFIG}"
  --split-name "${SPLIT_NAME}"
  --ed2-dataset-key "${ED2_DATASET_KEY}"
  --ed5-dataset-key "${ED5_DATASET_KEY}"
  --ed811-dataset-key "${ED811_DATASET_KEY}"
  --max-dataset-rows "${MAX_DATASET_ROWS}"
  --subsample-seed "${SUBSAMPLE_SEED}"
  --subsample-stratify-bins "${SUBSAMPLE_STRATIFY_BINS}"
  --batch-size "${BATCH_SIZE}"
)

if [[ "${SUBSAMPLE_EQUAL_WEIGHT}" == "1" ]]; then ARGS+=(--subsample-equal-weight); fi

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

if [[ "${FORCE_SPLIT_REBUILD}" == "1" ]]; then ARGS+=(--force-split-rebuild); fi

echo "[run] output dir: ${OUT_DIR}"
echo "[run] DMS config: ${DMS_CONFIG}"
echo "[run] DMS split: ${SPLIT_NAME}"
echo "[run] max dataset rows: ${MAX_DATASET_ROWS}"
uv run python scripts/analysis/raw_models_comparison.py "${ARGS[@]}"
