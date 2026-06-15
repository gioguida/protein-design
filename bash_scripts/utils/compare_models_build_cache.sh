#!/bin/bash
# Build the model-comparison cache one model at a time (no plots).
#
# Phase 1 of the two-step workflow: this populates the scratch cache
# (WT PPL + per-dataset PLL scores / Spearman / AUROC) for each 650M model so
# the combined comparison in compare_models.sh can read it back and run fast.
#
# Usage:
#   sbatch bash_scripts/utils/compare_models_build_cache.sh        # all models, sequentially
#   sbatch bash_scripts/utils/compare_models_build_cache.sh 0      # only MODELS[0]
# Submitting per-index lets you run the models one by one as separate jobs;
# each model writes to its own cache dir (keyed by label|size|checkpoint), so
# concurrent per-index jobs do not overwrite each other.

#SBATCH --job-name=compare_models_cache
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=bash_scripts/logs/compare_models_cache_%j.log

set -euo pipefail
SBATCH_SCRIPT_PATH="bash_scripts/utils/compare_models_build_cache.sh"
source bash_scripts/common_setup.sh

# ----------------------------- model specs -----------------------------------
# Format per model: "LABEL|SIZE|CHECKPOINT"
MODELS=(
  "vanilla|650M|facebook/esm2_t33_650M_UR50D"
  "evotuned|650M|/cluster/project/infk/krause/mdenegri/protein-design/checkpoints/oas_full_evo_650m/oas_full_evo_650m.pt"
  "just_dpo|650M|/cluster/project/infk/krause/gguidarini/protein-design/checkpoints/just_dpo_650M/step_8250.pt"
  "evo_dpo|650M|/cluster/project/infk/krause/gguidarini/protein-design/checkpoints/evo_dpo_650M/step_8250.pt"
)

# Optional single-model selection by index (0-based).
if [[ $# -ge 1 ]]; then
  IDX="$1"
  if ! [[ "${IDX}" =~ ^[0-9]+$ ]] || (( IDX < 0 || IDX >= ${#MODELS[@]} )); then
    echo "[error] model index '${IDX}' out of range (0..$(( ${#MODELS[@]} - 1 )))" >&2
    exit 1
  fi
  MODELS=("${MODELS[$IDX]}")
  echo "[run] single-model build, index ${IDX}"
fi

# ----------------------------- switches --------------------------------------
USE_ED2=1
USE_ED5=1
USE_ED811=1
USE_EXP=1

# Compute + cache everything expensive; skip plots (those run in the combined step).
RUN_WT_PPL=1
RUN_GOOD_PPL=1
RUN_SPEARMAN=1
RUN_PLOTS=0
RUN_VIOLIN_PLOTS=0
USE_CACHE=1
WRITE_CACHE=1
PLOTS_ONLY=0
FORCE_RECOMPUTE=0
CACHE_ROOT=""

# ----------------------------- data selection --------------------------------
DMS_CONFIG="conf/data/dms/default.yaml"
SPLIT_NAME="test"
ED2_DATASET_KEY="ed2_m22"
ED5_DATASET_KEY="ed5_m22"
ED811_DATASET_KEY="ed811_m22"
EXP_DATASET_KEY="exp"
FORCE_SPLIT_REBUILD=1
GOOD_THRESHOLD=5.190013461

# ----------------------------- runtime ---------------------------------------
BATCH_SIZE=64
STAMP="$(date +%Y%m%d_%H%M%S)"
# Append the SLURM job id so concurrent per-model jobs never share an OUT_DIR.
OUT_DIR="/cluster/project/infk/krause/${USER}/protein-design/reports/model_comparison/cache_build_${STAMP}_${SLURM_JOB_ID:-local}"
mkdir -p "${OUT_DIR}"

ARGS=(
  --output-dir "${OUT_DIR}"
  --dms-config "${DMS_CONFIG}"
  --split-name "${SPLIT_NAME}"
  --ed2-dataset-key "${ED2_DATASET_KEY}"
  --ed5-dataset-key "${ED5_DATASET_KEY}"
  --ed811-dataset-key "${ED811_DATASET_KEY}"
  --exp-dataset-key "${EXP_DATASET_KEY}"
  --batch-size "${BATCH_SIZE}"
  --good-threshold "${GOOD_THRESHOLD}"
)

for spec in "${MODELS[@]}"; do
  ARGS+=(--model "${spec}")
done

if [[ "${USE_ED2}" == "1" ]]; then ARGS+=(--include-dataset "ED2"); fi
if [[ "${USE_ED5}" == "1" ]]; then ARGS+=(--include-dataset "ED5"); fi
if [[ "${USE_ED811}" == "1" ]]; then ARGS+=(--include-dataset "ED811"); fi
if [[ "${USE_EXP}" == "1" ]]; then ARGS+=(--include-dataset "EXP"); fi

if [[ "${RUN_WT_PPL}" == "1" ]]; then ARGS+=(--run-wt-ppl); fi
if [[ "${RUN_GOOD_PPL}" == "1" ]]; then ARGS+=(--run-good-ppl); fi
if [[ "${RUN_SPEARMAN}" == "1" ]]; then ARGS+=(--run-spearman); fi
if [[ "${RUN_PLOTS}" == "1" ]]; then ARGS+=(--run-plots); fi
if [[ "${RUN_VIOLIN_PLOTS}" == "1" ]]; then ARGS+=(--run-violin-plots); fi
if [[ "${USE_CACHE}" == "1" ]]; then ARGS+=(--use-cache); fi
if [[ "${WRITE_CACHE}" == "1" ]]; then ARGS+=(--write-cache); fi
if [[ "${PLOTS_ONLY}" == "1" ]]; then ARGS+=(--plots-only); fi
if [[ "${FORCE_RECOMPUTE}" == "1" ]]; then ARGS+=(--force-recompute); fi
if [[ -n "${CACHE_ROOT}" ]]; then ARGS+=(--cache-root "${CACHE_ROOT}"); fi

if [[ "${FORCE_SPLIT_REBUILD}" == "1" ]]; then ARGS+=(--force-split-rebuild); fi

echo "[run] output dir: ${OUT_DIR}"
echo "[run] split: ${SPLIT_NAME}"
echo "[run] good-threshold: ${GOOD_THRESHOLD}"
echo "[run] models: ${MODELS[*]}"
uv run python scripts/analysis/compare_models.py "${ARGS[@]}"
