#!/bin/bash
# Compare fine-tuned checkpoints on WT PPL, good-sequence PPL, Spearman,
# and per-dataset PLL violin model-comparison plots.

#SBATCH --job-name=compare_models
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40g
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=bash_scripts/logs/compare_models_%j.log

set -euo pipefail
source bash_scripts/common_setup.sh

# ----------------------------- model specs -----------------------------------
# Format per model: "LABEL|SIZE|CHECKPOINT"
MODELS=(
  "vanilla|35M|facebook/esm2_t12_35M_UR50D"
  "evotuned|35M|/cluster/project/infk/krause/mdenegri/protein-design/checkpoints/oas_full_evo_35m/oas_full_evo_35m.pt"
  # "evo_C05|35M|/cluster/project/infk/krause/mdenegri/protein-design/checkpoints/oas_full_evo_35m_c05_cdrh3_blosum25"
  # "dpo_reduceLronPlateau|35M|/cluster/project/infk/krause/gguidarini/protein-design/checkpoints/dpo_reduceLRonPlateau/best.pt"
  # "dpo_linear_warmup|35M|/cluster/project/infk/krause/gguidarini/protein-design/checkpoints/dpo_linear_warmup/best.pt"
  # "dpo_linear_warmup_cosine|35M|/cluster/project/infk/krause/gguidarini/protein-design/checkpoints/dpo_linear_warmup_cosine/best.pt"
  # "dpo_step|35M|/cluster/project/infk/krause/gguidarini/protein-design/checkpoints/dpo_step/best.pt"
  # "dpo_one_epoch|35M|/cluster/project/infk/krause/gguidarini/protein-design/checkpoints/dpo_one_epoch_low_lr/best.pt"
  # "vanilla_dpo_4ep|35M|/cluster/project/infk/krause/gguidarini/protein-design/checkpoints/just_dpo_4ep/step_1376.pt"
  "just_dpo_1ep|35M|/cluster/project/infk/krause/gguidarini/protein-design/checkpoints/just_dpo/step_344.pt"
  "just_dpo_4ep|35M|/cluster/project/infk/krause/gguidarini/protein-design/checkpoints/just_dpo/step_1376.pt"
  "just_dpo_7ep|35M|/cluster/project/infk/krause/gguidarini/protein-design/checkpoints/just_dpo/step_2408.pt"
  "just_dpo_10ep|35M|/cluster/project/infk/krause/gguidarini/protein-design/checkpoints/just_dpo/step_3440.pt"
  
  "evo_dpo_1ep|35M|/cluster/project/infk/krause/gguidarini/protein-design/checkpoints/evo_dpo/step_344.pt"
  "evo_dpo_4ep|35M|/cluster/project/infk/krause/gguidarini/protein-design/checkpoints/evo_dpo/step_1376.pt"
  "evo_dpo_7ep|35M|/cluster/project/infk/krause/gguidarini/protein-design/checkpoints/evo_dpo/step_2408.pt"
  "evo_dpo_10ep|35M|/cluster/project/infk/krause/gguidarini/protein-design/checkpoints/evo_dpo/step_3440.pt"
)

# ----------------------------- switches --------------------------------------
USE_ED2=1
USE_ED5=1
USE_ED811=1

RUN_WT_PPL=1
RUN_GOOD_PPL=1
RUN_SPEARMAN=1
RUN_PLOTS=1
RUN_VIOLIN_PLOTS=1
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
FORCE_SPLIT_REBUILD=1
GOOD_THRESHOLD=5.190013461

# ----------------------------- runtime ---------------------------------------
BATCH_SIZE=64
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="/cluster/project/infk/krause/${USER}/protein-design/reports/model_comparison/${STAMP}"
mkdir -p "${OUT_DIR}"

ARGS=(
  --output-dir "${OUT_DIR}"
  --dms-config "${DMS_CONFIG}"
  --split-name "${SPLIT_NAME}"
  --ed2-dataset-key "${ED2_DATASET_KEY}"
  --ed5-dataset-key "${ED5_DATASET_KEY}"
  --ed811-dataset-key "${ED811_DATASET_KEY}"
  --batch-size "${BATCH_SIZE}"
  --good-threshold "${GOOD_THRESHOLD}"
)

for spec in "${MODELS[@]}"; do
  ARGS+=(--model "${spec}")
done

if [[ "${USE_ED2}" == "1" ]]; then ARGS+=(--include-dataset "ED2"); fi
if [[ "${USE_ED5}" == "1" ]]; then ARGS+=(--include-dataset "ED5"); fi
if [[ "${USE_ED811}" == "1" ]]; then ARGS+=(--include-dataset "ED811"); fi

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
uv run python scripts/analysis/compare_models.py "${ARGS[@]}"

