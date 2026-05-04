#!/bin/bash
#SBATCH --job-name=beam_array
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --partition=gpu
#SBATCH --array=0-8
#SBATCH --output=bash_scripts/logs/beam_array_%A_%a.out

# One beam-search job per model variant, all running in parallel as
# GPUs become available. Submit once with
# `sbatch bash_scripts/utils/run_beam_search_array.sh` and SLURM dispatches
# the array.
#
# Each task picks a variant from the VARIANTS array below using
# $SLURM_ARRAY_TASK_ID. To add a variant: append to VARIANTS and bump
# the --array=0-N range above.

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"

if [[ -f "${ROOT_DIR}/bash_scripts/common_setup.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/bash_scripts/common_setup.sh"
fi
cd "${ROOT_DIR}"

# ---------------------------------------------------------------------------
# Configure runs.
#
# GIBBS_CONFIG=distribution|fitness selects a preset that overrides any of the
# defaults below that you haven't explicitly set in the environment. Leave it
# unset to keep the legacy single-config behavior.
# ---------------------------------------------------------------------------
case "${GIBBS_CONFIG:-}" in
  distribution)
    BEAM_SIZE="${BEAM_SIZE:-15}"
    N_STEPS="${N_STEPS:-4}"
    SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-1}"
    START_MODE="${START_MODE:-dms}"
    OUT_DIR="${OUT_DIR:-outputs/beam_search/distribution}"
    ;;
  fitness)
    BEAM_SIZE="${BEAM_SIZE:-15}"
    N_STEPS="${N_STEPS:-4}"
    SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-1}"
    START_MODE="${START_MODE:-wt}"
    OUT_DIR="${OUT_DIR:-outputs/beam_search/fitness}"
    ;;
  "")
    ;;
  *)
    echo "[error] unknown GIBBS_CONFIG=${GIBBS_CONFIG} (expected: distribution|fitness|<empty>)" >&2
    exit 1
    ;;
esac

BEAM_SIZE="${BEAM_SIZE:-5}"
N_STEPS="${N_STEPS:-4}"
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
SEED="${SEED:-42}"
OUT_DIR="${OUT_DIR:-outputs/beam_search}"
START_MODE="${START_MODE:-wt}"

PROJECT_BASE_DIR="${PROJECT_DIR:-/cluster/project/infk/krause/${USER}/protein-design}"
DMS_M22_PATH="${DMS_M22_PATH:-${PROJECT_BASE_DIR}/datasets/scoring/D2_M22.csv}"
DMS_SI06_PATH="${DMS_SI06_PATH:-${PROJECT_BASE_DIR}/datasets/scoring/D2_SI06.csv}"

CKPT_ROOT="${CKPT_ROOT:-${PROJECT_BASE_DIR}/checkpoints}"

# Variants: "label|checkpoint". Empty checkpoint -> vanilla HF default.
# Index = $SLURM_ARRAY_TASK_ID; keep in sync with #SBATCH --array=0-N above.
VARIANTS=(
  "vanilla|"
  "evotuned|${CKPT_ROOT}/oas_dedup___esm2_t12_35M_UR50D__lr2e-05__ep3_48h_20260414_101859"
  "c05-finetuned|${CKPT_ROOT}/c05_c05_cdrh3_blosum25__evo_seed_20260424_191020"
  "dpo-from-evo|${CKPT_ROOT}/dpo__evo_base_20260425_190014"
  "dpo-from-c05|${CKPT_ROOT}/dpo__c05_ft_20260425_190143"
  "dpo-from-c05-ep6|${CKPT_ROOT}/dpo__c05_ft_20260425_190143_ep6"
  "giovanni-dpo|${CKPT_ROOT}/giovanni-dpo"
  "giovanni-dpo-less|${CKPT_ROOT}/giovanni-dpo-trained-less"
  "unlikelihood|${CKPT_ROOT}/unlikelihood-experiment"
)

# ---------------------------------------------------------------------------
# Pick this task's variant.
# ---------------------------------------------------------------------------
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
N_VARIANTS="${#VARIANTS[@]}"

if (( TASK_ID >= N_VARIANTS )); then
  echo "[error] TASK_ID=${TASK_ID} but only ${N_VARIANTS} variants configured." >&2
  exit 1
fi

ENTRY="${VARIANTS[$TASK_ID]}"
IFS='|' read -r LABEL CHECKPOINT <<<"${ENTRY}"

echo "============================================================"
echo "[task ${TASK_ID}] variant=${LABEL}  checkpoint=${CHECKPOINT:-<vanilla>}"
echo "============================================================"

mkdir -p "${OUT_DIR}"

# ---------------------------------------------------------------------------
# Run.
# ---------------------------------------------------------------------------
ARGS=(
  --model-variant "${LABEL}"
  --beam-size "${BEAM_SIZE}"
  --n-steps "${N_STEPS}"
  --snapshot-every "${SNAPSHOT_EVERY}"
  --temperature "${TEMPERATURE}"
  --seed "${SEED}"
  --start-mode "${START_MODE}"
  --output-path "${OUT_DIR}/${LABEL}.csv"
)
if [[ -n "${CHECKPOINT}" ]]; then
  ARGS+=(--checkpoint-path "${CHECKPOINT}")
fi
if [[ "${START_MODE}" == "dms" ]]; then
  ARGS+=(
    --dms-m22-path "${DMS_M22_PATH}"
    --dms-si06-path "${DMS_SI06_PATH}"
  )
fi

uv run python scripts/stochastic_beam_search.py "${ARGS[@]}"
