#!/bin/bash
#SBATCH --job-name=dpo_spearman_probe
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --time=24:00:00
#SBATCH --output=bash_scripts/logs/dpo_spearman_probe_%j.out

set -euo pipefail

# Slurm executes batch scripts from a spool path; anchor to submit directory.
ROOT_DIR="${SLURM_SUBMIT_DIR:-}"
if [[ -z "${ROOT_DIR}" ]]; then
  ROOT_DIR="$(pwd)"
fi
cd "${ROOT_DIR}"

if [[ -f "${ROOT_DIR}/bash_scripts/common_setup.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/bash_scripts/common_setup.sh"
fi
cd "${ROOT_DIR}"

# ---------------------------------------------------------------------------
# Configure your run here (no command-line args needed).
# ---------------------------------------------------------------------------
RUN_NAME="dpo_spearman_probe"
SPEARMAN_INTERVAL_STEPS=50
TRAINING_PRESET="fast_debug"   # e.g. default | fast_debug
DEVICE="cuda"                  # e.g. cuda | cpu

TRAINING_OVERRIDES=()
if [[ "${TRAINING_PRESET}" == "fast_debug" ]]; then
  TRAINING_OVERRIDES=(
    "training.batch_size=8"
    "training.num_epochs=2"
    "training.num_workers=0"
    "training.pin_memory=false"
    "training.persistent_workers=false"
    "training.prefetch_factor=null"
  )
fi

python tests/dpo_first_epoch_spearman_probe.py \
  "run.base_name=${RUN_NAME}" \
  "+probe.spearman_interval_steps=${SPEARMAN_INTERVAL_STEPS}" \
  "task=dpo" \
  "training.device=${DEVICE}" \
  "${TRAINING_OVERRIDES[@]}"

