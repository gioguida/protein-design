#!/bin/bash
#SBATCH --job-name=temperature_sweep
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --partition=gpu
#SBATCH --output=bash_scripts/logs/temperature_sweep_%j.out

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"

if [[ -f "${ROOT_DIR}/bash_scripts/common_setup.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/bash_scripts/common_setup.sh"
fi
cd "${ROOT_DIR}"

ANALYSIS_CONFIG="${ANALYSIS_CONFIG:-conf/analysis/temperature_sweep.yaml}"
DRY_RUN="${DRY_RUN:-0}"

ARGS=(--config "${ANALYSIS_CONFIG}")
if [[ "${DRY_RUN}" == "1" ]]; then
  ARGS+=(--dry-run)
fi

uv run python scripts/analysis/run_temperature_sweep.py "${ARGS[@]}"
