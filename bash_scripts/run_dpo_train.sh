#!/bin/bash
#SBATCH --job-name=train_dpo
#SBATCH --output=/cluster/home/%u/protein-design/bash_scripts/logs/%x-%j.out
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_4090:1	# a100:1 | a100_80gb:1 | pro_6000:1 | rtx_4090:1

set -euo pipefail

mkdir -p bash_scripts/logs/

# Shared environment setup (modules + conda env)
source "${HOME}/protein-design/bash_scripts/common_setup.sh"

cd "${DPO_PROJECT_ROOT}"

echo "Running on host: $(hostname)"
which python
nvidia-smi || true

# Pass any Hydra overrides from sbatch command line.
# Example:
#   sbatch bash_scripts/run_dpo_train.sh training.num_epochs=100 training.beta=0.2
TRAIN_ARGS=("$@")
if [ "${#TRAIN_ARGS[@]}" -gt 0 ] && [[ "${TRAIN_ARGS[0]}" != *=* ]]; then
    DPO_BASE_RUN_NAME="${TRAIN_ARGS[0]}"
    TRAIN_ARGS=("${TRAIN_ARGS[@]:1}")
fi
if [ -n "${DPO_BASE_RUN_NAME:-}" ]; then
    TRAIN_ARGS=("run.base_name=${DPO_BASE_RUN_NAME}" "${TRAIN_ARGS[@]}")
fi

if [ "${DPO_USE_UV:-0}" = "1" ]; then
    if ! command -v uv >/dev/null 2>&1; then
        echo "ERROR: DPO_USE_UV=1 but uv is not available in PATH."
        exit 1
    fi
    uv run python -m src.train_dpo \
		"${TRAIN_ARGS[@]}"
else
    python -m src.train_dpo \
		"${TRAIN_ARGS[@]}"
fi
