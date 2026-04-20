#!/bin/bash
#SBATCH --job-name=train_dpo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --time=24:00:00
#SBATCH --output=bash_scripts/logs/train_dpo_%j.out
#SBATCH --error=bash_scripts/logs/train_dpo_%j.err

set -euo pipefail
source bash_scripts/common_setup.sh

TRAIN_ARGS=("$@")
if [ "${#TRAIN_ARGS[@]}" -gt 0 ] && [[ "${TRAIN_ARGS[0]}" != *=* ]]; then
    TRAIN_ARGS=("run.base_name=${TRAIN_ARGS[0]}" "${TRAIN_ARGS[@]:1}")
fi

python scripts/train_dpo.py "${TRAIN_ARGS[@]}"
