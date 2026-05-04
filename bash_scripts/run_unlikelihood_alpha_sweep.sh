#!/bin/bash
#SBATCH --job-name=ul_alpha_sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --output=bash_scripts/logs/ul_alpha_sweep_%A_%a.out
#SBATCH --array=0-3

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"

source bash_scripts/common_setup.sh

ALPHAS=(0.1 0.3 1.0 3.0)
ALPHA="${ALPHAS[$SLURM_ARRAY_TASK_ID]}"

python scripts/train.py task=unlikelihood training.alpha="${ALPHA}"
