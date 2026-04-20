#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --time=01:00:00
#SBATCH --output=bash_scripts/logs/eval_%j.log

# Usage:
#   sbatch bash_scripts/eval.sbatch <checkpoint_path> [hydra overrides...]
#
# Example:
#   sbatch bash_scripts/eval.sbatch $PROJECT_DIR/checkpoints/my_run/best.pt scoring=d2

set -euo pipefail
source bash_scripts/common_setup.sh

CKPT="${1:?Usage: sbatch eval.sbatch <checkpoint_path> [hydra overrides...]}"
shift

python scripts/eval.py "+checkpoint=${CKPT}" "$@"
