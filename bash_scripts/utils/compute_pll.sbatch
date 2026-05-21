#!/bin/bash
#SBATCH --job-name=compute_pll
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40g
#SBATCH --time=1:00:00
#SBATCH --output=bash_scripts/logs/compute_pll_%j.log

# Usage:
#   sbatch bash_scripts/utils/compute_pll.sbatch \
#     --model-name evodpo_4ep_step1376 \
#     --checkpoint /cluster/.../checkpoints/evodpo_4ep/step_1376.pt \
#     --dataset all

set -euo pipefail
source bash_scripts/common_setup.sh

python scripts/analysis/compute_pll.py "$@"
