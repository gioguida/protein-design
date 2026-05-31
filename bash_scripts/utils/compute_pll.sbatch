#!/bin/bash
#SBATCH --job-name=compute_pll
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --time=1:00:00
#SBATCH --output=bash_scripts/logs/compute_pll_%j.log

# DEPRECATED: prefer  bash_scripts/extract.sbatch --what pll  (idempotent,
# registry-driven, writes to $ANALYSIS_DIR/<model>/pll/). Kept as a thin alias.
#
# Usage:
#   sbatch bash_scripts/utils/compute_pll.sbatch --model evo_35m --dataset all

set -euo pipefail
source bash_scripts/common_setup.sh

python scripts/analysis/compute_pll.py "$@"
