#!/bin/bash
#SBATCH --job-name=pll_vs_score
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --time=04:00:00
#SBATCH --output=bash_scripts/logs/pll_vs_score_%j.log

# Generic launcher for scripts/analysis/plot_pll_vs_m22_score.py.
# All arguments are passed through directly.
#
# Usage:
#   sbatch bash_scripts/plot_pll_vs_score.sbatch \
#     --sequences-csv <path> \
#     --seq-col aa \
#     --score-col <col> \
#     --checkpoint-path <ckpt> \
#     --model-variant <label> \
#     --output-dir <dir>

set -euo pipefail
source bash_scripts/common_setup.sh

uv run python scripts/analysis/plot_pll_vs_m22_score.py "$@"
