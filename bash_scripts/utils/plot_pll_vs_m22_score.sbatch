#!/bin/bash
#SBATCH --job-name=pll_vs_m22
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40g
#SBATCH --time=1:00:00
#SBATCH --output=bash_scripts/logs/pll_vs_m22_%j.log

# Usage:
#   sbatch bash_scripts/utils/plot_pll_vs_m22_score.sbatch \
#     --sequences-csv <path/to/gibbs.csv> \
#     --checkpoint-path <path/to/checkpoint> \
#     --model-variant evotuned+dpo \
#     --output-dir <path/to/output>
#
# Optional flags (defaults shown):
#   --scorer-ckpt  /cluster/project/infk/krause/ssussex/flu/models/M22_best.pt
#   --batch-size   32
#   --max-sequences <int>   (omit to score all unique CDR-H3s)

set -euo pipefail
source bash_scripts/common_setup.sh

python scripts/analysis/plot_pll_vs_m22_score.py "$@"
