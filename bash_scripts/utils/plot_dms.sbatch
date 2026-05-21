#!/bin/bash
#SBATCH --job-name=plot_dms
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0:15:00
#SBATCH --output=bash_scripts/logs/plot_dms_%j.log

# Usage:
#   sbatch bash_scripts/utils/plot_dms.sbatch --model-name evodpo_4ep_step1376 --dataset all
#
# CPU-only — runs locally too:
#   uv run python scripts/analysis/plot_dms_correlations.py --model-name X --dataset Y

set -euo pipefail
source bash_scripts/common_setup.sh

python scripts/analysis/plot_dms_correlations.py "$@"
