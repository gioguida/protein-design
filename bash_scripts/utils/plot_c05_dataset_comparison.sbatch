#!/bin/bash
#SBATCH --job-name=plot_c05_compare
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32G
#SBATCH --time=0:30:00
#SBATCH --output=bash_scripts/logs/plot_c05_dataset_comparison_%j.log

set -euo pipefail
source bash_scripts/common_setup.sh

python scripts/analysis/plot_c05_dataset_comparison.py "$@"
