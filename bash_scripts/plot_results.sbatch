#!/bin/bash
#SBATCH --job-name=plot_results
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --time=00:30:00
#SBATCH --output=bash_scripts/logs/plot_results_%j.out
#SBATCH --error=bash_scripts/logs/plot_results_%j.err

# Passes all arguments straight through to scripts/analysis/plot_results.py.
# Example:
#   sbatch bash_scripts/plot_results.sbatch \
#     scoring=d2 data=oas_full \
#     '+runs=[/path/to/evotuning_run,/path/to/c05_run]' \
#     '+labels=[evotuned,+C05]' \
#     +out_dir=$HOME/protein-design/plots/meeting

set -euo pipefail
source bash_scripts/common_setup.sh

python scripts/analysis/plot_results.py "$@"
