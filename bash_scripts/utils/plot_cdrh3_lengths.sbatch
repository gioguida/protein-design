#!/bin/bash
#SBATCH --job-name=plot_cdrh3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --time=0:30:00
#SBATCH --output=bash_scripts/logs/plot_cdrh3_%j.out
#SBATCH --error=bash_scripts/logs/plot_cdrh3_%j.err

set -euo pipefail
source bash_scripts/common_setup.sh

python scripts/analysis/plot_cdrh3_lengths.py
