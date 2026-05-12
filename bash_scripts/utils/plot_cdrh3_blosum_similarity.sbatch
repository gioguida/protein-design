#!/bin/bash
#SBATCH --job-name=plot_blosum_sim
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32G
#SBATCH --time=0:20:00
#SBATCH --output=bash_scripts/logs/plot_cdrh3_blosum_sim_%j.log

set -euo pipefail
source bash_scripts/common_setup.sh

python scripts/analysis/plot_cdrh3_blosum_similarity.py "$@"
