#!/bin/bash
#SBATCH --job-name=filter_oas
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --time=24:00:00
#SBATCH --output=bash_scripts/logs/filter_oas_%j.out
#SBATCH --error=bash_scripts/logs/filter_oas_%j.err

set -euo pipefail
source bash_scripts/common_setup.sh

python scripts/filter_oas.py
