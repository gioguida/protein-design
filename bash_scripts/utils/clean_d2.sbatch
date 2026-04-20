#!/bin/bash
#SBATCH --job-name=clean_d2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:30:00
#SBATCH --output=bash_scripts/logs/clean_d2_%j.log

set -euo pipefail
source bash_scripts/common_setup.sh

python scripts/data_prep/clean_d2.py
