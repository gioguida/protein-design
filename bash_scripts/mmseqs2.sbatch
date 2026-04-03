#!/bin/bash
#SBATCH --job-name=mmseqs2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

source .venv/bin/activate

mkdir -p logs

bash scripts/run_mmseqs2.sh
