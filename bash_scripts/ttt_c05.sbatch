#!/bin/bash
#SBATCH --job-name=ttt_c05
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=bash_scripts/logs/ttt_c05_%j.out
#SBATCH --error=bash_scripts/logs/ttt_c05_%j.err

set -euo pipefail
source bash_scripts/common_setup.sh

RUN_NAME="${1:-ttt_c05}"
shift || true
python scripts/train_ttt.py task=ttt data=c05_single run_name="$RUN_NAME" "$@"
