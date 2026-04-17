#!/bin/bash
#SBATCH --job-name=evotuning
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=bash_scripts/logs/evotuning_%j.out
#SBATCH --error=bash_scripts/logs/evotuning_%j.err

set -euo pipefail
source bash_scripts/common_setup.sh

RUN_NAME="${1:?Usage: sbatch evotuning.sbatch <run_name> [hydra overrides...]}"
shift
python scripts/train_evotuning.py task=evotuning data=oas_full run_name="$RUN_NAME" "$@"
