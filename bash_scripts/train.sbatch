#!/bin/bash
#SBATCH --job-name=train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --gpus=1
# Measured peak reserved memory at batch 64: 3.1 GB (8M), 7.7 GB (35M),
# 21.5 GB (150M). The peak comes from the recovery metric, which runs at a
# larger batch than training. Larger models need a bigger request on the
# command line; sweep_evotuning.sh sets it per model size.
#SBATCH --gres=gpumem:24g
#SBATCH --time=12:00:00
#SBATCH --output=bash_scripts/logs/train_%j.log

# Usage:
#   sbatch bash_scripts/train.sbatch <task_name> [hydra overrides...]
#
# Examples:
#   sbatch bash_scripts/train.sbatch evotuning run_name=my_run
#   sbatch bash_scripts/train.sbatch evotuning_c05 model.init.source=checkpoint model.init.checkpoint=/path/to/best.pt
#   sbatch bash_scripts/train.sbatch ttt model.init.source=checkpoint model.init.checkpoint=/path/to/best.pt
#   sbatch bash_scripts/train.sbatch dpo

set -euo pipefail
source bash_scripts/common_setup.sh

TASK="${1:?Usage: sbatch train.sbatch <task_name> [hydra overrides...]}"
shift

python scripts/train.py "task=${TASK}" "$@"
