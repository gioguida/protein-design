#!/bin/bash
#SBATCH --job-name=train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:80g
#SBATCH --time=04:00:00
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
