#!/bin/bash
#SBATCH --job-name=train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --time=24:00:00
#SBATCH --output=bash_scripts/logs/train_%j.out
#SBATCH --error=bash_scripts/logs/train_%j.err

# Usage:
#   sbatch bash_scripts/train.sbatch <pipeline_name> [hydra overrides...]
#
# Examples:
#   sbatch bash_scripts/train.sbatch evotuning run_name=my_run
#   sbatch bash_scripts/train.sbatch evotune_c05_ttt
#   sbatch bash_scripts/train.sbatch c05 pipeline.init_from=/path/to/best.pt
#   sbatch bash_scripts/train.sbatch evotune_c05_ttt \
#       pipeline.stages.1.overrides.training.learning_rate=1e-4

set -euo pipefail
source bash_scripts/common_setup.sh

PIPELINE="${1:?Usage: sbatch train.sbatch <pipeline_name> [hydra overrides...]}"
shift

python scripts/train.py "+evotuning/pipeline=${PIPELINE}" "$@"
