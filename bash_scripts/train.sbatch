#!/bin/bash
#SBATCH --job-name=train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
# Training runs in bf16, which needs tensor cores of the Ampere generation or
# newer. The cluster's Turing cards (Titan RTX, Quadro RTX 6000) have none and
# fall back to fp32 CUDA cores, which measured 4-7x slower on this workload, so
# the GPU model is pinned rather than left to whatever a memory request lands
# on. RTX 4090 is Ada, has bf16 tensor cores, and is the most plentiful card
# here. Peak reserved memory at batch 64 is 3.1 GB (8M), 7.7 GB (35M) and
# 21.5 GB (150M); the peak comes from the recovery metric, which runs at a
# larger batch than training. 150M needs a larger card or a smaller
# recovery_batch_size, which sweep_evotuning.sh handles per model size.
#SBATCH --gpus=nvidia_geforce_rtx_4090:1
#SBATCH --gres=gpumem:20g
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
