#!/bin/bash
#SBATCH --job-name=ttt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --time=08:00:00
#SBATCH --output=bash_scripts/logs/ttt_%j.log

# Run TTT + LoRA on C05. Evaluates CDR PPL and Spearman on ED2/ED5/ED811
# at step 0 (pre-training) and at each snapshot step.
#
# Usage:
#   sbatch bash_scripts/ttt.sbatch [hydra overrides...]
#
# Examples:
#   sbatch bash_scripts/ttt.sbatch \
#       finetune=/cluster/project/infk/krause/mdenegri/protein-design/checkpoints/oas_full_evo_35m/oas_full_evo_35m.pt \
#       run_name=ttt_evotuned35m
#
#   # Smoke run (small model, few steps):
#   sbatch bash_scripts/ttt.sbatch model=esm2_8m \
#       training.max_steps=20 'training.snapshot_steps=[5,10,20]' \
#       run_name=ttt_smoke

set -euo pipefail
source bash_scripts/common_setup.sh

uv run python scripts/train.py task=ttt "$@"
