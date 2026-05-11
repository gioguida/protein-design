#!/bin/bash
#SBATCH --job-name=ttt_sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --time=04:00:00
#SBATCH --output=bash_scripts/logs/ttt_sweep_%j.log

# Run the C05 TTT + LoRA sweep: baseline ESM2 / evotuned ESM2 / evotuned+TTT
# evaluated on ED2/ED5/ED8 test splits. ED2 selects the best TTT step;
# ED5/ED8 are the main reported numbers.
#
# Usage:
#   sbatch bash_scripts/ttt_sweep.sbatch finetune=/path/to/evotuned/best.pt [hydra overrides...]
#
# Examples:
#   sbatch bash_scripts/ttt_sweep.sbatch \
#       finetune=/cluster/project/infk/krause/mdenegri/protein-design/checkpoints/oas_full_evo_35m/oas_full_evo_35m.pt \
#       run_name=ttt_sweep_evotuned35m
#
#   # Smoke run (small model, short sweep, no evotuned ckpt):
#   sbatch bash_scripts/ttt_sweep.sbatch model=esm2_8m \
#       training.max_steps=10 training.snapshot_steps='[5,10]' \
#       +eval_sample_n=500 run_name=ttt_sweep_smoke

set -euo pipefail
source bash_scripts/common_setup.sh

python scripts/run_ttt_sweep.py task=ttt_sweep "$@"
