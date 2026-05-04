#!/bin/bash
#SBATCH --job-name=dpo_temp_sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --output=bash_scripts/logs/dpo_temp_sweep_%A_%a.out
#SBATCH --array=0-4

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"

source bash_scripts/common_setup.sh

TEMPERATURES=(0.5  1.0  2.0  5.0  10.0)
TEMPERATURE="${TEMPERATURES[$SLURM_ARRAY_TASK_ID]}"

CHECKPOINT="/cluster/project/infk/krause/mdenegri/protein-design/checkpoints/c05_c05_cdrh3_blosum25__evo_seed_20260424_191020/best.pt"

python scripts/train.py task=dpo training.temperature="${TEMPERATURE}" training.resume_checkpoint="${CHECKPOINT}"
