#!/bin/bash
#SBATCH --job-name=evotuning
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:40g
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail

# Load environment
source .venv/bin/activate

# Ensure logs directory exists
mkdir -p logs

python scripts/train_evotuning.py --config configs/evotuning_base.yaml
