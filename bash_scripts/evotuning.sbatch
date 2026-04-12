#!/bin/bash
#SBATCH --job-name=evotuning
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:80g
#SBATCH --time=120:00:00
#SBATCH --output=logs/evotuning_%j.out
#SBATCH --error=logs/evotuning_%j.err

set -euo pipefail
cd /cluster/home/mdenegri/protein-design
module load eth_proxy
set -a; source .env; set +a

source .venv/bin/activate
mkdir -p logs

CONFIG="${1:?Usage: sbatch evotuning.sbatch <config> <run_name>}"
RUN_NAME="${2:?Usage: sbatch evotuning.sbatch <config> <run_name>}"
python scripts/train_evotuning.py --config "$CONFIG" --run-name "$RUN_NAME"
