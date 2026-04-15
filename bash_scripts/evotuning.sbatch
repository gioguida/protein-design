#!/bin/bash
#SBATCH --job-name=evotuning
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g         # 24 GB VRAM (RTX 4090)
#SBATCH --time=48:00:00
#SBATCH --output=logs/evotuning_%j.out
#SBATCH --error=logs/evotuning_%j.err

set -euo pipefail
cd /cluster/home/mdenegri/protein-design
module load eth_proxy
# api.wandb.ai was added to no_proxy in a recent eth_proxy update, but compute
# nodes have no direct internet — route wandb through the proxy instead.
export no_proxy="${no_proxy//api.wandb.ai,/}"
export NO_PROXY="${no_proxy}"
set -a; source .env; set +a

source .venv/bin/activate
mkdir -p logs

CONFIG="${1:?Usage: sbatch evotuning.sbatch <config> <run_name>}"
RUN_NAME="${2:?Usage: sbatch evotuning.sbatch <config> <run_name>}"
python scripts/train_evotuning.py --config "$CONFIG" --run-name "$RUN_NAME"
