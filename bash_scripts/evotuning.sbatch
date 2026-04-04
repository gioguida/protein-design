#!/bin/bash
#SBATCH --job-name=evotuning
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --gpus=nvidia_rtx_pro_6000:1
#SBATCH --gres=gpumem:96g
#SBATCH --output=logs/evotuning_%j.out
#SBATCH --error=logs/evotuning_%j.err

set -euo pipefail
cd /cluster/home/mdenegri/protein-design
module load eth_proxy
set -a; source .env; set +a

# Load environment
source .venv/bin/activate

# Ensure logs directory exists
mkdir -p logs

python scripts/train_evotuning.py --config configs/evotuning_5k.yaml
