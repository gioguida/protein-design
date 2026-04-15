#!/bin/bash
#SBATCH --job-name=ttt_c05
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --time=01:00:00
#SBATCH --output=logs/ttt_c05_%j.out
#SBATCH --error=logs/ttt_c05_%j.err

set -euo pipefail
cd /cluster/home/mdenegri/protein-design
module load eth_proxy
set -a; source .env; set +a

source .venv/bin/activate
mkdir -p logs

CONFIG="${1:-configs/ttt_c05_single.yaml}"
RUN_NAME="${2:-ttt_c05}"
python scripts/train_ttt.py --config "$CONFIG" --run-name "$RUN_NAME"
