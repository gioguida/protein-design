#!/bin/bash
#SBATCH --job-name=plot_results
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --time=00:30:00
#SBATCH --output=logs/plot_results_%j.out
#SBATCH --error=logs/plot_results_%j.err

# Passes all arguments straight through to scripts/plot_results.py.
# Example:
#   sbatch bash_scripts/plot_results.sbatch \
#     --run $PROJECT_DIR/checkpoints/<evotuning_run> --label evotuned \
#     --run $PROJECT_DIR/checkpoints/<c05_run>       --label +C05 \
#     --run $TRAIN_DIR/<ttt_only_run>                --label +TTT \
#     --run $TRAIN_DIR/<c05_ttt_run>                 --label +C05+TTT \
#     --scoring-config configs/evotuning_base.yaml \
#     --out-dir $PROJECT_DIR/plots/meeting

set -euo pipefail
cd /cluster/home/mdenegri/protein-design
module load eth_proxy
set -a; source .env; set +a

source .venv/bin/activate
mkdir -p logs

python scripts/plot_results.py "$@"
