#!/bin/bash
#SBATCH --job-name=beam_dpo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --time=00:20:00
#SBATCH --output=bash_scripts/logs/beam_dpo_%j.log

set -euo pipefail
source bash_scripts/common_setup.sh

DPO_CKPT=/cluster/project/infk/krause/gguidarini/protein-design/checkpoints/dpo_lowLR_cosine_2ep/best.pt
BEAM_CSV=/cluster/scratch/mdenegri/protein-design/beam_dpo/beam_dpo.csv
PLOT_DIR=/cluster/project/infk/krause/mdenegri/protein-design/outputs/pll_vs_m22

mkdir -p "$(dirname "${BEAM_CSV}")" "${PLOT_DIR}"

echo "[step 1] beam search"
python scripts/stochastic_beam_search.py \
  --model-variant dpo \
  --checkpoint-path "${DPO_CKPT}" \
  --beam-size 500 \
  --n-steps 2 \
  --snapshot-every 1 \
  --temperature 1.0 \
  --seed 42 \
  --output-path "${BEAM_CSV}"

echo "[step 2] PLL vs M22 score"
python scripts/analysis/plot_pll_vs_m22_score.py \
  --sequences-csv "${BEAM_CSV}" \
  --checkpoint-path "${DPO_CKPT}" \
  --model-variant dpo \
  --output-dir "${PLOT_DIR}" \
  --batch-size 32
