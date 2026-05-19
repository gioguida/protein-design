#!/bin/bash
#SBATCH --job-name=pll_vs_exp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --time=04:00:00
#SBATCH --output=bash_scripts/logs/pll_vs_exp_%j.log

# Score exp_enrichment.csv with a given checkpoint and plot PLL vs delta_exp_enrichment_adj.
#
# Usage:
#   sbatch bash_scripts/plot_pll_vs_exp_enrichment.sbatch <checkpoint_path> <output_dir> [--batch-size N]
#
# Example:
#   sbatch bash_scripts/plot_pll_vs_exp_enrichment.sbatch \
#     /cluster/project/infk/krause/mdenegri/protein-design/checkpoints/oas_full_evo_35m/oas_full_evo_35m.pt \
#     /cluster/project/infk/krause/mdenegri/protein-design/plots/pll_vs_exp_enrichment/oas_full_evo_35m

set -euo pipefail
source bash_scripts/common_setup.sh

CKPT="${1:?Usage: sbatch plot_pll_vs_exp_enrichment.sbatch <checkpoint_path> <output_dir>}"
OUT_DIR="${2:?Usage: sbatch plot_pll_vs_exp_enrichment.sbatch <checkpoint_path> <output_dir>}"
shift 2

uv run python scripts/analysis/plot_pll_vs_m22_score.py \
    --sequences-csv /cluster/project/infk/krause/mdenegri/protein-design/data/dms_splits/exp/exp_enrichment.csv \
    --seq-col aa \
    --score-col delta_exp_enrichment_adj \
    --checkpoint-path "${CKPT}" \
    --model-variant "$(basename "$(dirname "${CKPT}")")" \
    --output-dir "${OUT_DIR}" \
    --batch-size 64 \
    "$@"
