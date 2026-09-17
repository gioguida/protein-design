#!/bin/bash
# Explicit A100-80GB launcher for batched report generation and PLL scoring.
#SBATCH --job-name=report_plot_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --gpus=a100_80gb:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=bash_scripts/logs/report_plot_a100_%j.log

set -euo pipefail
source bash_scripts/common_setup.sh
echo "[allocation] Slurm job ${SLURM_JOB_ID:-unknown} on ${SLURMD_NODENAME:-unknown}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python scripts/analysis/collect_report_plot_data.py --profile a100_80gb "$@"
