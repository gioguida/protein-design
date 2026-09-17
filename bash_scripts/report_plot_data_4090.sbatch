#!/bin/bash
# Explicit RTX 4090 launcher for report functional/preference work units.
#SBATCH --job-name=report_plot_4090
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --output=bash_scripts/logs/report_plot_4090_%j.log

set -euo pipefail
source bash_scripts/common_setup.sh
echo "[allocation] Slurm job ${SLURM_JOB_ID:-unknown} on ${SLURMD_NODENAME:-unknown}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python scripts/analysis/collect_report_plot_data.py --profile rtx_4090 "$@"
