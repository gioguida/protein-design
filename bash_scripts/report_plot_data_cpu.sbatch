#!/bin/bash
# CPU-only report baseline and reduction work units.
#SBATCH --job-name=report_plot_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --partition=cpu
#SBATCH --output=bash_scripts/logs/report_plot_cpu_%j.log

set -euo pipefail
source bash_scripts/common_setup.sh
echo "[allocation] Slurm job ${SLURM_JOB_ID:-unknown} on ${SLURMD_NODENAME:-unknown}"
python scripts/analysis/collect_report_plot_data.py "$@"
