#!/bin/bash
#SBATCH --job-name=report_plot_data
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=bash_scripts/logs/report_plot_data_%j.log

# The report notebook prints this command when one of its JSON inputs is
# absent.  All expensive model inference and sampling stays off the login node.
set -euo pipefail
source bash_scripts/common_setup.sh

python scripts/analysis/collect_report_plot_data.py "$@"
