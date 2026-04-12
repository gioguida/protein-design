#!/bin/bash
#SBATCH --job-name=plot_cdrh3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --time=0:30:00
#SBATCH --output=logs/plot_cdrh3_%j.out
#SBATCH --error=logs/plot_cdrh3_%j.err

set -euo pipefail
cd /cluster/home/mdenegri/protein-design

mkdir -p logs

source .venv/bin/activate
python scripts/plot_cdrh3_lengths.py
