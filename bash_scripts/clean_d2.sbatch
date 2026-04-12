#!/bin/bash
#SBATCH --job-name=clean_d2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/clean_d2_%j.out
#SBATCH --error=logs/clean_d2_%j.err

set -euo pipefail
cd /cluster/home/mdenegri/protein-design
module load eth_proxy
set -a; source .env; set +a

mkdir -p logs

source .venv/bin/activate
python scripts/clean_d2.py
