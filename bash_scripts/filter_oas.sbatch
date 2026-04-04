#!/bin/bash
#SBATCH --job-name=filter_oas
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=4:00:00
#SBATCH --output=logs/filter_oas_%j.out
#SBATCH --error=logs/filter_oas_%j.err

set -euo pipefail
cd /cluster/home/mdenegri/protein-design
module load eth_proxy
set -a; source .env; set +a

mkdir -p logs

source .venv/bin/activate
python scripts/filter_oas.py
