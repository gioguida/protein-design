#!/bin/bash
#SBATCH --job-name=prepare_oas
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=4:00:00
#SBATCH --output=logs/prepare_oas_%j.out
#SBATCH --error=logs/prepare_oas_%j.err

set -euo pipefail
cd /cluster/home/mdenegri/protein-design
module load eth_proxy
set -a; source .env; set +a

mkdir -p logs

# Step 1: Filter OAS sequences
source .venv/bin/activate
python scripts/filter_oas.py

# Step 2: Deduplicate with MMseqs2
module load stack/2024-06 gcc/12.2.0 mmseqs2/14-7e284
bash scripts/run_mmseqs2.sh
