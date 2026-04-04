#!/bin/bash
#SBATCH --job-name=mmseqs2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mmseqs2_%j.out
#SBATCH --error=logs/mmseqs2_%j.err

set -euo pipefail
cd /cluster/home/mdenegri/protein-design
set -a; source .env; set +a

mkdir -p logs

module load stack/2024-06 gcc/12.2.0 mmseqs2/14-7e284
bash scripts/run_mmseqs2.sh
