#!/bin/bash
#SBATCH --job-name=c05_search
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=12G
#SBATCH --time=04:00:00
#SBATCH --output=logs/c05_search_%j.out
#SBATCH --error=logs/c05_search_%j.err

set -euo pipefail
cd /cluster/home/mdenegri/protein-design
set -a; source .env; set +a

mkdir -p logs

module load stack/2024-06 gcc/12.2.0 mmseqs2/14-7e284
source .venv/bin/activate

python scripts/search_c05.py --threads 16
