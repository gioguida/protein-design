#!/bin/bash
#SBATCH --job-name=oas_download
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4:00:00
#SBATCH --output=logs/oas_download_%j.out
#SBATCH --error=logs/oas_download_%j.err

set -euo pipefail
cd /cluster/home/mdenegri/protein-design
module load eth_proxy
set -a; source .env; set +a

mkdir -p logs
bash scripts/download_oas.sh oas_ighg_urls.txt
