#!/bin/bash
#SBATCH --job-name=oas_download
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=24:00:00
#SBATCH --output=bash_scripts/logs/oas_download_%j.log

set -euo pipefail
source bash_scripts/common_setup.sh

bash scripts/data_prep/download_oas.sh oas_human_heavy_urls.txt
