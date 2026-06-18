#!/bin/bash
#SBATCH --job-name=oas_h3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=24G
#SBATCH --time=04:00:00
#SBATCH --output=bash_scripts/logs/cache_oas_h3_%j.log

# One-pass cache of unique OAS CDR-H3 (count + length) -> scratch parquet.
# Heavy memory because it holds a Counter over all unique H3s (~tens of millions).
set -euo pipefail
source bash_scripts/common_setup.sh

python scripts/data_prep/cache_oas_h3.py
