#!/bin/bash
#SBATCH --job-name=c05_search
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=12G
#SBATCH --time=04:00:00
#SBATCH --output=bash_scripts/logs/c05_search_%j.log

set -euo pipefail
source bash_scripts/common_setup.sh

module load stack/2024-06 gcc/12.2.0 mmseqs2/14-7e284
python scripts/data_prep/search_c05.py --threads 16
