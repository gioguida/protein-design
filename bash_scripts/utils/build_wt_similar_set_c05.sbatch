#!/bin/bash
#SBATCH --job-name=wt_similar_c05
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=08:00:00
#SBATCH --output=bash_scripts/logs/wt_similar_c05_%j.log

set -euo pipefail
source bash_scripts/common_setup.sh

python scripts/data_prep/build_wt_similar_set.py \
    --wt-name c05 \
    --wt-cdrh3 AKHMSMQQVVSAGWERADLVGDAFDV \
    --top-n 5000
