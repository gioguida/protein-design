#!/bin/bash
#SBATCH --job-name=linclust
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=4G
#SBATCH --tmp=200G
#SBATCH --time=48:00:00
#SBATCH --output=bash_scripts/logs/linclust_%j.log

set -euo pipefail
source bash_scripts/common_setup.sh

module load stack/2024-06 gcc/12.2.0 mmseqs2/14-7e284
bash scripts/data_prep/run_linclust.sh
