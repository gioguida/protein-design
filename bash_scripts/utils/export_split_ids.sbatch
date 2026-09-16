#!/bin/bash
#SBATCH --job-name=export_split_ids
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00
#SBATCH --output=bash_scripts/logs/export_split_ids_%j.log

set -euo pipefail
source bash_scripts/common_setup.sh

# Run after run_linclust.sh / linclust.sbatch has produced oas_dedup_rep_seq.fasta.
python scripts/data_prep/export_split_ids.py
