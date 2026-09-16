#!/bin/bash
#SBATCH --job-name=extract_dedup_metadata
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --time=04:00:00
#SBATCH --output=bash_scripts/logs/extract_dedup_metadata_%j.log

set -euo pipefail
source bash_scripts/common_setup.sh

# Run after run_linclust.sh / linclust.sbatch has produced oas_dedup_rep_seq.fasta.
python scripts/data_prep/extract_dedup_metadata.py
