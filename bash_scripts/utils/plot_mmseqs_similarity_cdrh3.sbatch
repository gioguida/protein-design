#!/bin/bash
#SBATCH --job-name=plot_mmseqs_cdrh3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=0:15:00
#SBATCH --output=bash_scripts/logs/plot_mmseqs_cdrh3_%j.log

set -euo pipefail
source bash_scripts/common_setup.sh

python scripts/analysis/plot_mmseqs_similarity.py \
  --tsv "$SCRATCH_DIR/c05_cdrh3_mmseqs/results.tsv" \
  --label "CDR-H3"
