#!/bin/bash
#SBATCH --job-name=c05_variants
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00
#SBATCH --output=bash_scripts/logs/c05_variants_%j.log

# Usage:
#   sbatch bash_scripts/utils/extract_c05_variants.sbatch preview
#   sbatch bash_scripts/utils/extract_c05_variants.sbatch fullseq_60
#   sbatch bash_scripts/utils/extract_c05_variants.sbatch cdrh3_sim <threshold>

set -euo pipefail
source bash_scripts/common_setup.sh

MODE="${1:-preview}"

case "$MODE" in
  preview)
    uv run python scripts/data_prep/extract_c05_variants.py --preview-cdrh3
    ;;
  fullseq_60)
    uv run python scripts/data_prep/extract_c05_variants.py --variant fullseq_60
    ;;
  cdrh3_sim)
    THRESH="${2:?ERROR: cdrh3_sim needs a threshold, e.g. sbatch ... cdrh3_sim 0.20}"
    uv run python scripts/data_prep/extract_c05_variants.py \
      --variant cdrh3_sim --cdrh3-threshold "$THRESH"
    ;;
  *)
    echo "Unknown mode: $MODE. Use preview | fullseq_60 | cdrh3_sim <threshold>" >&2
    exit 2
    ;;
esac
