#!/bin/bash
#SBATCH --job-name=c05_h3_blosum
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=08:00:00
#SBATCH --output=bash_scripts/logs/c05_cdrh3_blosum_%j.log

# Usage:
#   sbatch bash_scripts/utils/extract_c05_cdrh3_blosum.sbatch preview
#   sbatch bash_scripts/utils/extract_c05_cdrh3_blosum.sbatch build 0.70
#   sbatch bash_scripts/utils/extract_c05_cdrh3_blosum.sbatch build 0.70 --rebuild-scores

set -euo pipefail
source bash_scripts/common_setup.sh

MODE="${1:?Usage: sbatch ... preview OR build <threshold> [extra args]}"

case "$MODE" in
  preview)
    shift
    python scripts/data_prep/extract_c05_cdrh3_blosum.py --preview "$@"
    ;;
  build)
    THRESH="${2:?ERROR: build needs a threshold, e.g. sbatch ... build 0.70}"
    shift 2
    python scripts/data_prep/extract_c05_cdrh3_blosum.py --threshold "$THRESH" "$@"
    ;;
  *)
    echo "Unknown mode: $MODE. Use preview | build <threshold>" >&2
    exit 2
    ;;
esac
