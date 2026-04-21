#!/bin/bash
#SBATCH --job-name=c05_h3_mmseqs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=12G
#SBATCH --time=06:00:00
#SBATCH --output=bash_scripts/logs/c05_cdrh3_mmseqs_%j.log

# Usage:
#   sbatch bash_scripts/utils/extract_c05_cdrh3_mmseqs.sbatch preview
#   sbatch bash_scripts/utils/extract_c05_cdrh3_mmseqs.sbatch build 0.60
#   sbatch bash_scripts/utils/extract_c05_cdrh3_mmseqs.sbatch build 0.60 --rebuild-index

set -euo pipefail
source bash_scripts/common_setup.sh
module load stack/2024-06 gcc/12.2.0 mmseqs2/14-7e284

MODE="${1:?Usage: sbatch ... preview OR build <threshold> [extra args]}"

case "$MODE" in
  preview)
    shift
    python scripts/data_prep/extract_c05_cdrh3_mmseqs.py --preview --threads 16 "$@"
    ;;
  build)
    THRESH="${2:?ERROR: build needs a threshold, e.g. sbatch ... build 0.60}"
    shift 2
    python scripts/data_prep/extract_c05_cdrh3_mmseqs.py \
      --threshold "$THRESH" --threads 16 "$@"
    ;;
  *)
    echo "Unknown mode: $MODE. Use preview | build <threshold>" >&2
    exit 2
    ;;
esac
