#!/bin/bash
#SBATCH --job-name=pack_corpus
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=12:00:00
#SBATCH --output=bash_scripts/logs/pack_corpus_%j.log

# Pack a corpus into the memory-mapped form the trainer reads. Run once per
# (corpus, flank); every masking policy then shares it.
#
#   sbatch bash_scripts/utils/pack_corpus.sbatch
#   sbatch bash_scripts/utils/pack_corpus.sbatch \
#       --fasta $PROJECT_DIR/data/wt_similar/c05/c05_wt_similar_top5000.fasta \
#       --name c05_wt_similar_top5000

set -euo pipefail
source bash_scripts/common_setup.sh

python scripts/data_prep/pack_corpus.py --flank 5 "$@"
