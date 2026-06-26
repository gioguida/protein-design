#!/bin/bash
#SBATCH --job-name=score_uq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1:00:00
#SBATCH --output=bash_scripts/logs/score_uq_%j.log

# Score an arbitrary CSV of CDR-H3 sequences with Lorenz Kaiser's uncertainty
# scorer (ModelArcEnsemble, one-hot) from the private uncertainty_protein repo.
# Uncertainty-aware sibling of score_generated_with_esme.sbatch — all flags after
# the script are forwarded.
#
# Usage:
#   sbatch bash_scripts/utils/score_generated_with_uncertainty.sbatch \
#       --input-csv  /path/to/generated.csv \
#       --output-csv /path/to/uq_scores.csv
#   (optional: --seq-col cdrh3 --method ModelArcEnsemble --seed 42 --force)
#
# Output CSV columns: <seq_col>, pred, epistd, alestd, totalstd
#
# Runs on CPU only (the ensemble is tiny). Uses the scorer's dedicated venv at
# ~/uncertainty_protein/.venv (CPU torch + gpytorch/torchbnn/optuna/...).

set -euo pipefail
cd /cluster/home/${USER}/protein-design

mkdir -p bash_scripts/logs

VENV_PY=/cluster/home/${USER}/uncertainty_protein/.venv/bin/python
export WANDB_MODE=offline

$VENV_PY scripts/analysis/score_sequences_with_uncertainty.py "$@"
