#!/bin/bash
#SBATCH --job-name=score_random_sweep_uq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1:00:00
#SBATCH --array=0-4
#SBATCH --output=bash_scripts/logs/score_random_sweep_uq_%A_%a.log

# Score the random library-mutant baseline trust-radius sweep with Lorenz
# Kaiser's uncertainty scorer (ModelArcEnsemble, one-hot), one SLURM array task
# per trust radius. Uncertainty-aware sibling of the M22 sweep scoring; produces
# the per-radius UQ score CSVs the quality-vs-diversity frontier (UQ scorer)
# needs. R=11 is already scored and is reused by the scorer (idempotent unless
# --force).
#
# Usage:
#   sbatch bash_scripts/utils/score_random_sweep_with_uncertainty.sbatch
#
# The RADII array must match `sweep.trust_radii` in
# conf/analysis/random_baseline_sweep_dpo_comparison.yaml (and #SBATCH --array).
#
# Output CSV columns: cdrh3, pred, epistd, alestd, totalstd  (one row / unique seq)
# Runs on CPU only (the ensemble is tiny); uses the scorer's dedicated venv at
# ~/uncertainty_protein/.venv.

set -euo pipefail
cd /cluster/home/${USER}/protein-design

mkdir -p bash_scripts/logs

RADII=(2 4 6 8 11)
R="${RADII[$SLURM_ARRAY_TASK_ID]}"

SWEEP=/cluster/scratch/mdenegri/protein-design/random_vs_dpo_650m/random_sweep
SCORES=/cluster/scratch/mdenegri/protein-design/random_vs_dpo_650m/scores
mkdir -p "$SCORES"

VENV_PY=/cluster/home/${USER}/uncertainty_protein/.venv/bin/python
export WANDB_MODE=offline

echo "[task $SLURM_ARRAY_TASK_ID] UQ-scoring trust_radius=$R"
$VENV_PY scripts/analysis/score_sequences_with_uncertainty.py \
  --input-csv  "$SWEEP/radius_${R}/random_output.csv" \
  --output-csv "$SCORES/random_R${R}_uq.csv"
