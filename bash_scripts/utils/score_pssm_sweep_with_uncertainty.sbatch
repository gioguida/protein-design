#!/bin/bash
#SBATCH --job-name=score_pssm_sweep_uq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1:00:00
#SBATCH --array=0-9
#SBATCH --output=bash_scripts/logs/score_pssm_sweep_uq_%A_%a.log

# Score the PSSM temperature-sweep sequences with Lorenz Kaiser's uncertainty
# scorer (ModelArcEnsemble, one-hot), one SLURM array task per temperature
# (genuinely parallel). Uncertainty-aware sibling of
# score_pssm_sweep_with_esme.sbatch; produces the per-temperature UQ score CSVs
# the quality-vs-diversity frontier (UQ scorer) needs. T=0.9 is already scored
# and is reused by the scorer (idempotent unless --force).
#
# Usage:
#   sbatch bash_scripts/utils/score_pssm_sweep_with_uncertainty.sbatch
#
# The TEMPS array must match `sweep.temperatures` in
# conf/analysis/pssm_baseline_sweep_dpo_comparison.yaml (and #SBATCH --array).
#
# Output CSV columns: cdrh3, pred, epistd, alestd, totalstd  (one row / unique seq)
# Runs on CPU only (the ensemble is tiny); uses the scorer's dedicated venv at
# ~/uncertainty_protein/.venv.

set -euo pipefail
cd /cluster/home/${USER}/protein-design

mkdir -p bash_scripts/logs

TEMPS=(0.1 0.2 0.3 0.5 0.7 0.9 1.0 1.3 1.6 2.0)
T="${TEMPS[$SLURM_ARRAY_TASK_ID]}"

SWEEP=/cluster/scratch/mdenegri/protein-design/pssm_vs_dpo_650m/pssm_sweep
SCORES=/cluster/scratch/mdenegri/protein-design/pssm_vs_dpo_650m/scores
mkdir -p "$SCORES"

VENV_PY=/cluster/home/${USER}/uncertainty_protein/.venv/bin/python
export WANDB_MODE=offline

echo "[task $SLURM_ARRAY_TASK_ID] UQ-scoring T=$T"
$VENV_PY scripts/analysis/score_sequences_with_uncertainty.py \
  --input-csv  "$SWEEP/temp_${T}/pssm_output.csv" \
  --output-csv "$SCORES/pssm_T${T}_uq.csv"
