#!/bin/bash
#SBATCH --job-name=score_pssm_sweep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=0:30:00
#SBATCH --array=0-9
#SBATCH --output=bash_scripts/logs/score_pssm_sweep_%A_%a.log

# Score the PSSM temperature-sweep sequences with the M22 scorer, one SLURM
# array task per temperature (genuinely parallel). Each task scores the 2000
# sequences generated at one temperature. T=0.9 is already scored and is skipped
# by the scorer (idempotent unless --force).
#
# Usage:
#   sbatch bash_scripts/utils/score_pssm_sweep_with_esme.sbatch
#
# The TEMPS array must match `sweep.temperatures` in
# conf/analysis/pssm_baseline_sweep_dpo_comparison.yaml (and #SBATCH --array).

set -euo pipefail
cd /cluster/home/${USER}/protein-design

module load eth_proxy
export no_proxy="${no_proxy//api.wandb.ai,/}"
export NO_PROXY="${no_proxy}"

mkdir -p bash_scripts/logs

TEMPS=(0.1 0.2 0.3 0.5 0.7 0.9 1.0 1.3 1.6 2.0)
T="${TEMPS[$SLURM_ARRAY_TASK_ID]}"

SWEEP=/cluster/scratch/mdenegri/protein-design/pssm_vs_dpo_650m/pssm_sweep
SCORES=/cluster/scratch/mdenegri/protein-design/pssm_vs_dpo_650m/scores
mkdir -p "$SCORES"

CONDA_PY=/cluster/project/infk/krause/mdenegri/miniconda3/envs/flu/bin/python
echo "[task $SLURM_ARRAY_TASK_ID] scoring T=$T"
$CONDA_PY scripts/analysis/score_sequences_with_esme.py \
  --input-csv  "$SWEEP/temp_${T}/pssm_output.csv" \
  --output-csv "$SCORES/pssm_T${T}_m22.csv"
