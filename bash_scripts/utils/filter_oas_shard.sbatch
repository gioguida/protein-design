#!/bin/bash
#SBATCH --job-name=filter_oas_shard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=10G
#SBATCH --time=04:00:00
#SBATCH --array=0-31
#SBATCH --output=bash_scripts/logs/filter_oas_shard_%A_%a.log

set -euo pipefail
source bash_scripts/common_setup.sh

# --array above must match --num-shards passed to make_oas_shards.py (default 32).
SHARD_DIR="${SCRATCH_DIR}/oas_shards"

python scripts/data_prep/filter_oas.py \
    --input-list "${SHARD_DIR}/manifest_${SLURM_ARRAY_TASK_ID}.txt" \
    --output-fasta "${SHARD_DIR}/oas_filtered_shard${SLURM_ARRAY_TASK_ID}.fasta" \
    --output-csv "${SHARD_DIR}/oas_filtered_shard${SLURM_ARRAY_TASK_ID}.csv.gz" \
    --summary-json "${SHARD_DIR}/summary_${SLURM_ARRAY_TASK_ID}.json" \
    --no-csv-header
