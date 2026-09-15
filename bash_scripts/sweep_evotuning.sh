#!/bin/bash
# Learning-rate sweep over the masking ablation: one SLURM job per
# (dataset x policy x model size x lr). This is the first of the two waves.
# Every job here trains a full epoch under a batch-masking policy; the
# single-position continuations are launched afterwards by
# bash_scripts/branch_evotuning.sh, once each base run has a switch point.
#
# Jobs are independent and run in parallel as GPUs free up; this script does
# not block.
#
# Usage:
#   bash_scripts/sweep_evotuning.sh                # full grid
#   bash_scripts/sweep_evotuning.sh --dry-run      # print and write the manifest only
#   bash_scripts/sweep_evotuning.sh --lr-grid 1.0e-5 --model-grid esm2_35m

set -euo pipefail
cd "/cluster/home/${USER}/protein-design"

LR_GRID="5.0e-6,1.0e-5,2.0e-5,5.0e-5,1.0e-4"
MODEL_GRID="esm2_8m,esm2_35m,esm2_150m"
POLICY_GRID="cdr50,hybrid"
DATA_GRID="oas,c05_wt_similar"
NAME_PREFIX="evo"
# Per model size, from measured peak reserved memory (3.1 / 7.7 / 21.5 GB at
# batch 64). Requesting one large number for every job would leave the small
# models queueing behind nodes they do not need. --gpu-mem overrides all of it.
declare -A GPU_MEM_BY_MODEL=( [esm2_8m]=12g [esm2_35m]=24g [esm2_150m]=40g )
GPU_MEM=""
DRY_RUN=0
EXTRA_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lr-grid)     LR_GRID="$2"; shift 2 ;;
    --model-grid)  MODEL_GRID="$2"; shift 2 ;;
    --policy-grid) POLICY_GRID="$2"; shift 2 ;;
    --data-grid)   DATA_GRID="$2"; shift 2 ;;
    --name-prefix) NAME_PREFIX="$2"; shift 2 ;;
    --gpu-mem)     GPU_MEM="$2"; shift 2 ;;
    --dry-run)     DRY_RUN=1; shift ;;
    *)             EXTRA_OVERRIDES+=("$1"); shift ;;
  esac
done

IFS=',' read -r -a LR_ARR     <<< "${LR_GRID}"
IFS=',' read -r -a MODEL_ARR  <<< "${MODEL_GRID}"
IFS=',' read -r -a POLICY_ARR <<< "${POLICY_GRID}"
IFS=',' read -r -a DATA_ARR   <<< "${DATA_GRID}"

mkdir -p bash_scripts/logs
MANIFEST="bash_scripts/logs/sweep_evotuning_$(date +%Y%m%d_%H%M%S).csv"
echo "dataset,policy,model,lr,run_name,job_id" > "${MANIFEST}"

n=0
for data in "${DATA_ARR[@]}"; do
  for policy in "${POLICY_ARR[@]}"; do
    for model in "${MODEL_ARR[@]}"; do
      for lr in "${LR_ARR[@]}"; do
        mem="${GPU_MEM:-${GPU_MEM_BY_MODEL[$model]:-24g}}"
        SBATCH_OPTS=("--gres=gpumem:${mem}")
        run_name="${NAME_PREFIX}_${data}_${policy}_${model}_lr${lr}"
        overrides=(
          "data=evo/${data}_${policy}"
          "model=${model}"
          "training.learning_rate=${lr}"
          "run_name=${run_name}"
        )
        overrides+=("${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}")

        if [[ "${DRY_RUN}" -eq 1 ]]; then
          echo "[dry-run] sbatch ${SBATCH_OPTS[*]-} bash_scripts/train.sbatch evotuning ${overrides[*]}"
          echo "${data},${policy},${model},${lr},${run_name}," >> "${MANIFEST}"
        else
          out="$(sbatch "${SBATCH_OPTS[@]+"${SBATCH_OPTS[@]}"}" bash_scripts/train.sbatch evotuning "${overrides[@]}")"
          echo "${out}  [${run_name}]"
          echo "${data},${policy},${model},${lr},${run_name},$(awk '{print $NF}' <<< "${out}")" >> "${MANIFEST}"
        fi
        n=$((n+1))
      done
    done
  done
done

echo "----"
if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "DRY RUN: ${n} jobs would be submitted. Manifest: ${MANIFEST}"
else
  echo "Submitted ${n} jobs. Manifest: ${MANIFEST}"
fi
