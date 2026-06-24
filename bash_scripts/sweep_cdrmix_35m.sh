#!/bin/bash
# 35M analogue of sweep_cdrmix.sh. Light 2x2 sweep over learning rate and CDR
# keep-prob for the cdrmix 35M M22-Spearman starting point. Each run is the
# evotuning_c05_cdrmix_v2_35m task (sub-epoch eval + best_spearman.pt selection).
#
# Usage:
#   bash bash_scripts/sweep_cdrmix_35m.sh          # submit all runs
#   DRY_RUN=1 bash bash_scripts/sweep_cdrmix_35m.sh  # print the sbatch lines only
#
# After the runs finish, compare eval/spearman_mean (W&B / summary.json) and pick
# the run + checkpoint with the highest best_spearman; that run's best_spearman.pt
# is the DPO starting point. Register it in conf/analysis/models.yaml (e.g. key
# evo_c05_cdrmix_35m) and add it to the low-data DPO sweep.
set -euo pipefail
cd "$(dirname "$0")/.."

TASK=evotuning_c05_cdrmix_v2_35m

# (learning_rate, cdr_mask_prob) grid.
LRS=(2.0e-5 5.0e-5)
CMPS=(0.5 0.7)

for lr in "${LRS[@]}"; do
  for cmp in "${CMPS[@]}"; do
    run_name="cdrmix_v2_35m_lr${lr}_cmp${cmp}"
    cmd=(sbatch bash_scripts/train.sbatch "$TASK"
         "training.learning_rate=${lr}"
         "data.cdr_mask_prob=${cmp}"
         "run_name=${run_name}")
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      printf '%q ' "${cmd[@]}"; echo
    else
      "${cmd[@]}"
    fi
  done
done
