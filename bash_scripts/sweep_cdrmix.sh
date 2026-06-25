#!/bin/bash
# Light sweep over the two highest-leverage knobs for the cdrmix M22-Spearman
# starting point: learning rate and CDR keep-prob. Each run is the cdrmix_v2
# task (sub-epoch eval + best_spearman.pt selection).
#
# Usage:
#   bash bash_scripts/sweep_cdrmix.sh          # submit all runs
#   DRY_RUN=1 bash bash_scripts/sweep_cdrmix.sh  # print the sbatch lines only
#
# After the runs finish, compare eval/spearman_mean (W&B) and pick the run +
# checkpoint with the highest best_spearman; that run's best_spearman.pt is the
# DPO starting point (path is logged as "DPO starting checkpoint: ...").
set -euo pipefail
cd "$(dirname "$0")/.."

TASK=evotuning_c05_cdrmix_v2

# (learning_rate, cdr_mask_prob) grid.
LRS=(2.0e-5 5.0e-5)
CMPS=(0.5 0.7)

for lr in "${LRS[@]}"; do
  for cmp in "${CMPS[@]}"; do
    # Compact, filesystem-safe run name, e.g. cdrmix_v2_lr2.0e-5_cmp0.7
    run_name="cdrmix_v2_lr${lr}_cmp${cmp}"
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
