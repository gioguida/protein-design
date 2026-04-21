#!/bin/bash
#SBATCH --job-name=plot_dpo
#SBATCH --output=/cluster/home/%u/protein-design/bash_scripts/logs/%x-%j.out
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G

set -euo pipefail

mkdir -p bash_scripts/logs/

source "${HOME}/protein-design/bash_scripts/common_setup.sh"

PROJECT_ROOT="${DPO_PROJECT_ROOT:-${HOME}/protein-design}"
cd "${PROJECT_ROOT}"

mkdir -p plots

echo "Running on host: $(hostname)"
which python

# Edit these values before submitting.
PLOT_OUTPUT_DIR="${PLOT_OUTPUT_DIR:-${PROJECT_ROOT}/plots}"
PLOT_TRAINING_CURVES="${PLOT_TRAINING_CURVES:-true}"
PLOT_VALIDATION_CURVES="${PLOT_VALIDATION_CURVES:-true}"
PLOT_VALIDATION_SUMMARY="${PLOT_VALIDATION_SUMMARY:-true}"
PLOT_TEST_SUMMARY="${PLOT_TEST_SUMMARY:-true}"

RUN_TIMESTAMPS=(
    "20260419_120621"
    "20260419_165257"
    "20260419_232123"
)

RUN_LABELS=(
    "cross"
    "cross + wt"
    "cros + wt + pos"
)

TRAINING_METRICS=(loss reward_accuracy reward_margin)
VALIDATION_METRICS=(loss reward_accuracy reward_margin perplexity spearman_avg spearman_random ppl/val_pos ppl/val_neg ppl/val_wt)
VALIDATION_SUMMARY_METRICS=(val_ppl spearman_M22)
TEST_SUMMARY_METRICS=(test_reward_accuracy test_reward_margin test_implicit_kl test_perplexity)

PLOT_ARGS=(
    --output-dir "${PLOT_OUTPUT_DIR}"
    --train-root "${TRAIN_DIR}"
    --archive-root "${PROJECT_DIR}/checkpoints"
)

if [ "${PLOT_TRAINING_CURVES}" = "true" ]; then
    PLOT_ARGS+=(--plot-training-curves)
else
    PLOT_ARGS+=(--no-plot-training-curves)
fi

if [ "${PLOT_VALIDATION_CURVES}" = "true" ]; then
    PLOT_ARGS+=(--plot-validation-curves)
else
    PLOT_ARGS+=(--no-plot-validation-curves)
fi

if [ "${PLOT_VALIDATION_SUMMARY}" = "true" ]; then
    PLOT_ARGS+=(--plot-validation-summary)
else
    PLOT_ARGS+=(--no-plot-validation-summary)
fi

if [ "${PLOT_TEST_SUMMARY}" = "true" ]; then
    PLOT_ARGS+=(--plot-test-summary)
else
    PLOT_ARGS+=(--no-plot-test-summary)
fi

for index in "${!RUN_TIMESTAMPS[@]}"; do
    PLOT_ARGS+=(--run-id "${RUN_TIMESTAMPS[index]}")
    PLOT_ARGS+=(--run-label "${RUN_LABELS[index]}")
done

PLOT_ARGS+=(--training-metrics "${TRAINING_METRICS[@]}")
PLOT_ARGS+=(--validation-metrics "${VALIDATION_METRICS[@]}")
PLOT_ARGS+=(--validation-summary-metrics "${VALIDATION_SUMMARY_METRICS[@]}")
PLOT_ARGS+=(--test-summary-metrics "${TEST_SUMMARY_METRICS[@]}")

python scripts/plot_dpo_metrics.py "${PLOT_ARGS[@]}"