#!/bin/bash
# K-fold cross-validation low-data DPO sweep launcher.
#
# This is a THIN wrapper around bash_scripts/dpo_lowdata_sweep.sh (all the
# per-job machinery -- model resolution, low_data subsampling, checkpoint
# selection, summary.json -- is reused unchanged). For each held-out fold I it
# calls the base launcher once with:
#   - a fold-specific dataset key   <base>_cv<K>s<SEED>_f<I>   (train + test),
#   - the CV dms_config             conf/data/dms/cv.yaml,
#   - --base-suffix _cv<K>s<SEED>_f<I> so run.base_name becomes
#       lowdata_<model>_n<N>_s<seed>_cv<K>s<SEED>_f<I>
#     which the existing scanner (_BASE_RE with its trailing (?:_.*)?) already
#     tolerates, and whose resolved_config.yaml records the fold key as
#     data.test.dataset_key (so the notebook can group folds and pool them).
#
# Fold splits are pre-materialized ONCE here (sequentially, on the login node)
# before any job is submitted, so concurrent SLURM jobs never race to build the
# same fold directory. The headline metric (pooled out-of-fold Spearman over
# all K folds) is computed in the notebook / scripts/analysis/aggregate_cv_folds.py
# from each fold run's test_predictions.csv -- NOT the mean of per-fold rho.
#
# Usage:
#   bash_scripts/dpo_cv_sweep.sh --dry-run
#   bash_scripts/dpo_cv_sweep.sh --models vanilla_35m,evo_35m --n 20,50,100,200,366 \
#       --seeds 0,1,2 --n-folds 5 --fold-seed 0 --dataset ed1_m22
#
# Datasets currently wired (mirrors report/meetings/14-07-26.ipynb scope):
#   ed1_m22, cetuximab_h   (cetuximab_h adds its own delta thresholds + use_context=false)

set -euo pipefail
cd "/cluster/home/${USER}/protein-design"

UV_BIN="/cluster/project/infk/krause/gguidarini/uv-bin/uv"

# ---- defaults (override via flags) -----------------------------------------
MODELS="vanilla_35m,evo_35m"
SEEDS="0,1,2"
N_FOLDS=5
FOLD_SEED=0
MODEL_PRESET="esm2_35m"
TASK="lora_dpo"
DMS_CONFIG="conf/data/dms/cv.yaml"
DATASET="ed1_m22"          # ed1_m22 | cetuximab_h (loop by rerunning with --dataset)
N_GRID=""                  # default is filled in per-dataset below if empty
DRY_RUN=0
SKIP_BUILD=0               # skip the pre-materialization step (assume splits exist)
EXTRA_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)     MODELS="$2"; shift 2 ;;
    --seeds)      SEEDS="$2"; shift 2 ;;
    --n)          N_GRID="$2"; shift 2 ;;
    --n-folds)    N_FOLDS="$2"; shift 2 ;;
    --fold-seed)  FOLD_SEED="$2"; shift 2 ;;
    --model-preset) MODEL_PRESET="$2"; shift 2 ;;
    --task)       TASK="$2"; shift 2 ;;
    --dms-config) DMS_CONFIG="$2"; shift 2 ;;
    --dataset)    DATASET="$2"; shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --dry-run)    DRY_RUN=1; shift ;;
    *)            EXTRA_OVERRIDES+=("$1"); shift ;;
  esac
done

# Per-dataset defaults (N grid sized to each dataset's own size; cetuximab_h
# needs its own delta thresholds + no context, exactly as in the 14-07 notebook).
DATASET_EXTRA=()
case "${DATASET}" in
  ed1_m22)
    [[ -z "${N_GRID}" ]] && N_GRID="20,50,100,200,366"
    ;;
  cetuximab_h)
    [[ -z "${N_GRID}" ]] && N_GRID="20,50,100,200,447"
    DATASET_EXTRA=(
      "data.delta_based.strong_pos_threshold=2.5"
      "data.delta_based.strong_neg_threshold=-0.25"
      "model.use_context=false"
    )
    ;;
  *)
    [[ -z "${N_GRID}" ]] && { echo "ERROR: --n grid required for dataset '${DATASET}'." >&2; exit 2; }
    ;;
esac

# ---- 1) pre-materialize the K fold splits (sequential, no races) -----------
if [[ "${SKIP_BUILD}" -eq 0 ]]; then
  echo "# Building ${N_FOLDS} CV fold splits for ${DATASET} (seed ${FOLD_SEED})..."
  BUILD_CMD=("${UV_BIN}" run python scripts/data_prep/build_cv_splits.py
             --dms-config "${DMS_CONFIG}" --datasets "${DATASET}"
             --n-folds "${N_FOLDS}" --fold-seed "${FOLD_SEED}" --verify)
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] ${BUILD_CMD[*]}"
  else
    "${BUILD_CMD[@]}"
  fi
fi

# ---- 2) submit one base-launcher call per fold -----------------------------
DRY_FLAG=()
[[ "${DRY_RUN}" -eq 1 ]] && DRY_FLAG=(--dry-run)

for (( i=0; i<N_FOLDS; i++ )); do
  fold_key="${DATASET}_cv${N_FOLDS}s${FOLD_SEED}_f${i}"
  suffix="_cv${N_FOLDS}s${FOLD_SEED}_f${i}"
  echo "# --- fold ${i}/${N_FOLDS}: ${fold_key} ---"
  bash_scripts/dpo_lowdata_sweep.sh \
    --models "${MODELS}" --n "${N_GRID}" --seeds "${SEEDS}" \
    --model-preset "${MODEL_PRESET}" --task "${TASK}" \
    --base-suffix "${suffix}" \
    "${DRY_FLAG[@]}" \
    "data.dms_config=${DMS_CONFIG}" \
    "data.dpo_dataset_key=${fold_key}" \
    "data.test.dataset_key=${fold_key}" \
    "${DATASET_EXTRA[@]}" \
    "${EXTRA_OVERRIDES[@]}"
done

echo "----"
echo "CV sweep dispatch complete for ${DATASET} (${N_FOLDS} folds x models x N x seeds)."
