#!/bin/bash
# Generic (model-size- and dataset-agnostic) cdrmix evotuning sweep launcher.
#
# Submits ONE independent SLURM job (bash_scripts/train.sbatch evotuning_cdrmix)
# per combination in the cartesian product of (lr x cmp x flank x fmp x salt).
# Jobs are independent and run in parallel as GPUs free up -- this script does
# not block. Each run tracks best.pt by val perplexity ONLY (no DMS/Spearman in
# selection); collect winners afterwards from $TRAIN_DIR/<run_name>_<ts>/.
#
# Model size / init / freeze come from flags (vanilla = --init-source huggingface,
# which has no registry checkpoint), or from --seed-model <key> resolved against
# conf/analysis/models.yaml (like dpo_lowdata_sweep.sh). Explicit flags win.
#
# Dataset defaults to the task's data config (c05_cdrh3_blosum25); override with
# --fasta-path. The CDR-window cache auto-derives from (fasta, flank) at runtime
# (build_cdr_windows.py must have been run once per flank for that fasta).
#
# Usage:
#   # 81-run screening sweep, 35M vanilla init (default grids, salt oas-v1):
#   bash_scripts/sweep_evotuning_cdrmix.sh \
#       --model-preset esm2_35m --freeze-layers 0 --init-source huggingface \
#       --name-prefix vanilla_cdrmix_35m
#
#   # dry run first (prints every sbatch line, writes manifest, submits nothing):
#   bash_scripts/sweep_evotuning_cdrmix.sh ... --dry-run
#
#   # confirmation stage: restrict grids to the winning combos, add salts + scoring:
#   bash_scripts/sweep_evotuning_cdrmix.sh \
#       --model-preset esm2_35m --freeze-layers 0 --init-source huggingface \
#       --name-prefix vanilla_cdrmix_35m \
#       --lr-grid 5.0e-5 --cmp-grid 0.6 --flank-grid 3 --fmp-grid 0.05 \
#       --salt-grid oas-v2,oas-v3 --scoring c05

set -euo pipefail
cd "/cluster/home/${USER}/protein-design"

# ---- defaults (the 81-run screening grid; override via flags) ---------------
LR_GRID="2.0e-5,5.0e-5,8.0e-5"
CMP_GRID="0.5,0.6,0.7"
FLANK_GRID="1,3,5"
FMP_GRID="0.0,0.05,0.1"
SALT_GRID="oas-v1"
SCORING="none"

MODEL_PRESET="esm2_35m"
FREEZE_LAYERS="0"
INIT_SOURCE="huggingface"   # huggingface | checkpoint
INIT_CHECKPOINT=""
SEED_MODEL=""               # optional conf/analysis/models.yaml key to resolve
FASTA_PATH=""               # optional override of the task's default dataset
NAME_PREFIX="vanilla_cdrmix_35m"
GPU_MEM=""                  # e.g. "24g"; passed as sbatch --gres=gpumem:<val> CLI
                             # override (CLI wins over train.sbatch's #SBATCH
                             # header), so the shared 650M default stays untouched.
DRY_RUN=0
EXTRA_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-preset)    MODEL_PRESET="$2"; shift 2 ;;
    --freeze-layers)   FREEZE_LAYERS="$2"; shift 2 ;;
    --init-source)     INIT_SOURCE="$2"; shift 2 ;;
    --init-checkpoint) INIT_CHECKPOINT="$2"; shift 2 ;;
    --seed-model)      SEED_MODEL="$2"; shift 2 ;;
    --fasta-path)      FASTA_PATH="$2"; shift 2 ;;
    --lr-grid)         LR_GRID="$2"; shift 2 ;;
    --cmp-grid)        CMP_GRID="$2"; shift 2 ;;
    --flank-grid)      FLANK_GRID="$2"; shift 2 ;;
    --fmp-grid)        FMP_GRID="$2"; shift 2 ;;
    --salt-grid)       SALT_GRID="$2"; shift 2 ;;
    --scoring)         SCORING="$2"; shift 2 ;;
    --name-prefix)     NAME_PREFIX="$2"; shift 2 ;;
    --gpu-mem)         GPU_MEM="$2"; shift 2 ;;
    --dry-run)         DRY_RUN=1; shift ;;
    *)                 EXTRA_OVERRIDES+=("$1"); shift ;;
  esac
done

SBATCH_GRES_OPTS=()
if [[ -n "${GPU_MEM}" ]]; then
  SBATCH_GRES_OPTS=("--gres=gpumem:${GPU_MEM}")
fi

# ---- optional: resolve --seed-model key -> checkpoint + base_model ----------
# Explicit --init-source / --init-checkpoint flags take precedence if given.
if [[ -n "${SEED_MODEL}" ]]; then
  resolved="$(uv run python - "${SEED_MODEL}" <<'PY'
import sys, yaml
from pathlib import Path
reg = yaml.safe_load(Path("conf/analysis/models.yaml").read_text())
m = reg.get("models", {}).get(sys.argv[1])
if m is None:
    sys.stderr.write(f"Unknown model key {sys.argv[1]!r} in conf/analysis/models.yaml\n")
    sys.exit(2)
print((m.get("checkpoint") or "") + "|" + (m.get("base_model") or ""))
PY
)"
  seed_ckpt="${resolved%%|*}"
  if [[ -z "${INIT_CHECKPOINT}" && -n "${seed_ckpt}" ]]; then
    INIT_CHECKPOINT="${seed_ckpt}"
    INIT_SOURCE="checkpoint"
  fi
fi

if [[ "${INIT_SOURCE}" == "checkpoint" && -z "${INIT_CHECKPOINT}" ]]; then
  echo "ERROR: --init-source checkpoint requires --init-checkpoint <path> (or a --seed-model with a checkpoint)." >&2
  exit 1
fi

IFS=',' read -r -a LR_ARR    <<< "${LR_GRID}"
IFS=',' read -r -a CMP_ARR   <<< "${CMP_GRID}"
IFS=',' read -r -a FLANK_ARR <<< "${FLANK_GRID}"
IFS=',' read -r -a FMP_ARR   <<< "${FMP_GRID}"
IFS=',' read -r -a SALT_ARR  <<< "${SALT_GRID}"

SBATCH_SCRIPT="bash_scripts/train.sbatch"
TASK="evotuning_cdrmix"
mkdir -p bash_scripts/logs
MANIFEST="bash_scripts/logs/sweep_evotuning_cdrmix_$(date +%Y%m%d_%H%M%S).csv"
echo "lr,cmp,flank,fmp,salt,run_name,job_id" > "${MANIFEST}"

n_jobs=0
for lr in "${LR_ARR[@]}"; do
  for cmp in "${CMP_ARR[@]}"; do
    for flank in "${FLANK_ARR[@]}"; do
      for fmp in "${FMP_ARR[@]}"; do
        for salt in "${SALT_ARR[@]}"; do
          run_name="${NAME_PREFIX}_lr${lr}_cmp${cmp}_flank${flank}_fmp${fmp}_${salt}"
          overrides=(
            "model=${MODEL_PRESET}"
            "model.init.source=${INIT_SOURCE}"
            "model.freeze_first_n_layers=${FREEZE_LAYERS}"
            "scoring=${SCORING}"
            "training.learning_rate=${lr}"
            "data.cdr_mask_prob=${cmp}"
            "data.cdr_flank=${flank}"
            "data.framework_mask_prob=${fmp}"
            "data.split.salt=${salt}"
            "run_name=${run_name}"
          )
          if [[ "${INIT_SOURCE}" == "checkpoint" ]]; then
            overrides+=("model.init.checkpoint=${INIT_CHECKPOINT}")
          fi
          if [[ -n "${FASTA_PATH}" ]]; then
            overrides+=("data.fasta_path=${FASTA_PATH}")
          fi
          overrides+=("${EXTRA_OVERRIDES[@]}")

          if [[ "${DRY_RUN}" -eq 1 ]]; then
            echo "[dry-run] sbatch ${SBATCH_GRES_OPTS[*]} ${SBATCH_SCRIPT} ${TASK} ${overrides[*]}"
            echo "${lr},${cmp},${flank},${fmp},${salt},${run_name}," >> "${MANIFEST}"
          else
            out="$(sbatch "${SBATCH_GRES_OPTS[@]}" "${SBATCH_SCRIPT}" "${TASK}" "${overrides[@]}")"
            echo "${out}  [${run_name}]"
            job_id="$(awk '{print $NF}' <<< "${out}")"
            echo "${lr},${cmp},${flank},${fmp},${salt},${run_name},${job_id}" >> "${MANIFEST}"
          fi
          n_jobs=$((n_jobs+1))
        done
      done
    done
  done
done

echo "----"
if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "DRY RUN: ${n_jobs} jobs would be submitted. Manifest: ${MANIFEST}"
else
  echo "Submitted ${n_jobs} jobs. Manifest: ${MANIFEST}"
fi
