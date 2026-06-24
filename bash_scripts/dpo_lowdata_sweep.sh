#!/bin/bash
# Low-data DPO learning-curve sweep launcher.
#
# Submits ONE independent SLURM job (bash_scripts/dpo_lowdata.sbatch) per
# (model, n_train, seed) combination. The jobs are independent and run in
# parallel as GPUs free up — this script does not block. Each run writes
# summary.json (test_spearman_avg on ed5) to $TRAIN_DIR/<base_name>_<ts>/;
# collect them afterwards with scripts/analysis/collect_lowdata_curve.py.
#
# Model checkpoints/base sizes are resolved from conf/analysis/models.yaml
# (the single source of truth). low_data.seed is the repeat dimension and is
# identical across models, so evo/vanilla/cdrmix see the SAME train sequences;
# the run seed (pairing/training) is fixed.
#
# Usage:
#   bash_scripts/dpo_lowdata_sweep.sh --dry-run                 # list jobs only
#   bash_scripts/dpo_lowdata_sweep.sh                           # submit full grid
#   bash_scripts/dpo_lowdata_sweep.sh --models vanilla_650m,evo_650m --n 200,1000 --seeds 0
#   bash_scripts/dpo_lowdata_sweep.sh --epochs 5

set -euo pipefail
cd "/cluster/home/${USER}/protein-design"

# ---- defaults (override via flags) -----------------------------------------
MODELS="vanilla_650m,evo_650m,evo_c05_cdrmix,evo_c05_cdrmix_spearman"
N_GRID="50,100,200,500,1000,2000,5000"
SEEDS="0,1,2"
EPOCHS=3
RUN_SEED=42
MODEL_PRESET="esm2_650m" # conf/model preset (esm2_650m | esm2_35m | ...); must
                         # match the registry base_model size of the chosen keys
TASK="lora_dpo"          # lora_dpo (LoRA, fits 24G) | dpo (full-FT, OOMs at 650M)
BATCH_SIZE=16            # LoRA frees the AdamW optimizer state -> headroom on 24G
# Per-epoch val eval cost is dominated by the val PAIR loss loop: DPO logprob is
# masked PLL (~3s/pair), so 1000 pairs ≈ 50min. Keep val_pairs small; the cdr_pll
# val-Spearman eval is much lighter (CDR-only), so it can stay larger.
VAL_PAIRS_CAP=200
VAL_SPEARMAN_CAP=2000
# Final eval iterates ALL test pairs (~40min on 35M); cap it. The reported ed5
# test_spearman_avg is computed separately and stays uncapped.
TEST_PAIRS_CAP=1000
DRY_RUN=0
EXTRA_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)  MODELS="$2"; shift 2 ;;
    --n)       N_GRID="$2"; shift 2 ;;
    --seeds)   SEEDS="$2"; shift 2 ;;
    --epochs)  EPOCHS="$2"; shift 2 ;;
    --run-seed) RUN_SEED="$2"; shift 2 ;;
    --model-preset) MODEL_PRESET="$2"; shift 2 ;;
    --task)    TASK="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --val-pairs-cap) VAL_PAIRS_CAP="$2"; shift 2 ;;
    --val-spearman-cap) VAL_SPEARMAN_CAP="$2"; shift 2 ;;
    --test-pairs-cap) TEST_PAIRS_CAP="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *)         EXTRA_OVERRIDES+=("$1"); shift ;;
  esac
done

IFS=',' read -r -a MODEL_ARR <<< "${MODELS}"
IFS=',' read -r -a N_ARR <<< "${N_GRID}"
IFS=',' read -r -a SEED_ARR <<< "${SEEDS}"

SBATCH_SCRIPT="bash_scripts/dpo_lowdata.sbatch"
MANIFEST="bash_scripts/logs/dpo_lowdata_sweep_$(date +%Y%m%d_%H%M%S).csv"
mkdir -p bash_scripts/logs

# ---- resolve model key -> checkpoint + base_model from the registry --------
# Emits  key|checkpoint|base_model  (checkpoint empty => vanilla/HF base). A
# non-whitespace separator avoids `read` collapsing an empty checkpoint field.
resolve_models() {
  uv run python - "$@" <<'PY'
import sys, yaml
from pathlib import Path
reg = yaml.safe_load(Path("conf/analysis/models.yaml").read_text())
models = reg.get("models", {})
for key in sys.argv[1:]:
    if key not in models:
        sys.stderr.write(f"Unknown model key {key!r} (not in conf/analysis/models.yaml)\n")
        sys.exit(2)
    m = models[key]
    ckpt = m.get("checkpoint") or ""
    base = m.get("base_model") or ""
    print(f"{key}|{ckpt}|{base}")
PY
}

declare -A CKPT
declare -A BASE
while IFS='|' read -r key ckpt base; do
  CKPT["$key"]="$ckpt"
  BASE["$key"]="$base"
done < <(resolve_models "${MODEL_ARR[@]}")

# Warn if a model's registry base size doesn't match the chosen preset (e.g.
# esm2_35m preset with a 650M checkpoint). Derive the size token from the preset.
PRESET_SIZE="$(sed -E 's/.*_([0-9]+m)$/\1/' <<< "${MODEL_PRESET}")"   # esm2_35m -> 35m
for key in "${MODEL_ARR[@]}"; do
  if [[ "${BASE[$key],,}" != *"${PRESET_SIZE}"* ]]; then
    echo "WARNING: model '${key}' base is '${BASE[$key]}', which doesn't match preset '${MODEL_PRESET}' (${PRESET_SIZE})." >&2
  fi
done

echo "model,checkpoint,n_train,low_data_seed,base_name,job_id" > "${MANIFEST}"

n_jobs=0
for key in "${MODEL_ARR[@]}"; do
  ckpt="${CKPT[$key]}"
  for n in "${N_ARR[@]}"; do
    for s in "${SEED_ARR[@]}"; do
      base_name="lowdata_${key}_n${n}_s${s}"
      overrides=(
        "task=${TASK}"
        "model=${MODEL_PRESET}"
        "seed=${RUN_SEED}"
        "training.num_epochs=${EPOCHS}"
        "training.batch_size=${BATCH_SIZE}"
        "data.low_data.enabled=true"
        "data.low_data.n_train=${n}"
        "data.low_data.seed=${s}"
        "data.low_data.scheme=stratified"
        "data.pair_split.enforce_train_controlled_sizes=false"
        # Per-epoch val eval on the full ed2 val set costs hours; cap it to a
        # fixed size (final ed5 test eval is never capped).
        "data.low_data.val_pairs_cap=${VAL_PAIRS_CAP}"
        "data.low_data.val_spearman_cap=${VAL_SPEARMAN_CAP}"
        "data.low_data.test_pairs_cap=${TEST_PAIRS_CAP}"
        # Low-data total steps are tiny (≈10–1000), so the default
        # linear_warmup_cosine (warmup_steps=200) is infeasible. Use epoch-based
        # cosine: T_max=num_epochs, no warmup<total constraint, identical across
        # all N and models. Override via EXTRA_OVERRIDES if needed.
        "training.scheduler.name=cosine"
        "training.scheduler.interval=epoch"
        "run.base_name=${base_name}"
      )
      if [[ -n "${ckpt}" ]]; then
        overrides+=("model.init.source=checkpoint" "model.init.checkpoint=${ckpt}")
      else
        overrides+=("model.init.source=huggingface")
      fi
      overrides+=("${EXTRA_OVERRIDES[@]}")

      if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "[dry-run] sbatch ${SBATCH_SCRIPT} ${overrides[*]}"
        echo "${key},${ckpt},${n},${s},${base_name}," >> "${MANIFEST}"
      else
        out="$(sbatch "${SBATCH_SCRIPT}" "${overrides[@]}")"
        echo "${out}  [${base_name}]"
        job_id="$(awk '{print $NF}' <<< "${out}")"
        echo "${key},${ckpt},${n},${s},${base_name},${job_id}" >> "${MANIFEST}"
      fi
      n_jobs=$((n_jobs+1))
    done
  done
done

echo "----"
if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "DRY RUN: ${n_jobs} jobs would be submitted. Manifest: ${MANIFEST}"
else
  echo "Submitted ${n_jobs} jobs. Manifest: ${MANIFEST}"
fi
