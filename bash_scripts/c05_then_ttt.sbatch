#!/bin/bash
#SBATCH --job-name=c05_then_ttt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --time=02:00:00
#SBATCH --output=logs/c05_then_ttt_%j.out
#SBATCH --error=logs/c05_then_ttt_%j.err

# Usage: sbatch bash_scripts/c05_then_ttt.sbatch <EVOTUNING_RUN_NAME>
#   <EVOTUNING_RUN_NAME>: name of the evotuning run dir under $PROJECT_DIR/checkpoints/
#                         whose best.pt seeds the C05-5k finetune.
#
# Runs two stages in one SLURM allocation:
#   1. evotuning_c05_5k.yaml  (train_evotuning.py)  — continues MLM on C05-5k FASTA
#   2. ttt_c05_single_from_evotuned.yaml  (train_ttt.py)  — TTT on the single CDRH3
# Stage 2's finetune path is resolved from the run dir stage 1 just created.

set -euo pipefail
cd /cluster/home/mdenegri/protein-design
module load eth_proxy
export no_proxy="${no_proxy//api.wandb.ai,/}"
export NO_PROXY="${no_proxy}"
set -a; source .env; set +a

source .venv/bin/activate
mkdir -p logs configs/_generated

EVOTUNING_RUN="${1:?Usage: sbatch c05_then_ttt.sbatch <EVOTUNING_RUN_NAME>}"

# ── Stage 1: C05-5k finetune from evotuned best.pt ─────────────────────
STAGE1_CONFIG="configs/_generated/c05_5k_from_${EVOTUNING_RUN}.yaml"
STAGE1_NAME="c05_5k_from_${EVOTUNING_RUN}"
sed "s|REPLACE_WITH_EVOTUNING_RUN_NAME|${EVOTUNING_RUN}|g" \
    configs/evotuning_c05_5k.yaml > "$STAGE1_CONFIG"

echo "[stage 1] running evotuning on C05-5k, seeded from ${EVOTUNING_RUN}/best.pt"
python scripts/train_evotuning.py --config "$STAGE1_CONFIG" --run-name "$STAGE1_NAME"

# Stage 1 appended a timestamp; find the exact dir it just created.
STAGE1_RUN_DIR=$(ls -td "${TRAIN_DIR}/${STAGE1_NAME}"_* | head -1)
STAGE1_RUN_NAME=$(basename "$STAGE1_RUN_DIR")
echo "[stage 1] completed: $STAGE1_RUN_DIR"

# Verify best.pt exists (archive copy lives under $PROJECT_DIR/checkpoints/)
if [[ ! -f "${PROJECT_DIR}/checkpoints/${STAGE1_RUN_NAME}/best.pt" ]]; then
    echo "ERROR: stage 1 did not produce ${PROJECT_DIR}/checkpoints/${STAGE1_RUN_NAME}/best.pt" >&2
    exit 1
fi

# ── Stage 2: TTT on single C05 CDRH3, seeded from stage-1 best.pt ───────
STAGE2_CONFIG="configs/_generated/ttt_from_${STAGE1_RUN_NAME}.yaml"
STAGE2_NAME="ttt_from_${STAGE1_RUN_NAME}"
sed "s|REPLACE_WITH_C05_5K_RUN_NAME|${STAGE1_RUN_NAME}|g" \
    configs/ttt_c05_single_from_evotuned.yaml > "$STAGE2_CONFIG"

echo "[stage 2] running TTT, seeded from ${STAGE1_RUN_NAME}/best.pt"
python scripts/train_ttt.py --config "$STAGE2_CONFIG" --run-name "$STAGE2_NAME"

echo "[done] chain complete: $STAGE1_RUN_NAME -> $STAGE2_NAME"
