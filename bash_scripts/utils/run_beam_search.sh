#!/bin/bash
#SBATCH --job-name=beam_search
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --partition=gpu
#SBATCH --output=bash_scripts/logs/beam_search_%j.out

# Slurm executes batch scripts from a spool path; anchor to submit directory.
ROOT_DIR="${SLURM_SUBMIT_DIR:-}"
if [[ -z "${ROOT_DIR}" ]]; then
  ROOT_DIR="$(pwd)"
fi
cd "${ROOT_DIR}"

SBATCH_SCRIPT_PATH="bash_scripts/utils/run_beam_search.sh"
if [[ -f "${ROOT_DIR}/bash_scripts/common_setup.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/bash_scripts/common_setup.sh"
fi
cd "${ROOT_DIR}"

# ---------------------------------------------------------------------------
# Configure your run here.
# ---------------------------------------------------------------------------
BEAM_SIZE=5
N_STEPS=4
SNAPSHOT_EVERY=1
TEMPERATURE=1.0
SEED=42
OUT_DIR="outputs/gibbs"

EVOTUNED_CKPT="/cluster/project/infk/krause/mdenegri/protein-design/checkpoints/oas_dedup___esm2_t12_35M_UR50D__lr2e-05__ep3_48h_20260414_101859/"

mkdir -p "${OUT_DIR}"

run_variant() {
  local variant="$1"
  local ckpt_arg=("$@")
  ckpt_arg=("${ckpt_arg[@]:1}")  # drop $1

  uv run python scripts/stochastic_beam_search.py \
    --model-variant "${variant}" \
    --beam-size "${BEAM_SIZE}" \
    --n-steps "${N_STEPS}" \
    --snapshot-every "${SNAPSHOT_EVERY}" \
    --temperature "${TEMPERATURE}" \
    --seed "${SEED}" \
    --output-path "${OUT_DIR}/${variant}.csv" \
    "${ckpt_arg[@]}"
}

run_variant vanilla
run_variant evotuned --checkpoint-path "${EVOTUNED_CKPT}"

