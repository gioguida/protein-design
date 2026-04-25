#!/bin/bash
#SBATCH --job-name=embedding_analysis
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --partition=gpu
#SBATCH --output=bash_scripts/logs/embedding_analysis_%j.log

# Slurm executes batch scripts from a spool path; anchor to submit directory.
ROOT_DIR="${SLURM_SUBMIT_DIR:-}"
if [[ -z "${ROOT_DIR}" ]]; then
  ROOT_DIR="$(pwd)"
fi
cd "${ROOT_DIR}"

if [[ -f "${ROOT_DIR}/bash_scripts/common_setup.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/bash_scripts/common_setup.sh"
fi
cd "${ROOT_DIR}"

# ---------------------------------------------------------------------------
# Configure your run here.
# ---------------------------------------------------------------------------
EVO_CKPT="/cluster/project/infk/krause/mdenegri/protein-design/checkpoints/oas_dedup___esm2_t12_35M_UR50D__lr2e-05__ep3_48h_20260414_101859/"
C05_FT_CKPT="/cluster/project/infk/krause/mdenegri/protein-design/checkpoints/c05_c05_cdrh3_blosum25__evo_seed_20260424_191020/"

GIBBS_DIR="${ROOT_DIR}/outputs/gibbs"
EMB_DIR="${SCRATCH_DIR}/embedding_analysis/embeddings"
PROJ_DIR="${SCRATCH_DIR}/embedding_analysis/projections"
PLOT_DIR="${PROJECT_DIR}/plots/embedding_analysis"

mkdir -p "${EMB_DIR}" "${PROJ_DIR}" "${PLOT_DIR}"

# ---------------------------------------------------------------------------
# Step 1 — extract embeddings, one invocation per model variant.
# Gibbs paths are passed only when the file exists (graceful degradation).
# ---------------------------------------------------------------------------
extract() {
  local variant="$1"; shift
  local gibbs_arg=()
  local gibbs_path="${GIBBS_DIR}/${variant}.csv"
  if [[ -f "${gibbs_path}" ]]; then
    gibbs_arg=(--gibbs-path "${gibbs_path}")
  else
    echo "[info] no gibbs file for ${variant} at ${gibbs_path} — extracting without"
  fi
  uv run python scripts/analysis/extract_embeddings.py \
    --model-variant "${variant}" \
    --output-path "${EMB_DIR}/${variant}.npz" \
    "${gibbs_arg[@]}" \
    "$@"
}

extract vanilla
extract evotuned      --checkpoint-path "${EVO_CKPT}"
extract c05-finetuned --checkpoint-path "${C05_FT_CKPT}"

# ---------------------------------------------------------------------------
# Step 2 — fit PCA/UMAP on vanilla, project all variants.
# ---------------------------------------------------------------------------
uv run python scripts/analysis/compute_projections.py \
  "${EMB_DIR}"/*.npz \
  --output-dir "${PROJ_DIR}" \
  --vanilla-label vanilla

# ---------------------------------------------------------------------------
# Step 3 — render figures.
# ---------------------------------------------------------------------------
uv run python scripts/analysis/plot_projections.py \
  --projections-dir "${PROJ_DIR}" \
  --embeddings-dir "${EMB_DIR}" \
  --output-dir "${PLOT_DIR}"

echo "[done] plots → ${PLOT_DIR}"
