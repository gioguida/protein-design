#!/bin/bash
#SBATCH --job-name=embedding_analysis
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --partition=gpu
#SBATCH --output=bash_scripts/logs/embedding_analysis_%j.out

# Full analysis pipeline:
#   extract_embeddings × N variants
#   ↓
#   compute_projections          (existing: shared vanilla PCA + UMAP)
#   compute_per_model_pca        (DMS-only PCA per model)
#   compute_diff_vectors_pca     (uncentered SVD on embed − WT)
#   compute_cka                  (4×4 representational similarity)
#   compute_procrustes_displacement  (gated on CKA threshold)
#   compute_pll_pca              (joint PLL biplot + per-position loadings)
#   gibbs_diagnostics            (only when Gibbs CSVs exist)
#   ↓
#   plot_projections, plot_per_model_pca, plot_gibbs_per_model_pca,
#   plot_diff_vectors_pca, plot_oas_germline_umap, plot_pll_pca

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"

if [[ -f "${ROOT_DIR}/bash_scripts/common_setup.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/bash_scripts/common_setup.sh"
fi
cd "${ROOT_DIR}"

# ---------------------------------------------------------------------------
# Configure paths & variants. Override any of these via the environment
# before invoking sbatch (e.g. EVOTUNED_CKPT=/path sbatch ...).
# ---------------------------------------------------------------------------
SCRATCH_BASE="${SCRATCH_DIR:-/cluster/scratch/${USER}/protein-design}/embedding_analysis"
PROJECT_BASE="${PROJECT_DIR:-/cluster/project/infk/krause/${USER}/protein-design}/reports/embedding_analysis"

# Which DMS dataset feeds extract_embeddings/gibbs_diagnostics. All artifact
# dirs are scoped by this so two datasets don't clobber each other.
DMS_DATASET="${DMS_DATASET:-ed2}"
echo "DMS_DATASET=${DMS_DATASET}"

EMB_DIR="${EMB_DIR:-${SCRATCH_BASE}/embeddings/${DMS_DATASET}}"
PROJ_DIR="${PROJ_DIR:-${SCRATCH_BASE}/projections/${DMS_DATASET}}"
PER_MODEL_DIR="${PER_MODEL_DIR:-${SCRATCH_BASE}/per_model_pca/${DMS_DATASET}}"
DIFF_DIR="${DIFF_DIR:-${SCRATCH_BASE}/diff_pca/${DMS_DATASET}}"
CKA_DIR="${CKA_DIR:-${PROJECT_BASE}/plots/${DMS_DATASET}/cka}"
PROCRUSTES_DIR="${PROCRUSTES_DIR:-${SCRATCH_BASE}/procrustes/${DMS_DATASET}}"
PLL_DIR="${PLL_DIR:-${SCRATCH_BASE}/pll_pca/${DMS_DATASET}}"
GIBBS_DIAG_DIR="${GIBBS_DIAG_DIR:-${PROJECT_BASE}/plots/${DMS_DATASET}/gibbs_diagnostics}"

PLOTS_DIR="${PLOTS_DIR:-${PROJECT_BASE}/plots/${DMS_DATASET}}"

CKA_THRESHOLD="${CKA_THRESHOLD:-0.5}"
MAX_DMS="${MAX_DMS:-500}"
MAX_OAS="${MAX_OAS:-2000}"
MAX_GIBBS="${MAX_GIBBS:-200}"

# Skip the GPU-heavy steps if their primary output already exists. Set
# FORCE=1 to recompute everything regardless. The fast PCA/CKA/Procrustes
# and plotting steps always run — they're cheap and safe to re-do when
# plot code changes.
FORCE="${FORCE:-0}"

# Variants — edit the arrays below to add/remove models.
# Each entry is "label|checkpoint|gibbs_csv". Use empty checkpoint for vanilla.
# Use empty gibbs_csv to skip Gibbs diagnostics for that variant.
# Checkpoint dirs contain best.pt; the loaders strip the "model." prefix and
# fall back to a vanilla HF init when needed.
CKPT_ROOT="${CKPT_ROOT:-/cluster/project/infk/krause/${USER}/protein-design/checkpoints}"

EVOTUNED_CKPT="${EVOTUNED_CKPT:-${CKPT_ROOT}/oas_dedup___esm2_t12_35M_UR50D__lr2e-05__ep3_48h_20260414_101859}"
C05_FINETUNED_CKPT="${C05_FINETUNED_CKPT:-${CKPT_ROOT}/c05_c05_cdrh3_blosum25__evo_seed_20260424_191020}"
DPO_FROM_EVO_CKPT="${DPO_FROM_EVO_CKPT:-${CKPT_ROOT}/dpo__evo_base_20260425_190014}"
DPO_FROM_C05_CKPT="${DPO_FROM_C05_CKPT:-${CKPT_ROOT}/dpo__c05_ft_20260425_190143}"
DPO_FROM_C05_EP6_CKPT="${DPO_FROM_C05_EP6_CKPT:-${CKPT_ROOT}/dpo__c05_ft_20260425_190143_ep6}"
GIOVANNI_DPO_CKPT="${GIOVANNI_DPO_CKPT:-${CKPT_ROOT}/giovanni-dpo}"
GIOVANNI_DPO_LESS_CKPT="${GIOVANNI_DPO_LESS_CKPT:-${CKPT_ROOT}/giovanni-dpo-trained-less}"
UNLIKELIHOOD_CKPT="${UNLIKELIHOOD_CKPT:-${CKPT_ROOT}/unlikelihood-experiment}"

GIBBS_DIST_DIR="${GIBBS_DIST_DIR:-outputs/gibbs/distribution}"
GIBBS_FIT_DIR="${GIBBS_FIT_DIR:-outputs/gibbs/fitness}"

# Each entry: "label|checkpoint|distribution_csv|fitness_csv". Either CSV
# slot may point to a non-existent path; per-step logic below checks file
# existence and only passes flags for CSVs that exist.
VARIANTS=(
  "vanilla||${GIBBS_DIST_DIR}/vanilla.csv|${GIBBS_FIT_DIR}/vanilla.csv"
  "evotuned|${EVOTUNED_CKPT}|${GIBBS_DIST_DIR}/evotuned.csv|${GIBBS_FIT_DIR}/evotuned.csv"
  "c05-finetuned|${C05_FINETUNED_CKPT}|${GIBBS_DIST_DIR}/c05-finetuned.csv|${GIBBS_FIT_DIR}/c05-finetuned.csv"
  "dpo-from-evo|${DPO_FROM_EVO_CKPT}|${GIBBS_DIST_DIR}/dpo-from-evo.csv|${GIBBS_FIT_DIR}/dpo-from-evo.csv"
  "dpo-from-c05|${DPO_FROM_C05_CKPT}|${GIBBS_DIST_DIR}/dpo-from-c05.csv|${GIBBS_FIT_DIR}/dpo-from-c05.csv"
  "dpo-from-c05-ep6|${DPO_FROM_C05_EP6_CKPT}|${GIBBS_DIST_DIR}/dpo-from-c05-ep6.csv|${GIBBS_FIT_DIR}/dpo-from-c05-ep6.csv"
  "giovanni-dpo|${GIOVANNI_DPO_CKPT}|${GIBBS_DIST_DIR}/giovanni-dpo.csv|${GIBBS_FIT_DIR}/giovanni-dpo.csv"
  "giovanni-dpo-less|${GIOVANNI_DPO_LESS_CKPT}|${GIBBS_DIST_DIR}/giovanni-dpo-less.csv|${GIBBS_FIT_DIR}/giovanni-dpo-less.csv"
  "unlikelihood|${UNLIKELIHOOD_CKPT}|${GIBBS_DIST_DIR}/unlikelihood.csv|${GIBBS_FIT_DIR}/unlikelihood.csv"
)

mkdir -p "${EMB_DIR}" "${PROJ_DIR}" "${PER_MODEL_DIR}" "${DIFF_DIR}" \
         "${CKA_DIR}" "${PROCRUSTES_DIR}" "${PLL_DIR}" "${GIBBS_DIAG_DIR}" \
         "${PLOTS_DIR}"

# ---------------------------------------------------------------------------
# Step 1 — Extract embeddings per variant
# ---------------------------------------------------------------------------
EMBED_NPZ=()
PLL_VARIANT_ARGS=()
GIBBS_DIAG_ARGS=()

for entry in "${VARIANTS[@]}"; do
  IFS='|' read -r label checkpoint dist_csv fit_csv <<<"${entry}"
  npz_path="${EMB_DIR}/${label}.npz"
  EMBED_NPZ+=("${npz_path}")

  echo "============================================================"
  echo "[extract_embeddings] variant=${label}"
  echo "============================================================"
  if [[ "${FORCE}" != "1" && -f "${npz_path}" ]]; then
    echo "[extract_embeddings] ${npz_path} exists — skipping (set FORCE=1 to recompute)"
  else
    extract_args=(
      --model-variant "${label}"
      --output-path "${npz_path}"
      --dms-dataset "${DMS_DATASET}"
      --max-dms "${MAX_DMS}"
      --max-oas "${MAX_OAS}"
      --max-gibbs "${MAX_GIBBS}"
    )
    [[ -n "${checkpoint}" ]] && extract_args+=(--checkpoint-path "${checkpoint}")
    [[ -f "${dist_csv}" ]] && extract_args+=(--gibbs-path "distribution=${dist_csv}")
    [[ -f "${fit_csv}" ]] && extract_args+=(--gibbs-path "fitness=${fit_csv}")
    uv run python scripts/analysis/extract_embeddings.py "${extract_args[@]}"
  fi

  PLL_VARIANT_ARGS+=(--variant "${label}=${checkpoint}")
  if [[ -f "${dist_csv}" ]]; then
    GIBBS_DIAG_ARGS+=(--gibbs "${label}=${checkpoint}=${dist_csv}=distribution")
  fi
  if [[ -f "${fit_csv}" ]]; then
    GIBBS_DIAG_ARGS+=(--gibbs "${label}=${checkpoint}=${fit_csv}=fitness")
  fi
done

# ---------------------------------------------------------------------------
# Step 2 — Shared PCA / UMAP projections (existing pipeline)
# ---------------------------------------------------------------------------
echo "============================================================"
echo "[compute_projections]"
echo "============================================================"
uv run python scripts/analysis/compute_projections.py \
  "${EMBED_NPZ[@]}" \
  --output-dir "${PROJ_DIR}"

# ---------------------------------------------------------------------------
# Step 3 — Per-model DMS-only PCA (Phase 1)
# ---------------------------------------------------------------------------
echo "============================================================"
echo "[compute_per_model_pca]"
echo "============================================================"
uv run python scripts/analysis/compute_per_model_pca.py \
  "${EMBED_NPZ[@]}" \
  --output-dir "${PER_MODEL_DIR}"

# ---------------------------------------------------------------------------
# Step 4 — Diff-vector SVD (Phase 2)
# ---------------------------------------------------------------------------
echo "============================================================"
echo "[compute_diff_vectors_pca]"
echo "============================================================"
uv run python scripts/analysis/compute_diff_vectors_pca.py \
  "${EMBED_NPZ[@]}" \
  --output-dir "${DIFF_DIR}"

# ---------------------------------------------------------------------------
# Step 5 — Linear CKA (Phase 4)
# ---------------------------------------------------------------------------
echo "============================================================"
echo "[compute_cka]"
echo "============================================================"
uv run python scripts/analysis/compute_cka.py \
  "${EMBED_NPZ[@]}" \
  --output-dir "${CKA_DIR}"

# ---------------------------------------------------------------------------
# Step 6 — Procrustes (Phase 5; gated on CKA)
# ---------------------------------------------------------------------------
echo "============================================================"
echo "[compute_procrustes_displacement] (CKA gate ${CKA_THRESHOLD})"
echo "============================================================"
uv run python scripts/analysis/compute_procrustes_displacement.py \
  "${EMBED_NPZ[@]}" \
  --cka-dir "${CKA_DIR}" \
  --cka-threshold "${CKA_THRESHOLD}" \
  --output-dir "${PROCRUSTES_DIR}"

# ---------------------------------------------------------------------------
# Step 7 — Joint PLL PCA (Phase 6)
# ---------------------------------------------------------------------------
echo "============================================================"
echo "[compute_pll_pca]"
echo "============================================================"
PLL_OUT="${PLL_DIR}/pll_pca.npz"
if [[ "${FORCE}" != "1" && -f "${PLL_OUT}" ]]; then
  echo "[compute_pll_pca] ${PLL_OUT} exists — skipping (set FORCE=1 to recompute)"
else
  uv run python scripts/analysis/compute_pll_pca.py \
    "${PLL_VARIANT_ARGS[@]}" \
    --max-dms "${MAX_DMS}" \
    --output-path "${PLL_OUT}"
fi

# ---------------------------------------------------------------------------
# Step 8 — Gibbs diagnostics (Phase 7; only if Gibbs CSVs exist)
# ---------------------------------------------------------------------------
if (( ${#GIBBS_DIAG_ARGS[@]} )); then
  echo "============================================================"
  echo "[gibbs_diagnostics]"
  echo "============================================================"
  # Use the trajectory PNG as the skip marker: it's a newer output, so any
  # pre-existing GIBBS_DIAG_DIR from before the trajectory plot was added
  # will still trigger a re-run the first time after upgrade.
  GIBBS_MARKER="${GIBBS_DIAG_DIR}/gibbs_pll_trajectory.png"
  if [[ "${FORCE}" != "1" && -f "${GIBBS_MARKER}" ]]; then
    echo "[gibbs_diagnostics] ${GIBBS_MARKER} exists — skipping (set FORCE=1 to recompute)"
  else
    uv run python scripts/analysis/gibbs_diagnostics.py \
      "${GIBBS_DIAG_ARGS[@]}" \
      --dms-dataset "${DMS_DATASET}" \
      --max-dms "${MAX_DMS}" \
      --output-dir "${GIBBS_DIAG_DIR}"
  fi
else
  echo "[gibbs_diagnostics] no Gibbs CSVs found — skipping"
fi

# ---------------------------------------------------------------------------
# Step 9 — Plotting
# ---------------------------------------------------------------------------
echo "============================================================"
echo "[plot_projections]                  (shared PCA/UMAP)"
echo "============================================================"
uv run python scripts/analysis/plot_projections.py \
  --projections-dir "${PROJ_DIR}" \
  --embeddings-dir "${EMB_DIR}" \
  --output-dir "${PLOTS_DIR}/projections"

echo "============================================================"
echo "[plot_per_model_pca]"
echo "============================================================"
uv run python scripts/analysis/plot_per_model_pca.py \
  --projections-dir "${PER_MODEL_DIR}" \
  --output-dir "${PLOTS_DIR}/per_model_pca"

echo "============================================================"
echo "[plot_gibbs_per_model_pca]"
echo "============================================================"
uv run python scripts/analysis/plot_gibbs_per_model_pca.py \
  --per-model-pca-dir "${PER_MODEL_DIR}" \
  --embeddings-dir "${EMB_DIR}" \
  --output-dir "${PLOTS_DIR}/per_model_pca"

echo "============================================================"
echo "[plot_diff_vectors_pca]"
echo "============================================================"
uv run python scripts/analysis/plot_diff_vectors_pca.py \
  --projections-dir "${DIFF_DIR}" \
  --output-dir "${PLOTS_DIR}/diff_pca"

echo "============================================================"
echo "[plot_oas_germline_umap]"
echo "============================================================"
uv run python scripts/analysis/plot_oas_germline_umap.py \
  "${EMBED_NPZ[@]}" \
  --output-dir "${PLOTS_DIR}/oas_germline"

echo "============================================================"
echo "[plot_pll_pca]"
echo "============================================================"
uv run python scripts/analysis/plot_pll_pca.py \
  --input "${PLL_DIR}/pll_pca.npz" \
  --output-dir "${PLOTS_DIR}/pll_pca"

echo "============================================================"
echo "[done] all outputs under ${PLOTS_DIR}"
echo "============================================================"
