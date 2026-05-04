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
#   extract_oas_embeddings × N variants  (dataset-agnostic; cached under OAS_DIR)
#   extract_embeddings × N variants      (DMS+WT+Gibbs only; per DMS_DATASET)
#   ↓
#   compute_per_model_pca        (DMS-only PCA per model)
#   compute_diff_vectors_pca     (uncentered SVD on embed − WT)
#   compute_cka                  (representational similarity)
#   compute_procrustes_displacement  (gated on CKA threshold)
#   compute_pll_pca              (joint PLL biplot + per-position loadings)
#   gibbs_diagnostics            (only when Gibbs CSVs exist)
#   beam_diagnostics             (only when stochastic-beam CSVs exist)
#   ↓
#   plot_per_model_pca, plot_gibbs_per_model_pca, plot_diff_vectors_pca,
#   plot_oas_umap (×5 modes), plot_pll_pca, plot_beam_per_model_pca

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
# OAS embeddings are dataset-independent — cache outside any DMS_DATASET scope.
OAS_DIR="${OAS_DIR:-${SCRATCH_BASE}/embeddings/_oas}"
PER_MODEL_DIR="${PER_MODEL_DIR:-${SCRATCH_BASE}/per_model_pca/${DMS_DATASET}}"
DIFF_DIR="${DIFF_DIR:-${SCRATCH_BASE}/diff_pca/${DMS_DATASET}}"
CKA_DIR="${CKA_DIR:-${PROJECT_BASE}/plots/${DMS_DATASET}/cka}"
PROCRUSTES_DIR="${PROCRUSTES_DIR:-${SCRATCH_BASE}/procrustes/${DMS_DATASET}}"
PLL_DIR="${PLL_DIR:-${SCRATCH_BASE}/pll_pca/${DMS_DATASET}}"
GIBBS_DIAG_DIR="${GIBBS_DIAG_DIR:-${PROJECT_BASE}/plots/${DMS_DATASET}/gibbs_diagnostics}"
BEAM_DIAG_DIR="${BEAM_DIAG_DIR:-${PROJECT_BASE}/plots/${DMS_DATASET}/beam_diagnostics}"

PLOTS_DIR="${PLOTS_DIR:-${PROJECT_BASE}/plots/${DMS_DATASET}}"
BEAM_PLOTS_DIR="${BEAM_PLOTS_DIR:-${PLOTS_DIR}/per_model_pca_beam}"
BEAM_EMB_DIR="${BEAM_EMB_DIR:-${SCRATCH_BASE}/embeddings_beam/${DMS_DATASET}}"

CKA_THRESHOLD="${CKA_THRESHOLD:-0.5}"
MAX_DMS="${MAX_DMS:-500}"
MAX_OAS="${MAX_OAS:-2000}"
MAX_GIBBS="${MAX_GIBBS:-200}"

# Skip the GPU-heavy steps if their primary output already exists. Set
# FORCE=1 to recompute everything regardless. The fast PCA/CKA/Procrustes
# and plotting steps always run — they're cheap and safe to re-do when
# plot code changes.
FORCE="${FORCE:-0}"
# Set SKIP_OAS=1 to skip extract_oas_embeddings (Step 1a) and plot_oas_umap.
# Safe when OAS plots are not needed; all DMS/Gibbs steps are unaffected.
SKIP_OAS="${SKIP_OAS:-0}"
# Set SKIP_BEAM=1 to skip stochastic-beam extraction/diagnostics/plots.
SKIP_BEAM="${SKIP_BEAM:-0}"

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
BEAM_DIST_DIR="${BEAM_DIST_DIR:-outputs/beam_search/distribution}"
BEAM_FIT_DIR="${BEAM_FIT_DIR:-outputs/beam_search/fitness}"

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

mkdir -p "${EMB_DIR}" "${OAS_DIR}" "${PER_MODEL_DIR}" "${DIFF_DIR}" \
         "${CKA_DIR}" "${PROCRUSTES_DIR}" "${PLL_DIR}" "${GIBBS_DIAG_DIR}" \
         "${BEAM_DIAG_DIR}" "${PLOTS_DIR}" "${BEAM_PLOTS_DIR}" "${BEAM_EMB_DIR}"

append_sampler_paths() {
  local label="$1"
  local checkpoint="$2"
  local dist_csv="$3"
  local fit_csv="$4"
  local -n diag_args_ref="$5"
  local -n extract_args_ref="$6"

  if [[ -f "${dist_csv}" ]]; then
    diag_args_ref+=(--gibbs "${label}=${checkpoint}=${dist_csv}=distribution")
    extract_args_ref+=(--gibbs-path "distribution=${dist_csv}")
  fi
  if [[ -f "${fit_csv}" ]]; then
    diag_args_ref+=(--gibbs "${label}=${checkpoint}=${fit_csv}=fitness")
    extract_args_ref+=(--gibbs-path "fitness=${fit_csv}")
  fi
}

run_sampler_diagnostics() {
  local sampler_name="$1"
  local diag_dir="$2"
  local skip_flag="$3"
  local -n diag_args_ref="$4"

  if [[ "${skip_flag}" == "1" ]]; then
    echo "[${sampler_name}_diagnostics] skip flag enabled — skipping"
    return
  fi

  if (( ${#diag_args_ref[@]} )); then
    echo "============================================================"
    echo "[${sampler_name}_diagnostics]"
    echo "============================================================"
    local marker="${diag_dir}/gibbs_pll_trajectory.png"
    if [[ "${FORCE}" != "1" && -f "${marker}" ]]; then
      echo "[${sampler_name}_diagnostics] ${marker} exists — skipping (set FORCE=1 to recompute)"
    else
      uv run python scripts/analysis/gibbs_diagnostics.py \
        "${diag_args_ref[@]}" \
        --dms-dataset "${DMS_DATASET}" \
        --max-dms "${MAX_DMS}" \
        --output-dir "${diag_dir}"
    fi
  else
    echo "[${sampler_name}_diagnostics] no CSVs found — skipping"
  fi
}

# ---------------------------------------------------------------------------
# Step 1a — Extract OAS embeddings per variant (dataset-agnostic, cached)
# ---------------------------------------------------------------------------
OAS_NPZ=()
if [[ "${SKIP_OAS}" == "1" ]]; then
  echo "[extract_oas_embeddings] SKIP_OAS=1 — skipping all OAS extraction"
else
  for entry in "${VARIANTS[@]}"; do
    IFS='|' read -r label checkpoint dist_csv fit_csv <<<"${entry}"
    oas_npz="${OAS_DIR}/${label}.npz"
    OAS_NPZ+=("${oas_npz}")

    echo "============================================================"
    echo "[extract_oas_embeddings] variant=${label}"
    echo "============================================================"
    oas_args=(
      --model-variant "${label}"
      --output-path "${oas_npz}"
      --max-oas "${MAX_OAS}"
    )
    [[ -n "${checkpoint}" ]] && oas_args+=(--checkpoint-path "${checkpoint}")
    [[ "${FORCE}" != "1" ]] && oas_args+=(--skip-if-current)
    uv run python scripts/analysis/extract_oas_embeddings.py "${oas_args[@]}"
  done
fi

# ---------------------------------------------------------------------------
# Step 1b — Extract DMS+WT+Gibbs embeddings per variant (per DMS_DATASET)
# ---------------------------------------------------------------------------
EMBED_NPZ=()
PLL_VARIANT_ARGS=()
GIBBS_DIAG_ARGS=()
BEAM_DIAG_ARGS=()

for entry in "${VARIANTS[@]}"; do
  IFS='|' read -r label checkpoint dist_csv fit_csv <<<"${entry}"
  npz_path="${EMB_DIR}/${label}.npz"
  beam_npz_path="${BEAM_EMB_DIR}/${label}.npz"
  beam_dist_csv="${BEAM_DIST_DIR}/${label}.csv"
  beam_fit_csv="${BEAM_FIT_DIR}/${label}.csv"
  EMBED_NPZ+=("${npz_path}")

  echo "============================================================"
  echo "[extract_embeddings] variant=${label}"
  echo "============================================================"
  gibbs_extract_args=()
  append_sampler_paths "${label}" "${checkpoint}" "${dist_csv}" "${fit_csv}" \
    GIBBS_DIAG_ARGS gibbs_extract_args
  extract_args=(
    --model-variant "${label}"
    --output-path "${npz_path}"
    --dms-dataset "${DMS_DATASET}"
    --max-dms "${MAX_DMS}"
    --max-gibbs "${MAX_GIBBS}"
  )
  extract_args+=("${gibbs_extract_args[@]}")
  [[ -n "${checkpoint}" ]] && extract_args+=(--checkpoint-path "${checkpoint}")
  [[ "${FORCE}" != "1" ]] && extract_args+=(--skip-if-current)
  uv run python scripts/analysis/extract_embeddings.py "${extract_args[@]}"

  if [[ "${SKIP_BEAM}" != "1" ]]; then
    echo "============================================================"
    echo "[extract_embeddings_beam] variant=${label}"
    echo "============================================================"
    beam_extract_args=()
    append_sampler_paths "${label}" "${checkpoint}" "${beam_dist_csv}" "${beam_fit_csv}" \
      BEAM_DIAG_ARGS beam_extract_args
    if (( ${#beam_extract_args[@]} )); then
      beam_args=(
        --model-variant "${label}"
        --output-path "${beam_npz_path}"
        --dms-dataset "${DMS_DATASET}"
        --max-dms "${MAX_DMS}"
        --max-gibbs "${MAX_GIBBS}"
      )
      beam_args+=("${beam_extract_args[@]}")
      [[ -n "${checkpoint}" ]] && beam_args+=(--checkpoint-path "${checkpoint}")
      [[ "${FORCE}" != "1" ]] && beam_args+=(--skip-if-current)
      uv run python scripts/analysis/extract_embeddings.py "${beam_args[@]}"
    else
      echo "[extract_embeddings_beam] no beam CSVs for ${label} — skipping"
    fi
  fi

  PLL_VARIANT_ARGS+=(--variant "${label}=${checkpoint}")
done

# ---------------------------------------------------------------------------
# Step 2 — Per-model DMS-only PCA (Phase 1)
# ---------------------------------------------------------------------------
echo "============================================================"
echo "[compute_per_model_pca]"
echo "============================================================"
uv run python scripts/analysis/compute_per_model_pca.py \
  "${EMBED_NPZ[@]}" \
  --output-dir "${PER_MODEL_DIR}"

# ---------------------------------------------------------------------------
# Step 3 — Diff-vector SVD (Phase 2)
# ---------------------------------------------------------------------------
echo "============================================================"
echo "[compute_diff_vectors_pca]"
echo "============================================================"
uv run python scripts/analysis/compute_diff_vectors_pca.py \
  "${EMBED_NPZ[@]}" \
  --output-dir "${DIFF_DIR}"

# ---------------------------------------------------------------------------
# Step 4 — Linear CKA (Phase 4)
# ---------------------------------------------------------------------------
echo "============================================================"
echo "[compute_cka]"
echo "============================================================"
uv run python scripts/analysis/compute_cka.py \
  "${EMBED_NPZ[@]}" \
  --output-dir "${CKA_DIR}"

# ---------------------------------------------------------------------------
# Step 5 — Procrustes (Phase 5; gated on CKA)
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
# Step 6 — Joint PLL PCA (Phase 6)
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
# Step 7 — Sampler diagnostics (Phase 7; Gibbs + optional beam)
# ---------------------------------------------------------------------------
run_sampler_diagnostics "gibbs" "${GIBBS_DIAG_DIR}" "0" GIBBS_DIAG_ARGS
run_sampler_diagnostics "beam" "${BEAM_DIAG_DIR}" "${SKIP_BEAM}" BEAM_DIAG_ARGS

# ---------------------------------------------------------------------------
# Step 8 — Plotting
# ---------------------------------------------------------------------------
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
echo "[plot_beam_per_model_pca]"
echo "============================================================"
if [[ "${SKIP_BEAM}" == "1" ]]; then
  echo "[plot_beam_per_model_pca] SKIP_BEAM=1 — skipping"
else
  uv run python scripts/analysis/plot_gibbs_per_model_pca.py \
    --per-model-pca-dir "${PER_MODEL_DIR}" \
    --embeddings-dir "${BEAM_EMB_DIR}" \
    --output-dir "${BEAM_PLOTS_DIR}"
fi

echo "============================================================"
echo "[plot_diff_vectors_pca]"
echo "============================================================"
uv run python scripts/analysis/plot_diff_vectors_pca.py \
  --projections-dir "${DIFF_DIR}" \
  --output-dir "${PLOTS_DIR}/diff_pca"

echo "============================================================"
echo "[plot_oas_umap] (germline_family, j_gene, vgene_within IGHV3, shm_within IGHV3, cdr3_length)"
echo "============================================================"
if [[ "${SKIP_OAS}" == "1" ]]; then
  echo "[plot_oas_umap] SKIP_OAS=1 — skipping"
else
  OAS_PLOT_DIR="${PLOTS_DIR}/oas_umap"
  SHM_FAMILY="${SHM_FAMILY:-IGHV3}"
  VGENE_FAMILY="${VGENE_FAMILY:-IGHV3}"

  uv run python scripts/analysis/plot_oas_umap.py \
    "${OAS_NPZ[@]}" --output-dir "${OAS_PLOT_DIR}" --color-by germline_family

  uv run python scripts/analysis/plot_oas_umap.py \
    "${OAS_NPZ[@]}" --output-dir "${OAS_PLOT_DIR}" --color-by j_gene

  uv run python scripts/analysis/plot_oas_umap.py \
    "${OAS_NPZ[@]}" --output-dir "${OAS_PLOT_DIR}" \
    --color-by vgene_within_family --filter-family "${VGENE_FAMILY}"

  uv run python scripts/analysis/plot_oas_umap.py \
    "${OAS_NPZ[@]}" --output-dir "${OAS_PLOT_DIR}" \
    --color-by shm_within_family --filter-family "${SHM_FAMILY}"

  uv run python scripts/analysis/plot_oas_umap.py \
    "${OAS_NPZ[@]}" --output-dir "${OAS_PLOT_DIR}" --color-by cdr3_length
fi

echo "============================================================"
echo "[plot_pll_pca]"
echo "============================================================"
uv run python scripts/analysis/plot_pll_pca.py \
  --input "${PLL_DIR}/pll_pca.npz" \
  --output-dir "${PLOTS_DIR}/pll_pca"

echo "============================================================"
echo "[done] all outputs under ${PLOTS_DIR}"
echo "============================================================"
