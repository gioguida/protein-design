#!/bin/bash
# Common setup script for flu experiments

# Load environment variables
if [ -f "${HOME}/protein-design/.env.local" ]; then
    source "${HOME}/protein-design/.env.local"
else
    echo "ERROR: ${HOME}/protein-design/.env.local not found."
    echo "Please copy .env.template to .env.local and customize it:"
    echo "  cp ${HOME}/protein-design/.env.template ${HOME}/protein-design/.env.local"
    exit 1
fi

# Route run artifacts to scratch by default.
export DPO_OUTPUT_DIR="${DPO_OUTPUT_DIR:-${DPO_SCRATCH}/outputs}"
export DPO_BEST_MODEL_DIR="${DPO_BEST_MODEL_DIR:-${DPO_SCRATCH}/dpo_best_models}"
export DPO_LAST_MODEL_DIR="${DPO_LAST_MODEL_DIR:-${DPO_SCRATCH}/dpo_last_models}"

# Keep W&B caches/artifacts off home.
export DPO_WANDB_DIR="${DPO_WANDB_DIR:-${DPO_SCRATCH}/wandb}"
export DPO_WANDB_CACHE_DIR="${DPO_WANDB_CACHE_DIR:-${DPO_SCRATCH}/wandb-cache}"
export DPO_WANDB_DATA_DIR="${DPO_WANDB_DATA_DIR:-${DPO_SCRATCH}/wandb-data}"
export WANDB_DIR="${DPO_WANDB_DIR}"
export WANDB_CACHE_DIR="${DPO_WANDB_CACHE_DIR}"
export WANDB_DATA_DIR="${DPO_WANDB_DATA_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${DPO_OUTPUT_DIR}" "${DPO_BEST_MODEL_DIR}" "${DPO_LAST_MODEL_DIR}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_DATA_DIR}"

# Load required modules
# Some stacks hide specific Python modules; if unavailable, rely on conda Python.
if [ -n "${DPO_PYTHON_MODULE:-}" ]; then
    if module load "${DPO_PYTHON_MODULE}" >/dev/null 2>&1; then
        echo "Loaded optional Python module: ${DPO_PYTHON_MODULE}"
    else
        echo "WARNING: Could not load optional Python module: ${DPO_PYTHON_MODULE}"
        echo "Continuing without explicit python module; conda env Python will be used."
    fi
fi

module load eth_proxy
module load "${DPO_STACK_MODULE}" "${DPO_GCC_MODULE}"
module load "${DPO_CUDA_MODULE}"

# Activate conda environment
source "${DPO_CONDA_BASE}/bin/activate" "${DPO_CONDA_ENV}"

echo "DPO environment loaded for user: ${DPO_USER}"
echo "DPO output dir: ${DPO_OUTPUT_DIR}"
echo "DPO best-model export dir: ${DPO_BEST_MODEL_DIR}"
echo "DPO last-model export dir: ${DPO_LAST_MODEL_DIR}"
echo "W&B dir: ${WANDB_DIR}"
echo "PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF}"
