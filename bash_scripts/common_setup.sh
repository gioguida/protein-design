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

# Load required modules
# Some stacks hide specific Python modules; if unavailable, rely on conda Python.
if [ -n "${DPO_PYTHON_MODULE}" ]; then
    if module spider "${DPO_PYTHON_MODULE}" >/dev/null 2>&1; then
        module load "${DPO_PYTHON_MODULE}"
    else
        echo "WARNING: ${DPO_PYTHON_MODULE} is not available in this module stack."
        echo "Continuing without explicit python module; conda env Python will be used."
    fi
fi

module load eth_proxy
module load "${DPO_STACK_MODULE}" "${DPO_GCC_MODULE}"
module load "${DPO_CUDA_MODULE}"

# Activate conda environment
source ${DPO_CONDA_BASE}/bin/activate ${DPO_CONDA_ENV}

echo "DPO environment loaded for user: ${DPO_USER}"
