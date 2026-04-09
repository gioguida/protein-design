#!/bin/bash
#SBATCH --job-name=test_model
#SBATCH --output=slurm-outputs/%x-%j.out  
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=rtx_3090:1

# Make sure results folder exists
mkdir -p slurm-outputs

# common setup
source "${HOME}/protein-design/bash_scripts/common_setup.sh"
cd "${DPO_PROJECT_ROOT}"
nvidia-smi
which python

# run test
if [ "${DPO_USE_UV:-0}" = "1" ]; then
    if ! command -v uv >/dev/null 2>&1; then
        echo "ERROR: DPO_USE_UV=1 but uv is not available in PATH."
        exit 1
    fi
    uv run python -m src.tests.test_model
else
    python -m src.tests.test_model
fi
