#!/bin/bash
#SBATCH --job-name=test_esm
#SBATCH --output=slurm-outputs/%x-%j.out  
#SBATCH --error=slurm-outputs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=rtx_3090:1

# Make sure results folder exists
mkdir -p slurm-outputs

# common setup
source "${HOME}/protein-design/bash_scripts/common_setup.sh"
nvidia-smi
which python

# run test
python src/tests/test_model.py
