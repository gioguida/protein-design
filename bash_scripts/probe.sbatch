#!/bin/bash
#SBATCH --job-name=probe
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=12G
#SBATCH --time=4:00:00
#SBATCH --output=bash_scripts/logs/probe_%j.log

# Embedding linear-probe analysis on a compute node (NOT the login node).
# The emb artifacts are up to ~2 GB each (343k x 1280) and the probe fits over
# the full train split — this OOMs / hangs the shared login node. Run here.
#
# Computes the comparison table + figures and writes them to report/figures/:
#   full  -> probe_scores.csv, probe_spearman.pdf, embedding_pca.pdf
#   curve -> learning_curve.csv, learning_curve.pdf   (small-data regime, DPO base)
#
# Usage:
#   sbatch bash_scripts/probe.sbatch                 # full-data table + figures
#   sbatch bash_scripts/probe.sbatch --mode curve    # small-data learning curve
#   sbatch bash_scripts/probe.sbatch --mode both     # everything
#   sbatch bash_scripts/probe.sbatch --models vanilla_650m,evo_c05_cdrmix --no-pca
# Extra flags are forwarded to scripts/analysis/run_probe.py unchanged.

source bash_scripts/common_setup.sh

# Keep BLAS/sklearn within the allocated cpus (don't oversubscribe the node).
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

python scripts/analysis/run_probe.py "$@"
