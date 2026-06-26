#!/bin/bash
#SBATCH --job-name=pssm_sweep_dpo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00
#SBATCH --output=bash_scripts/logs/pssm_sweep_dpo_%j.log

# CPU-only PSSM baseline temperature sweep for the PSSM-vs-DPO-650M comparison.
# pssm_sampling.py / build_pssm_counts are pure numpy — no GPU needed. The T=0.9
# run (n_sequences=5000, matching the beam-search comparison set) doubles as the
# headline PSSM baseline.

set -euo pipefail
source bash_scripts/common_setup.sh

python scripts/analysis/run_pssm_baseline_sweep.py \
  --config conf/analysis/pssm_baseline_sweep_dpo_comparison.yaml
