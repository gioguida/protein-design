#!/bin/bash
#SBATCH --job-name=random_sweep_dpo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00
#SBATCH --output=bash_scripts/logs/random_sweep_dpo_%j.log

# CPU-only random library-mutant baseline trust-radius sweep for the
# random-vs-DPO-650M comparison. random_sampling.py is pure numpy -- no GPU
# needed. The radius=11 run (n_sequences=5000, matching the beam-search
# comparison set) doubles as the headline random baseline.

set -euo pipefail
source bash_scripts/common_setup.sh

python scripts/analysis/run_random_baseline_sweep.py \
  --config conf/analysis/random_baseline_sweep_dpo_comparison.yaml
