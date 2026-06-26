#!/bin/bash
#SBATCH --job-name=beam_dpo650m
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:32g
#SBATCH --time=02:00:00
#SBATCH --output=bash_scripts/logs/beam_dpo650m_%j.log

# Stochastic beam search from Giovanni's DPO 650M checkpoint (base ESM2-650M),
# for the PSSM-vs-DPO comparison
# (report/meetings/pssm_vs_dpo_650m_comparison.ipynb).
#
# Each beam step changes exactly one CDR-H3 position, so --n-steps is also the
# cap on how many mutations from WT the beam can reach. We run --n-steps 11 (up
# to 11 mutations) so the DPO library spans the same edit range as the PSSM
# baseline. Because the run is seeded (--seed 42), steps 0-5 reproduce the
# previous --n-steps 5 run exactly and 6-11 extend it (so this rerun is
# effectively a continuation).
#
# NOTE on resources: per-step cost is roughly constant (beam capped at 500), so
# 11 steps ~ 2.5x the heavy compute of the old 5-step run; budget 02:00:00.
# Peak GPU memory does not grow with --n-steps, so gpumem:32g is unchanged. If
# the job OOMs bump --gres=gpumem (e.g. 40g/80g); if it times out bump --time.

set -euo pipefail
source bash_scripts/common_setup.sh

DPO_CKPT=/cluster/project/infk/krause/gguidarini/protein-design/checkpoints/just_dpo_650M/step_8250.pt
BEAM_CSV=/cluster/scratch/mdenegri/protein-design/pssm_vs_dpo_650m/beam_dpo/beam_dpo_650m_step8250.csv

mkdir -p "$(dirname "${BEAM_CSV}")"

echo "[beam] DPO 650M step 8250"
python scripts/stochastic_beam_search.py \
  --model-variant just_dpo_650m_step8250 \
  --checkpoint-path "${DPO_CKPT}" \
  --beam-size 500 \
  --n-steps 11 \
  --snapshot-every 1 \
  --temperature 0.9 \
  --seed 42 \
  --start-mode wt \
  --output-path "${BEAM_CSV}"
