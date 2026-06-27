#!/bin/bash
#SBATCH --job-name=score_pll
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --time=1:00:00
#SBATCH --output=bash_scripts/logs/score_pll_%j.log

# Compute a model's CDR-H3 PLL for an arbitrary CSV of generated sequences.
# Reused for both the DPO and the PSSM generated sets (forwarding all flags) so
# the PSSM-vs-DPO notebook can correlate the DPO model's own likelihood with the
# M22 and UQ scorers.
#
# Usage:
#   sbatch bash_scripts/utils/score_generated_with_pll.sbatch \
#       --input-csv  /path/to/generated.csv \
#       --output-csv /path/to/pll.csv \
#       --checkpoint /cluster/project/infk/krause/gguidarini/protein-design/checkpoints/just_dpo_650M/step_8250.pt
#   (optional: --seq-col cdrh3 --base-model facebook/esm2_t33_650M_UR50D --batch-size 256 --force)

set -euo pipefail
source bash_scripts/common_setup.sh

python scripts/analysis/score_generated_with_pll.py "$@"
