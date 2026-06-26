#!/bin/bash
#SBATCH --job-name=score_generated
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=1:00:00
#SBATCH --output=bash_scripts/logs/score_generated_%j.log

# Score an arbitrary CSV of CDR-H3 sequences with the M22 scorer. Reused for both
# the DPO and PSSM generated sets — all flags after the script are forwarded.
#
# Usage:
#   sbatch bash_scripts/utils/score_generated_with_esme.sbatch \
#       --input-csv  /path/to/generated.csv \
#       --output-csv /path/to/scores.csv
#   (optional: --seq-col cdrh3 --scorer m22 --batch-size 32 --force)
#
# Uses the `flu` conda env (esme + flash_attn) so the scorer forward pass is
# bit-identical to training time. Ampere (rtx_3090/A100) required by flash_attn.

set -euo pipefail
cd /cluster/home/${USER}/protein-design

module load eth_proxy
export no_proxy="${no_proxy//api.wandb.ai,/}"
export NO_PROXY="${no_proxy}"

mkdir -p bash_scripts/logs

CONDA_PY=/cluster/project/infk/krause/mdenegri/miniconda3/envs/flu/bin/python
$CONDA_PY scripts/analysis/score_sequences_with_esme.py "$@"
