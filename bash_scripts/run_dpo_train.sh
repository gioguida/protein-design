#!/bin/bash
#SBATCH --job-name=train_dpo
#SBATCH --output=/cluster/home/%u/protein-design/slurm-outputs/%x-%j.out
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=rtx_4090:1	# tesla_a100:1 | tesla_a100_80gb:1 | rtx_pro_6000:1

set -euo pipefail

mkdir -p slurm-outputs

# Shared environment setup (modules + conda env)
source "${HOME}/protein-design/bash_scripts/common_setup.sh"

cd "${DPO_PROJECT_ROOT}"

echo "Running on host: $(hostname)"
which python
nvidia-smi || true

# Pass any Hydra overrides from sbatch command line.
# Example:
#   sbatch bash_scripts/run_dpo_train.sh training.num_epochs=100 training.beta=0.2
python -m src.train_dpo \
	output_dir="${DPO_OUTPUT_DIR}" \
	checkpointing.best_export_dir="${DPO_BEST_MODEL_DIR}" \
	checkpointing.last_export_dir="${DPO_LAST_MODEL_DIR}" \
	"$@"
