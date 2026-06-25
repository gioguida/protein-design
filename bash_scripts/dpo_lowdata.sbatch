#!/bin/bash
#SBATCH --job-name=dpo_lowdata
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --time=12:00:00
#SBATCH --output=bash_scripts/logs/dpo_lowdata_%j.log

# One low-data DPO run (a single point of the evo-vs-vanilla learning curve).
# This is a THIN wrapper: every Hydra override is forwarded from the launcher
# (bash_scripts/dpo_lowdata_sweep.sh), which submits one of these jobs per
# (model, n_train, seed) combination. Run it directly only for a smoke test:
#
#   sbatch bash_scripts/dpo_lowdata.sbatch \
#     task=dpo model=esm2_650m \
#     model.init.source=checkpoint model.init.checkpoint=<evo_650m.pt> \
#     data.low_data.enabled=true data.low_data.n_train=200 data.low_data.seed=0 \
#     data.pair_split.enforce_train_controlled_sizes=false \
#     run.base_name=lowdata_evo_650m_n200_s0
#
# The run writes summary.json (test_spearman_avg on ed5) to $TRAIN_DIR/<base_name>_<ts>/.

source bash_scripts/common_setup.sh

# DPO holds policy + frozen reference (2x 650M) in GPU memory; on a 24G card it
# sits right at the ceiling. expandable_segments curbs allocator fragmentation so
# the marginal 20MiB OOM at the start of training doesn't recur.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use scripts/train.py (the @hydra.main entrypoint defined in __main__, dispatches
# via train_dispatch -> run_dpo / run_lora_dpo). scripts/train_dpo.py imports the
# hydra main from a submodule, so Hydra resolves config_path module-relative and
# cannot find conf/. The task (dpo | lora_dpo) is passed by the launcher in "$@".
python scripts/train.py "$@"
