#!/bin/bash
#SBATCH --job-name=merge_oas_shards
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00
#SBATCH --output=bash_scripts/logs/merge_oas_shards_%j.log

set -euo pipefail
source bash_scripts/common_setup.sh

# Must match --array in filter_oas_shard.sbatch / --num-shards in make_oas_shards.py.
python scripts/data_prep/merge_oas_shards.py --num-shards 32
