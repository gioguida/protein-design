# ── Common setup sourced by every SBATCH script ──────────────────────
# Usage: add  `source bash_scripts/common_setup.sh`  after #SBATCH directives.

set -euo pipefail
cd /cluster/home/mdenegri/protein-design

# Networking — eth_proxy provides outbound access on compute nodes.
module load eth_proxy
# api.wandb.ai was added to no_proxy in a recent eth_proxy update, but compute
# nodes have no direct internet — route wandb through the proxy instead.
export no_proxy="${no_proxy//api.wandb.ai,/}"
export NO_PROXY="${no_proxy}"

# Environment variables (.env.local is a local copy of .env.template)
set -a; source .env.local; set +a

# Python
source .venv/bin/activate

# Logs directory (SBATCH --output/--error point here)
mkdir -p bash_scripts/logs
