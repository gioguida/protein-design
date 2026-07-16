#!/bin/bash
#SBATCH --job-name=build_cdr_windows
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=40G
#SBATCH --time=12:00:00
#SBATCH --output=bash_scripts/logs/build_cdr_windows_%j.log

# Usage:
#   sbatch bash_scripts/utils/build_cdr_windows.sbatch [build_cdr_windows.py args...]
#
# Builds the CDR-H3 +/- flank window cache the "cdr" / "cdr_mix" / "hybrid"
# masking modes need, for the full (OAS-scale) deduplicated corpus. Previously
# only ever run ad hoc for the much smaller C05-scale corpus; at OAS scale
# (~140M sequences) this is too heavy for the login node (streams the full
# oas_filtered.csv.gz to resolve cdr3_aa per seq_id, and holds the wanted/found
# seq_id sets in memory for the whole run — sized generously above for that).
#
# Example (default --fasta/--csv/--flank baked into build_cdr_windows.py match
# the promoted OAS corpus and the masking-ablation configs' default flank=3):
#   sbatch bash_scripts/utils/build_cdr_windows.sbatch \
#     --fasta ${PROJECT_DIR}/data/oas/oas_dedup_rep_seq.fasta --flank 3

set -euo pipefail
source bash_scripts/common_setup.sh

python scripts/data_prep/build_cdr_windows.py "$@"
