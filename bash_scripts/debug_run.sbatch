#!/bin/bash
#SBATCH --job-name=evo_debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1:00:00
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --output=logs/debug_%j.out
#SBATCH --error=logs/debug_%j.err

set -euo pipefail
cd /cluster/home/mdenegri/protein-design
set -a; source .env; set +a

mkdir -p logs

# Use a separate scratch dir so debug doesn't interfere with the main pipeline
DEBUG_SCRATCH="${SCRATCH_DIR}/debug"
mkdir -p "$DEBUG_SCRATCH"

# ── Step 1: Sample 10k sequences from the filtered FASTA ─────────────────────
echo "=== Sampling 10k sequences ==="
awk '/^>/{n++} n>10000{exit} {print}' \
    "${SCRATCH_DIR}/oas_filtered.fasta" \
    > "${DEBUG_SCRATCH}/oas_filtered.fasta"
echo "Sampled $(grep -c '^>' ${DEBUG_SCRATCH}/oas_filtered.fasta) sequences"

# ── Step 2: Deduplicate with MMseqs2 ─────────────────────────────────────────
echo "=== Running MMseqs2 ==="
module load eth_proxy
module load stack/2024-06 gcc/12.2.0 mmseqs2/14-7e284
mmseqs easy-linclust \
    "${DEBUG_SCRATCH}/oas_filtered.fasta" \
    "${DEBUG_SCRATCH}/oas_dedup" \
    "${DEBUG_SCRATCH}/mmseqs_tmp" \
    --min-seq-id 0.99 \
    --cov-mode 0 \
    -c 0.9 \
    --threads 4
echo "After dedup: $(grep -c '^>' ${DEBUG_SCRATCH}/oas_dedup_rep_seq.fasta) sequences"

# ── Step 3: Train ─────────────────────────────────────────────────────────────
echo "=== Starting debug training ==="
source .venv/bin/activate
SCRATCH_DIR="$DEBUG_SCRATCH" python scripts/train_evotuning.py \
    --config configs/evotuning_debug.yaml

echo "=== Debug run complete ==="
