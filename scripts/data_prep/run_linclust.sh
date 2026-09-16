#!/usr/bin/env bash
# Run MMseqs2 easy-linclust to deduplicate filtered OAS sequences at 95% identity
# (single-stage, following Talaei, Walker et al. 2025 Methods §4.1.1 — see
# report/evotuning.md "OAS filtering, download, deduplication, and splits").
# Usage: ./run_linclust.sh
set -euo pipefail

: "${SCRATCH_DIR:?Set SCRATCH_DIR in .env (see .env.template)}"
INPUT="${SCRATCH_DIR}/oas_filtered.fasta"
OUT_PREFIX="${SCRATCH_DIR}/oas_dedup"
# Node-local disk (requested via sbatch --tmp=) instead of Lustre: rescorediagonal's
# random-access pattern is I/O-bound on the shared filesystem. Falls back to
# SCRATCH_DIR only when run outside SLURM (no $TMPDIR).
TMP_DIR="${TMPDIR:-${SCRATCH_DIR}/mmseqs_tmp}/mmseqs_tmp"
THREADS="${SLURM_CPUS_PER_TASK:-32}"

if [[ ! -f "$INPUT" ]]; then
    echo "Error: Input file not found: $INPUT"
    exit 1
fi

before=$(grep -c "^>" "$INPUT")
echo "Input sequences: $before"
echo "TMP_DIR: $TMP_DIR (threads: $THREADS)"

mkdir -p "$TMP_DIR"

stdbuf -oL -eL mmseqs easy-linclust \
    "$INPUT" \
    "$OUT_PREFIX" \
    "$TMP_DIR" \
    --min-seq-id 0.95 \
    --cov-mode 0 \
    -c 0.9 \
    --db-load-mode 2 \
    --threads "$THREADS"

after=$(grep -c "^>" "${OUT_PREFIX}_rep_seq.fasta")
echo ""
echo "Before deduplication: $before"
echo "After deduplication:  $after"
echo "Reduction: $(echo "scale=1; (1 - $after / $before) * 100" | bc)%"
echo "Output: ${OUT_PREFIX}_rep_seq.fasta"
