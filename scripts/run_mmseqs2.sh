#!/usr/bin/env bash
# Run MMseqs2 easy-linclust to deduplicate filtered OAS sequences at 99% identity.
# Usage: ./run_mmseqs2.sh
set -euo pipefail

INPUT="${SCRATCH_DIR:?Set SCRATCH_DIR env var}/oas_filtered.fasta"
OUT_PREFIX="${SCRATCH_DIR}/oas_dedup"
TMP_DIR="${SCRATCH_DIR}/mmseqs_tmp"

if [[ ! -f "$INPUT" ]]; then
    echo "Error: Input file not found: $INPUT"
    exit 1
fi

before=$(grep -c "^>" "$INPUT")
echo "Input sequences: $before"

mkdir -p "$TMP_DIR"

mmseqs easy-linclust \
    "$INPUT" \
    "$OUT_PREFIX" \
    "$TMP_DIR" \
    --min-seq-id 0.99 \
    --cov-mode 0 \
    -c 0.9 \
    --threads 16

after=$(grep -c "^>" "${OUT_PREFIX}_rep_seq.fasta")
echo ""
echo "Before deduplication: $before"
echo "After deduplication:  $after"
echo "Reduction: $(echo "scale=1; (1 - $after / $before) * 100" | bc)%"
echo "Output: ${OUT_PREFIX}_rep_seq.fasta"
