#!/usr/bin/env bash
# One-time migration: restructure $PROJECT_DIR from flat layout to organized subdirectories.
# Usage: bash scripts/migrate_project_dir.sh
set -euo pipefail

if [ -f .env ]; then
    set -a; source .env; set +a
fi

: "${PROJECT_DIR:?Set PROJECT_DIR in .env (see .env.template)}"

echo "Restructuring: $PROJECT_DIR"
echo ""

# Create target directories
mkdir -p "$PROJECT_DIR/datasets/scoring"
mkdir -p "$PROJECT_DIR/checkpoints"
mkdir -p "$PROJECT_DIR/reports"

# Move training datasets
for f in oas_dedup_rep_seq.fasta oas_filtered.fasta oas_filtered.csv.gz; do
    src="$PROJECT_DIR/$f"
    dst="$PROJECT_DIR/datasets/$f"
    if [ -f "$src" ] && [ ! -f "$dst" ]; then
        echo "  mv $f -> datasets/$f"
        mv "$src" "$dst"
    fi
done

# Move scoring datasets
for f in D2_M22.csv D2_SI06.csv D2_exp.csv; do
    src="$PROJECT_DIR/$f"
    dst="$PROJECT_DIR/datasets/scoring/$f"
    if [ -f "$src" ] && [ ! -f "$dst" ]; then
        echo "  mv $f -> datasets/scoring/$f"
        mv "$src" "$dst"
    fi
done

# Move reports
for f in search_report.txt cdrh3_length_distribution.png; do
    src="$PROJECT_DIR/$f"
    dst="$PROJECT_DIR/reports/$f"
    if [ -f "$src" ] && [ ! -f "$dst" ]; then
        echo "  mv $f -> reports/$f"
        mv "$src" "$dst"
    fi
done

echo ""
echo "Done. New layout:"
find "$PROJECT_DIR" -maxdepth 3 -not -path '*/\.*' | head -30
