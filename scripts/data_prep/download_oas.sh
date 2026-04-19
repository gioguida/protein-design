#!/usr/bin/env bash
# Download OAS data units from a list of URLs.
# Usage: ./download_oas.sh <url_file>
set -euo pipefail

URL_FILE="${1:?Usage: download_oas.sh <url_file>}"
OUT_DIR="${SCRATCH_DIR:?Set SCRATCH_DIR env var}/oas_raw"

mkdir -p "$OUT_DIR"

total=0
ok=0
skipped=0
failed=0

while IFS= read -r url || [[ -n "$url" ]]; do
    # Skip empty lines and comments
    [[ -z "$url" || "$url" == \#* ]] && continue

    fname=$(basename "$url")
    total=$((total + 1))

    if [[ -f "$OUT_DIR/$fname" ]]; then
        echo "[SKIP] $fname (already exists)"
        skipped=$((skipped + 1))
        continue
    fi

    if wget -q --show-progress -O "$OUT_DIR/$fname" "$url"; then
        echo "[OK]   $fname"
        ok=$((ok + 1))
    else
        echo "[FAIL] $fname"
        rm -f "$OUT_DIR/$fname"
        failed=$((failed + 1))
    fi
done < "$URL_FILE"

echo ""
echo "Done. Total: $total | Downloaded: $ok | Skipped: $skipped | Failed: $failed"
