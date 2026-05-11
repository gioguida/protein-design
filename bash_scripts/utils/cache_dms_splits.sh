#!/bin/bash
# Materialize and cache DMS train/val/test splits for all datasets in the DMS config.
# Usage:
#   bash bash_scripts/utils/cache_dms_splits.sh [--config conf/data/dms/default.yaml] [--force]

set -euo pipefail
source bash_scripts/common_setup.sh

DMS_CONFIG="conf/data/dms/default.yaml"
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      DMS_CONFIG="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

echo "[cache] config: ${DMS_CONFIG}"
echo "[cache] force rebuild: ${FORCE}"

uv run python - <<PY
from pathlib import Path
from protein_design.dms_splitting import load_dms_config, ensure_dataset_splits

config_path = Path("${DMS_CONFIG}")
force = bool(${FORCE})
cfg = load_dms_config(config_path)

print(f"[cache] output_dir: {cfg.split.output_dir}")
for key in sorted(cfg.datasets.keys()):
    paths = ensure_dataset_splits(key, config_path=config_path, force=force)
    print(f"[cache] {key}:")
    for split_name in ("train", "val", "test"):
        print(f"  - {split_name}: {paths[split_name]}")
PY
