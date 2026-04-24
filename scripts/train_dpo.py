#!/usr/bin/env python
"""DPO training entrypoint."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protein_design.dpo.train import main


if __name__ == "__main__":
    if all(not arg.startswith("task=") for arg in sys.argv[1:]):
        sys.argv.insert(1, "task=dpo")
    main()
