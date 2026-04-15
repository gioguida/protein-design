#!/usr/bin/env python
"""Entry point for ESM2 test-time training (TTT)."""

import argparse
import logging
from datetime import datetime

from protein_design.train_ttt import train_ttt
from protein_design.utils import load_config


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="ESM2 test-time training (TTT)")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--run-name", required=True, help="Name for this training run")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.run_name}_{timestamp}"

    config = load_config(args.config)
    train_ttt(config, run_name)


if __name__ == "__main__":
    main()
