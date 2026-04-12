#!/usr/bin/env python
"""Entry point for ESM2 evotuning training."""

import argparse
import logging

from protein_design.train import train
from protein_design.utils import load_config


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="ESM2 evotuning training")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--run-name", required=True, help="Name for this training run")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, args.run_name)


if __name__ == "__main__":
    main()
