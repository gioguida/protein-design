#!/usr/bin/env python
"""Entry point for ESM2 test-time training (TTT).

Usage:
    python scripts/train_ttt.py task=ttt data=c05_single finetune=/path/to/best.pt
    python scripts/train_ttt.py task=ttt data=c05_single run_name=my_ttt
"""

import logging

import hydra
from omegaconf import DictConfig

from protein_design.evotuning.train_ttt import train_ttt
from protein_design.utils import flatten_config, generate_run_name


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run_name = generate_run_name(cfg)
    config = flatten_config(cfg)
    train_ttt(config, run_name)


if __name__ == "__main__":
    main()
