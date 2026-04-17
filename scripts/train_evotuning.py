#!/usr/bin/env python
"""Entry point for ESM2 evotuning training.

Usage:
    python scripts/train_evotuning.py task=evotuning data=oas_full run_name=my_run
    python scripts/train_evotuning.py task=evotuning_c05 data=c05_5k finetune=/path/to/best.pt
"""

import logging

import hydra
from omegaconf import DictConfig

from protein_design.evotuning.train import train
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
    train(config, run_name)


if __name__ == "__main__":
    main()
