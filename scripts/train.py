#!/usr/bin/env python
"""Training entry point. Runs one selected task.

Examples:
    python scripts/train.py task=evotuning run_name=my_run
    python scripts/train.py task=evotuning_c05 model.init.source=checkpoint model.init.checkpoint=/path/to/best.pt
    python scripts/train.py task=dpo
"""

import logging

import hydra
from omegaconf import DictConfig

from protein_design.train_dispatch import run_selected_task


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_selected_task(cfg)


if __name__ == "__main__":
    main()
