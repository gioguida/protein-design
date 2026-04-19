#!/usr/bin/env python
"""Training entry point. Runs a pipeline selected via `+pipeline=<name>`.

Examples:
    python scripts/train.py +pipeline=evotuning run_name=my_run
    python scripts/train.py +pipeline=evotune_c05_ttt
    python scripts/train.py +pipeline=c05 pipeline.init_from=/path/to/best.pt
"""

import logging

import hydra
from omegaconf import DictConfig

from protein_design.evotuning.pipeline import run_pipeline


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
