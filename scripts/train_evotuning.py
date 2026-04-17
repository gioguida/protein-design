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
from protein_design.utils import (
    build_data_config,
    build_model_config,
    build_run_config,
    build_scoring_config,
    build_training_config,
    generate_run_name,
)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run_name = generate_run_name(cfg)
    train(
        model_cfg=build_model_config(cfg),
        data_cfg=build_data_config(cfg),
        training_cfg=build_training_config(cfg),
        scoring_cfg=build_scoring_config(cfg),
        run_cfg=build_run_config(cfg),
        run_name=run_name,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
