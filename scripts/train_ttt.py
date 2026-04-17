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
    train_ttt(
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
