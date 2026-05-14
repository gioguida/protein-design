"""Task-based training dispatch.

Routes the composed Hydra config to either:
- evotuning/TTT runner (`run_stage`)
- DPO runner (`run_dpo`)
- unlikelihood runner (`run_unlikelihood`)
"""

from __future__ import annotations

import logging
from pathlib import Path

from omegaconf import DictConfig

from protein_design.config import (
    build_model_config,
    build_run_config,
    build_scoring_config,
    generate_run_name,
)
from protein_design.dpo.train import run_dpo
from protein_design.evotuning.config import build_data_config, build_training_config
from protein_design.evotuning.train import run_stage
from protein_design.lora_dpo.train import run_lora_dpo
from protein_design.unlikelihood.train import run_unlikelihood

logger = logging.getLogger(__name__)


def _task_name(cfg: DictConfig) -> str:
    if "task" in cfg and cfg.task is not None and cfg.task.get("name") is not None:
        return str(cfg.task.name).strip().lower()
    return ""


def _task_runner(cfg: DictConfig) -> str:
    if "task" in cfg and cfg.task is not None and cfg.task.get("runner") is not None:
        return str(cfg.task.runner).strip().lower()
    name = _task_name(cfg)
    if name.startswith("dpo"):
        return "dpo"
    return "evotuning"


def _stage_type(cfg: DictConfig) -> str:
    if "task" in cfg and cfg.task is not None and cfg.task.get("stage_type") is not None:
        return str(cfg.task.stage_type).strip().lower()
    return "ttt" if _task_name(cfg) == "ttt" else "evotuning"


def run_selected_task(cfg: DictConfig) -> Path:
    runner = _task_runner(cfg)
    if runner == "dpo":
        logger.info("Dispatching to DPO runner (task=%s)", _task_name(cfg) or "dpo")
        return run_dpo(cfg)
    if runner == "lora_dpo":
        logger.info(
            "Dispatching to LoRA-DPO runner (task=%s)",
            _task_name(cfg) or "lora_dpo",
        )
        return run_lora_dpo(cfg)
    if runner == "unlikelihood":
        logger.info(
            "Dispatching to unlikelihood runner (task=%s)",
            _task_name(cfg) or "unlikelihood",
        )
        return run_unlikelihood(cfg)

    stage_type = _stage_type(cfg)
    run_name = generate_run_name(cfg)
    logger.info(
        "Dispatching to evotuning runner (task=%s, stage_type=%s)",
        _task_name(cfg) or "evotuning",
        stage_type,
    )
    return run_stage(
        stage_type=stage_type,
        model_cfg=build_model_config(cfg),
        data_cfg=build_data_config(cfg),
        training_cfg=build_training_config(cfg),
        scoring_cfg=build_scoring_config(cfg),
        run_cfg=build_run_config(cfg),
        run_name=run_name,
        cfg=cfg,
    )
