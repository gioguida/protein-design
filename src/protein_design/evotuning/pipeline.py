"""Sequential training pipeline.

A pipeline config (conf/evotuning/pipeline/*.yaml) lists stages; each stage
names a task + data group. `run_pipeline` iterates stages, re-materializing
the per-stage config via Hydra compose, and threads the output checkpoint of
stage *n* as the `finetune` input to stage *n+1*.
"""

import logging
from pathlib import Path
from typing import List, Optional

from hydra import compose as hydra_compose
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from protein_design.config import (
    build_model_config,
    build_run_config,
    build_scoring_config,
    generate_run_name,
)
from protein_design.evotuning.config import build_data_config, build_training_config
from protein_design.evotuning.train import run_stage

logger = logging.getLogger(__name__)


def _infer_stage_type(stage: DictConfig) -> str:
    if "type" in stage and stage.type is not None:
        return str(stage.type)
    task_name = str(stage.task)
    if task_name == "ttt":
        return "ttt"
    return "evotuning"


def _filter_cli_overrides(overrides: List[str]) -> List[str]:
    """Drop pipeline-selector overrides — those are consumed by the runner
    and should not be re-applied per stage."""
    out = []
    for o in overrides:
        key = o.lstrip("+~").split("=", 1)[0]
        if (
            key == "pipeline"
            or key.startswith("pipeline.")
            or key == "evotuning/pipeline"
            or key.startswith("evotuning/pipeline.")
        ):
            continue
        out.append(o)
    return out


def compose_stage(
    stage: DictConfig,
    init_ckpt: Optional[str],
    cli_overrides: List[str],
) -> DictConfig:
    """Materialize a per-stage config by calling Hydra compose with the stage's
    task + data groups, then re-applying the user's CLI overrides on top so
    global tweaks like `training.max_steps=5` work end-to-end.

    Merge order: stage defaults (task/data yaml) → CLI overrides → stage.overrides.
    """
    base_overrides = [f"evotuning/task={stage.task}", f"evotuning/data={stage.data}"]
    cfg = hydra_compose(config_name="config", overrides=base_overrides + cli_overrides)

    stage_ov = stage.get("overrides")
    if stage_ov:
        cfg = OmegaConf.merge(cfg, stage_ov)

    cfg.finetune = str(init_ckpt) if init_ckpt else None
    cfg.pipeline = None
    return cfg


def run_pipeline(
    cfg: DictConfig,
    cli_overrides: Optional[List[str]] = None,
) -> Optional[Path]:
    """Execute all stages declared in cfg.pipeline sequentially.

    Returns the final stage handoff checkpoint path when available.
    """
    if cfg.get("pipeline") is None:
        raise ValueError(
            "No pipeline selected. Pass `+evotuning/pipeline=<name>` on the CLI "
            "(see conf/evotuning/pipeline/ for available pipelines)."
        )

    stages = cfg.pipeline.stages
    if not stages:
        raise ValueError("Pipeline has no stages.")

    if cli_overrides is None:
        try:
            original_overrides = list(HydraConfig.get().overrides.task)
        except ValueError:
            original_overrides = []
        effective_cli_overrides = _filter_cli_overrides(original_overrides)
    else:
        effective_cli_overrides = _filter_cli_overrides(list(cli_overrides))

    init_ckpt: Optional[str] = cfg.pipeline.get("init_from")
    pipeline_base_name = cfg.run_name or "pipeline"

    logger.info(
        "Starting pipeline with %d stage(s). Seed checkpoint: %s",
        len(stages), init_ckpt or "(base ESM2)",
    )
    if effective_cli_overrides:
        logger.info("CLI overrides re-applied to each stage: %s", effective_cli_overrides)

    for i, stage in enumerate(stages, start=1):
        stage_type = _infer_stage_type(stage)
        logger.info(
            "=== Stage %d/%d: %s (type=%s, task=%s, data=%s) ===",
            i, len(stages), stage.name, stage_type, stage.task, stage.data,
        )

        stage_cfg = compose_stage(stage, init_ckpt, effective_cli_overrides)
        stage_cfg.run_name = f"{pipeline_base_name}__{stage.name}"
        run_name = generate_run_name(stage_cfg)

        ckpt = run_stage(
            stage_type=stage_type,
            model_cfg=build_model_config(stage_cfg),
            data_cfg=build_data_config(stage_cfg),
            training_cfg=build_training_config(stage_cfg),
            scoring_cfg=build_scoring_config(stage_cfg),
            run_cfg=build_run_config(stage_cfg),
            run_name=run_name,
            cfg=stage_cfg,
        )

        init_ckpt = str(ckpt)
        logger.info("Stage %d complete. Handoff checkpoint: %s", i, init_ckpt)

    logger.info("Pipeline complete. Final checkpoint: %s", init_ckpt)
    return Path(init_ckpt) if init_ckpt else None
