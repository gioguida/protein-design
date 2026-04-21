#!/usr/bin/env python
"""Composable workflow runner for evotuning and DPO stages."""

import logging
from pathlib import Path
import sys
from typing import Any, List, Optional

import hydra
from hydra import compose as hydra_compose
from omegaconf import DictConfig, ListConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protein_design.dpo.train import run_dpo
from protein_design.evotuning.pipeline import run_pipeline


def _as_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, ListConfig)):
        return [str(v) for v in value]
    return [str(value)]


def _resolve_handoff_checkpoint(
    mode: str,
    configured_path: Optional[str],
    evotuning_checkpoint: Optional[Path],
) -> Optional[Path]:
    mode_lc = str(mode).strip().lower()
    if mode_lc not in {"best", "final", "path"}:
        raise ValueError(
            f"workflow.handoff.mode={mode!r} is invalid. Expected one of: best, final, path."
        )

    if mode_lc == "path":
        if configured_path is None:
            raise ValueError("workflow.handoff.path is required when workflow.handoff.mode='path'.")
        explicit = Path(configured_path).expanduser().resolve()
        if not explicit.exists():
            raise FileNotFoundError(f"workflow.handoff.path does not exist: {explicit}")
        return explicit

    if evotuning_checkpoint is None:
        return None

    target_name = "best.pt" if mode_lc == "best" else "final.pt"
    candidates = [
        evotuning_checkpoint,
        evotuning_checkpoint.with_name(target_name),
        evotuning_checkpoint.parent / target_name,
        evotuning_checkpoint.parent / "checkpoints" / target_name,
        evotuning_checkpoint.parent.parent / target_name,
        evotuning_checkpoint.parent.parent / "checkpoints" / target_name,
    ]

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.name == target_name and candidate.exists():
            return candidate

    if evotuning_checkpoint.exists():
        return evotuning_checkpoint

    raise FileNotFoundError(
        f"Could not resolve handoff checkpoint (mode={mode_lc}) from {evotuning_checkpoint}."
    )


@hydra.main(version_base=None, config_path="../conf", config_name="workflow")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("workflow")

    steps = [str(step).strip().lower() for step in cfg.workflow.steps]
    invalid_steps = [step for step in steps if step not in {"evotuning", "dpo"}]
    if invalid_steps:
        raise ValueError(f"Unsupported workflow step(s): {invalid_steps}")
    if not steps:
        raise ValueError("workflow.steps is empty.")

    evo_overrides = _as_string_list(cfg.workflow.get("evotuning_overrides"))
    dpo_overrides = _as_string_list(cfg.workflow.get("dpo_overrides"))

    evotuning_handoff: Optional[Path] = None
    if "evotuning" in steps:
        logger.info("Running workflow step: evotuning")
        evo_cfg = hydra_compose(config_name="evotuning/config", overrides=evo_overrides)
        if "selectors" in cfg and "evotuning" in cfg.selectors and "pipeline" in cfg.selectors.evotuning:
            evo_cfg.pipeline = OmegaConf.create(cfg.selectors.evotuning.pipeline)
        evotuning_handoff = run_pipeline(evo_cfg, cli_overrides=evo_overrides)
        logger.info("EvoTuning handoff checkpoint: %s", evotuning_handoff)

    if "dpo" in steps:
        logger.info("Running workflow step: dpo")
        dpo_cfg = hydra_compose(config_name="dpo")

        if "selectors" in cfg and "dpo" in cfg.selectors:
            for section in ("data", "model", "training", "logging", "checkpointing", "wandb"):
                if section in cfg.selectors.dpo:
                    dpo_cfg[section] = OmegaConf.create(cfg.selectors.dpo[section])

        if dpo_overrides:
            dpo_cfg = OmegaConf.merge(dpo_cfg, OmegaConf.from_dotlist(dpo_overrides))

        handoff_cfg = cfg.workflow.handoff
        handoff_checkpoint = _resolve_handoff_checkpoint(
            mode=str(handoff_cfg.mode),
            configured_path=None if handoff_cfg.path is None else str(handoff_cfg.path),
            evotuning_checkpoint=evotuning_handoff,
        )
        if handoff_checkpoint is not None:
            dpo_cfg.model.init.source = "checkpoint"
            dpo_cfg.model.init.checkpoint = str(handoff_checkpoint)
            logger.info("DPO model initialization checkpoint: %s", handoff_checkpoint)

        run_dpo(dpo_cfg)


if __name__ == "__main__":
    main()
