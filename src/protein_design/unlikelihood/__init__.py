"""Unlikelihood training pipeline for ESM-2 fine-tuning."""


def run_unlikelihood(cfg):
    from .train import run_unlikelihood as _run_unlikelihood

    return _run_unlikelihood(cfg)


__all__ = ["run_unlikelihood"]
