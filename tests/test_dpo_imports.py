"""Smoke tests for DPO package imports."""

import importlib


def test_dpo_package_imports() -> None:
    import protein_design.dpo.data_processing  # noqa: F401
    import protein_design.dpo.dataset  # noqa: F401

    for module_name in ("protein_design.dpo.loss", "protein_design.dpo.train"):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            # Some environments run lightweight tests without the full Hydra stack.
            assert exc.name in {"omegaconf", "hydra", "wandb"}
