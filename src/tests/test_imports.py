"""Smoke tests for package-mode imports."""


def test_src_package_imports() -> None:
    import src.dataset  # noqa: F401
    import src.loss  # noqa: F401
    import src.train_dpo  # noqa: F401
