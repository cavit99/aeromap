from __future__ import annotations

import tomllib
from pathlib import Path


def test_mypy_does_not_ignore_missing_imports_globally() -> None:
    config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    mypy_config = config["tool"]["mypy"]
    assert mypy_config.get("ignore_missing_imports") is not True
    overrides = mypy_config.get("overrides", [])
    assert overrides
