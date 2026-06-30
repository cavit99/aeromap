from __future__ import annotations

from pathlib import Path

import pytest

from aeromap.cfd.runner import compose_run_command


def test_compose_run_command_uses_supplied_project_relative_case_dir() -> None:
    case_dir = Path.cwd() / "tmp_cases" / "case_demo"

    command = compose_run_command(case_dir)

    assert command[-1] == "tmp_cases/case_demo/run_openfoam.sh"


def test_compose_run_command_rejects_case_dir_outside_project(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="inside project root"):
        compose_run_command(tmp_path / "case_demo")
