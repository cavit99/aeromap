"""OpenFOAM runner helpers."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def docker_available() -> tuple[bool, str]:
    docker = shutil.which("docker")
    if docker is None:
        return False, "docker CLI is not installed"
    result = subprocess.run(  # noqa: S603
        [docker, "info", "--format", "{{.ServerVersion}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False, result.stderr.strip() or result.stdout.strip()
    return True, result.stdout.strip()


def compose_run_command(case_dir: Path) -> list[str]:
    docker = shutil.which("docker") or "docker"
    project_root = Path.cwd().resolve()
    try:
        run_script = (case_dir.resolve() / "run_openfoam.sh").relative_to(project_root)
    except ValueError as exc:
        message = f"CFD case directory must be inside project root for Docker Compose: {case_dir}"
        raise RuntimeError(message) from exc
    return [
        docker,
        "compose",
        "run",
        "--rm",
        "cfd",
        run_script.as_posix(),
    ]


def run_case(case_dir: Path, *, dry_run: bool = False) -> int:
    command = compose_run_command(case_dir)
    if dry_run:
        print(" ".join(command))
        return 0

    available, reason = docker_available()
    if not available:
        message = f"Docker/OpenFOAM is unavailable: {reason}"
        raise RuntimeError(message)

    return subprocess.run(command, check=False).returncode  # noqa: S603
