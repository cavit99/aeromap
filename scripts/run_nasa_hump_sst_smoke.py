# ruff: noqa: S603,S607
"""Run the local NASA/TMR hump OpenFOAM SST smoke case."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from aeromap.cfd.nasa_hump import (
    plot3d_grid_input_name,
    split_plot3d_default_patch,
    write_conversion_scaffold,
    write_sst_smoke_case_template,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRID = ROOT / "artifacts/methodology/nasa_hump/raw/hump2newtop_noplenumZ103x28.p2dfmt.gz"
DEFAULT_CASE_DIR = ROOT / "artifacts/methodology/nasa_hump/sst_smoke_case"


def _case_path_for_container(case_dir: Path) -> str:
    return str(case_dir.resolve().relative_to(ROOT))


def _docker_openfoam(case_dir: Path, command: str) -> None:
    case_path = _case_path_for_container(case_dir)
    subprocess.run(
        [
            "docker",
            "compose",
            "run",
            "--rm",
            "cfd",
            (f"source /opt/openfoam13/etc/bashrc && cd {case_path} && {command}"),
        ],
        cwd=ROOT,
        check=True,
    )


def run_smoke_case(
    *,
    grid_path: Path,
    case_dir: Path,
    overwrite: bool,
    end_time: int,
) -> dict[str, object]:
    if case_dir.exists():
        if not overwrite:
            msg = f"case directory already exists: {case_dir}. Use --overwrite to regenerate it."
            raise FileExistsError(msg)
        shutil.rmtree(case_dir)
    if not grid_path.exists():
        msg = (
            f"grid file not found: {grid_path}. Run "
            "`uv run python scripts/prepare_nasa_hump_methodology.py` first."
        )
        raise FileNotFoundError(msg)

    grid_input_name = plot3d_grid_input_name(grid_path)
    write_conversion_scaffold(grid_path=grid_path, out_dir=case_dir)
    template = write_sst_smoke_case_template(case_dir, end_time=end_time)
    (case_dir / "logs").mkdir(parents=True, exist_ok=True)

    _docker_openfoam(
        case_dir,
        f"plot3dToFoam -noBlank -2D 0.1 input/{grid_input_name} > logs/plot3dToFoam.log 2>&1",
    )
    split = split_plot3d_default_patch(case_dir)
    _docker_openfoam(case_dir, "checkMesh > logs/checkMesh.log 2>&1")
    _docker_openfoam(case_dir, "checkMesh -meshQuality > logs/checkMesh_meshQuality.log 2>&1")
    _docker_openfoam(case_dir, "potentialFoam > logs/potentialFoam.log 2>&1")
    _docker_openfoam(case_dir, "foamRun > logs/foamRun.log 2>&1")
    _docker_openfoam(
        case_dir,
        "foamPostProcess -solver incompressibleFluid -func wallShearStress -latestTime "
        "> logs/wallShearStress.log 2>&1",
    )
    _docker_openfoam(
        case_dir,
        'foamToVTK -latestTime -noInternal -fields "(p U wallShearStress)" -ascii '
        "> logs/foamToVTK_boundary.log 2>&1",
    )
    return {
        "case_dir": str(case_dir),
        "end_time": end_time,
        "template": template,
        "patch_split": split,
        "report_command": (
            f"uv run python scripts/report_nasa_hump_sst_smoke.py --case-dir {case_dir}"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--end-time", type=int, default=80)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = run_smoke_case(
        grid_path=args.grid,
        case_dir=args.case_dir,
        overwrite=args.overwrite,
        end_time=args.end_time,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
