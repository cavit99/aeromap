from __future__ import annotations

import json
from pathlib import Path

import pytest

from aeromap.cfd.reference_lane import (
    parse_surface_check_log,
    reference_lane_command,
    summarize_reference_lane,
)


def test_reference_lane_command_uses_existing_openfoam_image(tmp_path: Path) -> None:
    out_dir = Path.cwd() / tmp_path.name

    command, script = reference_lane_command(
        case_name="drivaerFastback",
        mode="mesh",
        out_dir=out_dir,
    )

    assert "aeromap/openfoam13:dev" in command
    assert "drivaerFastback" in script
    assert "snappyHexMesh -overwrite" in script
    assert "non_headline" not in script


def test_reference_lane_rejects_output_outside_project(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="inside the project root"):
        reference_lane_command(
            case_name="drivaerFastback",
            mode="inspect",
            out_dir=tmp_path,
        )


def test_reference_lane_shell_quotes_output_path() -> None:
    out_dir = Path.cwd() / "artifacts" / "gate2" / "reference $(touch injected)" / "with space"

    _, script = reference_lane_command(
        case_name="drivaerFastback",
        mode="inspect",
        out_dir=out_dir,
    )

    assert "out_dir=/work/'artifacts/gate2/reference $(touch injected)/with space'" in script
    assert "out_dir=/work/artifacts/gate2/reference $(touch injected)/with space" not in script


def test_surface_check_parser_reads_openfoam_surface_summary(tmp_path: Path) -> None:
    log_path = tmp_path / "surfaceCheck_body.obj.log"
    log_path.write_text(
        """
Surface has no illegal triangles.
Triangle quality (equilateral=1, collapsed=0):
    0 .. 0.05  : 0.102954
    min 1.40133e-09 for triangle 924
Edges:
    min 1.99378e-05 for edge 3 points (0 0 0)(1 0 0)
Surface is closed. All edges connected to two faces.
Number of unconnected parts : 1
Statistics:
Triangles    : 76578
Vertices     : 38291
""",
        encoding="utf-8",
    )

    parsed = parse_surface_check_log(log_path)

    assert parsed["surface_check_ok"] is True
    assert parsed["triangles"] == 76578
    assert parsed["quality_0_to_0p05"] == pytest.approx(0.102954)
    assert parsed["min_quality"] == pytest.approx(1.40133e-09)


def test_reference_lane_summary_is_marked_non_headline(tmp_path: Path) -> None:
    out_dir = tmp_path / "reference"
    logs = out_dir / "logs"
    quality = out_dir / "quality"
    system = out_dir / "case" / "system"
    logs.mkdir(parents=True)
    quality.mkdir()
    system.mkdir(parents=True)
    (logs / "surfaceCheck_body.obj.log").write_text(
        "Surface has no illegal triangles.\nSurface is closed.\n",
        encoding="utf-8",
    )
    (logs / "checkMesh.log").write_text("Mesh OK.\n    cells: 123\n", encoding="utf-8")
    (quality / "checkMesh.returncode").write_text("0\n", encoding="utf-8")
    (system / "snappyHexMeshDict").write_text("castellatedMesh true;\n", encoding="utf-8")

    summary_path = summarize_reference_lane(
        case_name="drivaerFastback",
        mode="mesh",
        out_dir=out_dir,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["non_headline"] is True
    assert summary["check_mesh"]["checkMesh"]["mesh_ok"] is True
    assert summary["return_codes"]["checkMesh"] == 0
