from __future__ import annotations

import gzip
import json
import textwrap
from pathlib import Path

from aeromap.cfd.nasa_hump import (
    HUMP_REYNOLDS_NUMBER,
    RECORDED_103X28_CHECKMESH,
    correlation_eligibility,
    methodology_mesh_policy,
    write_conversion_scaffold,
    write_sst_smoke_case_template,
)
from scripts.extract_nasa_hump_cp_cf import extract_openfoam_wall_curve
from scripts.report_nasa_hump_sst_smoke import latest_hump_wall_vtk


def test_methodology_gate_keeps_global_gate_strict() -> None:
    policy = methodology_mesh_policy(RECORDED_103X28_CHECKMESH)

    assert policy["global_aeromap_gate_passed"] is False
    assert policy["methodology_gate_passed"] is True
    assert policy["mesh_quality"] == "accepted_with_methodology_warning"
    assert "headline_correlation" in policy["not_accepted_for"]
    assert "single_solver_smoke" in policy["accepted_for"]


def test_correlation_eligibility_marks_solver_work_pending() -> None:
    rows = {row["requirement"]: row["status"] for row in correlation_eligibility()}

    assert rows["experimental_cp_cf_parsed"] == "Pass"
    assert rows["methodology_mesh_gate_defined"] == "Pass"
    assert rows["solver_run_completed"] == "Pass: single-grid smoke only"
    assert rows["cp_cf_extracted_from_openfoam"] == "Pass: smoke-grid overlay only"
    assert rows["openfoam_vs_experiment_compared"] == "Pass: smoke-grid overlay metrics only"
    assert rows["grid_sensitivity_checked"] == "Not yet"


def test_conversion_scaffold_writes_patch_policy_and_commands(tmp_path: Path) -> None:
    grid = tmp_path / "grid.p2dfmt.gz"
    grid.write_bytes(gzip.compress(b"1\n2 2\n0 1 0 1\n0 0 1 1\n"))

    manifest = write_conversion_scaffold(grid_path=grid, out_dir=tmp_path / "case")

    manifest_path = Path(manifest["manifest_path"])
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["runs_openfoam"] is False

    commands = Path(payload["commands_path"]).read_text(encoding="utf-8")
    assert "plot3dToFoam -noBlank -2D 0.1" in commands
    assert "checkMesh > checkMesh.log" in commands

    patch_contract = json.loads(Path(payload["patch_contract_path"]).read_text(encoding="utf-8"))
    assert patch_contract["hump_wall"] == "wall"


def test_sst_smoke_template_writes_potential_initialisation_inputs(tmp_path: Path) -> None:
    case_dir = tmp_path / "sst_case"

    metadata = write_sst_smoke_case_template(case_dir, end_time=80)

    assert metadata["solver"] == "incompressibleFluid"
    assert metadata["turbulence_model"] == "kOmegaSST"
    assert metadata["reynolds_number"] == HUMP_REYNOLDS_NUMBER
    assert metadata["potential_flow_initialisation_required"] is True
    assert "endTime         80;" in (case_dir / "system/controlDict").read_text(encoding="utf-8")
    assert "Phi" in (case_dir / "system/fvSolution").read_text(encoding="utf-8")
    assert '#includeEtc "caseDicts/mesh/generation/meshQualityDict.cfg"' in (
        case_dir / "system/meshQualityDict"
    ).read_text(encoding="utf-8")
    assert "kOmegaSST" in (case_dir / "constant/momentumTransport").read_text(encoding="utf-8")


def test_latest_hump_wall_vtk_uses_numeric_time_suffix(tmp_path: Path) -> None:
    vtk_dir = tmp_path / "VTK" / "hump_wall"
    vtk_dir.mkdir(parents=True)
    (vtk_dir / "hump_wall_80.vtk").write_text("80", encoding="utf-8")
    (vtk_dir / "hump_wall_120.vtk").write_text("120", encoding="utf-8")
    (vtk_dir / "hump_wall_9.vtk").write_text("9", encoding="utf-8")

    assert latest_hump_wall_vtk(tmp_path).name == "hump_wall_120.vtk"


def test_cp_cf_extraction_applies_audited_negative_shear_transform(tmp_path: Path) -> None:
    vtk_path = tmp_path / "hump_wall_10.vtk"
    vtk_path.write_text(
        textwrap.dedent(
            """\
            # vtk DataFile Version 2.0
            hump_wall
            ASCII
            DATASET POLYDATA
            POINTS 24 float
            -6 0 0 -5 0 0 -5 0 1 -6 0 1
            -5 0 0 -4 0 0 -4 0 1 -5 0 1
            -4 0 0 -3 0 0 -3 0 1 -4 0 1
            -3 0 0 -2 0 0 -2 0 1 -3 0 1
            -2 0 0 -1 0 0 -1 0 1 -2 0 1
            -1 0 0 0 0 0 0 0 1 -1 0 1
            POLYGONS 6 30
            4 0 1 2 3
            4 4 5 6 7
            4 8 9 10 11
            4 12 13 14 15
            4 16 17 18 19
            4 20 21 22 23
            CELL_DATA 6
            FIELD attributes 2
            p 1 6 float
            0.5 0.5 0.5 0.5 0.5 0.5
            wallShearStress 3 6 float
            -0.1 0 0 -0.1 0 0 -0.1 0 0 -0.1 0 0 -0.1 0 0 -0.1 0 0
            """
        ),
        encoding="utf-8",
    )

    curve, audit = extract_openfoam_wall_curve(vtk_path)

    assert audit["cf_sign_audit"]["raw_attached_region_sign"] == "negative"
    assert audit["cf_sign_audit"]["sign_transform_applied"].startswith("Cf = -")
    assert curve.cp.tolist() == [1.0] * 6
    assert curve.cf.tolist() == [0.2] * 6
