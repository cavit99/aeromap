from __future__ import annotations

import gzip
import json
from pathlib import Path

from aeromap.cfd.nasa_hump import (
    RECORDED_103X28_CHECKMESH,
    correlation_eligibility,
    methodology_mesh_policy,
    write_conversion_scaffold,
)


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
    assert rows["solver_run_completed"] == "Not yet"
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
