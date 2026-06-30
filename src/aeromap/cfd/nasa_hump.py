"""NASA/TMR wall-mounted hump methodology helpers."""

from __future__ import annotations

import gzip
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

METHODOLOGY_GATE_ID = "NASA_TMR_BOUNDARY_LAYER_METHODOLOGY_GATE_V0_1"
CLASSIFICATION = "NASA_HUMP_METHODOLOGY_PREFLIGHT_V0_1"
RECORDED_TMR_WARNING_COUNT = 2

PATCH_CONTRACT: dict[str, str] = {
    "front": "empty",
    "back": "empty",
    "inlet": "patch",
    "outlet": "patch",
    "top_slip": "slip wall candidate",
    "hump_wall": "wall",
}

WAIVED_WARNINGS = [
    "high_aspect_ratio_expected_for_near_wall_RANS_grid",
    "small_determinant_cells_expected_from_boundary_layer_stretching",
]

FATAL_CHECKS = [
    "negative_or_zero_cell_volume",
    "invalid_or_nonfinite_coordinates",
    "broken_boundary_patch_topology",
    "missing_hump_wall_patch",
    "failed_plot3d_conversion",
    "solver_crash_before_first_iteration",
]

CORRELATION_REQUIREMENTS = [
    ("experimental_cp_cf_parsed", "Pass"),
    ("published_cfl3d_sa_sst_references_parsed", "Pass"),
    ("plot3d_grid_converted", "Pass"),
    ("patch_split_audited", "Pass"),
    ("methodology_mesh_gate_defined", "Pass"),
    ("openfoam_sst_setup_generated", "Pending"),
    ("solver_run_completed", "Not yet"),
    ("cp_cf_extracted_from_openfoam", "Not yet"),
    ("openfoam_vs_experiment_compared", "Not yet"),
    ("grid_sensitivity_checked", "Not yet"),
]

RECORDED_103X28_CHECKMESH: dict[str, Any] = {
    "boundary_patches": 6,
    "cells": 2754,
    "failed_mesh_checks": 2,
    "log_path": "artifacts/methodology/nasa_hump/smoke_case_p2d_noblank/checkMesh_split.log",
    "max_aspect_ratio": 19660.6,
    "mesh_ok": False,
    "small_determinant_cells": 382,
    "source": "recorded_local_plot3dToFoam_smoke_summary",
}


def methodology_mesh_policy(strict_mesh_gate: Mapping[str, Any] | None) -> dict[str, Any]:
    """Classify the official TMR near-wall grid without weakening the global gate."""

    if strict_mesh_gate is None:
        return {
            "gate": METHODOLOGY_GATE_ID,
            "official_tmr_boundary_layer_grid": True,
            "global_aeromap_gate_passed": False,
            "methodology_gate_passed": False,
            "mesh_quality": "not_evaluated",
            "accepted_for": [],
            "not_accepted_for": [
                "solver_smoke",
                "headline_correlation",
                "turbulence_model_recommendation",
            ],
            "waived_warnings": [],
            "fatal_checks": FATAL_CHECKS,
            "rationale": "No checkMesh evidence was supplied.",
        }

    failed_checks = strict_mesh_gate.get("failed_mesh_checks")
    mesh_ok = bool(strict_mesh_gate.get("mesh_ok"))
    global_gate_passed = mesh_ok and (failed_checks in (0, None))

    methodology_warning = not global_gate_passed and failed_checks == RECORDED_TMR_WARNING_COUNT
    return {
        "gate": METHODOLOGY_GATE_ID,
        "official_tmr_boundary_layer_grid": True,
        "global_aeromap_gate_passed": global_gate_passed,
        "methodology_gate_passed": methodology_warning or global_gate_passed,
        "mesh_quality": (
            "accepted_with_methodology_warning"
            if methodology_warning
            else ("accepted" if global_gate_passed else "fatal_or_unclassified")
        ),
        "accepted_for": [
            "conversion_smoke",
            "boundary_condition_smoke",
            "single_solver_smoke",
        ]
        if methodology_warning or global_gate_passed
        else [],
        "not_accepted_for": [
            "headline_correlation",
            "turbulence_model_recommendation",
            "grid_convergence_claim",
        ],
        "waived_warnings": WAIVED_WARNINGS if methodology_warning else [],
        "fatal_checks": FATAL_CHECKS,
        "rationale": (
            "The official 103 x 28 TMR boundary-layer grid is intentionally stretched. "
            "High aspect ratio and small determinant warnings are therefore treated as "
            "methodology warnings for smoke work only, not as correlation eligibility."
        ),
    }


def correlation_eligibility() -> list[dict[str, str]]:
    """Return the preflight status table used by reports and evidence."""

    return [
        {"requirement": requirement, "status": status}
        for requirement, status in CORRELATION_REQUIREMENTS
    ]


def write_conversion_scaffold(
    *,
    grid_path: Path,
    out_dir: Path,
    case_name: str = "nasa_hump_no_plenum_103x28",
) -> dict[str, Any]:
    """Write a small OpenFOAM conversion scaffold for the NASA/TMR hump grid.

    The scaffold does not run OpenFOAM. It records the patch contract, mesh policy and
    commands required for a local smoke conversion.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    input_dir = out_dir / "input"
    input_dir.mkdir(exist_ok=True)
    target_grid = input_dir / "hump2newtop_noplenumZ103x28.p2dfmt"
    if grid_path.suffix == ".gz":
        target_grid.write_bytes(gzip.decompress(grid_path.read_bytes()))
    else:
        shutil.copyfile(grid_path, target_grid)

    policy = methodology_mesh_policy(RECORDED_103X28_CHECKMESH)
    patch_path = out_dir / "patch_contract.json"
    policy_path = out_dir / "methodology_mesh_policy.json"
    commands_path = out_dir / "Allrun.convert"

    patch_path.write_text(
        json.dumps(PATCH_CONTRACT, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    policy_path.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    commands_path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "",
                "# Source OpenFOAM before running this script, for example:",
                "#   source /opt/openfoam13/etc/bashrc",
                "",
                f"case_name={case_name!r}",
                "grid=input/hump2newtop_noplenumZ103x28.p2dfmt",
                "",
                'plot3dToFoam -noBlank -2D 0.1 "$grid"',
                "# Apply the patch split described in patch_contract.json before solving.",
                "checkMesh > checkMesh.log 2>&1",
                "",
            ],
        ),
        encoding="utf-8",
    )
    commands_path.chmod(0o755)

    manifest = {
        "schema_version": "nasa_hump_openfoam_conversion_scaffold_v0.1.0",
        "case_name": case_name,
        "grid_path": str(target_grid),
        "patch_contract_path": str(patch_path),
        "methodology_mesh_policy_path": str(policy_path),
        "commands_path": str(commands_path),
        "runs_openfoam": False,
        "next_step": (
            "Run Allrun.convert in an OpenFOAM v13 environment, then audit boundary patches."
        ),
    }
    manifest_path = out_dir / "conversion_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest | {"manifest_path": str(manifest_path)}
