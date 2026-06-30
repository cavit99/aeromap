from __future__ import annotations

import json
from pathlib import Path

from aeromap.cfd.venturi_core import (
    VENTURI_CORE_CLASSIFICATION,
    VenturiCoreConfig,
    block_mesh_dict,
    build_venturi_core_case,
    field_u,
    profile_stations,
    write_venturi_core_design_report,
    write_venturi_core_grid_validation,
)


def test_venturi_core_profile_is_deterministic_and_physical() -> None:
    config = VenturiCoreConfig()

    stations = profile_stations(config.geometry)

    assert len(stations) == 6
    assert stations[0]["label"] == "inlet_plenum"
    assert stations[-1]["label"] == "outlet_recovery"
    assert float(stations[2]["gap_m"]) < float(stations[1]["gap_m"])
    assert float(stations[4]["gap_m"]) > float(stations[3]["gap_m"])


def test_venturi_core_allows_approved_screening_map_angles() -> None:
    config = VenturiCoreConfig.model_validate(
        {"geometry": {"ride_height_mm": 20.0, "diffuser_angle_deg": 14.0}},
    )

    assert config.geometry.diffuser_angle_deg == 14.0
    assert config.geometry.diffuser_exit_height_m > config.geometry.throat_height_m


def test_venturi_core_blockmesh_has_no_snappy_or_stl_contract() -> None:
    rendered = block_mesh_dict(VenturiCoreConfig())

    assert "blockMeshDict" in rendered
    assert "snappyHexMesh" not in rendered
    assert "triSurface" not in rendered
    assert "floor" in rendered
    assert "ground" in rendered
    assert "symmetryPlane" in rendered
    assert rendered.count("hex (") == 5


def test_venturi_core_velocity_keeps_only_ground_moving() -> None:
    rendered = field_u(VenturiCoreConfig())

    assert "ground      { type fixedValue; value uniform (40 0 0); }" in rendered
    assert "floor       { type noSlip; }" in rendered


def test_build_venturi_core_case_writes_structured_openfoam_layout(tmp_path: Path) -> None:
    artifacts = build_venturi_core_case(VenturiCoreConfig(), cases_dir=tmp_path)
    case_dir = artifacts.case_dir

    assert case_dir.name == artifacts.case_id
    assert (case_dir / "openfoam" / "system" / "blockMeshDict").exists()
    assert not (case_dir / "openfoam" / "system" / "snappyHexMeshDict").exists()
    assert (case_dir / "openfoam" / "0" / "U").exists()
    assert (case_dir / "run_core_mesh.sh").exists()
    assert (case_dir / "run_core_solver.sh").exists()
    manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["classification"] == VENTURI_CORE_CLASSIFICATION
    assert manifest["status"] == "BUILT_NOT_RUN"
    assert manifest["claim_boundary"]["full_3d_extension_accuracy"] is False
    assert manifest["claim_boundary"]["structured_venturi_underfloor_benchmark"] is True


def test_venturi_core_mesh_script_blocks_extended_checkmesh_failures(tmp_path: Path) -> None:
    artifacts = build_venturi_core_case(VenturiCoreConfig(), cases_dir=tmp_path)
    rendered = artifacts.run_mesh_script_path.read_text(encoding="utf-8")

    assert "checkMesh_extended.failed_checks" in rendered
    assert "extended checkMesh reported Failed" in rendered
    assert "exit 4" in rendered


def test_write_venturi_core_design_report_has_claim_boundary(tmp_path: Path) -> None:
    report = write_venturi_core_design_report(
        config=VenturiCoreConfig(),
        out=tmp_path / "core.md",
    )

    text = report.read_text(encoding="utf-8")
    payload = json.loads((tmp_path / "core.json").read_text(encoding="utf-8"))
    assert "Venturi Core / Venturi Lab" in text
    assert "No `snappyHexMesh`" in text
    assert payload["classification"] == VENTURI_CORE_CLASSIFICATION
    assert payload["claim_boundary"]["training_eligible_before_validation"] is False


def _write_core_metrics(
    case_dir: Path,
    *,
    grid: str,
    cd_mean: float,
    suction_mean: float,
    pressure_recovery: float,
    reverse_flow_class: str = "attached_pre_cliff",
) -> None:
    (case_dir / "outputs").mkdir(parents=True)
    payload = {
        "case_id": case_dir.name,
        "case_dir": str(case_dir),
        "grid": grid,
        "mesh": {
            "estimated_cells": 1000,
            "fatal_mesh_quality": {"mesh_ok": True},
            "extended_mesh": {"mesh_ok": True},
        },
        "mass_balance": {"relative_imbalance": 1.0e-6},
        "boundary_conditions": {
            "passed": True,
            "ground_patch": {
                "explicit_moving_belt_ok": True,
                "type": "fixedValue",
                "value": "uniform (40 0 0)",
            },
            "floor_patch": {"type": "noSlip"},
        },
        "force_coefficients_final_window": {
            "Cd": {"mean": cd_mean, "cv": 1.0e-4},
            "Cl": {"mean": suction_mean, "cv": 1.0e-4},
        },
        "floor_metrics": {
            "pressure_recovery_cp_exit_minus_cp_throat": pressure_recovery,
            "diffuser_raw_addendum_f_sep_tau_x_lt_0": 1.0,
            "attached_reference": {
                "mean_wall_shear_x": -1.0,
                "sign": "negative",
            },
            "diffuser_f_sep": 0.0,
            "diffuser_f_sep_regime": reverse_flow_class,
            "near_wall_diffuser_reverse_flow_fraction_u_x_lt_0": 0.0,
            "regions": {
                "diffuser": {
                    "y_plus_mean": 1.0,
                    "y_plus_max": 2.0,
                },
            },
        },
        "ground_metrics": {
            "y_plus_mean": 1.0,
            "y_plus_max": 2.0,
        },
    }
    (case_dir / "outputs" / "core_metrics.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_write_venturi_core_grid_validation_accepts_target_specific_core_reference(
    tmp_path: Path,
) -> None:
    coarse = tmp_path / "coarse"
    medium = tmp_path / "medium"
    fine = tmp_path / "fine"
    _write_core_metrics(
        coarse,
        grid="coarse",
        cd_mean=0.0162,
        suction_mean=1.10,
        pressure_recovery=1.88,
    )
    _write_core_metrics(
        medium,
        grid="medium",
        cd_mean=0.0163,
        suction_mean=1.098,
        pressure_recovery=1.877,
    )
    _write_core_metrics(
        fine,
        grid="fine",
        cd_mean=0.0164,
        suction_mean=1.088,
        pressure_recovery=1.874,
    )

    result = write_venturi_core_grid_validation(
        coarse_case=coarse,
        medium_case=medium,
        fine_case=fine,
        out=tmp_path / "validation.json",
    )

    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["classification"] == "VENTURI_CORE_ATTACHED_PRESSURE_LOAD_REFERENCE_V0"
    assert payload["accepted"] is True
    assert payload["eligible_targets"]["core_attached_drag"] is True
    assert payload["eligible_targets"]["core_cliff_boundary"] is False
    assert payload["eligible_targets"]["full_3d_extension"] is False


def test_write_venturi_core_grid_validation_rejects_grid_sensitive_core_reference(
    tmp_path: Path,
) -> None:
    coarse = tmp_path / "coarse"
    medium = tmp_path / "medium"
    fine = tmp_path / "fine"
    _write_core_metrics(
        coarse,
        grid="coarse",
        cd_mean=0.0162,
        suction_mean=1.10,
        pressure_recovery=1.88,
    )
    _write_core_metrics(
        medium,
        grid="medium",
        cd_mean=0.0163,
        suction_mean=1.098,
        pressure_recovery=1.877,
    )
    _write_core_metrics(
        fine,
        grid="fine",
        cd_mean=0.0164,
        suction_mean=0.90,
        pressure_recovery=1.874,
    )

    result = write_venturi_core_grid_validation(
        coarse_case=coarse,
        medium_case=medium,
        fine_case=fine,
        out=tmp_path / "validation.json",
    )

    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["classification"] == "VENTURI_CORE_ATTACHED_PRESSURE_LOAD_VALIDATION_FAILED"
    assert payload["accepted"] is False
    assert payload["eligible_targets"]["core_attached_drag"] is False
