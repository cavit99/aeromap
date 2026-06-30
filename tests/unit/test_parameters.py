from __future__ import annotations

import pytest
from pydantic import ValidationError

from aeromap.parameters import AeroParams, sobol_params


def test_parameter_bounds_reject_invalid_ride_height() -> None:
    with pytest.raises(ValidationError):
        AeroParams(
            ride_height_mm=10.0,
            pitch_deg=0.0,
            yaw_deg=0.0,
            throat_offset_mm=35.0,
            diffuser_angle_deg=1.25,
            edge_radius_mm=12.0,
        )


def test_advanced_family_preserves_original_bounds() -> None:
    with pytest.raises(ValidationError, match="advanced_challenge"):
        AeroParams(
            ride_height_mm=100.0,
            pitch_deg=0.0,
            yaw_deg=0.0,
            throat_offset_mm=35.0,
            diffuser_angle_deg=0.6,
            edge_radius_mm=12.0,
        )


def test_stable_reference_family_has_separate_identity_and_bounds() -> None:
    stable = AeroParams.stable_reference()
    advanced = AeroParams.canonical()

    assert stable.geometry_family == "stable_reference"
    assert stable.ride_height_mm == 100.0
    assert stable.diffuser_angle_deg == 0.6
    assert stable.geometry_id() != advanced.geometry_id()
    assert stable.state_id() != advanced.state_id()
    assert stable.geometry_payload()["floor_family"] == "stable_reference_underfloor_fixture"


def test_stable_reference_rejects_advanced_state_sweep_values() -> None:
    with pytest.raises(ValidationError, match="stable_reference"):
        AeroParams(**{**AeroParams.stable_reference().model_dump(), "yaw_deg": 1.0})


def test_case_id_and_family_are_stable() -> None:
    params = AeroParams.canonical()
    assert params.case_id() == AeroParams.canonical().case_id()
    assert params.geometry_family_id() == AeroParams.canonical().geometry_family_id()


def test_geometry_family_ignores_operating_state() -> None:
    base = AeroParams.canonical()
    changed = base.model_copy(update={"ride_height_mm": 55.0, "pitch_deg": 1.0, "yaw_deg": 2.0})
    assert changed.geometry_family_id() == base.geometry_family_id()
    assert changed.geometry_id() == base.geometry_id()
    assert changed.state_id() != base.state_id()
    assert changed.case_id() != base.case_id()


def test_geometry_identifiers_change_for_geometry_parameters() -> None:
    base = AeroParams.canonical()
    changed = base.model_copy(update={"diffuser_angle_deg": 2.0})

    assert changed.geometry_family_id() != base.geometry_family_id()
    assert changed.geometry_id() != base.geometry_id()


def test_simulation_identity_separates_geometry_state_and_solver_config() -> None:
    params = AeroParams.canonical()
    mesh_config = {"surface_level": [1, 2], "n_surface_layers": 3}
    surface_export_config = {"method": "cadquery_current"}
    solver_config = {"max_iterations": 250}

    payload = params.simulation_payload(
        mesh_config=mesh_config,
        surface_export_config=surface_export_config,
        solver_config=solver_config,
        openfoam_version="OpenFOAM Foundation v13",
    )

    assert payload["geometry_id"] == params.geometry_id()
    assert payload["state_id"] == params.state_id()
    assert payload["surface_export_config"] == surface_export_config
    assert "ride_height_mm" not in params.geometry_payload()
    assert params.simulation_id(
        mesh_config=mesh_config,
        surface_export_config=surface_export_config,
        solver_config=solver_config,
        openfoam_version="OpenFOAM Foundation v13",
    ).startswith("simulation_")
    baseline = params.simulation_id(
        mesh_config=mesh_config,
        surface_export_config=surface_export_config,
        solver_config=solver_config,
        openfoam_version="OpenFOAM Foundation v13",
    )
    assert (
        params.simulation_id(
            mesh_config={**mesh_config, "n_surface_layers": 6},
            surface_export_config=surface_export_config,
            solver_config=solver_config,
            openfoam_version="OpenFOAM Foundation v13",
        )
        != baseline
    )
    assert (
        params.simulation_id(
            mesh_config=mesh_config,
            surface_export_config={"method": "gmsh_occ_g0_no_healing"},
            solver_config=solver_config,
            openfoam_version="OpenFOAM Foundation v13",
        )
        != baseline
    )
    assert (
        params.simulation_id(
            mesh_config=mesh_config,
            surface_export_config=surface_export_config,
            solver_config={"max_iterations": 500},
            openfoam_version="OpenFOAM Foundation v13",
        )
        != baseline
    )
    assert (
        params.simulation_id(
            mesh_config=mesh_config,
            surface_export_config=surface_export_config,
            solver_config=solver_config,
            openfoam_version="OpenFOAM Foundation v14",
        )
        != baseline
    )


def test_sobol_params_are_deterministic_and_in_bounds() -> None:
    first = sobol_params(8, seed=7)
    second = sobol_params(8, seed=7)
    first_case_ids = [item.case_id() for item in first]
    assert len(first) == 8
    assert len(second) == 8
    assert len(set(first_case_ids)) == 8
    assert first_case_ids == [item.case_id() for item in second]
    for params in first:
        dumped = params.canonical_dict()
        for name, value in dumped.items():
            low, high = AeroParams.ranges[name]
            assert low <= value <= high
