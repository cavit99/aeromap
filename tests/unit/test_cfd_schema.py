from __future__ import annotations

import pytest
from pydantic import ValidationError

from aeromap.cfd.schema import (
    CfdConfig,
    MeshConfig,
    QualityConfig,
    RefinementBox,
    SolverConfig,
    SurfaceExportConfig,
)


def test_default_cfd_config_is_valid() -> None:
    config = CfdConfig()
    assert config.mesh.block_cells == (42, 16, 16)
    assert config.mesh.implicit_feature_snap is False
    assert config.mesh.explicit_feature_snap is True
    assert config.mesh.snap_solve_iterations == 30
    assert config.mesh.add_layers is True
    assert config.mesh.layer_relative_sizes is True
    assert config.surface_export.method == "cadquery_current"
    assert config.solver.max_iterations == 250
    assert config.quality.case_class == "CAMPAIGN_REFERENCE_CFD"
    assert config.quality.mesh_quality_fatal is True
    assert config.quality.extended_diagnostics_required is True
    assert config.quality.extended_diagnostics_fatal is False
    assert config.quality.region_mapping_min_coverage == pytest.approx(0.995)
    assert config.quality.region_mapping_max_distance_face_scale == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("block_cells", (0, 16, 16)),
        ("block_cells", (42, -1, 16)),
        ("max_global_cells", 0),
        ("surface_level", (3, 1)),
        ("surface_level", (-1, 1)),
        ("feature_angle_deg", 0.0),
        ("feature_angle_deg", 181.0),
        ("snap_solve_iterations", 0),
        ("n_surface_layers", 0),
        ("layer_expansion_ratio", 0.9),
        ("final_layer_thickness", 0.0),
        ("min_layer_thickness", 0.0),
        ("max_thickness_to_medial_ratio", 0.0),
        ("layer_feature_angle_deg", 0.0),
        ("layer_feature_angle_deg", 181.0),
        ("layer_slip_feature_angle_deg", 0.0),
        ("layer_slip_feature_angle_deg", 181.0),
        ("layer_n_relaxed_iter", -1),
        ("layer_n_medial_axis_iter", -1),
        ("layer_n_grow", -1),
        ("layer_n_buffer_cells_no_extrude", -1),
        ("n_cells_between_levels", 0),
    ],
)
def test_mesh_config_rejects_invalid_openfoam_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        MeshConfig.model_validate({field: value})


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "bad-name", "bounds_min": (0, 0, 0), "bounds_max": (1, 1, 1), "level": 1},
        {"name": "box", "bounds_min": (1, 0, 0), "bounds_max": (1, 1, 1), "level": 1},
        {"name": "box", "bounds_min": (0, 0, 0), "bounds_max": (1, 1, 1), "level": -1},
    ],
)
def test_refinement_box_rejects_invalid_values(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        RefinementBox.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"method": "unsupported"},
        {"mesh_size_min_m": 0.02, "mesh_size_max_m": 0.01},
        {"mesh_size_min_m": 0.0},
        {"mesh_algorithm": "delaunay"},
    ],
)
def test_surface_export_config_rejects_invalid_values(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SurfaceExportConfig.model_validate(payload)


@pytest.mark.parametrize("field", ["max_iterations", "write_interval", "force_window"])
def test_solver_config_rejects_non_positive_values(field: str) -> None:
    with pytest.raises(ValidationError):
        SolverConfig.model_validate({field: 0})


def test_quality_config_rejects_unknown_case_class() -> None:
    with pytest.raises(ValidationError):
        QualityConfig.model_validate({"case_class": "headline"})


@pytest.mark.parametrize(
    "payload",
    [
        {"region_mapping_min_coverage": 0.0},
        {"region_mapping_min_coverage": 1.1},
        {"region_mapping_max_distance_face_scale": 0.0},
    ],
)
def test_quality_config_rejects_invalid_region_mapping_thresholds(
    payload: dict[str, float],
) -> None:
    with pytest.raises(ValidationError):
        QualityConfig.model_validate(payload)
