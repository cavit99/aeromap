from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from aeromap.geometry.generator import (
    MIN_TUNNEL_SOLID_LIGAMENT_M,
    _mesh_from_cadquery,
    build_article,
    generate_geometry,
    minimum_upper_ligament_m,
)
from aeromap.geometry.regions import (
    DIFFUSER_EXIT_X_M,
    REGION_NAMES,
    THROAT_HEIGHT_M,
    classify_surface_regions,
    tunnel_design_metadata,
)
from aeromap.parameters import AeroParams, corner_params
from aeromap.transforms import apply_ride_height_pitch

if TYPE_CHECKING:
    import trimesh


def test_ride_height_controls_minimum_clearance() -> None:
    vertices = np.array([[0.0, 0.0, -0.02], [2.0, 0.0, 0.06]], dtype=np.float64)
    low = apply_ride_height_pitch(vertices, ride_height_mm=25.0, pitch_deg=0.0)
    high = apply_ride_height_pitch(vertices, ride_height_mm=75.0, pitch_deg=0.0)
    assert np.min(low[:, 2]) == pytest.approx(0.025)
    assert np.min(high[:, 2]) == pytest.approx(0.075)


def test_pitch_changes_front_rear_clearance_distribution() -> None:
    vertices = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    nose_down = apply_ride_height_pitch(vertices, ride_height_mm=40.0, pitch_deg=-1.0)
    rear_up = apply_ride_height_pitch(vertices, ride_height_mm=40.0, pitch_deg=1.5)
    assert nose_down[0, 2] > nose_down[1, 2]
    assert rear_up[1, 2] > rear_up[0, 2]


def test_throat_offset_moves_throat_position() -> None:
    forward = tunnel_design_metadata(
        AeroParams.canonical().model_copy(update={"throat_offset_mm": 20.0})
    )
    rearward = tunnel_design_metadata(
        AeroParams.canonical().model_copy(update={"throat_offset_mm": 50.0})
    )
    assert rearward.throat_x_m - forward.throat_x_m == pytest.approx(0.030)


def test_diffuser_angle_increases_exit_roof_height() -> None:
    shallow = tunnel_design_metadata(
        AeroParams.canonical().model_copy(update={"diffuser_angle_deg": 1.0})
    )
    steep = tunnel_design_metadata(
        AeroParams.canonical().model_copy(update={"diffuser_angle_deg": 2.0})
    )
    assert steep.diffuser_exit_roof_height_m > shallow.diffuser_exit_roof_height_m


def _roof_z_near(
    mesh: trimesh.Trimesh,
    *,
    x_m: float,
    y_m: float,
    expected_roof_z_m: float,
) -> float:
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    point = np.array([x_m, y_m], dtype=np.float64)
    a = triangles[:, 0, :2]
    b = triangles[:, 1, :2]
    c = triangles[:, 2, :2]
    ab = b - a
    ac = c - a
    ap = point - a
    denominator = ab[:, 0] * ac[:, 1] - ac[:, 0] * ab[:, 1]
    valid = np.abs(denominator) > 1e-12
    u = np.full(len(triangles), np.nan, dtype=np.float64)
    v = np.full(len(triangles), np.nan, dtype=np.float64)
    u[valid] = (ap[valid, 0] * ac[valid, 1] - ac[valid, 0] * ap[valid, 1]) / denominator[valid]
    v[valid] = (ab[valid, 0] * ap[valid, 1] - ap[valid, 0] * ab[valid, 1]) / denominator[valid]
    inside = valid & (u >= -1e-8) & (v >= -1e-8) & ((u + v) <= 1.0 + 1e-8)
    z_values = (
        triangles[:, 0, 2]
        + u * (triangles[:, 1, 2] - triangles[:, 0, 2])
        + v * (triangles[:, 2, 2] - triangles[:, 0, 2])
    )
    intersections = np.sort(z_values[inside])
    assert len(intersections) > 0
    nearest_index = int(np.argmin(np.abs(intersections - expected_roof_z_m)))
    roof_z = float(intersections[nearest_index])
    assert roof_z == pytest.approx(expected_roof_z_m, abs=0.015)
    return roof_z


def test_diffuser_angle_matches_measured_exported_geometry() -> None:
    params = AeroParams.canonical().model_copy(update={"diffuser_angle_deg": 2.0})
    design = tunnel_design_metadata(params)
    mesh = _mesh_from_cadquery(build_article(params))
    y_probe = design.tunnel_centres_y_m[0]
    x0 = design.throat_x_m + 0.25
    x1 = DIFFUSER_EXIT_X_M - 0.25
    z0_expected = THROAT_HEIGHT_M + np.tan(np.deg2rad(params.diffuser_angle_deg)) * (
        x0 - design.throat_x_m
    )
    z1_expected = THROAT_HEIGHT_M + np.tan(np.deg2rad(params.diffuser_angle_deg)) * (
        x1 - design.throat_x_m
    )
    z0 = _roof_z_near(mesh, x_m=x0, y_m=y_probe, expected_roof_z_m=z0_expected)
    z1 = _roof_z_near(mesh, x_m=x1, y_m=y_probe, expected_roof_z_m=z1_expected)
    measured_angle = np.rad2deg(np.arctan2(z1 - z0, x1 - x0))
    assert measured_angle == pytest.approx(params.diffuser_angle_deg, abs=0.20)


def test_edge_radius_changes_real_tunnel_edge_geometry_without_width_proxy() -> None:
    small = tunnel_design_metadata(
        AeroParams.canonical().model_copy(update={"edge_radius_mm": 5.0})
    )
    large = tunnel_design_metadata(
        AeroParams.canonical().model_copy(update={"edge_radius_mm": 25.0})
    )
    assert small.tunnel_half_width_m == large.tunnel_half_width_m

    small_mesh = _mesh_from_cadquery(
        build_article(AeroParams.canonical().model_copy(update={"edge_radius_mm": 5.0}))
    )
    large_mesh = _mesh_from_cadquery(
        build_article(AeroParams.canonical().model_copy(update={"edge_radius_mm": 25.0}))
    )
    assert large_mesh.volume > small_mesh.volume
    assert len(large_mesh.faces) != len(small_mesh.faces)


def test_corner_cases_include_worst_clearance() -> None:
    corners = corner_params()
    assert "worst_clearance" in corners
    assert corners["worst_clearance"].ride_height_mm == 25.0
    assert corners["worst_clearance"].pitch_deg == -1.0


def test_surface_region_metadata_has_required_regions() -> None:
    params = AeroParams.canonical()
    mesh = _mesh_from_cadquery(build_article(params))
    mesh.vertices = apply_ride_height_pitch(
        np.asarray(mesh.vertices, dtype=np.float64),
        ride_height_mm=params.ride_height_mm,
        pitch_deg=params.pitch_deg,
    )
    mesh.fix_normals()
    regions = classify_surface_regions(mesh, params)
    assert tuple(regions["region_names"]) == REGION_NAMES
    face_regions = regions["face_regions"]
    face_indices = [item["face_index"] for item in face_regions]
    per_region_counts: dict[str, int] = {}
    for item in face_regions:
        region = str(item["region"])
        per_region_counts[region] = per_region_counts.get(region, 0) + 1
    assert len(face_regions) == len(mesh.faces)
    assert len(set(face_indices)) == len(mesh.faces)
    assert sorted(face_indices) == list(range(len(mesh.faces)))
    assert sum(regions["counts"].values()) == len(mesh.faces)
    assert sum(per_region_counts.values()) == len(mesh.faces)
    for name in REGION_NAMES:
        assert regions["counts"][name] > 0


def test_v010_upper_fairing_has_required_tunnel_roof_ligament() -> None:
    params = AeroParams.canonical()
    mesh = _mesh_from_cadquery(build_article(params))
    assert mesh.bounds[0, 2] >= 0.0
    assert minimum_upper_ligament_m(params) >= MIN_TUNNEL_SOLID_LIGAMENT_M


def test_stable_reference_is_valid_separate_underfloor_fixture(tmp_path: Path) -> None:
    params = AeroParams.stable_reference()
    assert params.geometry_id() != AeroParams.canonical().geometry_id()
    assert minimum_upper_ligament_m(params) >= MIN_TUNNEL_SOLID_LIGAMENT_M

    artifacts = generate_geometry(params, tmp_path)
    assert artifacts.validation.valid
    metrics = artifacts.validation.metrics
    assert metrics is not None
    assert metrics.min_ground_clearance_m == pytest.approx(0.100)
    assert metrics.left_tunnel_half_width_m == pytest.approx(0.320)

    regions = json.loads(artifacts.regions_json_path.read_text(encoding="utf-8"))
    design = regions["design_metadata"]
    assert metrics.diffuser_exit_roof_height_m == pytest.approx(
        design["diffuser_exit_roof_height_m"],
    )
    assert design["diffuser_exit_roof_height_m"] > design["throat_roof_height_m"]
    assert regions["counts"]["tunnel_roofs"] > 0
    assert regions["counts"]["diffuser"] > 0
    assert regions["counts"]["keel"] == 0


def test_upward_facing_fairing_is_upper_body_region() -> None:
    params = AeroParams.canonical()
    mesh = _mesh_from_cadquery(build_article(params))
    regions = classify_surface_regions(mesh, params)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    upward = normals[:, 2] > 0.7
    region_names = np.asarray([item["region"] for item in regions["face_regions"]], dtype=object)
    assert np.count_nonzero(upward) > 0
    assert np.mean(region_names[upward] == "upper_body") > 0.95


def test_surface_region_counts_are_pose_independent(tmp_path: Path) -> None:
    base = AeroParams.canonical()
    posed = base.model_copy(update={"ride_height_mm": 70.0, "pitch_deg": 1.4})
    base_artifacts = generate_geometry(base, tmp_path)
    base_regions = json.loads(base_artifacts.regions_json_path.read_text(encoding="utf-8"))

    posed_artifacts = generate_geometry(posed, tmp_path)
    posed_regions = json.loads(posed_artifacts.regions_json_path.read_text(encoding="utf-8"))

    assert base_regions["classification_frame"] == "body_local"
    assert posed_regions["classification_frame"] == "body_local"
    assert posed_regions["counts"] == base_regions["counts"]
