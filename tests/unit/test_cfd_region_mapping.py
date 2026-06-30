from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import pyvista as pv
import trimesh

from aeromap.cfd.region_mapping import (
    RegionMappingError,
    map_surface_regions_to_vtp,
    map_wall_regions_analytically_to_vtp,
    trimesh_to_polydata,
)
from aeromap.parameters import AeroParams
from aeromap.transforms import apply_ride_height_pitch


def _two_triangle_surface() -> tuple[trimesh.Trimesh, dict[str, object]]:
    mesh = trimesh.Trimesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        process=False,
    )
    regions: dict[str, object] = {
        "face_regions": [
            {"face_index": 0, "region": "underfloor", "region_id": 2},
            {"face_index": 1, "region": "diffuser", "region_id": 0},
        ],
    }
    return mesh, regions


def _single_triangle_surface(
    region: str, region_id: int
) -> tuple[trimesh.Trimesh, dict[str, object]]:
    mesh = trimesh.Trimesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        process=False,
    )
    return mesh, {"face_regions": [{"face_index": 0, "region": region, "region_id": region_id}]}


def _posed_region_fixture() -> tuple[
    AeroParams,
    trimesh.Trimesh,
    dict[str, object],
    trimesh.Trimesh,
]:
    params = AeroParams.canonical().model_copy(update={"ride_height_mm": 60.0, "pitch_deg": 0.4})
    underfloor_z = 0.05
    upper_z = 0.14
    body_vertices = np.array(
        [
            [0.42, 0.078, underfloor_z],
            [0.52, 0.085, underfloor_z],
            [0.62, 0.078, underfloor_z],
            [0.42, -0.16, upper_z],
            [0.62, -0.16, upper_z],
            [0.52, -0.24, upper_z],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    source_faces = np.array(
        [
            [0, 1, 2],
            [3, 5, 4],
        ],
        dtype=np.int64,
    )
    target_faces_with_reversed_normals = source_faces[:, ::-1]
    posed_vertices = apply_ride_height_pitch(
        body_vertices,
        ride_height_mm=params.ride_height_mm,
        pitch_deg=params.pitch_deg,
    )
    source_mesh = trimesh.Trimesh(vertices=posed_vertices, faces=source_faces, process=False)
    target_mesh = trimesh.Trimesh(
        vertices=posed_vertices,
        faces=target_faces_with_reversed_normals,
        process=False,
    )
    regions: dict[str, object] = {
        "classification_frame": "body_local",
        "face_regions": [
            {"face_index": 0, "region": "underfloor", "region_id": 2},
            {"face_index": 1, "region": "upper_body", "region_id": 5},
        ],
    }
    return params, source_mesh, regions, target_mesh


def _tilted_target_with_centroid(
    centroid: np.ndarray,
    *,
    normal_z_component: float,
) -> trimesh.Trimesh:
    normal = np.array(
        [
            np.sqrt(1.0 - normal_z_component**2),
            0.0,
            normal_z_component,
        ],
        dtype=np.float64,
    )
    axis_u = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    axis_v = np.cross(normal, axis_u)
    axis_v /= np.linalg.norm(axis_v)
    radius = 0.1
    vertices = np.array(
        [
            centroid + radius * axis_u,
            centroid - 0.5 * radius * axis_u + radius * axis_v,
            centroid - 0.5 * radius * axis_u - radius * axis_v,
        ],
        dtype=np.float64,
    )
    return trimesh.Trimesh(vertices=vertices, faces=np.array([[0, 1, 2]]), process=False)


def test_surface_region_mapping_writes_region_id_arrays(tmp_path: Path) -> None:
    source_mesh, regions = _two_triangle_surface()
    result = map_surface_regions_to_vtp(
        source_mesh=source_mesh,
        source_regions=regions,
        target_surface=trimesh_to_polydata(source_mesh),
        output_vtp_path=tmp_path / "mapped.vtp",
        report_path=tmp_path / "mapping.json",
        max_distance_m=1e-9,
    )

    assert result.coverage == pytest.approx(1.0)
    assert result.unmapped_faces == 0
    assert result.normal_rejected_faces == 0
    assert result.missing_regions == ()
    assert result.per_region_area_m2["underfloor"] > 0.0
    assert result.per_region_area_m2["diffuser"] > 0.0
    mapped = pv.read(result.output_vtp_path)
    assert list(mapped.cell_data["region_id"]) == [2, 0]
    assert "source_region_distance_m" in mapped.cell_data
    assert "source_region_abs_normal_alignment" in mapped.cell_data
    assert "source_region_ambiguous" in mapped.cell_data
    assert "local_face_area_m2" in mapped.cell_data
    assert result.report_path.exists()


def test_analytic_wall_mapping_uses_body_local_pose_and_oriented_normals(
    tmp_path: Path,
) -> None:
    params, source_mesh, regions, target_mesh = _posed_region_fixture()

    result = map_wall_regions_analytically_to_vtp(
        source_mesh=source_mesh,
        source_regions=regions,
        target_surface=trimesh_to_polydata(target_mesh),
        params=params,
        output_vtp_path=tmp_path / "analytic_regions.vtp",
        report_path=tmp_path / "analytic_regions.json",
        required_regions=("underfloor", "upper_body"),
    )

    mapped = pv.read(result.output_vtp_path)
    assert result.classification_method == "body_local_analytic"
    assert result.area_coverage == pytest.approx(1.0)
    assert result.missing_regions == ()
    assert list(mapped.cell_data["surface_region"]) == ["underfloor", "upper_body"]
    assert list(mapped.cell_data["target_normal_orientation_flipped"]) == [1, 1]
    assert result.cross_check is not None
    assert result.cross_check["target_normal_orientation_flipped_faces"] == 2


def test_surface_region_mapping_follows_geometry_when_faces_are_permuted(tmp_path: Path) -> None:
    source_mesh, regions = _two_triangle_surface()
    target_mesh = trimesh.Trimesh(
        vertices=np.asarray(source_mesh.vertices, dtype=np.float64),
        faces=np.asarray(source_mesh.faces, dtype=np.int64)[::-1],
        process=False,
    )

    result = map_surface_regions_to_vtp(
        source_mesh=source_mesh,
        source_regions=regions,
        target_surface=trimesh_to_polydata(target_mesh),
        output_vtp_path=tmp_path / "mapped_permuted.vtp",
        report_path=tmp_path / "mapping_permuted.json",
        max_distance_m=1e-9,
    )

    mapped = pv.read(result.output_vtp_path)
    assert list(mapped.cell_data["region_id"]) == [0, 2]
    assert result.min_abs_normal_alignment == pytest.approx(1.0)


def test_surface_region_mapping_fails_below_required_coverage(tmp_path: Path) -> None:
    source_mesh, regions = _two_triangle_surface()
    shifted = source_mesh.copy()
    shifted.vertices = np.asarray(shifted.vertices, dtype=np.float64) + np.array([0.0, 0.0, 0.1])

    with pytest.raises(RegionMappingError):
        map_surface_regions_to_vtp(
            source_mesh=source_mesh,
            source_regions=regions,
            target_surface=trimesh_to_polydata(shifted),
            output_vtp_path=tmp_path / "mapped.vtp",
            report_path=tmp_path / "mapping.json",
            max_distance_m=0.001,
        )


def test_surface_region_mapping_uses_local_face_scale_distance_limit(tmp_path: Path) -> None:
    source_mesh, regions = _single_triangle_surface("underfloor", 2)
    shifted = source_mesh.copy()
    shifted.vertices = np.asarray(shifted.vertices, dtype=np.float64) + np.array([0.0, 0.0, 0.02])

    with pytest.raises(RegionMappingError):
        map_surface_regions_to_vtp(
            source_mesh=source_mesh,
            source_regions=regions,
            target_surface=trimesh_to_polydata(shifted),
            output_vtp_path=tmp_path / "mapped_scaled.vtp",
            report_path=tmp_path / "mapping_scaled.json",
            max_distance_face_scale=0.01,
        )

    report = json.loads((tmp_path / "mapping_scaled.json").read_text(encoding="utf-8"))
    assert report["distance_rejected_faces"] == 1
    assert "sqrt(local_face_area_m2)" in report["distance_rule"]


def test_surface_region_mapping_rejects_low_normal_primary_regions(tmp_path: Path) -> None:
    source_mesh, regions = _single_triangle_surface("underfloor", 2)
    target_mesh = _tilted_target_with_centroid(
        np.asarray(source_mesh.triangles_center[0], dtype=np.float64),
        normal_z_component=0.2,
    )

    with pytest.raises(RegionMappingError):
        map_surface_regions_to_vtp(
            source_mesh=source_mesh,
            source_regions=regions,
            target_surface=trimesh_to_polydata(target_mesh),
            output_vtp_path=tmp_path / "mapped_primary.vtp",
            report_path=tmp_path / "mapping_primary.json",
            max_distance_m=1e-9,
            min_abs_normal_alignment=0.5,
            edge_min_abs_normal_alignment=0.1,
        )


def test_surface_region_mapping_reports_edge_normal_exceptions(tmp_path: Path) -> None:
    source_mesh, regions = _single_triangle_surface("floor_edges", 4)
    target_mesh = _tilted_target_with_centroid(
        np.asarray(source_mesh.triangles_center[0], dtype=np.float64),
        normal_z_component=0.2,
    )

    result = map_surface_regions_to_vtp(
        source_mesh=source_mesh,
        source_regions=regions,
        target_surface=trimesh_to_polydata(target_mesh),
        output_vtp_path=tmp_path / "mapped_edge.vtp",
        report_path=tmp_path / "mapping_edge.json",
        max_distance_m=1e-9,
        min_abs_normal_alignment=0.5,
        edge_min_abs_normal_alignment=0.1,
    )

    assert result.coverage == pytest.approx(1.0)
    assert result.normal_exception_faces == 1
    assert result.normal_exception_regions == {"floor_edges": 1}
    mapped = pv.read(result.output_vtp_path)
    assert list(mapped.cell_data["source_region_normal_exception"]) == [1]


@pytest.mark.parametrize(
    ("min_abs_normal_alignment", "edge_min_abs_normal_alignment"),
    [(-0.1, 0.0), (0.5, -0.1), (0.5, 0.6)],
)
def test_surface_region_mapping_rejects_invalid_absolute_normal_thresholds(
    tmp_path: Path,
    min_abs_normal_alignment: float,
    edge_min_abs_normal_alignment: float,
) -> None:
    source_mesh, regions = _single_triangle_surface("floor_edges", 4)

    with pytest.raises(ValueError, match="normal_alignment"):
        map_surface_regions_to_vtp(
            source_mesh=source_mesh,
            source_regions=regions,
            target_surface=trimesh_to_polydata(source_mesh),
            output_vtp_path=tmp_path / "mapped_thresholds.vtp",
            report_path=tmp_path / "mapping_thresholds.json",
            min_abs_normal_alignment=min_abs_normal_alignment,
            edge_min_abs_normal_alignment=edge_min_abs_normal_alignment,
        )
