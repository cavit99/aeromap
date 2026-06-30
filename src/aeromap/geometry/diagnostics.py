"""CAD/BRep and STL surface diagnostics for mesh diagnostic work."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import cadquery as cq
import numpy as np
import pyvista as pv
import trimesh
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from aeromap.attempts import stable_id, write_attempt_manifest
from aeromap.cfd.schema import CfdConfig
from aeromap.constants import GEOMETRY_GENERATOR_VERSION
from aeromap.geometry.generator import build_article, generate_geometry
from aeromap.geometry.regions import REGION_NAMES
from aeromap.io import atomic_write_json, atomic_write_text, sha256_file
from aeromap.parameters import AeroParams
from aeromap.transforms import apply_ride_height_pitch

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

SURFACE_EXPORT_VERSION = "surface_export_v0.1.0"
SURFACE_DIAGNOSTIC_VERSION = "surface_diagnostics_v0.1.0"
CAD_TESSELLATION_TOLERANCE_M = 0.001
CAD_TESSELLATION_ANGULAR_TOLERANCE_RAD = 0.05
MICRO_EDGE_LENGTH_M = 5e-5
TINY_FACE_AREA_M2 = 1e-8
CAD_EDGE_LENGTH_THRESHOLDS_M = (5e-5, 1e-4, 5e-4, 1e-3)
CAD_FACE_AREA_THRESHOLDS_M2 = (1e-8, 1e-6, 1e-5, 1e-4)
LOW_TRIANGLE_QUALITY = 0.05
NEAR_ZERO_TRIANGLE_QUALITY = 1e-6
TRIANGLE_VERTEX_COUNT = 3


class _BoundBoxLike(Protocol):
    xmin: float
    ymin: float
    zmin: float
    xmax: float
    ymax: float
    zmax: float


class _VectorLike(Protocol):
    def toTuple(self) -> tuple[float, float, float]: ...  # noqa: N802


class _CadEntityLike(Protocol):
    def BoundingBox(self) -> _BoundBoxLike: ...  # noqa: N802

    def Center(self) -> _VectorLike: ...  # noqa: N802

    def geomType(self) -> object: ...  # noqa: N802


class _CadEdgeLike(_CadEntityLike, Protocol):
    def Length(self) -> float: ...  # noqa: N802


class _CadFaceLike(_CadEntityLike, Protocol):
    def Area(self) -> float: ...  # noqa: N802

    def Edges(self) -> Sequence[_CadEdgeLike]: ...  # noqa: N802

    def tessellate(
        self,
        tolerance: float,
        angularTolerance: float = 0.1,  # noqa: N803
    ) -> tuple[Sequence[_VectorLike], Sequence[tuple[int, int, int]]]: ...


class CadShapeLike(Protocol):
    def Faces(self) -> Sequence[_CadFaceLike]: ...  # noqa: N802

    def isValid(self) -> bool: ...  # noqa: N802


@dataclass(frozen=True)
class SurfaceDiagnosticArtifacts:
    attempt_id: str
    attempt_dir: Path
    metrics_path: Path
    attempt_manifest_path: Path
    cad_faces_vtp_path: Path
    stl_triangles_vtp_path: Path
    bad_triangles_csv_path: Path
    microscopic_edges_csv_path: Path
    tiny_faces_csv_path: Path


@dataclass(frozen=True)
class CadFaceSamples:
    points: FloatArray
    faces: IntArray
    face_ids: IntArray
    centroids: FloatArray


@dataclass(frozen=True)
class StlDiagnostics:
    mesh: trimesh.Trimesh
    metrics: dict[str, FloatArray]
    centroids: FloatArray
    region_ids: IntArray
    region_names: NDArray[np.object_]
    nearest_cad_face_id: IntArray
    low_quality: NDArray[np.bool_]
    near_zero_quality: NDArray[np.bool_]
    microscopic_edge: NDArray[np.bool_]
    min_quality_index: int
    vtp_path: Path
    bad_triangles_csv_path: Path


def _bbox_tuple(shape: _CadEntityLike) -> tuple[float, float, float, float, float, float]:
    bbox = shape.BoundingBox()
    return (
        float(bbox.xmin),
        float(bbox.ymin),
        float(bbox.zmin),
        float(bbox.xmax),
        float(bbox.ymax),
        float(bbox.zmax),
    )


def _vector_tuple(vector: _VectorLike) -> tuple[float, float, float]:
    values = vector.toTuple()
    return (float(values[0]), float(values[1]), float(values[2]))


def triangle_metrics(triangles: FloatArray) -> dict[str, FloatArray]:
    edges = np.stack(
        [
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
        ],
        axis=1,
    )
    min_edge = np.min(edges, axis=1)
    max_edge = np.max(edges, axis=1)
    edge_product = edges[:, 0] * edges[:, 1] * edges[:, 2]
    semi_perimeter = np.sum(edges, axis=1) / 2.0
    area = np.sqrt(
        np.maximum(
            semi_perimeter
            * (semi_perimeter - edges[:, 0])
            * (semi_perimeter - edges[:, 1])
            * (semi_perimeter - edges[:, 2]),
            0.0,
        ),
    )
    circumradius = np.minimum(edge_product / np.maximum(4.0 * area, 1e-300), 1e150)
    quality = area / np.maximum(0.75 * math.sqrt(3.0) * circumradius**2, 1e-300)
    aspect_ratio = max_edge / np.maximum(min_edge, 1e-300)
    return {
        "area_m2": area,
        "min_edge_m": min_edge,
        "max_edge_m": max_edge,
        "aspect_ratio": aspect_ratio,
        "triangle_quality": quality,
    }


def _mesh_polydata(vertices: FloatArray, faces: IntArray) -> pv.PolyData:
    cells = np.column_stack(
        [np.full(len(faces), TRIANGLE_VERTEX_COUNT, dtype=np.int64), faces],
    ).ravel()
    return pv.PolyData(vertices, cells)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        atomic_write_text(path, "")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _cad_face_records(shape: CadShapeLike) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    face_records: list[dict[str, Any]] = []
    edge_records: list[dict[str, Any]] = []
    for face_id, face in enumerate(shape.Faces()):
        face_records.append(
            {
                "cad_face_id": face_id,
                "area_m2": float(face.Area()),
                "bbox": _bbox_tuple(face),
                "center_m": _vector_tuple(face.Center()),
                "edge_count": len(face.Edges()),
                "geom_type": str(face.geomType()),
            },
        )
        for edge_id, edge in enumerate(face.Edges()):
            edge_records.append(
                {
                    "cad_face_id": face_id,
                    "face_edge_id": edge_id,
                    "length_m": float(edge.Length()),
                    "bbox": _bbox_tuple(edge),
                    "center_m": _vector_tuple(edge.Center()),
                    "geom_type": str(edge.geomType()),
                },
            )
    return face_records, edge_records


def _threshold_counts(
    records: list[dict[str, Any]],
    *,
    value_key: str,
    thresholds: tuple[float, ...],
) -> list[dict[str, Any]]:
    return [
        {
            "threshold": threshold,
            "count_below": sum(float(record[value_key]) < threshold for record in records),
        }
        for threshold in thresholds
    ]


def _minimum_record(records: list[dict[str, Any]], *, value_key: str) -> dict[str, Any]:
    if not records:
        return {}
    minimum = min(records, key=lambda record: float(record[value_key]))
    return dict(minimum)


def cad_topology_summary(shape: CadShapeLike) -> dict[str, Any]:
    """Summarise BRep topology with locations for smallest faces and edges."""

    face_records, edge_records = _cad_face_records(shape)
    return {
        "valid": bool(shape.isValid()),
        "face_count": len(face_records),
        "edge_count": len(edge_records),
        "min_edge": _minimum_record(edge_records, value_key="length_m"),
        "min_face": _minimum_record(face_records, value_key="area_m2"),
        "edge_length_threshold_counts": _threshold_counts(
            edge_records,
            value_key="length_m",
            thresholds=CAD_EDGE_LENGTH_THRESHOLDS_M,
        ),
        "face_area_threshold_counts": _threshold_counts(
            face_records,
            value_key="area_m2",
            thresholds=CAD_FACE_AREA_THRESHOLDS_M2,
        ),
        "microscopic_edge_threshold_m": MICRO_EDGE_LENGTH_M,
        "microscopic_edge_count": sum(
            float(row["length_m"]) < MICRO_EDGE_LENGTH_M for row in edge_records
        ),
        "tiny_face_area_threshold_m2": TINY_FACE_AREA_M2,
        "tiny_face_count": sum(float(row["area_m2"]) < TINY_FACE_AREA_M2 for row in face_records),
    }


def cad_face_samples(
    shape: CadShapeLike,
    params: AeroParams,
    *,
    tolerance_m: float = CAD_TESSELLATION_TOLERANCE_M,
    angular_tolerance_rad: float = CAD_TESSELLATION_ANGULAR_TOLERANCE_RAD,
) -> CadFaceSamples:
    all_points: list[tuple[float, float, float]] = []
    all_faces: list[tuple[int, int, int]] = []
    face_ids: list[int] = []
    for face_id, face in enumerate(shape.Faces()):
        vertices, face_triangles = face.tessellate(tolerance_m, angular_tolerance_rad)
        offset = len(all_points)
        all_points.extend(_vector_tuple(vertex) for vertex in vertices)
        all_faces.extend((a + offset, b + offset, c + offset) for a, b, c in face_triangles)
        face_ids.extend([face_id] * len(face_triangles))

    points = np.asarray(all_points, dtype=np.float64)
    points = apply_ride_height_pitch(
        points,
        ride_height_mm=params.ride_height_mm,
        pitch_deg=params.pitch_deg,
    )
    faces = np.asarray(all_faces, dtype=np.int64)
    triangles = points[faces]
    return CadFaceSamples(
        points=points,
        faces=faces,
        face_ids=np.asarray(face_ids, dtype=np.int64),
        centroids=np.mean(triangles, axis=1),
    )


def _source_regions_by_face(
    regions_path: Path, face_count: int
) -> tuple[IntArray, NDArray[np.object_]]:
    regions = json.loads(regions_path.read_text(encoding="utf-8"))
    face_regions = regions["face_regions"]
    if len(face_regions) != face_count:
        message = "surface region face count does not match STL face count"
        raise ValueError(message)
    region_ids = np.asarray([item["region_id"] for item in face_regions], dtype=np.int64)
    region_names = np.asarray([item["region"] for item in face_regions], dtype=object)
    return region_ids, region_names


def _surface_export_payload() -> dict[str, Any]:
    return {
        "surface_export_version": SURFACE_EXPORT_VERSION,
        "surface_diagnostic_version": SURFACE_DIAGNOSTIC_VERSION,
        "exporter": "cadquery.exporters.export",
        "linear_tolerance_m": CAD_TESSELLATION_TOLERANCE_M,
        "angular_tolerance_rad": CAD_TESSELLATION_ANGULAR_TOLERANCE_RAD,
        "relative": "cadquery_default",
        "parallel": "cadquery_default",
    }


def _write_cad_faces_vtp(samples: CadFaceSamples, path: Path) -> None:
    cad_poly = _mesh_polydata(samples.points, samples.faces)
    cad_poly.cell_data["cad_face_id"] = samples.face_ids.astype(np.int32)
    cad_poly.save(path)


def _write_stl_diagnostics(
    *,
    geometry_stl_path: Path,
    regions_json_path: Path,
    cad_samples: CadFaceSamples,
    output_dir: Path,
) -> StlDiagnostics:
    mesh = trimesh.load_mesh(geometry_stl_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    metrics = triangle_metrics(triangles)
    centroids = np.asarray(mesh.triangles_center, dtype=np.float64)
    _, nearest_cad = cKDTree(cad_samples.centroids).query(centroids, k=1)
    nearest_cad_face_id = cad_samples.face_ids[np.asarray(nearest_cad, dtype=np.int64)]
    region_ids, region_names = _source_regions_by_face(regions_json_path, len(mesh.faces))

    low_quality = metrics["triangle_quality"] < LOW_TRIANGLE_QUALITY
    near_zero_quality = metrics["triangle_quality"] < NEAR_ZERO_TRIANGLE_QUALITY
    microscopic_edge = metrics["min_edge_m"] < MICRO_EDGE_LENGTH_M
    min_quality_index = int(np.argmin(metrics["triangle_quality"]))

    stl_poly = _mesh_polydata(np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces))
    for name, values in metrics.items():
        stl_poly.cell_data[name] = values
    stl_poly.cell_data["surface_region_id"] = region_ids.astype(np.int32)
    stl_poly.cell_data["surface_region"] = region_names
    stl_poly.cell_data["nearest_cad_face_id"] = nearest_cad_face_id.astype(np.int32)
    stl_poly.cell_data["quality_lt_0p05"] = low_quality.astype(np.int8)
    stl_poly.cell_data["quality_lt_1e_minus_6"] = near_zero_quality.astype(np.int8)
    stl_poly.cell_data["min_edge_lt_0p05_mm"] = microscopic_edge.astype(np.int8)
    stl_poly.cell_data["is_min_quality_triangle"] = (
        np.arange(len(mesh.faces)) == min_quality_index
    ).astype(np.int8)
    vtp_path = output_dir / "stl_triangle_diagnostics.vtp"
    stl_poly.save(vtp_path)

    bad_rows = _bad_triangle_rows(
        metrics=metrics,
        bad_indices=np.where(low_quality | near_zero_quality | microscopic_edge)[0],
        centroids=centroids,
        region_ids=region_ids,
        region_names=region_names,
        nearest_cad_face_id=nearest_cad_face_id,
    )
    bad_triangles_csv_path = output_dir / "bad_stl_triangles.csv"
    _write_csv(bad_triangles_csv_path, bad_rows)
    return StlDiagnostics(
        mesh=mesh,
        metrics=metrics,
        centroids=centroids,
        region_ids=region_ids,
        region_names=region_names,
        nearest_cad_face_id=nearest_cad_face_id,
        low_quality=low_quality,
        near_zero_quality=near_zero_quality,
        microscopic_edge=microscopic_edge,
        min_quality_index=min_quality_index,
        vtp_path=vtp_path,
        bad_triangles_csv_path=bad_triangles_csv_path,
    )


def _bad_triangle_rows(
    *,
    metrics: dict[str, FloatArray],
    bad_indices: IntArray,
    centroids: FloatArray,
    region_ids: IntArray,
    region_names: NDArray[np.object_],
    nearest_cad_face_id: IntArray,
) -> list[dict[str, Any]]:
    return [
        {
            "triangle_index": int(index),
            "triangle_quality": float(metrics["triangle_quality"][index]),
            "area_m2": float(metrics["area_m2"][index]),
            "min_edge_m": float(metrics["min_edge_m"][index]),
            "max_edge_m": float(metrics["max_edge_m"][index]),
            "aspect_ratio": float(metrics["aspect_ratio"][index]),
            "surface_region": str(region_names[index]),
            "surface_region_id": int(region_ids[index]),
            "nearest_cad_face_id": int(nearest_cad_face_id[index]),
            "centroid_m": [float(value) for value in centroids[index]],
        }
        for index in bad_indices
    ]


def _surface_summary(
    *,
    surface_export_id: str,
    surface_export_payload: dict[str, Any],
    shape: CadShapeLike,
    microscopic_edges: list[dict[str, Any]],
    tiny_faces: list[dict[str, Any]],
    stl: StlDiagnostics,
    source_stl_path: Path,
    source_step_path: Path,
) -> dict[str, Any]:
    triangles = np.asarray(stl.mesh.triangles, dtype=np.float64)
    min_index = stl.min_quality_index
    cad_summary = cad_topology_summary(shape)
    return {
        "geometry_generator_version": GEOMETRY_GENERATOR_VERSION,
        "surface_diagnostic_version": SURFACE_DIAGNOSTIC_VERSION,
        "surface_export_id": surface_export_id,
        "surface_export": surface_export_payload,
        "cad": {
            **cad_summary,
            "min_edge_length_m": float(cad_summary["min_edge"]["length_m"]),
            "min_face_area_m2": float(cad_summary["min_face"]["area_m2"]),
            "microscopic_edge_count": len(microscopic_edges),
            "tiny_face_count": len(tiny_faces),
        },
        "stl": {
            "stl_path": str(source_stl_path),
            "stl_sha256": sha256_file(source_stl_path),
            "step_path": str(source_step_path),
            "step_sha256": sha256_file(source_step_path),
            "face_count": len(stl.mesh.faces),
            "vertex_count": len(stl.mesh.vertices),
            "watertight": bool(stl.mesh.is_watertight),
            "connected_components": len(stl.mesh.split(only_watertight=False)),
            "min_triangle_quality": float(np.min(stl.metrics["triangle_quality"])),
            "triangle_quality_formula": (
                "OpenFOAM triangle::quality area/(0.75*sqrt(3)*circumradius^2)"
            ),
            "low_quality_triangle_count": int(np.count_nonzero(stl.low_quality)),
            "low_quality_triangle_fraction": float(np.mean(stl.low_quality)),
            "near_zero_quality_triangle_count": int(np.count_nonzero(stl.near_zero_quality)),
            "microscopic_edge_triangle_count": int(np.count_nonzero(stl.microscopic_edge)),
            "min_quality_triangle_index": min_index,
            "min_quality_triangle_vertices_m": [
                [float(value) for value in vertex] for vertex in triangles[min_index]
            ],
            "min_quality_triangle_centroid_m": [float(value) for value in stl.centroids[min_index]],
            "min_quality_triangle_nearest_cad_face_id": int(
                stl.nearest_cad_face_id[min_index],
            ),
            "region_names": list(REGION_NAMES),
        },
    }


def diagnose_surface(
    *,
    params: AeroParams,
    attempts_dir: Path,
    parent_attempt_id: str | None = None,
    openfoam_image_digest: str = "unknown",
) -> SurfaceDiagnosticArtifacts:
    """Generate immutable CAD and STL diagnostics for the canonical surface."""

    surface_export_payload = _surface_export_payload()
    surface_export_id = stable_id(
        "surface_export",
        {"geometry_id": params.geometry_id(), **surface_export_payload},
    )
    attempt_id = stable_id(
        "attempt",
        {
            "kind": "surface_diagnostic",
            "surface_diagnostic_version": SURFACE_DIAGNOSTIC_VERSION,
            "geometry_id": params.geometry_id(),
            "surface_export_id": surface_export_id,
            "parent_attempt_id": parent_attempt_id,
        },
    )
    attempt_dir = attempts_dir / attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=True)

    geometry = generate_geometry(params, attempt_dir / "source")
    article = build_article(params)
    shape = cast("CadShapeLike", article.val())
    cq.exporters.export(cast("Any", article), str(attempt_dir / "article_body_datum.step"))

    face_records, edge_records = _cad_face_records(shape)
    microscopic_edges = [
        row for row in edge_records if float(row["length_m"]) < MICRO_EDGE_LENGTH_M
    ]
    tiny_faces = [row for row in face_records if float(row["area_m2"]) < TINY_FACE_AREA_M2]

    cad_faces_vtp_path = attempt_dir / "cad_faces_by_id.vtp"
    cad_samples = cad_face_samples(shape, params)
    _write_cad_faces_vtp(cad_samples, cad_faces_vtp_path)

    stl = _write_stl_diagnostics(
        geometry_stl_path=geometry.stl_path,
        regions_json_path=geometry.regions_json_path,
        cad_samples=cad_samples,
        output_dir=attempt_dir,
    )
    microscopic_edges_csv_path = attempt_dir / "microscopic_cad_edges.csv"
    tiny_faces_csv_path = attempt_dir / "tiny_cad_faces.csv"
    _write_csv(microscopic_edges_csv_path, microscopic_edges)
    _write_csv(tiny_faces_csv_path, tiny_faces)

    summary = _surface_summary(
        surface_export_id=surface_export_id,
        surface_export_payload=surface_export_payload,
        shape=shape,
        microscopic_edges=microscopic_edges,
        tiny_faces=tiny_faces,
        stl=stl,
        source_stl_path=geometry.stl_path,
        source_step_path=geometry.step_path,
    )
    metrics_path = attempt_dir / "surface_diagnostics.json"
    atomic_write_json(metrics_path, summary)

    manifest_path = write_attempt_manifest(
        attempt_dir=attempt_dir,
        attempt_id=attempt_id,
        geometry_id=params.geometry_id(),
        surface_export_id=surface_export_id,
        mesh_config=CfdConfig().mesh.model_dump(),
        openfoam_image_digest=openfoam_image_digest,
        parent_attempt_id=parent_attempt_id,
        configuration_diff={
            "surface": "current_cadquery_export_control",
            "surface_diagnostic_version": SURFACE_DIAGNOSTIC_VERSION,
        },
        status="SURFACE_DIAGNOSED",
        artifacts={
            "cad_faces_vtp": cad_faces_vtp_path,
            "stl_triangles_vtp": stl.vtp_path,
            "surface_metrics_json": metrics_path,
            "bad_triangles_csv": stl.bad_triangles_csv_path,
            "microscopic_edges_csv": microscopic_edges_csv_path,
            "tiny_faces_csv": tiny_faces_csv_path,
            "source_step": geometry.step_path,
            "source_stl": geometry.stl_path,
            "source_regions_json": geometry.regions_json_path,
        },
        metrics_path=metrics_path,
    )
    return SurfaceDiagnosticArtifacts(
        attempt_id=attempt_id,
        attempt_dir=attempt_dir,
        metrics_path=metrics_path,
        attempt_manifest_path=manifest_path,
        cad_faces_vtp_path=cad_faces_vtp_path,
        stl_triangles_vtp_path=stl.vtp_path,
        bad_triangles_csv_path=stl.bad_triangles_csv_path,
        microscopic_edges_csv_path=microscopic_edges_csv_path,
        tiny_faces_csv_path=tiny_faces_csv_path,
    )
