"""Volume-mesh diagnostics for failed OpenFOAM checkMesh sets."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyvista as pv
import trimesh
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from aeromap.attempts import stable_id, write_attempt_manifest
from aeromap.cfd.schema import CfdConfig
from aeromap.geometry.diagnostics import CadShapeLike, cad_face_samples, triangle_metrics
from aeromap.geometry.generator import build_article
from aeromap.io import atomic_write_json, atomic_write_text, sha256_file
from aeromap.parameters import AeroParams

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
MESH_DIAGNOSTIC_VERSION = "mesh_diagnostics_v0.1.0"
DEFAULT_FEATURE_ANGLE_DEG = 30.0
DEFAULT_NEAR_FEATURE_DISTANCE_M = 0.01
CHECKMESH_COUNT_PATTERNS = {
    "concaveFaces": re.compile(r"There are\s+(?P<count>\d+)\s+faces with concave angles"),
    "underdeterminedCells": re.compile(
        r"Cells with small determinant .* number of cells:\s*(?P<count>\d+)",
    ),
    "concaveCells": re.compile(
        r"Concave cells \(using face planes\) found, number of cells:\s*(?P<count>\d+)",
    ),
}


@dataclass(frozen=True)
class MeshDiagnosticArtifacts:
    attempt_id: str
    attempt_dir: Path
    metrics_path: Path
    attempt_manifest_path: Path


@dataclass(frozen=True)
class MeshDiagnosticContext:
    source_tree: cKDTree
    nearest_cad_face_id: IntArray
    region_ids: IntArray
    region_names: NDArray[np.object_]
    source_metrics: dict[str, FloatArray]
    feature_segments: FloatArray
    region_boundary_segments: FloatArray


@dataclass(frozen=True)
class CheckMeshSetDiagnostic:
    summary: dict[str, Any]
    artifacts: dict[str, Path]


@dataclass(frozen=True)
class CellCentreDiagnostic:
    summary: dict[str, Any]
    artifacts: dict[str, Path]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        atomic_write_text(path, "")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _manifest_path(
    case_dir: Path,
    manifest: dict[str, Any],
    manifest_key: str,
    fallback: str,
) -> Path:
    cfd_surface = manifest.get("cfd_surface")
    if isinstance(cfd_surface, dict):
        value = cfd_surface.get(manifest_key)
        if isinstance(value, str):
            return case_dir / value
    return case_dir / fallback


def _source_stl_path(case_dir: Path, manifest: dict[str, Any]) -> Path:
    return _manifest_path(case_dir, manifest, "stl_path", "geometry/article.stl")


def _source_regions_path(case_dir: Path, manifest: dict[str, Any]) -> Path:
    return _manifest_path(case_dir, manifest, "regions_json_path", "geometry/surface_regions.json")


def _source_step_path(case_dir: Path, manifest: dict[str, Any]) -> Path:
    return _manifest_path(
        case_dir, manifest, "source_step_path", "geometry/article_body_datum.step"
    )


def _load_source_mesh(case_dir: Path, manifest: dict[str, Any]) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(_source_stl_path(case_dir, manifest), process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def _source_regions(
    case_dir: Path,
    manifest: dict[str, Any],
    face_count: int,
) -> tuple[IntArray, NDArray[np.object_]]:
    regions_path = _source_regions_path(case_dir, manifest)
    regions = json.loads(regions_path.read_text(encoding="utf-8"))
    face_regions = regions["face_regions"]
    if len(face_regions) != face_count:
        message = "surface region face count does not match source mesh face count"
        raise ValueError(message)
    return (
        np.asarray([item["region_id"] for item in face_regions], dtype=np.int64),
        np.asarray([item["region"] for item in face_regions], dtype=object),
    )


def _feature_segments(mesh: trimesh.Trimesh, *, feature_angle_deg: float) -> FloatArray:
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    if len(adjacency) == 0:
        return np.empty((0, 2, 3), dtype=np.float64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    dot = np.einsum("ij,ij->i", normals[adjacency[:, 0]], normals[adjacency[:, 1]])
    dot = np.clip(dot, -1.0, 1.0)
    angles = np.rad2deg(np.arccos(dot))
    feature_mask = angles >= feature_angle_deg
    edge_vertex_ids = np.asarray(mesh.face_adjacency_edges, dtype=np.int64)[feature_mask]
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    return np.asarray(vertices[edge_vertex_ids], dtype=np.float64)


def _region_boundary_segments(mesh: trimesh.Trimesh, region_ids: IntArray) -> FloatArray:
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    if len(adjacency) == 0:
        return np.empty((0, 2, 3), dtype=np.float64)
    boundary_mask = region_ids[adjacency[:, 0]] != region_ids[adjacency[:, 1]]
    edge_vertex_ids = np.asarray(mesh.face_adjacency_edges, dtype=np.int64)[boundary_mask]
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    return np.asarray(vertices[edge_vertex_ids], dtype=np.float64)


def _min_distance_to_segments(points: FloatArray, segments: FloatArray) -> FloatArray:
    if len(segments) == 0:
        return np.full(len(points), np.inf, dtype=np.float64)
    best = np.full(len(points), np.inf, dtype=np.float64)
    for start, end in segments:
        direction = end - start
        length_sq = float(np.dot(direction, direction))
        if length_sq == 0.0:
            distance = np.linalg.norm(points - start, axis=1)
        else:
            t = np.clip(((points - start) @ direction) / length_sq, 0.0, 1.0)
            closest = start + t[:, None] * direction
            distance = np.linalg.norm(points - closest, axis=1)
        best = np.minimum(best, distance)
    return best


def _params_from_case(case_dir: Path) -> AeroParams:
    params = json.loads((case_dir / "params.json").read_text(encoding="utf-8"))
    return AeroParams(**params)


def _checkmesh_set_paths(case_dir: Path) -> list[Path]:
    root = case_dir / "openfoam" / "postProcessing" / "checkMesh" / "constant"
    if not root.exists():
        return []
    return sorted(root.glob("*Cells.vtk")) + sorted(root.glob("*Faces.vtk"))


def _reported_checkmesh_counts(case_dir: Path) -> dict[str, int]:
    log_path = case_dir / "logs" / "checkMesh.log"
    if not log_path.exists():
        return {}
    log = log_path.read_text(encoding="utf-8", errors="replace")
    counts: dict[str, int] = {}
    for set_name, pattern in CHECKMESH_COUNT_PATTERNS.items():
        match = pattern.search(log)
        if match is not None:
            counts[set_name] = int(match.group("count"))
    return counts


def _optional_sha256(path: Path) -> str | None:
    return sha256_file(path) if path.exists() else None


def _mesh_input_hashes(case_dir: Path) -> dict[str, str | None]:
    return {
        "system/snappyHexMeshDict": _optional_sha256(
            case_dir / "openfoam" / "system" / "snappyHexMeshDict",
        ),
        "system/blockMeshDict": _optional_sha256(
            case_dir / "openfoam" / "system" / "blockMeshDict",
        ),
        "system/surfaceFeaturesDict": _optional_sha256(
            case_dir / "openfoam" / "system" / "surfaceFeaturesDict",
        ),
        "logs/checkMesh.log": _optional_sha256(case_dir / "logs" / "checkMesh.log"),
        "logs/snappyHexMesh.log": _optional_sha256(case_dir / "logs" / "snappyHexMesh.log"),
    }


def _read_openfoam_label_list(path: Path) -> list[int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    compact = re.search(r"(?m)^\s*(?P<count>\d+)\s*\{\s*(?P<value>-?\d+)\s*\}\s*$", text)
    if compact is not None:
        return [int(compact.group("value"))] * int(compact.group("count"))
    lines = text.splitlines()
    start: int | None = None
    count: int | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.isdigit() and index + 1 < len(lines) and lines[index + 1].strip() == "(":
            count = int(stripped)
            start = index + 2
            break
    if start is None or count is None:
        message = f"OpenFOAM labelList block not found: {path}"
        raise ValueError(message)
    values: list[int] = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped == ")":
            break
        if stripped:
            values.append(int(stripped))
    if len(values) != count:
        message = f"OpenFOAM labelList count mismatch in {path}: {len(values)} != {count}"
        raise ValueError(message)
    return values


def _read_openfoam_internal_vectors(path: Path) -> FloatArray:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start: int | None = None
    count: int | None = None
    for index, line in enumerate(lines):
        if line.strip().startswith("internalField") and "nonuniform List<vector>" in line:
            count = int(lines[index + 1].strip())
            start = index + 3
            break
    if start is None or count is None:
        message = f"OpenFOAM nonuniform internal vector field not found: {path}"
        raise ValueError(message)
    values = [
        [float(component) for component in line.strip().strip("()").split()]
        for line in lines[start : start + count]
    ]
    return np.asarray(values, dtype=np.float64)


def _cell_centre_field_path(case_dir: Path) -> Path | None:
    for relative in ["openfoam/constant/C", "openfoam/0/C"]:
        candidate = case_dir / relative
        if candidate.exists():
            return candidate
    return None


def _bounds(points: FloatArray) -> dict[str, list[float] | None]:
    if len(points) == 0:
        return {"min_m": None, "max_m": None, "mean_m": None}
    return {
        "min_m": [float(value) for value in np.min(points, axis=0)],
        "max_m": [float(value) for value in np.max(points, axis=0)],
        "mean_m": [float(value) for value in np.mean(points, axis=0)],
    }


def _optional_cell_levels(case_dir: Path, cell_ids: list[int]) -> dict[str, int] | None:
    levels_path = case_dir / "openfoam" / "constant" / "polyMesh" / "cellLevel"
    if not levels_path.exists():
        return None
    levels = _read_openfoam_label_list(levels_path)
    counts: dict[str, int] = {}
    for cell_id in cell_ids:
        value = str(levels[cell_id])
        counts[value] = counts.get(value, 0) + 1
    return counts


def _diagnose_cell_centres(
    *,
    set_name: str,
    case_dir: Path,
    attempt_dir: Path,
    context: MeshDiagnosticContext,
) -> CellCentreDiagnostic | None:
    cell_set_path = case_dir / "openfoam" / "constant" / "polyMesh" / "sets" / set_name
    centres_path = _cell_centre_field_path(case_dir)
    if not cell_set_path.exists() or centres_path is None:
        return None

    cell_ids = _read_openfoam_label_list(cell_set_path)
    all_centres = _read_openfoam_internal_vectors(centres_path)
    selected = all_centres[np.asarray(cell_ids, dtype=np.int64)]
    nearest_distance, nearest_triangle = context.source_tree.query(selected, k=1)
    nearest_triangle = np.asarray(nearest_triangle, dtype=np.int64)
    nearest_distance = np.asarray(nearest_distance, dtype=np.float64)
    rows = [
        {
            "set_name": set_name,
            "cell_id": int(cell_id),
            "x_m": float(point[0]),
            "y_m": float(point[1]),
            "z_m": float(point[2]),
            "nearest_stl_triangle": int(triangle_id),
            "nearest_cad_face_id": int(context.nearest_cad_face_id[triangle_id]),
            "surface_region_id": int(context.region_ids[triangle_id]),
            "surface_region": str(context.region_names[triangle_id]),
            "distance_to_surface_m": float(nearest_distance[index]),
            "source_triangle_quality": float(
                context.source_metrics["triangle_quality"][triangle_id],
            ),
        }
        for index, (cell_id, point, triangle_id) in enumerate(
            zip(cell_ids, selected, nearest_triangle, strict=True),
        )
    ]
    csv_path = attempt_dir / f"{set_name}_cell_centres_mapped.csv"
    _write_csv(csv_path, rows)

    poly = pv.PolyData(selected)
    poly.point_data["openfoam_cell_id"] = np.asarray(cell_ids, dtype=np.int32)
    poly.point_data["nearest_stl_triangle"] = nearest_triangle.astype(np.int32)
    poly.point_data["nearest_cad_face_id"] = context.nearest_cad_face_id[nearest_triangle].astype(
        np.int32
    )
    poly.point_data["surface_region_id"] = context.region_ids[nearest_triangle].astype(np.int32)
    poly.point_data["distance_to_surface_m"] = nearest_distance
    vtp_path = attempt_dir / f"{set_name}_cell_centres_mapped.vtp"
    poly.save(vtp_path)

    region_counts = _count_by_key(rows, "surface_region")
    cad_face_counts = _count_by_key(rows, "nearest_cad_face_id")
    summary = {
        "set_name": set_name,
        "cell_count": len(cell_ids),
        "cell_centre_field": str(centres_path),
        "bounds_m": _bounds(selected),
        "max_distance_to_surface_m": float(np.max(nearest_distance)),
        "mean_distance_to_surface_m": float(np.mean(nearest_distance)),
        "region_counts": region_counts,
        "cad_face_counts": cad_face_counts,
        "cell_level_counts": _optional_cell_levels(case_dir, cell_ids),
        "worst_source_triangle_quality": float(
            np.min(context.source_metrics["triangle_quality"][nearest_triangle]),
        ),
    }
    return CellCentreDiagnostic(
        summary=summary,
        artifacts={
            f"{set_name}_cell_centres_mapped_csv": csv_path,
            f"{set_name}_cell_centres_mapped_vtp": vtp_path,
        },
    )


def _diagnostic_context(
    *,
    case_dir: Path,
    params: AeroParams,
    manifest: dict[str, Any],
) -> MeshDiagnosticContext:
    source_mesh = _load_source_mesh(case_dir, manifest)
    region_ids, region_names = _source_regions(case_dir, manifest, len(source_mesh.faces))
    source_centres = np.asarray(source_mesh.triangles_center, dtype=np.float64)
    shape = cast("CadShapeLike", build_article(params).val())
    cad_samples = cad_face_samples(shape, params)
    _, nearest_cad = cKDTree(cad_samples.centroids).query(source_centres, k=1)
    nearest_cad_face_id = cad_samples.face_ids[np.asarray(nearest_cad, dtype=np.int64)]
    feature_angle = float(manifest["cfd_config"]["mesh"]["feature_angle_deg"])
    return MeshDiagnosticContext(
        source_tree=cKDTree(source_centres),
        nearest_cad_face_id=nearest_cad_face_id,
        region_ids=region_ids,
        region_names=region_names,
        source_metrics=triangle_metrics(np.asarray(source_mesh.triangles, dtype=np.float64)),
        feature_segments=_feature_segments(source_mesh, feature_angle_deg=feature_angle),
        region_boundary_segments=_region_boundary_segments(source_mesh, region_ids),
    )


def _mapped_rows(
    *,
    set_name: str,
    centres: FloatArray,
    nearest_triangle: IntArray,
    nearest_distance: FloatArray,
    nearest_cad_face_id: IntArray,
    region_ids: IntArray,
    region_names: NDArray[np.object_],
    feature_distance: FloatArray,
    region_boundary_distance: FloatArray,
    near_feature_distance_m: float,
    target_cell_width_m: float,
) -> list[dict[str, Any]]:
    return [
        {
            "set_name": set_name,
            "diagnostic_element_index": int(index),
            "x_m": float(point[0]),
            "y_m": float(point[1]),
            "z_m": float(point[2]),
            "nearest_stl_triangle": int(triangle_id),
            "nearest_cad_face_id": int(nearest_cad_face_id[triangle_id]),
            "surface_region_id": int(region_ids[triangle_id]),
            "surface_region": str(region_names[triangle_id]),
            "distance_to_surface_m": float(nearest_distance[index]),
            "distance_to_feature_edge_m": float(feature_distance[index]),
            "distance_to_region_boundary_m": float(region_boundary_distance[index]),
            "near_feature_edge": bool(feature_distance[index] <= near_feature_distance_m),
            "within_one_target_cell_width": bool(
                min(feature_distance[index], region_boundary_distance[index])
                <= target_cell_width_m,
            ),
            "within_two_target_cell_widths": bool(
                min(feature_distance[index], region_boundary_distance[index])
                <= 2.0 * target_cell_width_m,
            ),
        }
        for index, (point, triangle_id) in enumerate(zip(centres, nearest_triangle, strict=True))
    ]


def _count_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        counts[value] = counts.get(value, 0) + 1
    return counts


def _diagnose_checkmesh_set(
    *,
    set_path: Path,
    case_dir: Path,
    attempt_dir: Path,
    context: MeshDiagnosticContext,
    near_feature_distance_m: float,
    target_cell_width_m: float,
    openfoam_reported_count: int | None,
) -> CheckMeshSetDiagnostic:
    set_mesh = pv.read(set_path)
    centres = np.asarray(set_mesh.cell_centers().points, dtype=np.float64)
    if len(centres) == 0:
        centres = np.asarray(set_mesh.points, dtype=np.float64)
    nearest_distance, nearest_triangle = context.source_tree.query(centres, k=1)
    nearest_triangle = np.asarray(nearest_triangle, dtype=np.int64)
    nearest_distance = np.asarray(nearest_distance, dtype=np.float64)
    feature_distance = _min_distance_to_segments(centres, context.feature_segments)
    region_boundary_distance = _min_distance_to_segments(
        centres,
        context.region_boundary_segments,
    )

    rows = _mapped_rows(
        set_name=set_path.stem,
        centres=centres,
        nearest_triangle=nearest_triangle,
        nearest_distance=nearest_distance,
        nearest_cad_face_id=context.nearest_cad_face_id,
        region_ids=context.region_ids,
        region_names=context.region_names,
        feature_distance=feature_distance,
        region_boundary_distance=region_boundary_distance,
        near_feature_distance_m=near_feature_distance_m,
        target_cell_width_m=target_cell_width_m,
    )
    csv_path = attempt_dir / f"{set_path.stem}_mapped.csv"
    _write_csv(csv_path, rows)

    mapped = set_mesh.copy(deep=True)
    mapped.cell_data["nearest_stl_triangle"] = nearest_triangle.astype(np.int32)
    mapped.cell_data["nearest_cad_face_id"] = context.nearest_cad_face_id[nearest_triangle].astype(
        np.int32
    )
    mapped.cell_data["surface_region_id"] = context.region_ids[nearest_triangle].astype(np.int32)
    mapped.cell_data["distance_to_surface_m"] = nearest_distance
    mapped.cell_data["distance_to_feature_edge_m"] = feature_distance
    mapped.cell_data["distance_to_region_boundary_m"] = region_boundary_distance
    mapped.cell_data["near_feature_edge"] = (feature_distance <= near_feature_distance_m).astype(
        np.int8,
    )
    nearest_feature_or_region = np.minimum(feature_distance, region_boundary_distance)
    mapped.cell_data["within_one_target_cell_width"] = (
        nearest_feature_or_region <= target_cell_width_m
    ).astype(np.int8)
    mapped.cell_data["within_two_target_cell_widths"] = (
        nearest_feature_or_region <= 2.0 * target_cell_width_m
    ).astype(np.int8)
    mapped_path = attempt_dir / f"{set_path.stem}_mapped.vtk"
    mapped.save(mapped_path)
    cell_centre_diagnostic = (
        _diagnose_cell_centres(
            set_name=set_path.stem,
            case_dir=case_dir,
            attempt_dir=attempt_dir,
            context=context,
        )
        if set_path.stem.endswith("Cells")
        else None
    )

    return CheckMeshSetDiagnostic(
        summary={
            "set_name": set_path.stem,
            "source_vtk": str(set_path),
            "openfoam_reported_problem_count": openfoam_reported_count,
            "diagnostic_vtk_element_count": len(rows),
            "max_distance_to_surface_m": float(np.max(nearest_distance)),
            "mean_distance_to_surface_m": float(np.mean(nearest_distance)),
            "near_feature_count": int(
                np.count_nonzero(feature_distance <= near_feature_distance_m),
            ),
            "target_cell_width_m": target_cell_width_m,
            "within_one_target_cell_width_count": int(
                np.count_nonzero(nearest_feature_or_region <= target_cell_width_m),
            ),
            "within_two_target_cell_widths_count": int(
                np.count_nonzero(nearest_feature_or_region <= 2.0 * target_cell_width_m),
            ),
            "region_counts": _count_by_key(rows, "surface_region"),
            "cad_face_counts": _count_by_key(rows, "nearest_cad_face_id"),
            "worst_source_triangle_quality": float(
                np.min(context.source_metrics["triangle_quality"][nearest_triangle]),
            ),
            "cell_centre_summary": (
                cell_centre_diagnostic.summary if cell_centre_diagnostic is not None else None
            ),
        },
        artifacts={
            f"{set_path.stem}_mapped_csv": csv_path,
            f"{set_path.stem}_mapped_vtk": mapped_path,
            **(cell_centre_diagnostic.artifacts if cell_centre_diagnostic is not None else {}),
        },
    )


def diagnose_mesh(
    *,
    case_dir: Path,
    attempts_dir: Path,
    parent_attempt_id: str | None = None,
    openfoam_image_digest: str = "unknown",
    near_feature_distance_m: float = DEFAULT_NEAR_FEATURE_DISTANCE_M,
    target_cell_width_m: float = 0.012,
) -> MeshDiagnosticArtifacts:
    """Map OpenFOAM checkMesh problem sets back to STL triangles and CAD faces."""

    manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    params = _params_from_case(case_dir)
    context = _diagnostic_context(case_dir=case_dir, params=params, manifest=manifest)

    surface_export_id = manifest.get("surface_export_id")
    if not isinstance(surface_export_id, str):
        surface_export_id = stable_id(
            "surface_export",
            {
                "geometry_id": manifest["geometry_id"],
                "source_stl_sha256": sha256_file(_source_stl_path(case_dir, manifest)),
            },
        )
    mesh_input_hashes = _mesh_input_hashes(case_dir)
    attempt_id = stable_id(
        "attempt",
        {
            "kind": "mesh_diagnostic",
            "mesh_diagnostic_version": MESH_DIAGNOSTIC_VERSION,
            "simulation_id": manifest["simulation_id"],
            "surface_export_id": surface_export_id,
            "mesh_input_hashes": mesh_input_hashes,
            "parent_attempt_id": parent_attempt_id,
        },
    )
    attempt_dir = attempts_dir / attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=True)

    set_paths = _checkmesh_set_paths(case_dir)
    if not set_paths:
        message = (
            "No checkMesh VTK sets found. Run checkMesh with "
            "-writeSurfaces -writeSets -surfaceFormat vtk -setFormat vtk first."
        )
        raise FileNotFoundError(message)

    summaries: list[dict[str, Any]] = []
    mapped_artifacts: dict[str, Path] = {}
    reported_counts = _reported_checkmesh_counts(case_dir)
    for set_path in set_paths:
        diagnostic = _diagnose_checkmesh_set(
            set_path=set_path,
            case_dir=case_dir,
            attempt_dir=attempt_dir,
            context=context,
            near_feature_distance_m=near_feature_distance_m,
            target_cell_width_m=target_cell_width_m,
            openfoam_reported_count=reported_counts.get(set_path.stem),
        )
        summaries.append(diagnostic.summary)
        mapped_artifacts.update(diagnostic.artifacts)

    summary: dict[str, Any] = {
        "case_dir": str(case_dir),
        "simulation_id": manifest["simulation_id"],
        "geometry_id": manifest["geometry_id"],
        "surface_export_id": surface_export_id,
        "mesh_diagnostic_version": MESH_DIAGNOSTIC_VERSION,
        "mesh_input_hashes": mesh_input_hashes,
        "near_feature_distance_m": near_feature_distance_m,
        "target_cell_width_m": target_cell_width_m,
        "openfoam_reported_problem_counts": reported_counts,
        "checkmesh_sets": summaries,
    }
    metrics_path = attempt_dir / "mesh_diagnostics.json"
    atomic_write_json(metrics_path, summary)

    artifact_paths = {
        "mesh_metrics_json": metrics_path,
        "source_stl": _source_stl_path(case_dir, manifest),
        "source_step": _source_step_path(case_dir, manifest),
        "surface_regions_json": _source_regions_path(case_dir, manifest),
        "surface_check_log": case_dir / "logs" / "surfaceCheck.log",
        "snappy_log": case_dir / "logs" / "snappyHexMesh.log",
        "check_mesh_log": case_dir / "logs" / "checkMesh.log",
        **mapped_artifacts,
    }
    manifest_path = write_attempt_manifest(
        attempt_dir=attempt_dir,
        attempt_id=attempt_id,
        geometry_id=manifest["geometry_id"],
        surface_export_id=surface_export_id,
        mesh_config=CfdConfig(**manifest["cfd_config"]).mesh.model_dump(),
        openfoam_image_digest=openfoam_image_digest,
        parent_attempt_id=parent_attempt_id,
        configuration_diff={
            "mesh": "diagnose_existing_checkmesh_problem_sets",
            "mesh_diagnostic_version": MESH_DIAGNOSTIC_VERSION,
            "mesh_input_hashes": mesh_input_hashes,
        },
        status="MESH_DIAGNOSED_FAILED_CHECKMESH",
        artifacts=artifact_paths,
        metrics_path=metrics_path,
    )
    return MeshDiagnosticArtifacts(
        attempt_id=attempt_id,
        attempt_dir=attempt_dir,
        metrics_path=metrics_path,
        attempt_manifest_path=manifest_path,
    )
