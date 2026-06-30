"""Controlled surface-export candidates for mesh diagnostic work."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cadquery as cq
import numpy as np
import pyvista as pv
import trimesh
from scipy.spatial import cKDTree

from aeromap.attempts import stable_id
from aeromap.geometry.diagnostics import CadShapeLike, cad_face_samples, triangle_metrics
from aeromap.geometry.generator import build_article
from aeromap.geometry.regions import REGION_NAMES, classify_surface_regions, write_surface_regions
from aeromap.geometry.validate import validate_mesh
from aeromap.io import atomic_write_json, sha256_file
from aeromap.parameters import AeroParams
from aeromap.transforms import apply_ride_height_pitch

CADQUERY_FIXED_TOLERANCE_MATRIX = (
    (0.0005, 0.05),
    (0.0005, 0.10),
    (0.0010, 0.05),
    (0.0010, 0.10),
    (0.0020, 0.05),
    (0.0020, 0.10),
)
SURFACE_CANDIDATE_SCHEMA_VERSION = "surface_candidate_matrix_v0.1.0"
LOW_TRIANGLE_QUALITY = 0.05
NEAR_ZERO_TRIANGLE_QUALITY = 1e-6
MICROSCOPIC_EDGE_M = 5e-5
TRIANGLE_VERTEX_COUNT = 3
GMSH_MESH_SIZE_MIN_M = 0.001
GMSH_MESH_SIZE_MAX_M = 0.012
GMSH_QUALITY_BINS = (0.0, 1e-6, 0.01, 0.05, 0.10, 0.20, 0.50, 1.01)
REFERENCE_DEVIATION_METHOD = (
    "absolute_implicit_distance_from_candidate_face_centres_to_current_cadquery_surface"
)
GMSH_SURFACE_VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "variant": "g0_no_healing",
        "healing_policy": "disabled",
        "occ_options": {
            "Geometry.OCCAutoFix": 0,
            "Geometry.OCCFixDegenerated": 0,
            "Geometry.OCCFixSmallEdges": 0,
            "Geometry.OCCFixSmallFaces": 0,
            "Geometry.OCCSewFaces": 0,
            "Geometry.OCCMakeSolids": 0,
        },
    },
    {
        "variant": "g1_conservative_autofix",
        "healing_policy": "orientation_autofix_only",
        "occ_options": {
            "Geometry.OCCAutoFix": 1,
            "Geometry.OCCFixDegenerated": 0,
            "Geometry.OCCFixSmallEdges": 0,
            "Geometry.OCCFixSmallFaces": 0,
            "Geometry.OCCSewFaces": 0,
            "Geometry.OCCMakeSolids": 0,
        },
    },
)


@dataclass(frozen=True)
class SurfaceCandidateResult:
    candidate_id: str
    candidate_dir: Path
    status: str
    stl_path: Path | None
    metrics_path: Path
    regions_json_path: Path | None = None
    regions_vtp_path: Path | None = None


@dataclass(frozen=True)
class SurfaceCandidateMatrix:
    out_dir: Path
    manifest_path: Path
    candidates: tuple[SurfaceCandidateResult, ...]


def surface_candidate_id(payload: dict[str, Any]) -> str:
    return stable_id("surface_candidate", payload, length=12)


def _candidate_name(payload: dict[str, Any]) -> str:
    kind = str(payload["kind"])
    if kind == "cadquery_fixed_tolerance":
        linear_mm = 1000.0 * float(payload["linear_tolerance_m"])
        angular = float(payload["angular_tolerance_rad"])
        return f"cadquery_fixed_tol_{linear_mm:g}mm_{angular:g}rad".replace(".", "p")
    if kind == "gmsh_occ_surface_remesh":
        return f"gmsh_occ_{payload['variant']}"
    return kind


def _load_mesh(stl_path: Path, *, process: bool) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(stl_path, process=process)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def _load_exported_mesh(stl_path: Path) -> trimesh.Trimesh:
    mesh = _load_mesh(stl_path, process=False)
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def _quality_histogram(values: np.ndarray) -> list[dict[str, Any]]:
    counts, edges = np.histogram(values, bins=np.asarray(GMSH_QUALITY_BINS, dtype=np.float64))
    return [
        {
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "count": int(count),
        }
        for index, count in enumerate(counts)
    ]


def _region_coverage(body_mesh: trimesh.Trimesh, params: AeroParams) -> dict[str, Any]:
    regions = classify_surface_regions(body_mesh, params)
    total = len(regions["face_regions"])
    counts = regions["counts"]
    return {
        "classification_frame": "body_local",
        "counts": counts,
        "fractions": {name: counts[name] / total if total else 0.0 for name in REGION_NAMES},
        "missing_regions": [name for name in REGION_NAMES if counts[name] == 0],
    }


def _cad_sample_deviation(mesh: trimesh.Trimesh, cad_samples_points: np.ndarray) -> dict[str, Any]:
    candidate_centroids = np.asarray(mesh.triangles_center, dtype=np.float64)
    if len(candidate_centroids) == 0 or len(cad_samples_points) == 0:
        return {
            "method": "nearest_neighbour_between_candidate_centroids_and_cad_face_samples",
            "candidate_to_cad": None,
            "cad_to_candidate": None,
        }

    cad_tree = cKDTree(cad_samples_points)
    candidate_tree = cKDTree(candidate_centroids)
    candidate_to_cad = np.asarray(cad_tree.query(candidate_centroids, k=1)[0], dtype=np.float64)
    cad_to_candidate = np.asarray(
        candidate_tree.query(cad_samples_points, k=1)[0], dtype=np.float64
    )

    def stats(values: np.ndarray) -> dict[str, float]:
        return {
            "mean_m": float(np.mean(values)),
            "p95_m": float(np.quantile(values, 0.95)),
            "max_m": float(np.max(values)),
        }

    return {
        "method": "nearest_neighbour_between_candidate_centroids_and_cad_face_samples",
        "candidate_to_cad": stats(candidate_to_cad),
        "cad_to_candidate": stats(cad_to_candidate),
    }


def _reference_surface_deviation(
    mesh: trimesh.Trimesh,
    reference_mesh: trimesh.Trimesh,
) -> dict[str, Any]:
    reference_faces = np.asarray(reference_mesh.faces, dtype=np.int64)
    reference_cells = np.column_stack(
        [np.full(len(reference_faces), TRIANGLE_VERTEX_COUNT, dtype=np.int64), reference_faces],
    ).ravel()
    reference_surface = pv.PolyData(np.asarray(reference_mesh.vertices), reference_cells)
    query_points = pv.PolyData(np.asarray(mesh.triangles_center, dtype=np.float64))
    distances = np.abs(
        query_points.compute_implicit_distance(reference_surface, inplace=False).point_data[
            "implicit_distance"
        ],
    )
    return {
        "method": REFERENCE_DEVIATION_METHOD,
        "mean_m": float(np.mean(distances)),
        "p95_m": float(np.quantile(distances, 0.95)),
        "max_m": float(np.max(distances)),
    }


def _surface_metrics(
    *,
    mesh: trimesh.Trimesh,
    body_mesh: trimesh.Trimesh,
    stl_path: Path,
    params: AeroParams,
    cad_sample_centroids: np.ndarray,
    reference_mesh: trimesh.Trimesh,
) -> dict[str, Any]:
    raw_mesh = _load_mesh(stl_path, process=False)
    raw_metrics = triangle_metrics(np.asarray(raw_mesh.triangles, dtype=np.float64))
    metrics = triangle_metrics(np.asarray(mesh.triangles, dtype=np.float64))
    validation = validate_mesh(mesh)
    low_quality = metrics["triangle_quality"] < LOW_TRIANGLE_QUALITY
    near_zero_quality = metrics["triangle_quality"] < NEAR_ZERO_TRIANGLE_QUALITY
    microscopic_edge = metrics["min_edge_m"] < MICROSCOPIC_EDGE_M
    return {
        "stl_path": str(stl_path),
        "stl_sha256": sha256_file(stl_path),
        "raw_export_metrics": {
            "face_count": len(raw_mesh.faces),
            "vertex_count": len(raw_mesh.vertices),
            "min_triangle_quality": float(np.min(raw_metrics["triangle_quality"])),
            "low_quality_triangle_count": int(
                np.count_nonzero(raw_metrics["triangle_quality"] < LOW_TRIANGLE_QUALITY),
            ),
            "near_zero_quality_triangle_count": int(
                np.count_nonzero(raw_metrics["triangle_quality"] < NEAR_ZERO_TRIANGLE_QUALITY),
            ),
            "min_triangle_area_m2": float(np.min(raw_metrics["area_m2"])),
            "min_edge_m": float(np.min(raw_metrics["min_edge_m"])),
            "max_aspect_ratio": float(np.max(raw_metrics["aspect_ratio"])),
        },
        "cleaned_mesh_delta": {
            "removed_faces": int(len(raw_mesh.faces) - len(mesh.faces)),
            "removed_vertices": int(len(raw_mesh.vertices) - len(mesh.vertices)),
        },
        "validation": validation.model_dump(mode="json"),
        "face_count": len(mesh.faces),
        "vertex_count": len(mesh.vertices),
        "watertight": bool(mesh.is_watertight),
        "connected_components": len(mesh.split(only_watertight=False)),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "min_triangle_quality": float(np.min(metrics["triangle_quality"])),
        "quality_histogram": _quality_histogram(metrics["triangle_quality"]),
        "low_quality_triangle_count": int(np.count_nonzero(low_quality)),
        "low_quality_triangle_fraction": float(np.mean(low_quality)),
        "near_zero_quality_triangle_count": int(np.count_nonzero(near_zero_quality)),
        "microscopic_edge_triangle_count": int(np.count_nonzero(microscopic_edge)),
        "min_triangle_area_m2": float(np.min(metrics["area_m2"])),
        "min_edge_m": float(np.min(metrics["min_edge_m"])),
        "max_aspect_ratio": float(np.max(metrics["aspect_ratio"])),
        "surface_region_coverage": _region_coverage(body_mesh, params),
        "cad_sample_deviation": _cad_sample_deviation(mesh, cad_sample_centroids),
        "reference_surface_deviation": _reference_surface_deviation(mesh, reference_mesh),
    }


def _write_candidate_region_sidecars(
    *,
    mesh: trimesh.Trimesh,
    classification_mesh: trimesh.Trimesh,
    params: AeroParams,
    candidate_dir: Path,
) -> dict[str, Any]:
    regions_json_path = candidate_dir / "surface_regions.json"
    regions_vtp_path = candidate_dir / "surface_regions.vtp"
    write_surface_regions(
        mesh,
        params,
        regions_json_path,
        regions_vtp_path,
        classification_mesh=classification_mesh,
    )
    return {
        "surface_regions_json_path": str(regions_json_path),
        "surface_regions_json_sha256": sha256_file(regions_json_path),
        "surface_regions_vtp_path": str(regions_vtp_path),
        "surface_regions_vtp_sha256": sha256_file(regions_vtp_path),
    }


def _write_cadquery_candidate(
    *,
    params: AeroParams,
    out_dir: Path,
    payload: dict[str, Any],
    cad_sample_centroids: np.ndarray,
    reference_mesh: trimesh.Trimesh | None = None,
) -> SurfaceCandidateResult:
    candidate_id = surface_candidate_id(payload)
    candidate_dir = out_dir / f"{_candidate_name(payload)}__{candidate_id}"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    body_stl_path = candidate_dir / "article_body_datum.stl"
    stl_path = candidate_dir / "article.stl"

    shape = build_article(params).val()
    ok = cast("Any", shape).exportStl(
        str(body_stl_path),
        tolerance=float(payload["linear_tolerance_m"]),
        angularTolerance=float(payload["angular_tolerance_rad"]),
        ascii=False,
        relative=bool(payload["relative"]),
        parallel=bool(payload["parallel"]),
    )
    if not ok:
        message = f"CadQuery exportStl returned false for {candidate_id}"
        raise RuntimeError(message)

    body_mesh = _load_exported_mesh(body_stl_path)
    mesh = body_mesh.copy()
    mesh.vertices = apply_ride_height_pitch(
        np.asarray(mesh.vertices, dtype=np.float64),
        ride_height_mm=params.ride_height_mm,
        pitch_deg=params.pitch_deg,
    )
    mesh.fix_normals()
    mesh.export(stl_path, file_type="stl")
    body_stl_path.unlink(missing_ok=True)
    reference_mesh = reference_mesh or mesh
    region_artifacts = _write_candidate_region_sidecars(
        mesh=mesh,
        classification_mesh=body_mesh,
        params=params,
        candidate_dir=candidate_dir,
    )

    metrics = {
        "schema_version": SURFACE_CANDIDATE_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "status": "EXPORTED",
        "export_payload": payload,
        "surface_region_artifacts": region_artifacts,
        "surface_metrics": _surface_metrics(
            mesh=mesh,
            body_mesh=body_mesh,
            stl_path=stl_path,
            params=params,
            cad_sample_centroids=cad_sample_centroids,
            reference_mesh=reference_mesh,
        ),
    }
    metrics_path = candidate_dir / "surface_candidate_metrics.json"
    atomic_write_json(metrics_path, metrics)
    return SurfaceCandidateResult(
        candidate_id=candidate_id,
        candidate_dir=candidate_dir,
        status="EXPORTED",
        stl_path=stl_path,
        metrics_path=metrics_path,
        regions_json_path=candidate_dir / "surface_regions.json",
        regions_vtp_path=candidate_dir / "surface_regions.vtp",
    )


def _gmsh_executable(explicit_path: Path | None) -> Path | None:
    if explicit_path is not None:
        resolved = explicit_path.expanduser()
        return resolved if resolved.is_file() and os.access(resolved, os.X_OK) else None
    discovered = shutil.which("gmsh")
    return Path(discovered) if discovered else None


def _gmsh_version(gmsh_path: Path) -> str:
    try:
        completed = subprocess.run(  # noqa: S603
            [str(gmsh_path), "-version"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return f"unavailable: {exc}"
    return completed.stdout.strip() or "unknown"


def _cadquery_reference_mesh(params: AeroParams, out_dir: Path) -> trimesh.Trimesh:
    body_stl_path = out_dir / "_cadquery_reference_body.stl"
    shape = build_article(params).val()
    ok = cast("Any", shape).exportStl(
        str(body_stl_path),
        tolerance=0.001,
        angularTolerance=0.05,
        ascii=False,
        relative=True,
        parallel=True,
    )
    if not ok:
        message = "CadQuery reference exportStl returned false"
        raise RuntimeError(message)
    body_mesh = _load_exported_mesh(body_stl_path)
    body_stl_path.unlink(missing_ok=True)
    mesh = body_mesh.copy()
    mesh.vertices = apply_ride_height_pitch(
        np.asarray(mesh.vertices, dtype=np.float64),
        ride_height_mm=params.ride_height_mm,
        pitch_deg=params.pitch_deg,
    )
    mesh.fix_normals()
    return mesh


def write_gmsh_g0_surface_export(
    *,
    params: AeroParams,
    out_dir: Path,
    gmsh_path: Path | None = None,
    mesh_size_min_m: float = GMSH_MESH_SIZE_MIN_M,
    mesh_size_max_m: float = GMSH_MESH_SIZE_MAX_M,
    mesh_algorithm: str = "front2d",
    mesh_optimize: int = 1,
) -> SurfaceCandidateResult:
    """Export the bounded Gmsh G0 surface candidate for a CFD case."""

    executable = _gmsh_executable(gmsh_path)
    if executable is None:
        requested = str(gmsh_path) if gmsh_path is not None else "PATH"
        message = f"gmsh executable was not found at {requested}"
        raise FileNotFoundError(message)

    out_dir.mkdir(parents=True, exist_ok=True)
    article = build_article(params)
    shape = cast("CadShapeLike", article.val())
    cad_sample_points = cad_face_samples(shape, params).points
    step_path = out_dir / "article_body_datum.step"
    cq.exporters.export(cast("Any", article), str(step_path))
    reference_mesh = _cadquery_reference_mesh(params, out_dir)
    version = _gmsh_version(executable)
    variant = GMSH_SURFACE_VARIANTS[0]
    payload = {
        "kind": "gmsh_occ_surface_remesh",
        "geometry_id": params.geometry_id(),
        "source_step": str(step_path),
        "gmsh_executable": str(executable),
        "gmsh_version": version,
        "mesh_algorithm": mesh_algorithm,
        "mesh_size_min_m": mesh_size_min_m,
        "mesh_size_max_m": mesh_size_max_m,
        "mesh_optimize": mesh_optimize,
        **variant,
    }
    return _write_gmsh_candidate(
        gmsh_path=executable,
        step_path=step_path,
        params=params,
        out_dir=out_dir,
        payload=payload,
        cad_sample_centroids=cad_sample_points,
        reference_mesh=reference_mesh,
    )


def _write_blocked_gmsh_candidate(out_dir: Path, payload: dict[str, Any]) -> SurfaceCandidateResult:
    candidate_id = surface_candidate_id(payload)
    candidate_dir = out_dir / f"{_candidate_name(payload)}__{candidate_id}"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = candidate_dir / "surface_candidate_metrics.json"
    atomic_write_json(
        metrics_path,
        {
            "schema_version": SURFACE_CANDIDATE_SCHEMA_VERSION,
            "candidate_id": candidate_id,
            "status": "BLOCKED_GMSH_NOT_FOUND",
            "export_payload": payload,
            "blocker": "gmsh executable was not found on PATH or at the requested path",
        },
    )
    return SurfaceCandidateResult(
        candidate_id=candidate_id,
        candidate_dir=candidate_dir,
        status="BLOCKED_GMSH_NOT_FOUND",
        stl_path=None,
        metrics_path=metrics_path,
        regions_json_path=None,
        regions_vtp_path=None,
    )


def _write_gmsh_candidate(
    *,
    gmsh_path: Path,
    step_path: Path,
    params: AeroParams,
    out_dir: Path,
    payload: dict[str, Any],
    cad_sample_centroids: np.ndarray,
    reference_mesh: trimesh.Trimesh,
) -> SurfaceCandidateResult:
    candidate_id = surface_candidate_id(payload)
    candidate_dir = out_dir / f"{_candidate_name(payload)}__{candidate_id}"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    body_stl_path = candidate_dir / "article_body_datum.stl"
    stl_path = candidate_dir / "article.stl"
    log_path = candidate_dir / "gmsh.log"

    command = [str(gmsh_path), "-nopopup", "-nt", "1"]
    set_numbers = {
        "Mesh.MeshSizeMin": payload["mesh_size_min_m"],
        "Mesh.MeshSizeMax": payload["mesh_size_max_m"],
        "Mesh.Optimize": payload["mesh_optimize"],
        **payload["occ_options"],
    }
    for name, value in set_numbers.items():
        command.extend(["-setnumber", name, str(value)])
    command.extend(
        [
            str(step_path),
            "-2",
            "-algo",
            str(payload["mesh_algorithm"]),
            "-format",
            "stl",
            "-o",
            str(body_stl_path),
        ],
    )
    try:
        completed = subprocess.run(  # noqa: S603
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        log_path.write_text(str(exc), encoding="utf-8")
        metrics_path = candidate_dir / "surface_candidate_metrics.json"
        atomic_write_json(
            metrics_path,
            {
                "schema_version": SURFACE_CANDIDATE_SCHEMA_VERSION,
                "candidate_id": candidate_id,
                "status": "GMSH_FAILED",
                "export_payload": payload,
                "gmsh_error": str(exc),
                "gmsh_log": str(log_path),
            },
        )
        return SurfaceCandidateResult(
            candidate_id=candidate_id,
            candidate_dir=candidate_dir,
            status="GMSH_FAILED",
            stl_path=None,
            metrics_path=metrics_path,
        )
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        atomic_write_json(
            candidate_dir / "surface_candidate_metrics.json",
            {
                "schema_version": SURFACE_CANDIDATE_SCHEMA_VERSION,
                "candidate_id": candidate_id,
                "status": "GMSH_FAILED",
                "export_payload": payload,
                "gmsh_return_code": completed.returncode,
                "gmsh_log": str(log_path),
            },
        )
        return SurfaceCandidateResult(
            candidate_id=candidate_id,
            candidate_dir=candidate_dir,
            status="GMSH_FAILED",
            stl_path=None,
            metrics_path=candidate_dir / "surface_candidate_metrics.json",
        )

    body_mesh = _load_exported_mesh(body_stl_path)
    mesh = body_mesh.copy()
    mesh.vertices = apply_ride_height_pitch(
        np.asarray(mesh.vertices, dtype=np.float64),
        ride_height_mm=params.ride_height_mm,
        pitch_deg=params.pitch_deg,
    )
    mesh.fix_normals()
    mesh.export(stl_path, file_type="stl")
    body_stl_path.unlink(missing_ok=True)
    region_artifacts = _write_candidate_region_sidecars(
        mesh=mesh,
        classification_mesh=body_mesh,
        params=params,
        candidate_dir=candidate_dir,
    )
    metrics_path = candidate_dir / "surface_candidate_metrics.json"
    atomic_write_json(
        metrics_path,
        {
            "schema_version": SURFACE_CANDIDATE_SCHEMA_VERSION,
            "candidate_id": candidate_id,
            "status": "EXPORTED",
            "export_payload": payload,
            "gmsh_command": command,
            "gmsh_log": str(log_path),
            "gmsh_log_sha256": sha256_file(log_path),
            "surface_region_artifacts": region_artifacts,
            "surface_metrics": _surface_metrics(
                mesh=mesh,
                body_mesh=body_mesh,
                stl_path=stl_path,
                params=params,
                cad_sample_centroids=cad_sample_centroids,
                reference_mesh=reference_mesh,
            ),
        },
    )
    return SurfaceCandidateResult(
        candidate_id=candidate_id,
        candidate_dir=candidate_dir,
        status="EXPORTED",
        stl_path=stl_path,
        metrics_path=metrics_path,
        regions_json_path=candidate_dir / "surface_regions.json",
        regions_vtp_path=candidate_dir / "surface_regions.vtp",
    )


def generate_surface_candidates(
    *,
    params: AeroParams,
    out_dir: Path,
    include_gmsh: bool = True,
    gmsh_path: Path | None = None,
    include_cadquery_fixed_matrix: bool = True,
) -> SurfaceCandidateMatrix:
    """Generate bounded surface tessellation candidates without altering CAD."""

    out_dir.mkdir(parents=True, exist_ok=True)
    article = build_article(params)
    shape = cast("CadShapeLike", article.val())
    cad_sample_points = cad_face_samples(shape, params).points
    step_path = out_dir / "article_body_datum.step"
    cq.exporters.export(cast("Any", article), str(step_path))

    results: list[SurfaceCandidateResult] = [
        _write_cadquery_candidate(
            params=params,
            out_dir=out_dir,
            payload={
                "kind": "cadquery_current_control",
                "geometry_id": params.geometry_id(),
                "linear_tolerance_m": 0.001,
                "angular_tolerance_rad": 0.05,
                "relative": True,
                "parallel": True,
            },
            cad_sample_centroids=cad_sample_points,
        ),
    ]
    current_control_stl = results[0].stl_path
    if current_control_stl is None:
        message = "current control candidate did not produce an STL"
        raise RuntimeError(message)
    reference_mesh = _load_exported_mesh(current_control_stl)
    if include_cadquery_fixed_matrix:
        for linear_tolerance_m, angular_tolerance_rad in CADQUERY_FIXED_TOLERANCE_MATRIX:
            results.append(
                _write_cadquery_candidate(
                    params=params,
                    out_dir=out_dir,
                    payload={
                        "kind": "cadquery_fixed_tolerance",
                        "geometry_id": params.geometry_id(),
                        "linear_tolerance_m": linear_tolerance_m,
                        "angular_tolerance_rad": angular_tolerance_rad,
                        "relative": False,
                        "parallel": False,
                    },
                    cad_sample_centroids=cad_sample_points,
                    reference_mesh=reference_mesh,
                ),
            )

    if include_gmsh:
        executable = _gmsh_executable(gmsh_path)
        if executable is None:
            for variant in GMSH_SURFACE_VARIANTS:
                payload = {
                    "kind": "gmsh_occ_surface_remesh",
                    "geometry_id": params.geometry_id(),
                    "source_step": str(step_path),
                    "mesh_algorithm": "front2d",
                    "mesh_size_min_m": GMSH_MESH_SIZE_MIN_M,
                    "mesh_size_max_m": GMSH_MESH_SIZE_MAX_M,
                    "mesh_optimize": 1,
                    **variant,
                }
                results.append(_write_blocked_gmsh_candidate(out_dir, payload))
        else:
            version = _gmsh_version(executable)
            for variant in GMSH_SURFACE_VARIANTS:
                payload = {
                    "kind": "gmsh_occ_surface_remesh",
                    "geometry_id": params.geometry_id(),
                    "source_step": str(step_path),
                    "gmsh_executable": str(executable),
                    "gmsh_version": version,
                    "mesh_algorithm": "front2d",
                    "mesh_size_min_m": GMSH_MESH_SIZE_MIN_M,
                    "mesh_size_max_m": GMSH_MESH_SIZE_MAX_M,
                    "mesh_optimize": 1,
                    **variant,
                }
                results.append(
                    _write_gmsh_candidate(
                        gmsh_path=executable,
                        step_path=step_path,
                        params=params,
                        out_dir=out_dir,
                        payload=payload,
                        cad_sample_centroids=cad_sample_points,
                        reference_mesh=reference_mesh,
                    ),
                )

    manifest_path = out_dir / "surface_candidate_matrix.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": SURFACE_CANDIDATE_SCHEMA_VERSION,
            "geometry_id": params.geometry_id(),
            "state_id": params.state_id(),
            "params": params.model_dump(),
            "step_path": str(step_path),
            "step_sha256": sha256_file(step_path),
            "include_cadquery_fixed_matrix": include_cadquery_fixed_matrix,
            "candidates": [
                {
                    "candidate_id": item.candidate_id,
                    "candidate_dir": str(item.candidate_dir),
                    "status": item.status,
                    "stl_path": str(item.stl_path) if item.stl_path is not None else None,
                    "metrics_path": str(item.metrics_path),
                    "regions_json_path": (
                        str(item.regions_json_path) if item.regions_json_path is not None else None
                    ),
                    "regions_vtp_path": (
                        str(item.regions_vtp_path) if item.regions_vtp_path is not None else None
                    ),
                }
                for item in results
            ],
        },
    )
    return SurfaceCandidateMatrix(
        out_dir=out_dir,
        manifest_path=manifest_path,
        candidates=tuple(results),
    )
