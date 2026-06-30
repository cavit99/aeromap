"""Trimesh geometry validation gates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from aeromap.constants import GEOMETRY_GENERATOR_VERSION
from aeromap.geometry.schema import GeometryMetrics, GeometryValidation

MANIFOLD_EDGE_COUNT = 2
DEGENERATE_FACE_AREA_M2 = 1e-12
MIN_GROUND_CLEARANCE_M = 0.015
GROUND_TOLERANCE_M = -1e-9
X_BOUNDS_RANGE_M = (1.75, 2.15)
Y_BOUNDS_RANGE_M = (0.80, 1.15)
Z_BOUNDS_RANGE_M = (0.05, 0.40)
VOLUME_RANGE_M3 = (0.01, 0.45)
SURFACE_AREA_RANGE_M2 = (1.0, 6.0)
ARRAY_NDIM = 2
XYZ_COLUMNS = 3


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def _edge_counts(mesh: trimesh.Trimesh) -> np.ndarray:
    inverse = mesh.edges_unique_inverse
    return np.bincount(inverse, minlength=len(mesh.edges_unique))


def _bounds_tuple(bounds: np.ndarray, row: int) -> tuple[float, float, float]:
    return (float(bounds[row, 0]), float(bounds[row, 1]), float(bounds[row, 2]))


def validate_mesh(mesh: trimesh.Trimesh) -> GeometryValidation:
    reasons: list[str] = []
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)

    if vertices.ndim != ARRAY_NDIM or vertices.shape[-1] != XYZ_COLUMNS or len(vertices) == 0:
        reasons.append("empty_or_invalid_vertices")
    elif not np.isfinite(vertices).all():
        reasons.append("non_finite_vertices")

    if faces.ndim != ARRAY_NDIM or faces.shape[-1] != XYZ_COLUMNS or len(faces) == 0:
        reasons.append("empty_or_invalid_faces")

    if reasons:
        return GeometryValidation(valid=False, reasons=tuple(reasons), metrics=None)

    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        return GeometryValidation(valid=False, reasons=("invalid_bounds",), metrics=None)

    try:
        components = mesh.split(only_watertight=False)
        face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
        edge_counts = _edge_counts(mesh)
        min_ground_clearance = float(np.min(vertices[:, 2]))
        volume = float(mesh.volume)
        area = float(mesh.area)
    except (IndexError, TypeError, ValueError) as exc:
        return GeometryValidation(
            valid=False,
            reasons=(f"unmeasurable_mesh:{exc.__class__.__name__}",),
            metrics=None,
        )

    if (
        not np.isfinite(face_areas).all()
        or not np.isfinite(edge_counts).all()
        or not np.isfinite([min_ground_clearance, volume, area]).all()
    ):
        return GeometryValidation(valid=False, reasons=("non_finite_mesh_metrics",), metrics=None)

    if not mesh.is_watertight:
        reasons.append("not_watertight")
    if not mesh.is_winding_consistent:
        reasons.append("inconsistent_orientation")
    if np.any(edge_counts != MANIFOLD_EDGE_COUNT):
        reasons.append("non_manifold_edges")
    if np.any(face_areas <= DEGENERATE_FACE_AREA_M2):
        reasons.append("degenerate_faces")
    if len(components) != 1:
        reasons.append("disconnected_components")
    if min_ground_clearance < MIN_GROUND_CLEARANCE_M:
        reasons.append("min_ground_clearance_below_15mm")
    if bounds[0, 2] < GROUND_TOLERANCE_M:
        reasons.append("diffuser_or_body_intersects_ground")
    if not (X_BOUNDS_RANGE_M[0] <= bounds[1, 0] - bounds[0, 0] <= X_BOUNDS_RANGE_M[1]):
        reasons.append("x_bounds_outside_sanity_limits")
    if not (Y_BOUNDS_RANGE_M[0] <= bounds[1, 1] - bounds[0, 1] <= Y_BOUNDS_RANGE_M[1]):
        reasons.append("y_bounds_outside_sanity_limits")
    if not (Z_BOUNDS_RANGE_M[0] <= bounds[1, 2] - bounds[0, 2] <= Z_BOUNDS_RANGE_M[1]):
        reasons.append("z_bounds_outside_sanity_limits")
    if not (VOLUME_RANGE_M3[0] <= volume <= VOLUME_RANGE_M3[1]):
        reasons.append("volume_outside_sanity_limits")
    if not (SURFACE_AREA_RANGE_M2[0] <= area <= SURFACE_AREA_RANGE_M2[1]):
        reasons.append("surface_area_outside_sanity_limits")

    metrics = GeometryMetrics(
        watertight=bool(mesh.is_watertight),
        winding_consistent=bool(mesh.is_winding_consistent),
        body_count=len(components),
        vertex_count=len(mesh.vertices),
        face_count=len(mesh.faces),
        bounds_min_m=_bounds_tuple(bounds, 0),
        bounds_max_m=_bounds_tuple(bounds, 1),
        volume_m3=volume,
        surface_area_m2=area,
        min_ground_clearance_m=min_ground_clearance,
        diffuser_region_x_m=(1.05, 1.92),
        generator_version=GEOMETRY_GENERATOR_VERSION,
    )
    return GeometryValidation(valid=not reasons, reasons=tuple(reasons), metrics=metrics)


def validate_stl(path: Path) -> GeometryValidation:
    return validate_mesh(load_mesh(path))
