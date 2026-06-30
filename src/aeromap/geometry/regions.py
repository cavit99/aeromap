"""Stable surface-region metadata for AeroCliff geometry."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyvista as pv
from numpy.typing import NDArray

from aeromap.constants import GEOMETRY_GENERATOR_VERSION
from aeromap.io import atomic_write_json
from aeromap.parameters import AeroParams

if TYPE_CHECKING:
    import trimesh

FloatArray = NDArray[np.float64]

REGION_NAMES: tuple[str, ...] = (
    "diffuser",
    "tunnel_roofs",
    "underfloor",
    "keel",
    "floor_edges",
    "upper_body",
)
REGION_IDS = {name: index for index, name in enumerate(REGION_NAMES)}
FACE_HASH_DECIMALS = 6
FACE_HASH_HEXDIGITS = 16
TRIANGLE_VERTEX_COUNT = 3
POINT_ARRAY_NDIM = 2
XYZ_COMPONENTS = 3
TUNNEL_HALF_WIDTH_BASE_M = 0.142
TUNNEL_CENTRES_Y_M: tuple[float, ...] = (-0.245, 0.245)
STABLE_REFERENCE_TUNNEL_HALF_WIDTH_M = 0.320
STABLE_REFERENCE_TUNNEL_CENTRES_Y_M: tuple[float, ...] = (0.0,)
EDGE_RADIUS_WIDTH_SCALE = 0.35
EDGE_RADIUS_MAX_WIDTH_LOSS_M = 0.008
THROAT_BASE_X_M = 0.72
THROAT_HEIGHT_M = 0.018
STABLE_REFERENCE_THROAT_HEIGHT_M = 0.042
DIFFUSER_EXIT_X_M = 1.90
TUNNEL_FLOOR_Z_M = -0.090
TUNNEL_REGION_MARGIN_M = 0.015
KEEL_HALF_WIDTH_M = 0.075
KEEL_MAX_Z_M = 0.09
FLOOR_EDGE_MIN_ABS_Y_M = 0.385
FLOOR_EDGE_MAX_Z_M = 0.12
DIFFUSER_MIN_X_M = 1.05
DIFFUSER_MAX_X_M = 1.94
DIFFUSER_MAX_Z_M = 0.13
DIFFUSER_HALF_WIDTH_M = 0.44
TUNNEL_ROOF_MAX_Z_M = 0.15
TUNNEL_ROOF_MAX_NORMAL_Z = -0.20
LOWER_OR_SIDE_SURFACE_MAX_NORMAL_Z = 0.35
UNDERFLOOR_MAX_X_M = DIFFUSER_MIN_X_M
UNDERFLOOR_MAX_Z_M = 0.115
UNDERFLOOR_HALF_WIDTH_M = 0.39


@dataclass(frozen=True)
class TunnelDesignMetadata:
    throat_x_m: float
    diffuser_exit_x_m: float
    diffuser_length_m: float
    tunnel_centres_y_m: tuple[float, ...]
    tunnel_half_width_m: float
    throat_roof_height_m: float
    diffuser_exit_roof_height_m: float
    diffuser_angle_deg: float
    edge_radius_m: float


def tunnel_design_metadata(params: AeroParams) -> TunnelDesignMetadata:
    edge_radius_m = params.edge_radius_mm / 1000.0
    throat_x = THROAT_BASE_X_M + params.throat_offset_mm / 1000.0
    diffuser_length = DIFFUSER_EXIT_X_M - throat_x
    if params.geometry_family == "stable_reference":
        tunnel_centres = STABLE_REFERENCE_TUNNEL_CENTRES_Y_M
        tunnel_half_width = STABLE_REFERENCE_TUNNEL_HALF_WIDTH_M
        throat_height = STABLE_REFERENCE_THROAT_HEIGHT_M
    else:
        tunnel_centres = TUNNEL_CENTRES_Y_M
        tunnel_half_width = TUNNEL_HALF_WIDTH_BASE_M
        throat_height = THROAT_HEIGHT_M
    exit_height = throat_height + np.tan(np.deg2rad(params.diffuser_angle_deg)) * diffuser_length
    return TunnelDesignMetadata(
        throat_x_m=throat_x,
        diffuser_exit_x_m=DIFFUSER_EXIT_X_M,
        diffuser_length_m=diffuser_length,
        tunnel_centres_y_m=tunnel_centres,
        tunnel_half_width_m=tunnel_half_width,
        throat_roof_height_m=throat_height,
        diffuser_exit_roof_height_m=float(exit_height),
        diffuser_angle_deg=params.diffuser_angle_deg,
        edge_radius_m=edge_radius_m,
    )


def _face_hash(centroid: FloatArray, normal: FloatArray) -> str:
    payload = np.concatenate(
        [np.round(centroid, FACE_HASH_DECIMALS), np.round(normal, FACE_HASH_DECIMALS)]
    ).astype(np.float64)
    return hashlib.sha256(payload.tobytes()).hexdigest()[:FACE_HASH_HEXDIGITS]


def classify_region_arrays(
    centroids_m: FloatArray,
    normals: FloatArray,
    params: AeroParams,
) -> NDArray[np.int32]:
    """Classify body-local face centroids/normals into AeroCliff surface regions."""

    centroids = np.asarray(centroids_m, dtype=np.float64)
    normal_array = np.asarray(normals, dtype=np.float64)
    if centroids.ndim != POINT_ARRAY_NDIM or centroids.shape[1] != XYZ_COMPONENTS:
        msg = "centroids_m must be an (n, 3) array"
        raise ValueError(msg)
    if normal_array.shape != centroids.shape:
        msg = "normals must match centroids_m shape"
        raise ValueError(msg)

    design = tunnel_design_metadata(params)
    x = centroids[:, 0]
    y = centroids[:, 1]
    z = centroids[:, 2]
    nz = normal_array[:, 2]

    near_tunnel = np.zeros(len(centroids), dtype=bool)
    for centre in design.tunnel_centres_y_m:
        near_tunnel |= np.abs(y - centre) <= design.tunnel_half_width_m + TUNNEL_REGION_MARGIN_M

    labels = np.full(len(centroids), REGION_IDS["upper_body"], dtype=np.int32)

    lower_or_side_surface = nz < LOWER_OR_SIDE_SURFACE_MAX_NORMAL_Z
    if params.geometry_family == "stable_reference":
        keel = np.zeros(len(centroids), dtype=bool)
    else:
        keel = (np.abs(y) <= KEEL_HALF_WIDTH_M) & (z < KEEL_MAX_Z_M) & lower_or_side_surface
    floor_edges = (
        (np.abs(y) >= FLOOR_EDGE_MIN_ABS_Y_M) & (z < FLOOR_EDGE_MAX_Z_M) & lower_or_side_surface
    )
    diffuser = (
        (x >= DIFFUSER_MIN_X_M)
        & (x <= DIFFUSER_MAX_X_M)
        & (z < DIFFUSER_MAX_Z_M)
        & (np.abs(y) <= DIFFUSER_HALF_WIDTH_M)
        & lower_or_side_surface
    )
    tunnel_roofs = near_tunnel & (z < TUNNEL_ROOF_MAX_Z_M) & (nz < TUNNEL_ROOF_MAX_NORMAL_Z)
    underfloor = (
        (x < UNDERFLOOR_MAX_X_M)
        & (z < UNDERFLOOR_MAX_Z_M)
        & (np.abs(y) < UNDERFLOOR_HALF_WIDTH_M)
        & ~keel
        & lower_or_side_surface
    )

    labels[underfloor] = REGION_IDS["underfloor"]
    labels[tunnel_roofs] = REGION_IDS["tunnel_roofs"]
    labels[diffuser] = REGION_IDS["diffuser"]
    labels[floor_edges] = REGION_IDS["floor_edges"]
    labels[keel] = REGION_IDS["keel"]
    return labels


def classify_surface_regions(mesh: trimesh.Trimesh, params: AeroParams) -> dict[str, Any]:
    """Classify body-datum mesh faces into stable aerodynamic surface regions."""

    centroids = np.asarray(mesh.triangles_center, dtype=np.float64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    labels = classify_region_arrays(centroids, normals, params)
    design = tunnel_design_metadata(params)

    face_hashes = [
        _face_hash(centroid, normal) for centroid, normal in zip(centroids, normals, strict=True)
    ]
    counts = {
        name: int(np.count_nonzero(labels == region_id)) for name, region_id in REGION_IDS.items()
    }
    face_regions = [
        {
            "face_index": int(index),
            "region": REGION_NAMES[int(region_id)],
            "region_id": int(region_id),
            "face_hash": face_hashes[index],
            "centroid_m": [float(value) for value in centroids[index]],
        }
        for index, region_id in enumerate(labels)
    ]
    return {
        "schema_version": "surface_regions_v0.1.0",
        "geometry_generator_version": GEOMETRY_GENERATOR_VERSION,
        "classification_frame": "body_local",
        "region_names": list(REGION_NAMES),
        "region_ids": REGION_IDS,
        "counts": counts,
        "design_metadata": {
            "throat_x_m": design.throat_x_m,
            "diffuser_exit_x_m": design.diffuser_exit_x_m,
            "diffuser_length_m": design.diffuser_length_m,
            "tunnel_centres_y_m": list(design.tunnel_centres_y_m),
            "tunnel_half_width_m": design.tunnel_half_width_m,
            "throat_roof_height_m": design.throat_roof_height_m,
            "diffuser_exit_roof_height_m": design.diffuser_exit_roof_height_m,
            "diffuser_angle_deg": design.diffuser_angle_deg,
            "edge_radius_m": design.edge_radius_m,
        },
        "face_regions": face_regions,
    }


def write_region_vtp(mesh: trimesh.Trimesh, regions: dict[str, Any], path: Path) -> None:
    """Write a VTP surface with cell region arrays for downstream conversion checks."""

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    cell_faces = np.column_stack(
        [np.full(len(faces), TRIANGLE_VERTEX_COUNT, dtype=np.int64), faces]
    ).ravel()
    poly = pv.PolyData(vertices, cell_faces)
    region_ids = np.array(
        [item["region_id"] for item in regions["face_regions"]],
        dtype=np.int32,
    )
    region_names = np.array(
        [item["region"] for item in regions["face_regions"]],
        dtype=object,
    )
    poly.cell_data["surface_region_id"] = region_ids
    poly.cell_data["region_id"] = region_ids
    poly.cell_data["surface_region"] = region_names
    path.parent.mkdir(parents=True, exist_ok=True)
    poly.save(path)


def write_surface_regions(
    mesh: trimesh.Trimesh,
    params: AeroParams,
    json_path: Path,
    vtp_path: Path,
    *,
    classification_mesh: trimesh.Trimesh | None = None,
) -> dict[str, Any]:
    regions = classify_surface_regions(classification_mesh or mesh, params)
    atomic_write_json(json_path, regions)
    write_region_vtp(mesh, regions, vtp_path)
    return regions
