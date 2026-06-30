"""CFD-only OpenFOAM patch surface export for layered patch-surface experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyvista as pv
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from aeromap.io import atomic_write_json, sha256_file

if TYPE_CHECKING:
    import trimesh

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

GATE2B_PATCH_NAMES = (
    "tunnel_roofs_core",
    "diffuser_core",
    "underfloor_core",
    "upper_body",
    "floor_edges",
    "keel",
    "layer_transition_band",
)
CRITICAL_UNDERFLOOR_PATCH_NAMES = (
    "critical_underfloor",
    "upper_body",
    "floor_edges",
    "keel",
    "layer_transition_band",
)
CRITICAL_CORE_PATCHES = ("tunnel_roofs_core", "diffuser_core", "underfloor_core")
CRITICAL_UNDERFLOOR_CORE_PATCHES = ("critical_underfloor",)
ZERO_LAYER_PATCHES = ("floor_edges", "keel", "layer_transition_band")
GATE2B_REGION_TO_PATCH = {
    "tunnel_roofs": "tunnel_roofs_core",
    "diffuser": "diffuser_core",
    "underfloor": "underfloor_core",
    "upper_body": "upper_body",
    "floor_edges": "floor_edges",
    "keel": "keel",
}
CRITICAL_UNDERFLOOR_REGION_TO_PATCH = {
    "tunnel_roofs": "critical_underfloor",
    "diffuser": "critical_underfloor",
    "underfloor": "critical_underfloor",
    "upper_body": "upper_body",
    "floor_edges": "floor_edges",
    "keel": "keel",
}
TRIANGLE_VERTEX_COUNT = 3
NEAREST_SEGMENT_QUERY_COUNT = 8
MULTI_PATCH_MODES = ("gate2b_core_transition", "critical_underfloor")


@dataclass(frozen=True)
class PatchSurfaceArtifacts:
    obj_path: Path
    metrics_path: Path
    vtp_path: Path
    patch_names: tuple[str, ...]


def article_patch_names(*, patch_mode: str) -> tuple[str, ...]:
    if patch_mode == "gate2b_core_transition":
        return GATE2B_PATCH_NAMES
    if patch_mode == "critical_underfloor":
        return CRITICAL_UNDERFLOOR_PATCH_NAMES
    return ("article",)


def critical_core_patches(*, patch_mode: str) -> tuple[str, ...]:
    if patch_mode == "critical_underfloor":
        return CRITICAL_UNDERFLOOR_CORE_PATCHES
    if patch_mode == "gate2b_core_transition":
        return CRITICAL_CORE_PATCHES
    return ("article",)


def is_multi_patch_mode(patch_mode: str) -> bool:
    return patch_mode in MULTI_PATCH_MODES


def _region_to_patch(*, patch_mode: str) -> dict[str, str]:
    if patch_mode == "critical_underfloor":
        return CRITICAL_UNDERFLOOR_REGION_TO_PATCH
    if patch_mode == "gate2b_core_transition":
        return GATE2B_REGION_TO_PATCH
    message = f"unsupported multi-patch mode: {patch_mode}"
    raise ValueError(message)


def _face_region_names(regions: dict[str, Any]) -> NDArray[np.object_]:
    return np.asarray([item["region"] for item in regions["face_regions"]], dtype=object)


def _triangle_areas(mesh: trimesh.Trimesh) -> FloatArray:
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    return np.asarray(
        0.5
        * np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
            axis=1,
        ),
        dtype=np.float64,
    )


def _edge_segments(
    mesh: trimesh.Trimesh,
    regions: NDArray[np.object_],
    *,
    patch_labels: NDArray[np.object_] | None = None,
    patch_mode: str = "gate2b_core_transition",
    feature_angle_deg: float,
) -> tuple[FloatArray, FloatArray]:
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    if len(adjacency) == 0:
        empty = np.empty((0, 2, 3), dtype=np.float64)
        return empty, empty

    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    dot = np.einsum("ij,ij->i", normals[adjacency[:, 0]], normals[adjacency[:, 1]])
    angles = np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0)))
    region_boundary = regions[adjacency[:, 0]] != regions[adjacency[:, 1]]
    sharp_feature = angles >= feature_angle_deg

    edge_vertex_ids = np.asarray(mesh.face_adjacency_edges, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)

    if patch_mode == "critical_underfloor":
        if patch_labels is None:
            message = "critical_underfloor transition classification requires patch labels"
            raise ValueError(message)
        patch_boundary = patch_labels[adjacency[:, 0]] != patch_labels[adjacency[:, 1]]
        both_critical = (patch_labels[adjacency[:, 0]] == "critical_underfloor") & (
            patch_labels[adjacency[:, 1]] == "critical_underfloor"
        )
        feature_mask = sharp_feature & ~both_critical
        boundary_mask = patch_boundary
    else:
        feature_mask = sharp_feature
        boundary_mask = region_boundary

    feature_segments = vertices[edge_vertex_ids[feature_mask]]
    boundary_segments = vertices[edge_vertex_ids[boundary_mask]]
    return feature_segments, boundary_segments


def _distance_to_segments(points: FloatArray, segments: FloatArray) -> FloatArray:
    if len(segments) == 0:
        return np.full(len(points), np.inf, dtype=np.float64)
    midpoints = np.mean(segments, axis=1)
    query_count = min(NEAREST_SEGMENT_QUERY_COUNT, len(segments))
    _, indices = cKDTree(midpoints).query(points, k=query_count)
    index_2d = np.asarray(indices, dtype=np.int64)
    if index_2d.ndim == 1:
        index_2d = index_2d[:, None]

    best = np.full(len(points), np.inf, dtype=np.float64)
    for column in range(index_2d.shape[1]):
        segment_subset = segments[index_2d[:, column]]
        start = segment_subset[:, 0, :]
        end = segment_subset[:, 1, :]
        direction = end - start
        length_sq = np.einsum("ij,ij->i", direction, direction)
        valid = length_sq > 0.0
        projection = np.zeros(len(points), dtype=np.float64)
        projection[valid] = (
            np.einsum(
                "ij,ij->i",
                points[valid] - start[valid],
                direction[valid],
            )
            / length_sq[valid]
        )
        projection = np.clip(projection, 0.0, 1.0)
        closest = start + projection[:, None] * direction
        best = np.minimum(best, np.linalg.norm(points - closest, axis=1))
    return best


def _patch_labels(
    mesh: trimesh.Trimesh,
    regions: dict[str, Any],
    *,
    transition_band_width_m: float,
    feature_angle_deg: float,
    patch_mode: str,
) -> tuple[NDArray[np.object_], dict[str, FloatArray]]:
    region_names = _face_region_names(regions)
    if len(region_names) != len(mesh.faces):
        message = "region face count does not match mesh face count"
        raise ValueError(message)

    region_to_patch = _region_to_patch(patch_mode=patch_mode)
    patch_labels = np.asarray(
        [region_to_patch[str(region)] for region in region_names],
        dtype=object,
    )
    feature_segments, boundary_segments = _edge_segments(
        mesh,
        region_names,
        patch_labels=patch_labels,
        patch_mode=patch_mode,
        feature_angle_deg=feature_angle_deg,
    )
    centroids = np.asarray(mesh.triangles_center, dtype=np.float64)
    feature_distance = _distance_to_segments(centroids, feature_segments)
    boundary_distance = _distance_to_segments(centroids, boundary_segments)
    transition = (np.minimum(feature_distance, boundary_distance) <= transition_band_width_m) & (
        region_names != "upper_body"
    )

    patch_labels[transition] = "layer_transition_band"
    return patch_labels, {
        "distance_to_sharp_feature_m": feature_distance,
        "distance_to_region_boundary_m": boundary_distance,
        "transition_mask": transition.astype(np.float64),
    }


def _write_obj(
    path: Path,
    mesh: trimesh.Trimesh,
    patch_labels: NDArray[np.object_],
    *,
    patch_names: tuple[str, ...],
) -> None:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    lines = [
        "# Wavefront OBJ file",
        "# AeroCliff CFD-only patch export; DoMINO single-part STL is separate.",
        "# Regions:",
        *[f"#     {index}    {patch_name}" for index, patch_name in enumerate(patch_names)],
        f"# points    : {len(vertices)}",
        f"# triangles : {len(faces)}",
        "",
    ]
    lines.extend(f"v {point[0]:.12g} {point[1]:.12g} {point[2]:.12g}" for point in vertices)
    for patch_name in patch_names:
        face_ids = np.flatnonzero(patch_labels == patch_name)
        lines.append(f"g {patch_name}")
        lines.extend(
            f"f {faces[face_id, 0] + 1} {faces[face_id, 1] + 1} {faces[face_id, 2] + 1}"
            for face_id in face_ids
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_vtp(
    path: Path,
    mesh: trimesh.Trimesh,
    regions: dict[str, Any],
    patch_labels: NDArray[np.object_],
    distances: dict[str, FloatArray],
) -> None:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    cells = np.column_stack(
        [np.full(len(faces), TRIANGLE_VERTEX_COUNT, dtype=np.int64), faces],
    ).ravel()
    poly = pv.PolyData(np.asarray(mesh.vertices, dtype=np.float64), cells)
    poly.cell_data["openfoam_patch"] = patch_labels
    poly.cell_data["source_surface_region"] = _face_region_names(regions)
    poly.cell_data["distance_to_sharp_feature_m"] = distances["distance_to_sharp_feature_m"]
    poly.cell_data["distance_to_region_boundary_m"] = distances["distance_to_region_boundary_m"]
    poly.cell_data["is_layer_transition_band"] = distances["transition_mask"].astype(np.int8)
    path.parent.mkdir(parents=True, exist_ok=True)
    poly.save(path)


def write_openfoam_patch_surface(
    *,
    mesh: trimesh.Trimesh,
    regions: dict[str, Any],
    out_dir: Path,
    transition_band_width_m: float,
    feature_angle_deg: float,
    patch_mode: str,
) -> PatchSurfaceArtifacts:
    """Write a connected watertight OBJ with real OpenFOAM patch groups."""

    out_dir.mkdir(parents=True, exist_ok=True)
    patch_names = article_patch_names(patch_mode=patch_mode)
    if patch_mode not in MULTI_PATCH_MODES:
        message = f"unsupported OpenFOAM patch surface mode: {patch_mode}"
        raise ValueError(message)
    patch_labels, distances = _patch_labels(
        mesh,
        regions,
        transition_band_width_m=transition_band_width_m,
        feature_angle_deg=feature_angle_deg,
        patch_mode=patch_mode,
    )
    obj_path = out_dir / "article.obj"
    vtp_path = out_dir / "article_openfoam_patches.vtp"
    metrics_path = out_dir / "article_openfoam_patches.json"
    _write_obj(obj_path, mesh, patch_labels, patch_names=patch_names)
    _write_vtp(vtp_path, mesh, regions, patch_labels, distances)

    areas = _triangle_areas(mesh)
    region_names = _face_region_names(regions)
    patch_metrics = {}
    for patch_name in patch_names:
        mask = patch_labels == patch_name
        patch_metrics[patch_name] = {
            "triangle_count": int(np.count_nonzero(mask)),
            "area_m2": float(np.sum(areas[mask])),
            "source_regions": {
                str(region): int(np.count_nonzero(mask & (region_names == region)))
                for region in sorted({str(value) for value in region_names[mask]})
            },
        }

    transition_mask = patch_labels == "layer_transition_band"
    metrics = {
        "schema_version": "openfoam_patch_surface_v0.2.0",
        "patch_mode": patch_mode,
        "obj_path": str(obj_path),
        "obj_sha256": sha256_file(obj_path),
        "vtp_path": str(vtp_path),
        "transition_band_width_m": transition_band_width_m,
        "feature_angle_deg": feature_angle_deg,
        "patch_names": list(patch_names),
        "critical_core_patches": list(critical_core_patches(patch_mode=patch_mode)),
        "zero_layer_patches": list(ZERO_LAYER_PATCHES),
        "connected_watertight_shell_preserved": bool(mesh.is_watertight),
        "patch_metrics": patch_metrics,
        "transition_band": {
            "triangle_count": int(np.count_nonzero(transition_mask)),
            "area_m2": float(np.sum(areas[transition_mask])),
            "area_fraction": float(np.sum(areas[transition_mask]) / np.sum(areas)),
        },
    }
    atomic_write_json(metrics_path, metrics)
    return PatchSurfaceArtifacts(
        obj_path=obj_path,
        metrics_path=metrics_path,
        vtp_path=vtp_path,
        patch_names=patch_names,
    )


def write_gate2b_patch_surface(
    *,
    mesh: trimesh.Trimesh,
    regions: dict[str, Any],
    out_dir: Path,
    transition_band_width_m: float,
    feature_angle_deg: float,
) -> PatchSurfaceArtifacts:
    """Backward-compatible writer for the original patch split."""

    return write_openfoam_patch_surface(
        mesh=mesh,
        regions=regions,
        out_dir=out_dir,
        transition_band_width_m=transition_band_width_m,
        feature_angle_deg=feature_angle_deg,
        patch_mode="gate2b_core_transition",
    )
