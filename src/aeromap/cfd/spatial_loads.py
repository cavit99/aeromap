"""Offline spatial load decomposition from exported article-wall fields."""

from __future__ import annotations

import gzip
import json
import re
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from aeromap.constants import REF
from aeromap.geometry.regions import (
    DIFFUSER_EXIT_X_M,
    REGION_NAMES,
    classify_region_arrays,
    tunnel_design_metadata,
)
from aeromap.io import atomic_write_json
from aeromap.parameters import AeroParams
from aeromap.transforms import (
    infer_ride_height_pitch_z_shift,
    inverse_pitch_normals,
    inverse_ride_height_pitch,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]

SCHEMA_VERSION = "spatial_load_decomposition_v0.1.0"
DEFAULT_STREAMWISE_BINS = 16
MIN_COP_FORCE_N = 1.0e-12
XYZ_COMPONENTS = 3
MIN_PHASE_SAMPLES = 3
TUNNEL_LOAD_REGIONS = frozenset({"tunnel_roofs", "diffuser", "underfloor"})
CRITICAL_UNDERFLOOR_REGIONS = frozenset(
    {"tunnel_roofs", "diffuser", "underfloor", "keel", "floor_edges"},
)
THROAT_BAND_HALF_WIDTH_M = 0.08
DIFFUSER_EXIT_BAND_UPSTREAM_M = 0.12
DIFFUSER_EXIT_BAND_DOWNSTREAM_M = 0.04
ARTICLE_PATCH_CANDIDATES = (
    "article",
    "critical_underfloor",
    "tunnel_roofs_core",
    "diffuser_core",
    "underfloor_core",
    "upper_body",
    "floor_edges",
    "keel",
    "layer_transition_band",
)
OPENFOAM_VECTOR_RE = re.compile(r"\(([^()]+)\)")
OPENFOAM_TIME_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
OPENFOAM_WORD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _read_text_maybe_gzip(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8", errors="replace")


def _numbers(text: str) -> list[float]:
    return [float(value) for value in FLOAT_RE.findall(text)]


def _as_polydata(path: Path) -> pv.PolyData:
    loaded = pv.read(path)
    if isinstance(loaded, pv.MultiBlock):
        msg = f"expected a wall PolyData export, got MultiBlock: {path}"
        raise TypeError(msg)
    return loaded if isinstance(loaded, pv.PolyData) else loaded.extract_surface()


def _polydata_cell_normals(poly: pv.PolyData) -> FloatArray:
    with_normals = poly.compute_normals(
        cell_normals=True,
        point_normals=False,
        auto_orient_normals=False,
        consistent_normals=False,
    )
    normals = np.asarray(with_normals.cell_data["Normals"], dtype=np.float64)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 0.0
    normals[valid] /= lengths[valid, None]
    return normals


def _cell_area_vectors(poly: pv.PolyData) -> tuple[FloatArray, FloatArray]:
    faces = np.asarray(poly.faces, dtype=np.int64)
    points = np.asarray(poly.points, dtype=np.float64)
    vectors: list[FloatArray] = []
    cursor = 0
    while cursor < len(faces):
        vertex_count = int(faces[cursor])
        vertex_ids = faces[cursor + 1 : cursor + 1 + vertex_count]
        cursor += vertex_count + 1
        vertices = points[vertex_ids]
        area_vector = np.zeros(3, dtype=np.float64)
        for start, end in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
            area_vector += np.cross(start, end)
        vectors.append(0.5 * area_vector)
    area_vectors = np.asarray(vectors, dtype=np.float64)
    areas = np.linalg.norm(area_vectors, axis=1)
    return area_vectors, areas


def _openfoam_patch_blocks(text: str) -> dict[str, str]:
    lines = text.splitlines()
    blocks: dict[str, str] = {}
    index = 0
    while index < len(lines):
        name = lines[index].strip()
        if not OPENFOAM_WORD_RE.fullmatch(name):
            index += 1
            continue
        open_index = index + 1
        while open_index < len(lines) and not lines[open_index].strip():
            open_index += 1
        if open_index >= len(lines) or lines[open_index].strip() != "{":
            index += 1
            continue
        depth = 0
        block_lines: list[str] = []
        close_index = open_index
        for close_index in range(open_index, len(lines)):
            line = lines[close_index]
            depth += line.count("{")
            depth -= line.count("}")
            block_lines.append(line)
            if depth == 0:
                break
        blocks[name] = "\n".join(block_lines)
        index += 1
    return blocks


def _openfoam_boundary_specs(mesh_dir: Path) -> dict[str, dict[str, int]]:
    blocks = _openfoam_patch_blocks((mesh_dir / "boundary").read_text(encoding="utf-8"))
    specs: dict[str, dict[str, int]] = {}
    for name, body in blocks.items():
        n_faces = re.search(r"\bnFaces\s+(\d+)\s*;", body)
        start_face = re.search(r"\bstartFace\s+(\d+)\s*;", body)
        if n_faces and start_face:
            specs[name] = {
                "n_faces": int(n_faces.group(1)),
                "start_face": int(start_face.group(1)),
            }
    if not specs:
        msg = f"no OpenFOAM boundary patch specs found in {mesh_dir / 'boundary'}"
        raise ValueError(msg)
    return specs


def _select_article_patches(
    specs: dict[str, dict[str, int]],
    patches: Sequence[str] | None,
) -> tuple[str, ...]:
    if patches is not None:
        missing = [patch for patch in patches if patch not in specs]
        if missing:
            msg = "OpenFOAM mesh lacks requested article patches: " + ", ".join(missing)
            raise ValueError(msg)
        return tuple(patches)
    selected = tuple(patch for patch in ARTICLE_PATCH_CANDIDATES if patch in specs)
    if not selected:
        msg = (
            "could not infer article wall patches from boundary file; expected one of "
            + ", ".join(ARTICLE_PATCH_CANDIDATES)
        )
        raise ValueError(msg)
    return selected


def _read_openfoam_points(points_path: Path) -> FloatArray:
    count: int | None = None
    points: FloatArray | None = None
    index = 0
    started = False
    with points_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if count is None:
                if stripped.isdigit():
                    count = int(stripped)
                continue
            if not started:
                if stripped == "(":
                    points = np.empty((count, 3), dtype=np.float64)
                    started = True
                continue
            if stripped == ")":
                break
            match = OPENFOAM_VECTOR_RE.search(stripped)
            if match is None:
                continue
            values = np.fromstring(match.group(1), sep=" ", dtype=np.float64)
            if len(values) != XYZ_COMPONENTS:
                msg = f"invalid OpenFOAM point line in {points_path}: {stripped}"
                raise ValueError(msg)
            assert points is not None
            points[index] = values
            index += 1
    if points is None or count is None or index != count:
        msg = f"expected {count} OpenFOAM points in {points_path}, read {index}"
        raise ValueError(msg)
    return points


def _read_boundary_faces(
    faces_path: Path,
    specs: dict[str, dict[str, int]],
    patches: Sequence[str],
) -> tuple[list[IntArray], list[str]]:
    ranges = sorted(
        (
            (
                specs[patch]["start_face"],
                specs[patch]["start_face"] + specs[patch]["n_faces"],
                patch,
            )
            for patch in patches
        ),
        key=lambda item: item[0],
    )
    min_start = min(start for start, _, _ in ranges)
    max_end = max(end for _, end, _ in ranges)
    faces: list[IntArray] = []
    face_patches: list[str] = []
    count: int | None = None
    started = False
    face_index = 0
    range_index = 0
    with faces_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if count is None:
                if stripped.isdigit():
                    count = int(stripped)
                continue
            if not started:
                if stripped == "(":
                    started = True
                continue
            if stripped == ")" or face_index >= max_end:
                break
            if face_index >= min_start:
                while range_index < len(ranges) and face_index >= ranges[range_index][1]:
                    range_index += 1
                if range_index < len(ranges):
                    start, end, patch = ranges[range_index]
                    if start <= face_index < end:
                        match = OPENFOAM_VECTOR_RE.search(stripped)
                        if match is None:
                            msg = f"invalid OpenFOAM face line in {faces_path}: {stripped}"
                            raise ValueError(msg)
                        faces.append(
                            np.fromstring(match.group(1), sep=" ", dtype=np.int64),
                        )
                        face_patches.append(patch)
            face_index += 1
    expected = sum(specs[patch]["n_faces"] for patch in patches)
    if len(faces) != expected:
        msg = f"expected {expected} selected boundary faces in {faces_path}, read {len(faces)}"
        raise ValueError(msg)
    return faces, face_patches


def _read_boundary_owner_cells(
    owner_path: Path,
    specs: dict[str, dict[str, int]],
    patches: Sequence[str],
) -> IntArray:
    ranges = sorted(
        (
            (
                specs[patch]["start_face"],
                specs[patch]["start_face"] + specs[patch]["n_faces"],
            )
            for patch in patches
        ),
        key=lambda item: item[0],
    )
    min_start = min(start for start, _ in ranges)
    max_end = max(end for _, end in ranges)
    owners: list[int] = []
    count: int | None = None
    started = False
    face_index = 0
    range_index = 0
    with owner_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if count is None:
                if stripped.isdigit():
                    count = int(stripped)
                continue
            if not started:
                if stripped == "(":
                    started = True
                continue
            if stripped == ")" or face_index >= max_end:
                break
            if face_index >= min_start:
                while range_index < len(ranges) and face_index >= ranges[range_index][1]:
                    range_index += 1
                if range_index < len(ranges):
                    start, end = ranges[range_index]
                    if start <= face_index < end:
                        owners.append(int(stripped))
            face_index += 1
    expected = sum(specs[patch]["n_faces"] for patch in patches)
    if len(owners) != expected:
        msg = f"expected {expected} selected boundary owners in {owner_path}, read {len(owners)}"
        raise ValueError(msg)
    return np.asarray(owners, dtype=np.int64)


def _boundary_mesh_arrays(
    *,
    openfoam_dir: Path,
    patches: Sequence[str] | None,
) -> dict[str, Any]:
    mesh_dir = openfoam_dir / "constant" / "polyMesh"
    specs = _openfoam_boundary_specs(mesh_dir)
    selected = _select_article_patches(specs, patches)
    points = _read_openfoam_points(mesh_dir / "points")
    faces, face_patches = _read_boundary_faces(mesh_dir / "faces", specs, selected)
    owner_cells = _read_boundary_owner_cells(mesh_dir / "owner", specs, selected)
    area_vectors: list[FloatArray] = []
    centres: list[FloatArray] = []
    for face in faces:
        vertices = points[face]
        area_vector = np.zeros(3, dtype=np.float64)
        for start, end in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
            area_vector += np.cross(start, end)
        area_vectors.append(0.5 * area_vector)
        centres.append(np.mean(vertices, axis=0))
    area_array = np.asarray(area_vectors, dtype=np.float64)
    areas = np.linalg.norm(area_array, axis=1)
    normals = np.zeros_like(area_array)
    valid = areas > 0.0
    normals[valid] = area_array[valid] / areas[valid, None]
    patch_slices: dict[str, tuple[int, int]] = {}
    offset = 0
    for patch in selected:
        n_faces = int(specs[patch]["n_faces"])
        patch_slices[patch] = (offset, offset + n_faces)
        offset += n_faces
    return {
        "patches": selected,
        "patch_specs": {patch: specs[patch] for patch in selected},
        "patch_slices": patch_slices,
        "face_patches": np.asarray(face_patches, dtype=object),
        "owner_cells": owner_cells,
        "area_vectors": area_array,
        "geometric_areas": areas,
        "mapped_areas": areas,
        "centres": np.asarray(centres, dtype=np.float64),
        "raw_normals": normals,
    }


def _patch_block_from_field(text: str, patch: str, path: Path) -> str:
    blocks = _openfoam_patch_blocks(text)
    try:
        return blocks[patch]
    except KeyError as exc:
        msg = f"patch {patch!r} not found in OpenFOAM field {path}"
        raise ValueError(msg) from exc


def _field_values(
    field_path: Path,
    *,
    patch: str,
    expected_count: int,
    components: int,
    owner_cells: IntArray | None = None,
    internal_scalar_values: FloatArray | None = None,
) -> FloatArray:
    block = _patch_block_from_field(_read_text_maybe_gzip(field_path), patch, field_path)
    if components == 1:
        uniform = re.search(r"value\s+uniform\s+([^;()]+)\s*;", block)
        if uniform:
            value = np.asarray([float(uniform.group(1))], dtype=np.float64)
            return np.repeat(value, expected_count)
        nonuniform = re.search(
            r"value\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;",
            block,
            flags=re.DOTALL,
        )
        if not nonuniform:
            if (
                owner_cells is not None
                and internal_scalar_values is not None
                and re.search(r"\btype\s+zeroGradient\s*;", block)
            ):
                if len(internal_scalar_values) == 1:
                    return np.repeat(internal_scalar_values[0], expected_count)
                return internal_scalar_values[owner_cells]
            msg = f"patch {patch!r} has no scalar value list in {field_path}"
            raise ValueError(msg)
        count = int(nonuniform.group(1))
        values = np.asarray(_numbers(nonuniform.group(2)), dtype=np.float64)
        if count != expected_count or len(values) != expected_count:
            msg = (
                f"expected {expected_count} scalar values for patch {patch!r}, "
                f"got count={count}, parsed={len(values)}"
            )
            raise ValueError(msg)
        return values

    uniform_vec = re.search(r"value\s+uniform\s+\(([^()]+)\)\s*;", block)
    if uniform_vec:
        vector = np.fromstring(uniform_vec.group(1), sep=" ", dtype=np.float64)
        if len(vector) != components:
            msg = f"uniform vector for patch {patch!r} in {field_path} has wrong length"
            raise ValueError(msg)
        return np.repeat(vector[None, :], expected_count, axis=0)
    nonuniform_vec = re.search(
        r"value\s+nonuniform\s+List<vector>\s+(\d+)\s*\((.*?)\)\s*;",
        block,
        flags=re.DOTALL,
    )
    if not nonuniform_vec:
        msg = f"patch {patch!r} has no vector value list in {field_path}"
        raise ValueError(msg)
    count = int(nonuniform_vec.group(1))
    rows = [
        np.fromstring(match, sep=" ", dtype=np.float64)
        for match in OPENFOAM_VECTOR_RE.findall(nonuniform_vec.group(2))
    ]
    values = np.asarray(rows, dtype=np.float64)
    if count != expected_count or values.shape != (expected_count, components):
        msg = (
            f"expected ({expected_count}, {components}) vector values for patch {patch!r}, "
            f"got count={count}, shape={values.shape}"
        )
        raise ValueError(msg)
    return values


def _internal_scalar_field_values(field_path: Path) -> FloatArray:
    text = _read_text_maybe_gzip(field_path)
    uniform = re.search(r"internalField\s+uniform\s+([^;()]+)\s*;", text)
    if uniform:
        return np.asarray([float(uniform.group(1))], dtype=np.float64)
    nonuniform = re.search(
        r"internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;",
        text,
        flags=re.DOTALL,
    )
    if not nonuniform:
        msg = f"field has no scalar internalField values: {field_path}"
        raise ValueError(msg)
    expected_count = int(nonuniform.group(1))
    values = np.fromstring(nonuniform.group(2), sep=" ", dtype=np.float64)
    if len(values) != expected_count:
        msg = f"expected {expected_count} internal scalar values in {field_path}, got {len(values)}"
        raise ValueError(msg)
    return values


def _field_path_for_time(openfoam_dir: Path, time_dir: str, field: str) -> Path:
    plain = openfoam_dir / time_dir / field
    gzipped = openfoam_dir / time_dir / f"{field}.gz"
    if plain.exists():
        return plain
    if gzipped.exists():
        return gzipped
    msg = f"OpenFOAM field {field!r} not found for time {time_dir!r} under {openfoam_dir}"
    raise FileNotFoundError(msg)


def _boundary_field_arrays(
    *,
    openfoam_dir: Path,
    time_dir: str,
    mesh_arrays: dict[str, Any],
) -> tuple[FloatArray, FloatArray]:
    pressure_path = _field_path_for_time(openfoam_dir, time_dir, "p")
    shear_path = _field_path_for_time(openfoam_dir, time_dir, "wallShearStress")
    pressure: list[FloatArray] = []
    shear: list[FloatArray] = []
    specs = mesh_arrays["patch_specs"]
    pressure_internal: FloatArray | None = None
    for patch in mesh_arrays["patches"]:
        n_faces = int(specs[patch]["n_faces"])
        start, end = mesh_arrays["patch_slices"][patch]
        owner_cells = mesh_arrays["owner_cells"][start:end]
        try:
            pressure_values = _field_values(
                pressure_path,
                patch=patch,
                expected_count=n_faces,
                components=1,
                owner_cells=owner_cells,
                internal_scalar_values=pressure_internal,
            )
        except ValueError as exc:
            if "has no scalar value list" not in str(exc):
                raise
            pressure_internal = _internal_scalar_field_values(pressure_path)
            pressure_values = _field_values(
                pressure_path,
                patch=patch,
                expected_count=n_faces,
                components=1,
                owner_cells=owner_cells,
                internal_scalar_values=pressure_internal,
            )
        pressure.append(pressure_values)
        shear.append(
            _field_values(shear_path, patch=patch, expected_count=n_faces, components=3),
        )
    return np.concatenate(pressure), np.concatenate(shear)


def _source_surface(openfoam_dir: Path) -> tuple[pv.PolyData, Path]:
    for path in (
        openfoam_dir / "constant" / "triSurface" / "article_surface_regions.vtp",
        openfoam_dir / "constant" / "triSurface" / "article.stl",
    ):
        if path.exists():
            return _as_polydata(path), path
    msg = f"no source article surface found under {openfoam_dir / 'constant' / 'triSurface'}"
    raise FileNotFoundError(msg)


def _classify_boundary_mesh(
    *,
    openfoam_dir: Path,
    mesh_arrays: dict[str, Any],
    params: AeroParams,
) -> tuple[NDArray[np.object_], FloatArray, FloatArray, dict[str, Any]]:
    source, source_path = _source_surface(openfoam_dir)
    source_centres = np.asarray(source.cell_centers().points, dtype=np.float64)
    source_normals = _polydata_cell_normals(source)
    source_points = np.asarray(source.points, dtype=np.float64)
    distances, indices = cKDTree(source_centres).query(mesh_arrays["centres"], k=1)
    raw_normals = np.asarray(mesh_arrays["raw_normals"], dtype=np.float64)
    orientation_dot = np.einsum("ij,ij->i", raw_normals, source_normals[indices])
    normals_for_classification = raw_normals.copy()
    flipped = orientation_dot < 0.0
    normals_for_classification[flipped] *= -1.0
    z_shift_m = infer_ride_height_pitch_z_shift(source_points, pitch_deg=params.pitch_deg)
    body_centres = inverse_ride_height_pitch(
        mesh_arrays["centres"],
        pitch_deg=params.pitch_deg,
        z_shift_m=z_shift_m,
    )
    body_normals = inverse_pitch_normals(normals_for_classification, pitch_deg=params.pitch_deg)
    region_ids = classify_region_arrays(body_centres, body_normals, params)
    region_names = np.asarray([REGION_NAMES[int(region_id)] for region_id in region_ids])
    context = {
        "source_surface": str(source_path),
        "classification_method": "openfoam_boundary_body_local_analytic",
        "normal_orientation_flipped_faces": int(np.count_nonzero(flipped)),
        "normal_orientation_flipped_fraction": float(np.count_nonzero(flipped) / len(flipped))
        if len(flipped)
        else 0.0,
        "nearest_source_distance_m": {
            "max": float(np.max(distances)) if len(distances) else 0.0,
            "mean": float(np.mean(distances)) if len(distances) else 0.0,
        },
    }
    return region_names, body_centres, body_normals, context


def _required_cell_data(poly: pv.PolyData, names: tuple[str, ...]) -> None:
    missing = [name for name in names if name not in poly.cell_data]
    if missing:
        msg = "mapped wall VTP lacks required cell fields: " + ", ".join(missing)
        raise KeyError(msg)


def _cell_arrays(poly: pv.PolyData) -> dict[str, Any]:
    _required_cell_data(
        poly,
        (
            "p",
            "wallShearStress",
            "surface_region",
            "local_face_area_m2",
            "body_local_centroid_m",
        ),
    )
    area_vectors, geometric_areas = _cell_area_vectors(poly)
    mapped_areas = np.asarray(poly.cell_data["local_face_area_m2"], dtype=np.float64).reshape(-1)
    return {
        "area_vectors": area_vectors,
        "geometric_areas": geometric_areas,
        "mapped_areas": mapped_areas,
        "centres": np.asarray(poly.cell_centers().points, dtype=np.float64),
        "pressure": np.asarray(poly.cell_data["p"], dtype=np.float64).reshape(-1),
        "shear": np.asarray(poly.cell_data["wallShearStress"], dtype=np.float64),
        "regions": np.asarray(poly.cell_data["surface_region"], dtype=object),
        "body_centres": np.asarray(poly.cell_data["body_local_centroid_m"], dtype=np.float64),
    }


def _force_components(arrays: dict[str, Any], mask: BoolArray) -> tuple[FloatArray, FloatArray]:
    pressure = arrays["pressure"][mask]
    shear = arrays["shear"][mask]
    area_vectors = arrays["area_vectors"][mask]
    areas = arrays["geometric_areas"][mask]
    pressure_force = REF.rho_kg_m3 * np.sum(pressure[:, None] * area_vectors, axis=0)
    viscous_force = -REF.rho_kg_m3 * np.sum(shear * areas[:, None], axis=0)
    return pressure_force, viscous_force


def _moment_components(
    arrays: dict[str, Any],
    mask: BoolArray,
    pressure_force_cells: FloatArray,
    viscous_force_cells: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    centres = arrays["centres"][mask]
    moment_arm = centres - np.asarray([REF.l_ref_m / 2.0, 0.0, 0.0], dtype=np.float64)
    return (
        np.sum(np.cross(moment_arm, pressure_force_cells), axis=0),
        np.sum(np.cross(moment_arm, viscous_force_cells), axis=0),
    )


def _coefficients(force: FloatArray, moment: FloatArray) -> dict[str, float]:
    denom_force = REF.q_inf_pa * REF.a_ref_m2
    denom_moment = denom_force * REF.l_ref_m
    return {
        "c_d": float(force[0] / denom_force),
        "c_y": float(force[1] / denom_force),
        "c_df": float(-force[2] / denom_force),
        "c_m_roll": float(moment[0] / denom_moment),
        "c_m_pitch": float(moment[1] / denom_moment),
        "c_m_yaw": float(moment[2] / denom_moment),
    }


def _load_summary(arrays: dict[str, Any], mask: BoolArray) -> dict[str, Any]:
    cell_count = int(np.count_nonzero(mask))
    area_m2 = float(np.sum(arrays["geometric_areas"][mask]))
    if cell_count == 0:
        nan_vec = [float("nan"), float("nan"), float("nan")]
        return {
            "cell_count": 0,
            "area_m2": 0.0,
            "pressure_n": nan_vec,
            "viscous_n": nan_vec,
            "total_n": nan_vec,
            "pressure_moment_nm": nan_vec,
            "viscous_moment_nm": nan_vec,
            "total_moment_nm": nan_vec,
            "coefficients": {
                "c_d": float("nan"),
                "c_y": float("nan"),
                "c_df": float("nan"),
                "c_m_roll": float("nan"),
                "c_m_pitch": float("nan"),
                "c_m_yaw": float("nan"),
            },
            "x_cp_m": float("nan"),
        }

    pressure = arrays["pressure"][mask]
    shear = arrays["shear"][mask]
    area_vectors = arrays["area_vectors"][mask]
    areas = arrays["geometric_areas"][mask]
    pressure_force_cells = REF.rho_kg_m3 * pressure[:, None] * area_vectors
    viscous_force_cells = -REF.rho_kg_m3 * shear * areas[:, None]
    pressure_force = np.sum(pressure_force_cells, axis=0)
    viscous_force = np.sum(viscous_force_cells, axis=0)
    pressure_moment, viscous_moment = _moment_components(
        arrays,
        mask,
        pressure_force_cells,
        viscous_force_cells,
    )
    total_force = pressure_force + viscous_force
    total_moment = pressure_moment + viscous_moment
    x_cp = (
        float(REF.l_ref_m / 2.0 - total_moment[1] / total_force[2])
        if abs(float(total_force[2])) >= MIN_COP_FORCE_N
        else float("nan")
    )
    return {
        "cell_count": cell_count,
        "area_m2": area_m2,
        "pressure_n": pressure_force.tolist(),
        "viscous_n": viscous_force.tolist(),
        "total_n": total_force.tolist(),
        "pressure_moment_nm": pressure_moment.tolist(),
        "viscous_moment_nm": viscous_moment.tolist(),
        "total_moment_nm": total_moment.tolist(),
        "coefficients": _coefficients(total_force, total_moment),
        "x_cp_m": x_cp,
    }


def _region_mask(regions: NDArray[np.object_], names: frozenset[str]) -> BoolArray:
    mask = np.zeros(len(regions), dtype=bool)
    for name in names:
        mask |= regions == name
    return mask


def _streamwise_bins(
    arrays: dict[str, Any], *, mask: BoolArray, bin_count: int
) -> list[dict[str, Any]]:
    x = arrays["body_centres"][:, 0]
    masked_x = x[mask]
    if len(masked_x) == 0:
        return []
    lower = float(np.min(masked_x))
    upper = float(np.max(masked_x))
    if np.isclose(lower, upper):
        upper = lower + 1.0e-9
    edges = np.linspace(lower, upper, bin_count + 1)
    bins: list[dict[str, Any]] = []
    for index, (x_min, x_max) in enumerate(pairwise(edges)):
        if index == bin_count - 1:
            bin_mask = mask & (x >= x_min) & (x <= x_max)
        else:
            bin_mask = mask & (x >= x_min) & (x < x_max)
        bins.append(
            {
                "index": index,
                "x_min_m": float(x_min),
                "x_max_m": float(x_max),
                **_load_summary(arrays, bin_mask),
            },
        )
    return bins


def _named_groups(arrays: dict[str, Any], params: AeroParams) -> dict[str, Any]:
    x = arrays["body_centres"][:, 0]
    y = arrays["body_centres"][:, 1]
    regions = arrays["regions"]
    design = tunnel_design_metadata(params)
    tunnel_mask = _region_mask(regions, TUNNEL_LOAD_REGIONS)
    critical_mask = _region_mask(regions, CRITICAL_UNDERFLOOR_REGIONS)
    throat_min = design.throat_x_m - THROAT_BAND_HALF_WIDTH_M
    throat_max = design.throat_x_m + THROAT_BAND_HALF_WIDTH_M
    exit_min = DIFFUSER_EXIT_X_M - DIFFUSER_EXIT_BAND_UPSTREAM_M
    exit_max = DIFFUSER_EXIT_X_M + DIFFUSER_EXIT_BAND_DOWNSTREAM_M
    groups = {
        "left_tunnel_y_negative": tunnel_mask & (y < 0.0),
        "right_tunnel_y_positive": tunnel_mask & (y > 0.0),
        "throat_band": critical_mask & (x >= throat_min) & (x <= throat_max),
        "diffuser_ramp": critical_mask & (x >= design.throat_x_m) & (x <= DIFFUSER_EXIT_X_M),
        "diffuser_exit_band": critical_mask & (x >= exit_min) & (x <= exit_max),
    }
    return {
        "definitions": {
            "left_tunnel_y_negative": (
                "surface_region in tunnel_roofs/diffuser/underfloor and body-local y < 0"
            ),
            "right_tunnel_y_positive": (
                "surface_region in tunnel_roofs/diffuser/underfloor and body-local y > 0"
            ),
            "throat_band": (
                f"critical underfloor regions with x in [{throat_min:g}, {throat_max:g}] m"
            ),
            "diffuser_ramp": (
                "critical underfloor regions with x in "
                f"[{design.throat_x_m:g}, {DIFFUSER_EXIT_X_M:g}] m"
            ),
            "diffuser_exit_band": (
                f"critical underfloor regions with x in [{exit_min:g}, {exit_max:g}] m"
            ),
        },
        "loads": {name: _load_summary(arrays, mask) for name, mask in groups.items()},
    }


def integrate_spatial_loads(
    wall_vtp: Path,
    *,
    params: AeroParams,
    streamwise_bins: int = DEFAULT_STREAMWISE_BINS,
) -> dict[str, Any]:
    """Integrate pressure/viscous loads over analytical regions and streamwise bins."""

    if streamwise_bins < 1:
        msg = "streamwise_bins must be positive"
        raise ValueError(msg)
    poly = _as_polydata(wall_vtp)
    arrays = _cell_arrays(poly)
    cell_count = int(poly.n_cells)
    all_mask = np.ones(cell_count, dtype=bool)
    critical_mask = _region_mask(arrays["regions"], CRITICAL_UNDERFLOOR_REGIONS)
    mapped_area = arrays["mapped_areas"]
    geometric_area = arrays["geometric_areas"]
    area_delta = np.abs(mapped_area - geometric_area)
    denominator = np.maximum(geometric_area, 1.0e-20)
    return {
        "schema_version": SCHEMA_VERSION,
        "source_wall_vtp": str(wall_vtp),
        "cell_count": cell_count,
        "surface_area_m2": float(np.sum(geometric_area)),
        "area_consistency": {
            "max_abs_delta_m2": float(np.max(area_delta)) if len(area_delta) else 0.0,
            "max_relative_delta": float(np.max(area_delta / denominator))
            if len(area_delta)
            else 0.0,
        },
        "reference": {
            "rho_kg_m3": REF.rho_kg_m3,
            "u_inf_m_s": REF.u_inf_m_s,
            "q_inf_pa": REF.q_inf_pa,
            "a_ref_m2": REF.a_ref_m2,
            "l_ref_m": REF.l_ref_m,
            "moment_reference_m": [REF.l_ref_m / 2.0, 0.0, 0.0],
            "downforce_coefficient": "C_DF = -Fz / (q_inf A_ref)",
        },
        "regions_present": sorted({str(name) for name in arrays["regions"]}),
        "total": _load_summary(arrays, all_mask),
        "critical_underfloor": _load_summary(arrays, critical_mask),
        "named_groups": _named_groups(arrays, params),
        "streamwise_bins": {
            "bin_count": streamwise_bins,
            "all_article": _streamwise_bins(arrays, mask=all_mask, bin_count=streamwise_bins),
            "critical_underfloor": _streamwise_bins(
                arrays,
                mask=critical_mask,
                bin_count=streamwise_bins,
            ),
        },
        "phase_relation": {
            "status": "UNAVAILABLE_SINGLE_SNAPSHOT",
            "reason": (
                "left/right phase requires a time series of exported wall surfaces; this report "
                "integrates one wall snapshot."
            ),
        },
        "notes": [
            (
                "Pressure and wallShearStress are OpenFOAM incompressible kinematic "
                "fields multiplied by rho."
            ),
            (
                "Pressure uses exported polygon area vectors; viscous force uses "
                "polygon area and wallShearStress."
            ),
            "Moments use exported case-frame cell centres about CofR = (1, 0, 0) m.",
            "Groups use body-local centroids and analytical AeroCliff surface_region labels.",
        ],
    }


def integrate_openfoam_boundary_spatial_loads(
    *,
    openfoam_dir: Path,
    time_dir: str,
    params: AeroParams,
    patches: Sequence[str] | None = None,
    streamwise_bins: int = DEFAULT_STREAMWISE_BINS,
) -> dict[str, Any]:
    """Integrate spatial loads directly from OpenFOAM boundary fields."""

    if streamwise_bins < 1:
        msg = "streamwise_bins must be positive"
        raise ValueError(msg)
    mesh_arrays = _boundary_mesh_arrays(openfoam_dir=openfoam_dir, patches=patches)
    regions, body_centres, _body_normals, classification = _classify_boundary_mesh(
        openfoam_dir=openfoam_dir,
        mesh_arrays=mesh_arrays,
        params=params,
    )
    return _openfoam_boundary_snapshot(
        openfoam_dir=openfoam_dir,
        time_dir=time_dir,
        params=params,
        mesh_arrays=mesh_arrays,
        regions=regions,
        body_centres=body_centres,
        classification=classification,
        streamwise_bins=streamwise_bins,
    )


def _openfoam_boundary_snapshot(
    *,
    openfoam_dir: Path,
    time_dir: str,
    params: AeroParams,
    mesh_arrays: dict[str, Any],
    regions: NDArray[np.object_],
    body_centres: FloatArray,
    classification: dict[str, Any],
    streamwise_bins: int,
) -> dict[str, Any]:
    pressure, shear = _boundary_field_arrays(
        openfoam_dir=openfoam_dir,
        time_dir=time_dir,
        mesh_arrays=mesh_arrays,
    )
    arrays = {
        **mesh_arrays,
        "pressure": pressure,
        "shear": shear,
        "regions": regions,
        "body_centres": body_centres,
    }
    all_mask = np.ones(len(pressure), dtype=bool)
    critical_mask = _region_mask(regions, CRITICAL_UNDERFLOOR_REGIONS)
    return {
        "schema_version": "openfoam_boundary_spatial_load_decomposition_v0.1.0",
        "openfoam_dir": str(openfoam_dir),
        "time_dir": time_dir,
        "patches": list(mesh_arrays["patches"]),
        "cell_count": len(pressure),
        "surface_area_m2": float(np.sum(mesh_arrays["geometric_areas"])),
        "classification": classification,
        "regions_present": sorted({str(name) for name in regions}),
        "per_region_area_m2": {
            name: float(np.sum(mesh_arrays["geometric_areas"][regions == name]))
            for name in sorted({str(value) for value in regions})
        },
        "reference": {
            "rho_kg_m3": REF.rho_kg_m3,
            "u_inf_m_s": REF.u_inf_m_s,
            "q_inf_pa": REF.q_inf_pa,
            "a_ref_m2": REF.a_ref_m2,
            "l_ref_m": REF.l_ref_m,
            "downforce_coefficient": "C_DF = -Fz / (q_inf A_ref)",
        },
        "total": _load_summary(arrays, all_mask),
        "critical_underfloor": _load_summary(arrays, critical_mask),
        "named_groups": _named_groups(arrays, params),
        "streamwise_bins": {
            "bin_count": streamwise_bins,
            "all_article": _streamwise_bins(arrays, mask=all_mask, bin_count=streamwise_bins),
            "critical_underfloor": _streamwise_bins(
                arrays,
                mask=critical_mask,
                bin_count=streamwise_bins,
            ),
        },
        "notes": [
            "OpenFOAM pressure and wallShearStress are kinematic fields multiplied by rho.",
            "Face geometry comes from constant/polyMesh boundary faces, not force histories.",
            "Region groups are analytical body-local classifications of OpenFOAM wall faces.",
            (
                "This is a compact diagnostic extractor; it does not establish stationarity, "
                "repeatability or label eligibility."
            ),
        ],
    }


def _time_dirs_with_fields(openfoam_dir: Path) -> list[str]:
    candidates: list[tuple[float, str]] = []
    for path in openfoam_dir.iterdir():
        if not path.is_dir() or not OPENFOAM_TIME_RE.fullmatch(path.name):
            continue
        if _numeric_time_dir_value(path.name) <= 0.0:
            continue
        has_pressure = (path / "p").exists() or (path / "p.gz").exists()
        has_shear = (path / "wallShearStress").exists() or (path / "wallShearStress.gz").exists()
        if has_pressure and has_shear:
            candidates.append((float(path.name), path.name))
    return [name for _, name in sorted(candidates)]


def _numeric_time_dir_value(name: str) -> float:
    return float(name)


def _compact_snapshot_row(snapshot: dict[str, Any]) -> dict[str, Any]:
    groups = snapshot["named_groups"]["loads"]
    return {
        "time_s": float(snapshot["time_dir"]),
        "total": snapshot["total"],
        "critical_underfloor": snapshot["critical_underfloor"],
        "left_tunnel_y_negative": groups["left_tunnel_y_negative"],
        "right_tunnel_y_positive": groups["right_tunnel_y_positive"],
        "throat_band": groups["throat_band"],
        "diffuser_ramp": groups["diffuser_ramp"],
        "diffuser_exit_band": groups["diffuser_exit_band"],
        "streamwise_bins": snapshot["streamwise_bins"],
    }


def _phase_relation_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) < MIN_PHASE_SAMPLES:
        return {"status": "INSUFFICIENT_SAMPLES", "sample_count": len(rows)}
    left = np.asarray(
        [row["left_tunnel_y_negative"]["coefficients"]["c_df"] for row in rows],
        dtype=np.float64,
    )
    right = np.asarray(
        [row["right_tunnel_y_positive"]["coefficients"]["c_df"] for row in rows],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        return {"status": "NONFINITE_VALUES", "sample_count": len(rows)}
    left_d = left - float(np.mean(left))
    right_d = right - float(np.mean(right))
    denom = float(np.linalg.norm(left_d) * np.linalg.norm(right_d))
    if denom <= 0.0:
        return {"status": "ZERO_VARIANCE", "sample_count": len(rows)}
    corr = np.correlate(left_d, right_d, mode="full") / denom
    lags = np.arange(-len(left_d) + 1, len(left_d))
    best = int(np.argmax(np.abs(corr)))
    times = np.asarray([row["time_s"] for row in rows], dtype=np.float64)
    dt = float(np.median(np.diff(times))) if len(times) > 1 else float("nan")
    return {
        "status": "DIAGNOSTIC_ONLY",
        "sample_count": len(rows),
        "best_lag_samples": int(lags[best]),
        "best_lag_s": float(lags[best] * dt) if np.isfinite(dt) else float("nan"),
        "best_normalized_correlation": float(corr[best]),
        "zero_lag_correlation": float(corr[len(left_d) - 1]),
        "limits": (
            "Short retained-field histories and coarse write intervals limit phase "
            "interpretation; this is not physical-frequency evidence."
        ),
    }


def write_urans_spatial_load_history(
    *,
    work_case: Path,
    params: AeroParams,
    out_json: Path | None = None,
    time_dirs: Sequence[str] | None = None,
    patches: Sequence[str] | None = None,
    streamwise_bins: int = DEFAULT_STREAMWISE_BINS,
) -> dict[str, Any]:
    """Write compact left/right and streamwise URANS load diagnostics."""

    openfoam_dir = work_case / "openfoam"
    if not openfoam_dir.exists():
        msg = f"URANS work case has no OpenFOAM directory: {openfoam_dir}"
        raise FileNotFoundError(msg)
    selected_times = (
        list(time_dirs) if time_dirs is not None else _time_dirs_with_fields(openfoam_dir)
    )
    if not selected_times:
        msg = f"no transient time directories with p/wallShearStress fields found in {openfoam_dir}"
        raise FileNotFoundError(msg)
    mesh_arrays = _boundary_mesh_arrays(openfoam_dir=openfoam_dir, patches=patches)
    regions, body_centres, _body_normals, classification = _classify_boundary_mesh(
        openfoam_dir=openfoam_dir,
        mesh_arrays=mesh_arrays,
        params=params,
    )
    snapshots = [
        _openfoam_boundary_snapshot(
            openfoam_dir=openfoam_dir,
            time_dir=time_dir,
            params=params,
            mesh_arrays=mesh_arrays,
            regions=regions,
            body_centres=body_centres,
            classification=classification,
            streamwise_bins=streamwise_bins,
        )
        for time_dir in selected_times
    ]
    rows = [_compact_snapshot_row(snapshot) for snapshot in snapshots]
    report = {
        "schema_version": "urans_spatial_load_history_v0.1.0",
        "work_case": str(work_case),
        "accepted": False,
        "training_eligible": False,
        "time_dirs": selected_times,
        "row_count": len(rows),
        "streamwise_bins": streamwise_bins,
        "patches": snapshots[0]["patches"],
        "regions_present": snapshots[0]["regions_present"],
        "per_region_area_m2": snapshots[0]["per_region_area_m2"],
        "classification": snapshots[0]["classification"],
        "rows": rows,
        "phase_relation": _phase_relation_from_rows(rows),
        "analysis_limits": [
            "This report integrates retained OpenFOAM wall fields; it does not use forces.dat.",
            "It is a compact diagnostic input only, not a campaign label or accepted mean.",
            "Wall-shear and separation outputs remain diagnostic until y+ and layer evidence pass.",
        ],
    }
    output = out_json or work_case / "quality" / "urans_spatial_load_history.json"
    atomic_write_json(output, report)
    return report


def write_spatial_loads_report(
    *,
    wall_vtp: Path,
    params: AeroParams,
    out_json: Path,
    streamwise_bins: int = DEFAULT_STREAMWISE_BINS,
) -> dict[str, Any]:
    report = integrate_spatial_loads(
        wall_vtp,
        params=params,
        streamwise_bins=streamwise_bins,
    )
    atomic_write_json(out_json, report)
    return report


def write_case_spatial_loads_report(
    *,
    case_dir: Path,
    out_json: Path | None = None,
    latest_time: str | None = None,
    streamwise_bins: int = DEFAULT_STREAMWISE_BINS,
) -> dict[str, Any]:
    manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    params = AeroParams.model_validate(manifest["params"])
    if latest_time is None:
        candidates = sorted((case_dir / "outputs").glob("article_wall_regions_*.vtp"))
        if not candidates:
            msg = f"no mapped wall VTPs found under {case_dir / 'outputs'}"
            raise FileNotFoundError(msg)
        wall_vtp = candidates[-1]
    else:
        wall_vtp = case_dir / "outputs" / f"article_wall_regions_{latest_time}.vtp"
    if not wall_vtp.exists():
        msg = f"mapped wall VTP not found: {wall_vtp}"
        raise FileNotFoundError(msg)
    output = out_json or case_dir / "quality" / "spatial_loads.json"
    return write_spatial_loads_report(
        wall_vtp=wall_vtp,
        params=params,
        out_json=output,
        streamwise_bins=streamwise_bins,
    )
