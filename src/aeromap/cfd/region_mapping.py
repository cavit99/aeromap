"""Map source STL surface-region labels onto post-mesh OpenFOAM wall surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyvista as pv
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from aeromap.geometry.regions import REGION_NAMES, classify_region_arrays
from aeromap.io import atomic_write_json
from aeromap.parameters import AeroParams
from aeromap.transforms import (
    infer_ride_height_pitch_z_shift,
    inverse_pitch_normals,
    inverse_ride_height_pitch,
)

if TYPE_CHECKING:
    import trimesh

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
MIN_MAPPING_COVERAGE = 0.995
TRIANGLE_VERTEX_COUNT = 3
NEAREST_NEIGHBOUR_COUNT = 2
DEFAULT_MIN_ABS_NORMAL_ALIGNMENT = 0.50
DEFAULT_EDGE_MIN_ABS_NORMAL_ALIGNMENT = 0.10
DEFAULT_MAX_DISTANCE_FACE_SCALE = 1.0
NORMAL_EXCEPTION_REGIONS = frozenset({"floor_edges", "keel", "upper_body"})


class RegionMappingError(RuntimeError):
    """Raised when post-mesh region mapping fails the required coverage gate."""


@dataclass(frozen=True)
class RegionMappingResult:
    total_faces: int
    mapped_faces: int
    unmapped_faces: int
    ambiguous_faces: int
    coverage: float
    area_coverage: float
    max_distance_m: float
    mean_distance_m: float
    max_allowed_distance_m: float
    mean_allowed_distance_m: float
    distance_rejected_faces: int
    distance_rule: str
    min_abs_normal_alignment: float
    mean_abs_normal_alignment: float
    normal_rejected_faces: int
    normal_exception_faces: int
    normal_exception_regions: dict[str, int]
    per_region_area_m2: dict[str, float]
    missing_regions: tuple[str, ...]
    output_vtp_path: Path
    report_path: Path
    classification_method: str = "nearest_source_triangle"
    classification_frame: str = "posed_target_to_body_local"
    cross_check: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "classification_method": self.classification_method,
            "classification_frame": self.classification_frame,
            "total_faces": self.total_faces,
            "mapped_faces": self.mapped_faces,
            "unmapped_faces": self.unmapped_faces,
            "ambiguous_faces": self.ambiguous_faces,
            "coverage": self.coverage,
            "area_coverage": self.area_coverage,
            "max_distance_m": self.max_distance_m,
            "mean_distance_m": self.mean_distance_m,
            "max_allowed_distance_m": self.max_allowed_distance_m,
            "mean_allowed_distance_m": self.mean_allowed_distance_m,
            "distance_rejected_faces": self.distance_rejected_faces,
            "distance_rule": self.distance_rule,
            "min_abs_normal_alignment": self.min_abs_normal_alignment,
            "mean_abs_normal_alignment": self.mean_abs_normal_alignment,
            "normal_rejected_faces": self.normal_rejected_faces,
            "normal_exception_faces": self.normal_exception_faces,
            "normal_exception_regions": self.normal_exception_regions,
            "per_region_area_m2": self.per_region_area_m2,
            "missing_regions": list(self.missing_regions),
            "output_vtp_path": str(self.output_vtp_path),
            "report_path": str(self.report_path),
            "cross_check": self.cross_check,
        }


@dataclass(frozen=True)
class _NearestFaces:
    distance_m: FloatArray
    index: IntArray
    ambiguous: NDArray[np.bool_]


@dataclass(frozen=True)
class _MappingArrays:
    mapped: NDArray[np.bool_]
    mapped_region_ids: NDArray[np.int32]
    mapped_region_names: NDArray[np.object_]
    nearest_distance_m: FloatArray
    normal_alignment: FloatArray
    abs_normal_alignment: FloatArray
    distance_limit_m: FloatArray
    ambiguous: NDArray[np.bool_]
    normal_exception: NDArray[np.bool_]
    target_areas_m2: FloatArray


def _cell_centres(poly: pv.PolyData) -> FloatArray:
    centres = poly.cell_centers().points
    return np.asarray(centres, dtype=np.float64)


def _source_region_arrays(regions: dict[str, Any]) -> tuple[IntArray, NDArray[np.object_]]:
    face_regions = regions["face_regions"]
    ids = np.array([item["region_id"] for item in face_regions], dtype=np.int64)
    names = np.array([item["region"] for item in face_regions], dtype=object)
    return ids, names


def _target_as_polydata(target_surface: pv.DataSet) -> pv.PolyData:
    if isinstance(target_surface, pv.PolyData):
        return target_surface
    return target_surface.extract_surface()


def _cell_normals(poly: pv.PolyData) -> FloatArray:
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


def _cell_areas(poly: pv.PolyData) -> FloatArray:
    with_sizes = poly.compute_cell_sizes(length=False, area=True, volume=False)
    return np.asarray(with_sizes.cell_data["Area"], dtype=np.float64)


def _nearest_source_faces(
    *,
    source_centres: FloatArray,
    target_centres: FloatArray,
    ambiguous_distance_m: float,
) -> _NearestFaces:
    query_count = NEAREST_NEIGHBOUR_COUNT if len(source_centres) > 1 else 1
    distances, indices = cKDTree(source_centres).query(target_centres, k=query_count)
    distances_2d = np.atleast_2d(distances)
    indices_2d = np.atleast_2d(indices)
    if distances_2d.shape[0] == 1 and len(target_centres) != 1:
        distances_2d = distances_2d.T
        indices_2d = indices_2d.T

    nearest_distance = np.asarray(distances_2d[:, 0], dtype=np.float64)
    nearest_index = np.asarray(indices_2d[:, 0], dtype=np.int64)
    if query_count == NEAREST_NEIGHBOUR_COUNT:
        second_distance = np.asarray(distances_2d[:, 1], dtype=np.float64)
        ambiguous = (second_distance - nearest_distance) <= ambiguous_distance_m
    else:
        ambiguous = np.zeros(len(target_centres), dtype=bool)
    return _NearestFaces(
        distance_m=nearest_distance,
        index=nearest_index,
        ambiguous=ambiguous,
    )


def _build_mapping_arrays(
    *,
    nearest: _NearestFaces,
    source_normals: FloatArray,
    target_normals: FloatArray,
    target_areas: FloatArray,
    source_region_ids: IntArray,
    source_region_names: NDArray[np.object_],
    min_abs_normal_alignment: float,
    edge_min_abs_normal_alignment: float,
    normal_exception_regions: frozenset[str],
    max_distance_face_scale: float,
    max_distance_m: float | None,
) -> _MappingArrays:
    normal_alignment = np.einsum(
        "ij,ij->i",
        target_normals,
        source_normals[nearest.index],
    )
    abs_normal_alignment = np.abs(normal_alignment)
    nearest_region_names = source_region_names[nearest.index]
    strict_normal_consistent = abs_normal_alignment >= min_abs_normal_alignment
    edge_region = np.isin(nearest_region_names, list(normal_exception_regions))
    normal_exception = (
        ~strict_normal_consistent
        & edge_region
        & (abs_normal_alignment >= edge_min_abs_normal_alignment)
    )
    normal_consistent = strict_normal_consistent | normal_exception
    local_distance_limit = max_distance_face_scale * np.sqrt(target_areas)
    if max_distance_m is not None:
        local_distance_limit = np.minimum(local_distance_limit, max_distance_m)
    mapped = (nearest.distance_m <= local_distance_limit) & normal_consistent
    ambiguous = mapped & nearest.ambiguous

    mapped_region_ids = np.full(len(target_areas), -1, dtype=np.int32)
    mapped_region_names = np.full(len(target_areas), "unmapped", dtype=object)
    mapped_region_ids[mapped] = source_region_ids[nearest.index[mapped]].astype(np.int32)
    mapped_region_names[mapped] = source_region_names[nearest.index[mapped]]
    return _MappingArrays(
        mapped=mapped,
        mapped_region_ids=mapped_region_ids,
        mapped_region_names=mapped_region_names,
        nearest_distance_m=nearest.distance_m,
        normal_alignment=normal_alignment,
        abs_normal_alignment=abs_normal_alignment,
        distance_limit_m=local_distance_limit,
        ambiguous=ambiguous,
        normal_exception=mapped & normal_exception,
        target_areas_m2=target_areas,
    )


def _write_mapped_polydata(
    poly: pv.PolyData,
    arrays: _MappingArrays,
    output_vtp_path: Path,
) -> None:
    poly.cell_data["region_id"] = arrays.mapped_region_ids
    poly.cell_data["surface_region_id"] = arrays.mapped_region_ids
    poly.cell_data["surface_region"] = arrays.mapped_region_names
    poly.cell_data["source_region_distance_m"] = arrays.nearest_distance_m
    poly.cell_data["source_region_distance_limit_m"] = arrays.distance_limit_m
    poly.cell_data["source_region_normal_alignment"] = arrays.normal_alignment
    poly.cell_data["source_region_abs_normal_alignment"] = arrays.abs_normal_alignment
    poly.cell_data["source_region_ambiguous"] = arrays.ambiguous.astype(np.int8)
    poly.cell_data["source_region_normal_exception"] = arrays.normal_exception.astype(np.int8)
    poly.cell_data["local_face_area_m2"] = arrays.target_areas_m2
    output_vtp_path.parent.mkdir(parents=True, exist_ok=True)
    poly.save(output_vtp_path)


def _write_analytic_polydata(
    poly: pv.PolyData,
    *,
    region_ids: NDArray[np.int32],
    region_names: NDArray[np.object_],
    target_areas: FloatArray,
    body_centres: FloatArray,
    body_normals: FloatArray,
    normal_orientation_flipped: NDArray[np.bool_],
    output_vtp_path: Path,
    cross_check_arrays: _MappingArrays | None = None,
) -> None:
    poly.cell_data["region_id"] = region_ids
    poly.cell_data["surface_region_id"] = region_ids
    poly.cell_data["surface_region"] = region_names
    poly.cell_data["local_face_area_m2"] = target_areas
    poly.cell_data["body_local_centroid_m"] = body_centres
    poly.cell_data["body_local_normal"] = body_normals
    poly.cell_data["target_normal_orientation_flipped"] = normal_orientation_flipped.astype(
        np.int8,
    )
    poly.cell_data["region_classification_method"] = np.full(
        len(region_ids),
        "body_local_analytic",
        dtype=object,
    )
    if cross_check_arrays is not None:
        poly.cell_data["nearest_source_region_id"] = cross_check_arrays.mapped_region_ids
        poly.cell_data["nearest_source_region"] = cross_check_arrays.mapped_region_names
        poly.cell_data["nearest_source_region_distance_m"] = cross_check_arrays.nearest_distance_m
        poly.cell_data["nearest_source_region_distance_limit_m"] = (
            cross_check_arrays.distance_limit_m
        )
        poly.cell_data["nearest_source_region_abs_normal_alignment"] = (
            cross_check_arrays.abs_normal_alignment
        )
        poly.cell_data["nearest_source_region_ambiguous"] = cross_check_arrays.ambiguous.astype(
            np.int8
        )
        poly.cell_data["nearest_source_region_mapped"] = cross_check_arrays.mapped.astype(np.int8)
    output_vtp_path.parent.mkdir(parents=True, exist_ok=True)
    poly.save(output_vtp_path)


def _region_mapping_result(
    *,
    arrays: _MappingArrays,
    source_region_names: NDArray[np.object_],
    min_abs_normal_alignment: float,
    max_distance_face_scale: float,
    max_distance_m: float | None,
    output_vtp_path: Path,
    report_path: Path,
) -> RegionMappingResult:
    mapped_faces = int(np.count_nonzero(arrays.mapped))
    total_faces = len(arrays.mapped)
    total_area = float(np.sum(arrays.target_areas_m2))
    mapped_area = float(np.sum(arrays.target_areas_m2[arrays.mapped]))
    per_region_area = {
        str(name): float(
            np.sum(
                arrays.target_areas_m2[arrays.mapped & (arrays.mapped_region_names == name)],
            ),
        )
        for name in sorted({str(value) for value in source_region_names})
    }
    missing_regions = tuple(name for name, area in per_region_area.items() if area <= 0.0)
    mapped_abs_normals = arrays.abs_normal_alignment[arrays.mapped]
    distance_rejected = arrays.nearest_distance_m > arrays.distance_limit_m
    distance_rule = f"nearest_distance_m <= {max_distance_face_scale:g} * sqrt(local_face_area_m2)"
    if max_distance_m is not None:
        distance_rule += f", capped at {max_distance_m:g} m"
    normal_exception_regions = {
        str(name): int(
            np.count_nonzero(
                arrays.normal_exception & (arrays.mapped_region_names == name),
            ),
        )
        for name in sorted({str(value) for value in arrays.mapped_region_names})
        if name != "unmapped"
        and np.count_nonzero(arrays.normal_exception & (arrays.mapped_region_names == name)) > 0
    }
    return RegionMappingResult(
        total_faces=total_faces,
        mapped_faces=mapped_faces,
        unmapped_faces=total_faces - mapped_faces,
        ambiguous_faces=int(np.count_nonzero(arrays.ambiguous)),
        coverage=mapped_faces / total_faces,
        area_coverage=mapped_area / total_area if total_area > 0.0 else 0.0,
        max_distance_m=float(np.max(arrays.nearest_distance_m)),
        mean_distance_m=float(np.mean(arrays.nearest_distance_m)),
        max_allowed_distance_m=float(np.max(arrays.distance_limit_m)),
        mean_allowed_distance_m=float(np.mean(arrays.distance_limit_m)),
        distance_rejected_faces=int(np.count_nonzero(distance_rejected)),
        distance_rule=distance_rule,
        min_abs_normal_alignment=(
            float(np.min(mapped_abs_normals)) if len(mapped_abs_normals) else 0.0
        ),
        mean_abs_normal_alignment=(
            float(np.mean(mapped_abs_normals)) if len(mapped_abs_normals) else 0.0
        ),
        normal_rejected_faces=int(
            np.count_nonzero(
                (arrays.abs_normal_alignment < min_abs_normal_alignment) & ~arrays.normal_exception,
            ),
        ),
        normal_exception_faces=int(np.count_nonzero(arrays.normal_exception)),
        normal_exception_regions=normal_exception_regions,
        per_region_area_m2=per_region_area,
        missing_regions=missing_regions,
        output_vtp_path=output_vtp_path,
        report_path=report_path,
    )


def _analytic_region_mapping_result(
    *,
    region_ids: NDArray[np.int32],
    region_names: NDArray[np.object_],
    target_areas: FloatArray,
    required_regions: tuple[str, ...],
    output_vtp_path: Path,
    report_path: Path,
    cross_check_arrays: _MappingArrays | None,
    normal_orientation_flipped: NDArray[np.bool_],
) -> RegionMappingResult:
    total_faces = len(region_ids)
    total_area = float(np.sum(target_areas))
    mapped = region_ids >= 0
    per_region_area = {
        name: float(np.sum(target_areas[mapped & (region_names == name)])) for name in REGION_NAMES
    }
    missing_regions = tuple(
        name for name in required_regions if per_region_area.get(name, 0.0) <= 0.0
    )
    if cross_check_arrays is not None:
        cross_mapped = cross_check_arrays.mapped
        mismatch = cross_mapped & (cross_check_arrays.mapped_region_ids != region_ids)
        mismatch_area = float(np.sum(target_areas[mismatch]))
        cross_check = {
            "method": "nearest_source_triangle",
            "mapped_faces": int(np.count_nonzero(cross_mapped)),
            "unmapped_faces": int(np.count_nonzero(~cross_mapped)),
            "face_coverage": float(np.mean(cross_mapped)) if len(cross_mapped) else 0.0,
            "area_coverage": float(np.sum(target_areas[cross_mapped]) / total_area)
            if total_area > 0.0
            else 0.0,
            "distance_rejected_faces": int(
                np.count_nonzero(
                    cross_check_arrays.nearest_distance_m > cross_check_arrays.distance_limit_m,
                ),
            ),
            "normal_rejected_faces": int(
                np.count_nonzero(
                    (cross_check_arrays.abs_normal_alignment < DEFAULT_MIN_ABS_NORMAL_ALIGNMENT)
                    & ~cross_check_arrays.normal_exception,
                ),
            ),
            "ambiguous_faces": int(np.count_nonzero(cross_check_arrays.ambiguous)),
            "label_mismatch_faces": int(np.count_nonzero(mismatch)),
            "label_mismatch_area_fraction": mismatch_area / total_area if total_area > 0.0 else 0.0,
            "max_distance_m": float(np.max(cross_check_arrays.nearest_distance_m)),
            "mean_distance_m": float(np.mean(cross_check_arrays.nearest_distance_m)),
            "target_normal_orientation_flipped_faces": int(
                np.count_nonzero(normal_orientation_flipped),
            ),
            "target_normal_orientation_flipped_fraction": float(
                np.mean(normal_orientation_flipped),
            )
            if len(normal_orientation_flipped)
            else 0.0,
        }
        max_distance = float(np.max(cross_check_arrays.nearest_distance_m))
        mean_distance = float(np.mean(cross_check_arrays.nearest_distance_m))
        max_allowed = float(np.max(cross_check_arrays.distance_limit_m))
        mean_allowed = float(np.mean(cross_check_arrays.distance_limit_m))
        mean_abs_normal = float(np.mean(cross_check_arrays.abs_normal_alignment))
    else:
        cross_check = None
        max_distance = 0.0
        mean_distance = 0.0
        max_allowed = 0.0
        mean_allowed = 0.0
        mean_abs_normal = 1.0
    mapped_faces = int(np.count_nonzero(mapped))
    mapped_area = float(np.sum(target_areas[mapped]))
    return RegionMappingResult(
        total_faces=total_faces,
        mapped_faces=mapped_faces,
        unmapped_faces=total_faces - mapped_faces,
        ambiguous_faces=0,
        coverage=mapped_faces / total_faces if total_faces else 0.0,
        area_coverage=mapped_area / total_area if total_area > 0.0 else 0.0,
        max_distance_m=max_distance,
        mean_distance_m=mean_distance,
        max_allowed_distance_m=max_allowed,
        mean_allowed_distance_m=mean_allowed,
        distance_rejected_faces=0,
        distance_rule="body-local analytic classification; nearest distance is diagnostic only",
        min_abs_normal_alignment=1.0,
        mean_abs_normal_alignment=mean_abs_normal,
        normal_rejected_faces=0,
        normal_exception_faces=0,
        normal_exception_regions={},
        per_region_area_m2=per_region_area,
        missing_regions=missing_regions,
        output_vtp_path=output_vtp_path,
        report_path=report_path,
        classification_method="body_local_analytic",
        classification_frame="posed_target_to_body_local",
        cross_check=cross_check,
    )


def map_surface_regions_to_vtp(
    *,
    source_mesh: trimesh.Trimesh,
    source_regions: dict[str, Any],
    target_surface: pv.DataSet,
    output_vtp_path: Path,
    report_path: Path,
    max_distance_m: float | None = None,
    max_distance_face_scale: float = DEFAULT_MAX_DISTANCE_FACE_SCALE,
    ambiguous_distance_m: float = 1e-6,
    min_abs_normal_alignment: float = DEFAULT_MIN_ABS_NORMAL_ALIGNMENT,
    edge_min_abs_normal_alignment: float = DEFAULT_EDGE_MIN_ABS_NORMAL_ALIGNMENT,
    normal_exception_regions: frozenset[str] = NORMAL_EXCEPTION_REGIONS,
    min_coverage: float = MIN_MAPPING_COVERAGE,
) -> RegionMappingResult:
    """Nearest-centroid transfer of source STL ``region_id`` to post-mesh wall faces."""

    if max_distance_m is not None and max_distance_m <= 0.0:
        message = "max_distance_m must be positive when set"
        raise ValueError(message)
    if max_distance_face_scale <= 0.0:
        message = "max_distance_face_scale must be positive"
        raise ValueError(message)
    if not 0.0 < min_coverage <= 1.0:
        message = "min_coverage must be in (0, 1]"
        raise ValueError(message)
    if not 0.0 <= min_abs_normal_alignment <= 1.0:
        message = "min_abs_normal_alignment must be in [0, 1]"
        raise ValueError(message)
    if not 0.0 <= edge_min_abs_normal_alignment <= min_abs_normal_alignment:
        message = "edge_min_abs_normal_alignment must be in [0, min_abs_normal_alignment]"
        raise ValueError(message)

    poly = _target_as_polydata(target_surface).copy(deep=True)
    source_centres = np.asarray(source_mesh.triangles_center, dtype=np.float64)
    source_normals = np.asarray(source_mesh.face_normals, dtype=np.float64)
    target_centres = _cell_centres(poly)
    target_normals = _cell_normals(poly)
    target_areas = _cell_areas(poly)
    source_region_ids, source_region_names = _source_region_arrays(source_regions)

    if len(source_centres) != len(source_region_ids):
        message = "source region face count does not match source mesh face count"
        raise ValueError(message)
    if len(source_centres) == 0 or len(target_centres) == 0:
        message = "source and target surfaces must both contain faces"
        raise ValueError(message)

    nearest = _nearest_source_faces(
        source_centres=source_centres,
        target_centres=target_centres,
        ambiguous_distance_m=ambiguous_distance_m,
    )
    arrays = _build_mapping_arrays(
        nearest=nearest,
        source_normals=source_normals,
        target_normals=target_normals,
        target_areas=target_areas,
        source_region_ids=source_region_ids,
        source_region_names=source_region_names,
        min_abs_normal_alignment=min_abs_normal_alignment,
        edge_min_abs_normal_alignment=edge_min_abs_normal_alignment,
        normal_exception_regions=normal_exception_regions,
        max_distance_face_scale=max_distance_face_scale,
        max_distance_m=max_distance_m,
    )
    _write_mapped_polydata(poly, arrays, output_vtp_path)
    result = _region_mapping_result(
        arrays=arrays,
        source_region_names=source_region_names,
        min_abs_normal_alignment=min_abs_normal_alignment,
        max_distance_face_scale=max_distance_face_scale,
        max_distance_m=max_distance_m,
        output_vtp_path=output_vtp_path,
        report_path=report_path,
    )
    atomic_write_json(report_path, result.as_dict())

    if result.area_coverage < min_coverage:
        message = (
            f"surface region mapping area coverage {result.area_coverage:.6f} is below required "
            f"{min_coverage:.6f}; report: {report_path}"
        )
        raise RegionMappingError(message)
    if result.missing_regions:
        message = (
            "surface region mapping missed source regions "
            f"{', '.join(result.missing_regions)}; report: {report_path}"
        )
        raise RegionMappingError(message)
    return result


def map_wall_regions_analytically_to_vtp(
    *,
    source_mesh: trimesh.Trimesh,
    source_regions: dict[str, Any],
    target_surface: pv.DataSet,
    params: AeroParams,
    output_vtp_path: Path,
    report_path: Path,
    max_distance_face_scale: float = DEFAULT_MAX_DISTANCE_FACE_SCALE,
    min_coverage: float = MIN_MAPPING_COVERAGE,
    required_regions: tuple[str, ...] = REGION_NAMES,
) -> RegionMappingResult:
    """Classify final OpenFOAM wall faces in AeroCliff body-local coordinates.

    The nearest source-triangle transfer is retained only as a cross-check.
    Final ``surface_region`` values come from analytical AeroCliff region rules
    applied to the posed wall face centroids/normals after inverse pitch/height.
    """

    if max_distance_face_scale <= 0.0:
        message = "max_distance_face_scale must be positive"
        raise ValueError(message)
    if not 0.0 < min_coverage <= 1.0:
        message = "min_coverage must be in (0, 1]"
        raise ValueError(message)

    poly = _target_as_polydata(target_surface).copy(deep=True)
    if poly.n_cells == 0:
        message = "target surface must contain faces"
        raise ValueError(message)
    target_centres = _cell_centres(poly)
    raw_target_normals = _cell_normals(poly)
    target_areas = _cell_areas(poly)
    source_centres = np.asarray(source_mesh.triangles_center, dtype=np.float64)
    source_normals = np.asarray(source_mesh.face_normals, dtype=np.float64)
    source_region_ids, source_region_names = _source_region_arrays(source_regions)
    if len(source_centres) != len(source_region_ids):
        message = "source region face count does not match source mesh face count"
        raise ValueError(message)
    if len(source_centres) == 0:
        message = "source surface must contain faces"
        raise ValueError(message)
    nearest = _nearest_source_faces(
        source_centres=source_centres,
        target_centres=target_centres,
        ambiguous_distance_m=1e-6,
    )
    nearest_source_normals = source_normals[nearest.index]
    orientation_dot = np.einsum("ij,ij->i", raw_target_normals, nearest_source_normals)
    normal_orientation_flipped = orientation_dot < 0.0
    target_normals = raw_target_normals.copy()
    target_normals[normal_orientation_flipped] *= -1.0
    z_shift_m = infer_ride_height_pitch_z_shift(
        np.asarray(source_mesh.vertices, dtype=np.float64),
        pitch_deg=params.pitch_deg,
    )
    body_centres = inverse_ride_height_pitch(
        target_centres,
        pitch_deg=params.pitch_deg,
        z_shift_m=z_shift_m,
    )
    body_normals = inverse_pitch_normals(target_normals, pitch_deg=params.pitch_deg)
    region_ids = classify_region_arrays(body_centres, body_normals, params)
    region_names = np.asarray(
        [REGION_NAMES[int(region_id)] for region_id in region_ids], dtype=object
    )

    cross_check_arrays = _build_mapping_arrays(
        nearest=nearest,
        source_normals=source_normals,
        target_normals=target_normals,
        target_areas=target_areas,
        source_region_ids=source_region_ids,
        source_region_names=source_region_names,
        min_abs_normal_alignment=DEFAULT_MIN_ABS_NORMAL_ALIGNMENT,
        edge_min_abs_normal_alignment=DEFAULT_EDGE_MIN_ABS_NORMAL_ALIGNMENT,
        normal_exception_regions=NORMAL_EXCEPTION_REGIONS,
        max_distance_face_scale=max_distance_face_scale,
        max_distance_m=None,
    )

    _write_analytic_polydata(
        poly,
        region_ids=region_ids,
        region_names=region_names,
        target_areas=target_areas,
        body_centres=body_centres,
        body_normals=body_normals,
        normal_orientation_flipped=normal_orientation_flipped,
        output_vtp_path=output_vtp_path,
        cross_check_arrays=cross_check_arrays,
    )
    result = _analytic_region_mapping_result(
        region_ids=region_ids,
        region_names=region_names,
        target_areas=target_areas,
        required_regions=required_regions,
        output_vtp_path=output_vtp_path,
        report_path=report_path,
        cross_check_arrays=cross_check_arrays,
        normal_orientation_flipped=normal_orientation_flipped,
    )
    atomic_write_json(report_path, result.as_dict())

    if result.area_coverage < min_coverage:
        message = (
            f"surface region analytical area coverage {result.area_coverage:.6f} is below "
            f"required {min_coverage:.6f}; report: {report_path}"
        )
        raise RegionMappingError(message)
    if result.missing_regions:
        message = (
            "surface region analytical classification missed required regions "
            f"{', '.join(result.missing_regions)}; report: {report_path}"
        )
        raise RegionMappingError(message)
    return result


def trimesh_to_polydata(mesh: trimesh.Trimesh) -> pv.PolyData:
    """Build a triangular ``PolyData`` from a trimesh surface for tests and adapters."""

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    cells = np.column_stack(
        [np.full(len(faces), TRIANGLE_VERTEX_COUNT, dtype=np.int64), faces]
    ).ravel()
    return pv.PolyData(vertices, cells)
