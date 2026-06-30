"""Small parsers for OpenFOAM smoke evidence."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyvista as pv

if TYPE_CHECKING:
    from numpy.typing import NDArray

    FloatArray = NDArray[np.float64]
else:
    FloatArray = np.ndarray

MESH_OK_RE = re.compile(r"Mesh OK", re.IGNORECASE)
CELL_COUNT_RE = re.compile(r"cells:\s+(\d+)")
FAILED_MESH_CHECKS_RE = re.compile(r"Failed\s+(\d+)\s+mesh checks?", re.IGNORECASE)
NEGATIVE_VOLUME_RE = re.compile(
    r"(?:negative[-\s]+(?:cell[-\s]+)?volume|negative[-\s]+volume[-\s]+cells?|"
    r"zero[-\s]+or[-\s]+negative[-\s]+(?:cell[-\s]+)?volume)",
    re.IGNORECASE,
)
NEGATIVE_VOLUME_COUNT_RE = re.compile(
    r"(?:negative[-\s]+(?:cell[-\s]+)?volume|negative[-\s]+volume[-\s]+cells?|"
    r"zero[-\s]+or[-\s]+negative[-\s]+(?:cell[-\s]+)?volume)"
    r"[^0-9\n]*(?:number\s+of\s+(?:cells|faces)\s*[:=]\s*)?(?P<count>\d+)",
    re.IGNORECASE,
)
ILLEGAL_FACE_NEGATIVE_VOLUME_RE = re.compile(
    r"Detected\s+(?P<count>\d+)\s+illegal\s+faces.*negative.*volume",
    re.IGNORECASE,
)
FEATURE_EDGES_RE = re.compile(r"Set displacement to zero for points on\s+(\d+)\s+feature edges")
EXTRUSION_RE = re.compile(
    r"Extruding\s+(?P<faces>\d+)\s+out of\s+(?P<total>\d+)\s+faces\s+"
    r"\((?P<percent>[0-9.]+)%\)\. Removed extrusion at\s+(?P<removed>\d+)\s+faces\.",
)
ADDED_CELLS_RE = re.compile(
    r"Added\s+(?P<cells>\d+)\s+out of\s+(?P<total>\d+)\s+cells\s+"
    r"\((?P<percent>[0-9.]+)%\)\.",
)
MEDIAL_REDUCTION_RE = re.compile(
    r"Reducing layer thickness at\s+(?P<count>\d+)\s+nodes where thickness to medial axis",
)
ISOLATED_POINTS_RE = re.compile(
    r"Number of isolated points extrusion stopped\s+:\s+(?P<count>\d+)",
)
ARTICLE_LAYER_RE = re.compile(
    r"^article\s+(?P<faces>\d+)\s+(?P<layers>[0-9.eE+-]+)\s+"
    r"(?P<thickness_m>[0-9.eE+-]+)\s+(?P<thickness_percent>[0-9.eE+-]+)",
    re.MULTILINE,
)
PATCH_LAYER_RE = re.compile(
    r"^(?P<patch>[A-Za-z_][A-Za-z0-9_]*)\s+(?P<faces>\d+)\s+"
    r"(?P<layers>[0-9.eE+-]+)\s+(?P<thickness_m>[0-9.eE+-]+)\s+"
    r"(?P<thickness_percent>[0-9.eE+-]+)",
    re.MULTILINE,
)
LAYER_ITERATION_RE = re.compile(r"^Layer addition iteration\s+(?P<iteration>\d+)")
ILLEGAL_FACES_RE = re.compile(
    r"Detected\s+(?P<count>\d+)\s+illegal faces "
    r"\(concave, zero area or negative cell pyramid volume\)",
)
QUALITY_COUNT_RE = re.compile(r"^\s*(?P<label>.+?)\s+:\s+(?P<count>\d+)\s*$")
CRITICAL_GATE2B_PATCHES = ("tunnel_roofs_core", "diffuser_core", "underfloor_core")
CRITICAL_UNDERFLOOR_PATCHES = ("critical_underfloor",)
GATE2B_AREA_COVERAGE_MIN = 0.80

QUALITY_LABELS = {
    "non-orthogonality": "non_orthogonality_faces",
    "face-decomposition tet quality": "low_tet_quality_faces",
    "concavity": "concave_faces",
    "skewness": "skew_faces",
    "interpolation weights": "low_interpolation_weight_faces",
    "volume ratio": "low_volume_ratio_faces",
    "face twist": "low_face_twist_faces",
    "determinant": "low_determinant_faces",
}


def parse_check_mesh_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    cell_match = CELL_COUNT_RE.search(text)
    failed_match = FAILED_MESH_CHECKS_RE.search(text)
    return {
        "mesh_ok": bool(MESH_OK_RE.search(text)),
        "cells": int(cell_match.group(1)) if cell_match else None,
        "failed_mesh_checks": int(failed_match.group(1)) if failed_match else 0,
        "contains_negative_volume": contains_negative_volume_failure(text),
    }


def contains_negative_volume_failure(text: str) -> bool:
    """Return true only for explicit positive-count negative-volume diagnostics."""

    for line in text.splitlines():
        illegal_match = ILLEGAL_FACE_NEGATIVE_VOLUME_RE.search(line)
        if illegal_match and int(illegal_match.group("count")) > 0:
            return True
        if not NEGATIVE_VOLUME_RE.search(line):
            continue
        lowered = line.lower()
        if "no negative" in lowered or "without negative" in lowered:
            continue
        count_match = NEGATIVE_VOLUME_COUNT_RE.search(line)
        if count_match and int(count_match.group("count")) > 0:
            return True
    return False


def _summary(values: list[int]) -> dict[str, int | None]:
    if not values:
        return {"first": None, "last": None, "max": None}
    return {"first": values[0], "last": values[-1], "max": max(values)}


def parse_snappy_layer_log(path: Path) -> dict[str, Any]:
    """Parse OpenFOAM snappyHexMesh layer-attempt evidence from a log file."""

    text = path.read_text(encoding="utf-8", errors="replace")
    extrusion_rows = [
        {
            "faces": int(match.group("faces")),
            "total_faces": int(match.group("total")),
            "percent": float(match.group("percent")),
            "removed_faces": int(match.group("removed")),
        }
        for match in EXTRUSION_RE.finditer(text)
    ]
    added_cell_rows = [
        {
            "cells": int(match.group("cells")),
            "total_cells": int(match.group("total")),
            "percent": float(match.group("percent")),
        }
        for match in ADDED_CELLS_RE.finditer(text)
    ]
    medial_reductions = [int(match.group("count")) for match in MEDIAL_REDUCTION_RE.finditer(text)]
    isolated_points = [int(match.group("count")) for match in ISOLATED_POINTS_RE.finditer(text)]
    feature_edges = FEATURE_EDGES_RE.search(text)
    article_layer_rows = list(ARTICLE_LAYER_RE.finditer(text))
    patch_layer_rows = list(PATCH_LAYER_RE.finditer(text))
    article_layers = article_layer_rows[-1] if article_layer_rows else None
    final_patch_layers = {
        match.group("patch"): {
            "faces": int(match.group("faces")),
            "average_layers": float(match.group("layers")),
            "overall_thickness_m": float(match.group("thickness_m")),
            "overall_thickness_percent": float(match.group("thickness_percent")),
        }
        for match in patch_layer_rows
    }
    final_extrusion = extrusion_rows[-1] if extrusion_rows else None
    final_added_cells = added_cell_rows[-1] if added_cell_rows else None
    final_article_layers = (
        {
            "faces": int(article_layers.group("faces")),
            "average_layers": float(article_layers.group("layers")),
            "overall_thickness_m": float(article_layers.group("thickness_m")),
            "overall_thickness_percent": float(article_layers.group("thickness_percent")),
        }
        if article_layers is not None
        else None
    )
    return {
        "schema_version": "snappy_layer_attempt_v0.1.0",
        "path": str(path),
        "feature_edges_zeroed": int(feature_edges.group(1)) if feature_edges else None,
        "iteration_count": len(extrusion_rows),
        "first_extrusion": extrusion_rows[0] if extrusion_rows else None,
        "final_extrusion": final_extrusion,
        "final_added_cells": final_added_cells,
        "final_article_layers": final_article_layers,
        "final_patch_layers": final_patch_layers,
        "max_extruded_faces": max((row["faces"] for row in extrusion_rows), default=None),
        "medial_axis_reduction_nodes": medial_reductions,
        "medial_axis_reduction_summary": _summary(medial_reductions),
        "isolated_points_stopped": isolated_points,
        "isolated_points_summary": _summary(isolated_points),
        "contains_layer_mesh": "Layer mesh :" in text,
    }


def _quality_key(label: str) -> str | None:
    for needle, key in QUALITY_LABELS.items():
        if needle in label:
            return key
    return None


def _new_layer_iteration(iteration: int) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "medial_axis_reduction_nodes": None,
        "isolated_points_stopped": None,
        "illegal_faces": None,
        "extruded_faces": None,
        "total_faces": None,
        "extruded_face_fraction": None,
        "removed_faces_reported": None,
        "added_cells": None,
        "total_cells": None,
        "added_cell_fraction": None,
    }


def _update_layer_iteration(
    row: dict[str, Any],
    quality_counts: dict[str, int],
    line: str,
) -> None:
    medial_match = MEDIAL_REDUCTION_RE.search(line)
    if medial_match:
        row["medial_axis_reduction_nodes"] = int(medial_match.group("count"))
        return
    isolated_match = ISOLATED_POINTS_RE.search(line)
    if isolated_match:
        row["isolated_points_stopped"] = int(isolated_match.group("count"))
        return
    illegal_match = ILLEGAL_FACES_RE.search(line)
    if illegal_match:
        row["illegal_faces"] = int(illegal_match.group("count"))
        return
    extrusion_match = EXTRUSION_RE.search(line)
    if extrusion_match:
        faces = int(extrusion_match.group("faces"))
        total_faces = int(extrusion_match.group("total"))
        row["extruded_faces"] = faces
        row["total_faces"] = total_faces
        row["extruded_face_fraction"] = faces / total_faces if total_faces else 0.0
        row["removed_faces_reported"] = int(extrusion_match.group("removed"))
        return
    added_match = ADDED_CELLS_RE.search(line)
    if added_match:
        cells = int(added_match.group("cells"))
        total_cells = int(added_match.group("total"))
        row["added_cells"] = cells
        row["total_cells"] = total_cells
        row["added_cell_fraction"] = cells / total_cells if total_cells else 0.0
        return
    quality_match = QUALITY_COUNT_RE.match(line)
    if quality_match:
        key = _quality_key(quality_match.group("label"))
        if key is not None:
            quality_counts[key] = int(quality_match.group("count"))


def _append_layer_iteration(
    rows: list[dict[str, Any]],
    row: dict[str, Any] | None,
    quality_counts: dict[str, int],
) -> None:
    if row is None:
        return
    row["quality_counts"] = dict(quality_counts)
    rows.append(row)


def _add_layer_iteration_deltas(rows: list[dict[str, Any]]) -> None:
    previous_faces: int | None = None
    for row in rows:
        faces = row["extruded_faces"]
        row["extruded_faces_delta_from_previous"] = (
            None if previous_faces is None or faces is None else faces - previous_faces
        )
        if faces is not None:
            previous_faces = int(faces)


def parse_snappy_layer_retention_log(path: Path) -> dict[str, Any]:
    """Parse per-iteration layer retention and rejection metrics from snappy logs."""

    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    current_quality: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        iteration_match = LAYER_ITERATION_RE.match(line)
        if iteration_match:
            _append_layer_iteration(rows, current, current_quality)
            current = _new_layer_iteration(int(iteration_match.group("iteration")))
            current_quality = {}
            continue
        if current is None:
            continue
        _update_layer_iteration(current, current_quality, line)

    _append_layer_iteration(rows, current, current_quality)
    _add_layer_iteration_deltas(rows)
    extruded = [int(row["extruded_faces"]) for row in rows if row["extruded_faces"] is not None]
    first_faces = extruded[0] if extruded else None
    final_faces = extruded[-1] if extruded else None
    max_faces = max(extruded) if extruded else None
    total_faces = next((int(row["total_faces"]) for row in rows if row["total_faces"]), None)
    return {
        "schema_version": "snappy_layer_retention_v0.1.0",
        "path": str(path),
        "iteration_count": len(rows),
        "iterations": rows,
        "summary": {
            "first_extruded_faces": first_faces,
            "final_extruded_faces": final_faces,
            "max_extruded_faces": max_faces,
            "total_faces": total_faces,
            "first_extruded_face_fraction": (
                first_faces / total_faces if first_faces is not None and total_faces else None
            ),
            "final_extruded_face_fraction": (
                final_faces / total_faces if final_faces is not None and total_faces else None
            ),
            "retained_fraction_of_first_extrusion": (
                final_faces / first_faces if final_faces is not None and first_faces else None
            ),
            "total_extruded_faces_lost": (
                first_faces - final_faces
                if first_faces is not None and final_faces is not None
                else None
            ),
        },
    }


def _cell_scalar(poly: pv.DataSet, name: str) -> FloatArray:
    if name in poly.cell_data:
        return np.asarray(poly.cell_data[name], dtype=np.float64).reshape(-1)
    if name in poly.point_data:
        point_values = np.asarray(poly.point_data[name], dtype=np.float64).reshape(-1)
        cells = np.asarray(poly.faces, dtype=np.int64)
        values: list[float] = []
        cursor = 0
        while cursor < len(cells):
            count = int(cells[cursor])
            ids = cells[cursor + 1 : cursor + 1 + count]
            cursor += count + 1
            values.append(float(np.mean(point_values[ids])))
        return np.asarray(values, dtype=np.float64)
    message = f"VTK patch has no scalar field {name!r}"
    raise KeyError(message)


def _cell_areas(poly: pv.PolyData) -> FloatArray:
    with_sizes = poly.compute_cell_sizes(length=False, area=True, volume=False)
    return np.asarray(with_sizes.cell_data["Area"], dtype=np.float64)


def mesh_layer_coverage_from_vtk(
    *,
    case_dir: Path,
    patch_names: tuple[str, ...],
    critical_patches: tuple[str, ...] | None = None,
    min_area_coverage: float = GATE2B_AREA_COVERAGE_MIN,
    time_name: str = "0",
) -> dict[str, Any]:
    """Report area-weighted layer coverage from foamToVTK boundary patch outputs."""

    if critical_patches is None:
        critical_patches = (
            CRITICAL_UNDERFLOOR_PATCHES
            if "critical_underfloor" in patch_names
            else CRITICAL_GATE2B_PATCHES
        )

    vtk_root = case_dir / "openfoam" / "VTK"
    per_patch: dict[str, Any] = {}
    missing: list[str] = []
    for patch in patch_names:
        vtk_path = vtk_root / patch / f"{patch}_{time_name}.vtk"
        if not vtk_path.exists():
            missing.append(patch)
            continue
        loaded = pv.read(vtk_path)
        poly = loaded if isinstance(loaded, pv.PolyData) else loaded.extract_surface()
        layers = _cell_scalar(poly, "nSurfaceLayers")
        areas = _cell_areas(poly)
        if len(layers) != len(areas):
            message = f"patch {patch} layer/area length mismatch: {len(layers)} != {len(areas)}"
            raise ValueError(message)
        layered = layers > 0.0
        total_area = float(np.sum(areas))
        layered_area = float(np.sum(areas[layered]))
        per_patch[patch] = {
            "vtk_path": str(vtk_path),
            "face_count": int(poly.n_cells),
            "area_m2": total_area,
            "layered_area_m2": layered_area,
            "area_fraction_with_layers": layered_area / total_area if total_area > 0.0 else 0.0,
            "faces_with_layers": int(np.count_nonzero(layered)),
            "face_fraction_with_layers": float(np.mean(layered)) if len(layered) else 0.0,
            "mean_layers_area_weighted": float(np.sum(layers * areas) / total_area)
            if total_area > 0.0
            else float("nan"),
            "max_layers": float(np.max(layers)) if len(layers) else 0.0,
        }

    critical = {
        patch: per_patch.get(patch, {}).get("area_fraction_with_layers", 0.0)
        for patch in critical_patches
    }
    missing_critical = [patch for patch in critical_patches if patch in missing]
    return {
        "schema_version": "mesh_layer_coverage_v0.1.0",
        "source": str(vtk_root),
        "time_name": time_name,
        "patches": per_patch,
        "missing_patches": missing,
        "missing_critical_patches": missing_critical,
        "critical_patches": list(critical_patches),
        "min_area_coverage": min_area_coverage,
        "critical_area_coverage": critical,
        "critical_area_coverage_ok": bool(
            not missing_critical
            and all(coverage >= min_area_coverage for coverage in critical.values())
        ),
    }
