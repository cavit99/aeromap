"""Surface topology-localisation report for surface/mesh unblock decisions."""

from __future__ import annotations

import ast
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from aeromap.geometry.diagnostics import CadShapeLike, cad_topology_summary
from aeromap.geometry.generator import build_article
from aeromap.io import atomic_write_json, atomic_write_text, sha256_file
from aeromap.parameters import AeroParams

TOPOLOGY_REPORT_SCHEMA_VERSION = "gate2a_topology_report_v0.1.0"
REPORT_EDGE_THRESHOLD_M = 1e-3
REPORT_FACE_AREA_THRESHOLD_M2 = 1e-4
TOP_COUNT_LIMIT = 8
SAMPLE_LOCATION_LIMIT = 8
SINGLE_FACE_CLUSTER_FRACTION = 0.60
MULTI_FACE_CLUSTER_FRACTION = 0.75
SINGLE_REGION_CLUSTER_FRACTION = 0.60
BAD_TRIANGLE_COLUMNS = {
    "triangle_index",
    "triangle_quality",
    "area_m2",
    "min_edge_m",
    "aspect_ratio",
    "surface_region",
    "nearest_cad_face_id",
    "centroid_m",
}
MESH_SET_COLUMNS = {
    "diagnostic_element_index",
    "x_m",
    "y_m",
    "z_m",
    "nearest_stl_triangle",
    "nearest_cad_face_id",
    "surface_region",
    "distance_to_surface_m",
}


@dataclass(frozen=True)
class TopologyReportArtifacts:
    report_json_path: Path
    report_markdown_path: Path


def _load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        message = f"expected JSON object: {path}"
        raise TypeError(message)
    return loaded


def _read_csv(path: Path, *, required_columns: set[str]) -> list[dict[str, str]]:
    if not path.exists():
        message = f"required diagnostic CSV is missing: {path}"
        raise FileNotFoundError(message)
    if path.stat().st_size == 0:
        message = f"required diagnostic CSV is empty: {path}"
        raise ValueError(message)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(required_columns - fieldnames)
        if missing:
            message = f"diagnostic CSV {path} is missing required columns: {missing}"
            raise ValueError(message)
        return list(reader)


def _reported_problem_count(raw_summary: dict[str, Any] | None) -> int:
    if raw_summary is None:
        return 0
    for key in ("openfoam_reported_problem_count", "diagnostic_vtk_element_count"):
        value = raw_summary.get(key)
        if isinstance(value, int):
            return value
    return 0


def _mapped_mesh_csvs(mesh_dir: Path) -> dict[str, Path]:
    suffix = "_mapped.csv"
    return {path.name[: -len(suffix)]: path for path in mesh_dir.glob(f"*{suffix}")}


def _as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _parse_centroid(value: str) -> list[float]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        message = f"centroid is not a list: {value}"
        raise TypeError(message)
    return [float(item) for item in parsed]


def _top_counts(rows: list[dict[str, str]], key: str) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row[key]] = counts.get(row[key], 0) + 1
    total = len(rows)
    return [
        {"value": value, "count": count, "fraction": count / total if total else 0.0}
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[
            :TOP_COUNT_LIMIT
        ]
    ]


def _bounds_from_rows(rows: list[dict[str, str]]) -> dict[str, list[float] | None]:
    if not rows:
        return {"min_m": None, "max_m": None}
    xs = [_as_float(row, "x_m") for row in rows]
    ys = [_as_float(row, "y_m") for row in rows]
    zs = [_as_float(row, "z_m") for row in rows]
    return {
        "min_m": [min(xs), min(ys), min(zs)],
        "max_m": [max(xs), max(ys), max(zs)],
    }


def _mesh_locations(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "index": int(row["diagnostic_element_index"]),
            "location_m": [_as_float(row, "x_m"), _as_float(row, "y_m"), _as_float(row, "z_m")],
            "nearest_stl_triangle": int(row["nearest_stl_triangle"]),
            "nearest_cad_face_id": int(row["nearest_cad_face_id"]),
            "surface_region": row["surface_region"],
            "distance_to_surface_m": _as_float(row, "distance_to_surface_m"),
        }
        for row in rows[:SAMPLE_LOCATION_LIMIT]
    ]


def _bad_triangle_locations(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda row: float(row["triangle_quality"]))
    return [
        {
            "triangle_index": int(row["triangle_index"]),
            "triangle_quality": float(row["triangle_quality"]),
            "area_m2": float(row["area_m2"]),
            "min_edge_m": float(row["min_edge_m"]),
            "aspect_ratio": float(row["aspect_ratio"]),
            "surface_region": row["surface_region"],
            "nearest_cad_face_id": int(row["nearest_cad_face_id"]),
            "centroid_m": _parse_centroid(row["centroid_m"]),
        }
        for row in sorted_rows[:SAMPLE_LOCATION_LIMIT]
    ]


def _cluster_label(top_cad_faces: list[dict[str, Any]], top_regions: list[dict[str, Any]]) -> str:
    top_face_fraction = top_cad_faces[0]["fraction"] if top_cad_faces else 0.0
    top_three_face_fraction = sum(item["fraction"] for item in top_cad_faces[:3])
    top_region_fraction = top_regions[0]["fraction"] if top_regions else 0.0
    if top_face_fraction >= SINGLE_FACE_CLUSTER_FRACTION:
        return "single_cad_face_dominant"
    if top_three_face_fraction >= MULTI_FACE_CLUSTER_FRACTION:
        return "multi_cad_face_local_cluster"
    if top_region_fraction >= SINGLE_REGION_CLUSTER_FRACTION:
        return "surface_region_dominant"
    return "distributed_across_multiple_faces_regions"


def _summarize_bad_triangles(rows: list[dict[str, str]]) -> dict[str, Any]:
    top_cad_faces = _top_counts(rows, "nearest_cad_face_id")
    top_regions = _top_counts(rows, "surface_region")
    return {
        "row_count": len(rows),
        "top_cad_faces": top_cad_faces,
        "top_regions": top_regions,
        "cluster_label": _cluster_label(top_cad_faces, top_regions),
        "worst_locations": _bad_triangle_locations(rows),
    }


def _summarize_mesh_set(
    *,
    set_name: str,
    rows: list[dict[str, str]],
    raw_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    top_cad_faces = _top_counts(rows, "nearest_cad_face_id")
    top_regions = _top_counts(rows, "surface_region")
    return {
        "set_name": set_name,
        "openfoam_reported_problem_count": None
        if raw_summary is None
        else raw_summary.get("openfoam_reported_problem_count"),
        "diagnostic_vtk_element_count": len(rows),
        "bounds_m": _bounds_from_rows(rows),
        "top_cad_faces": top_cad_faces,
        "top_regions": top_regions,
        "cluster_label": _cluster_label(top_cad_faces, top_regions),
        "sample_locations": _mesh_locations(rows),
        "raw_summary": raw_summary or {},
    }


def _cad_is_clean_for_gmsh_screen(cad: dict[str, Any]) -> bool:
    edge_counts = cad["edge_length_threshold_counts"]
    face_counts = cad["face_area_threshold_counts"]
    below_edge = [
        item
        for item in edge_counts
        if float(item["threshold"]) <= REPORT_EDGE_THRESHOLD_M and int(item["count_below"]) > 0
    ]
    below_face = [
        item
        for item in face_counts
        if float(item["threshold"]) <= REPORT_FACE_AREA_THRESHOLD_M2
        and int(item["count_below"]) > 0
    ]
    return bool(cad["valid"]) and not below_edge and not below_face


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    cad = report["cad_topology"]
    decision = report["decision"]
    bad = report["bad_stl_triangles"]
    mesh = report["mesh_problem_sets"]
    top_bad_region = bad["top_regions"][0] if bad["top_regions"] else {}
    top_bad_face = bad["top_cad_faces"][0] if bad["top_cad_faces"] else {}
    lines = [
        "# Surface Topology Report",
        "",
        f"- Decision: `{decision['next_step']}`",
        f"- Reason: {decision['reason']}",
        f"- CAD valid: `{cad['valid']}`",
        f"- Minimum CAD edge: `{cad['min_edge']['length_m']} m` at `{cad['min_edge']['center_m']}`",
        f"- Minimum CAD face area: `{cad['min_face']['area_m2']} m^2` at "
        f"`{cad['min_face']['center_m']}`",
        f"- Bad STL triangle cluster: `{bad['cluster_label']}`",
        f"- Bad STL top region: `{top_bad_region}`",
        f"- Bad STL top CAD face: `{top_bad_face}`",
        f"- Construction-feature resolution: {report['construction_feature_resolution']}",
        "",
        "## Mesh Problem Sets",
        "",
    ]
    for set_name, summary in mesh.items():
        top_region = summary["top_regions"][0] if summary["top_regions"] else {}
        top_face = summary["top_cad_faces"][0] if summary["top_cad_faces"] else {}
        lines.extend(
            [
                f"### {set_name}",
                "",
                f"- OpenFOAM count: `{summary['openfoam_reported_problem_count']}`",
                f"- Diagnostic VTK elements: `{summary['diagnostic_vtk_element_count']}`",
                f"- Cluster label: `{summary['cluster_label']}`",
                f"- Top region: `{top_region}`",
                f"- Top CAD face: `{top_face}`",
                "",
            ],
        )
    atomic_write_text(path, "\n".join(lines) + "\n")


def write_topology_report(
    *,
    case_dir: Path,
    surface_diagnostics_path: Path,
    mesh_diagnostics_path: Path,
    out_dir: Path,
) -> TopologyReportArtifacts:
    """Write the bounded surface topology report required before Gmsh experiments."""

    params = AeroParams(**_load_json(case_dir / "params.json"))
    manifest = _load_json(case_dir / "manifest.json")
    shape = cast("CadShapeLike", build_article(params).val())
    cad = cad_topology_summary(shape)
    surface = _load_json(surface_diagnostics_path)
    mesh = _load_json(mesh_diagnostics_path)

    surface_dir = surface_diagnostics_path.parent
    mesh_dir = mesh_diagnostics_path.parent
    bad_rows = _read_csv(
        surface_dir / "bad_stl_triangles.csv",
        required_columns=BAD_TRIANGLE_COLUMNS,
    )
    raw_sets = {
        item["set_name"]: item
        for item in mesh.get("checkmesh_sets", [])
        if isinstance(item, dict) and isinstance(item.get("set_name"), str)
    }
    mapped_csvs = _mapped_mesh_csvs(mesh_dir)
    mesh_sets: dict[str, Any] = {}
    for name in sorted(set(raw_sets) | set(mapped_csvs)):
        raw_summary = raw_sets.get(name)
        csv_path = mapped_csvs.get(name)
        if csv_path is None:
            if _reported_problem_count(raw_summary) > 0:
                message = f"reported checkMesh set {name!r} is missing mapped CSV evidence"
                raise FileNotFoundError(message)
            rows: list[dict[str, str]] = []
        else:
            rows = _read_csv(csv_path, required_columns=MESH_SET_COLUMNS)
        mesh_sets[name] = _summarize_mesh_set(
            set_name=name,
            rows=rows,
            raw_summary=raw_summary,
        )

    cad_clean = _cad_is_clean_for_gmsh_screen(cad)
    decision = {
        "cad_clean_for_bounded_gmsh_screen": cad_clean,
        "next_step": (
            "run_bounded_gmsh_g0_g1_surface_export" if cad_clean else "repair_cad_before_gmsh"
        ),
        "reason": (
            "No CAD edges below 1 mm and no CAD faces below 1e-4 m^2 were found; "
            "use Gmsh only as a bounded surface tessellation comparison."
            if cad_clean
            else "Microscopic CAD topology is present at the configured thresholds; "
            "repair the CAD construction before changing surface exporters."
        ),
        "do_not_run_solver": True,
        "do_not_add_layers_until_no_layer_mesh_passes": True,
    }

    report: dict[str, Any] = {
        "schema_version": TOPOLOGY_REPORT_SCHEMA_VERSION,
        "case_dir": str(case_dir),
        "simulation_id": manifest["simulation_id"],
        "geometry_id": manifest["geometry_id"],
        "surface_diagnostics_path": str(surface_diagnostics_path),
        "mesh_diagnostics_path": str(mesh_diagnostics_path),
        "input_hashes": {
            "surface_diagnostics": sha256_file(surface_diagnostics_path),
            "mesh_diagnostics": sha256_file(mesh_diagnostics_path),
            "source_step": sha256_file(case_dir / "geometry" / "article_body_datum.step"),
            "source_stl": sha256_file(case_dir / "geometry" / "article.stl"),
        },
        "cad_topology": cad,
        "surface_diagnostics_summary": surface,
        "bad_stl_triangles": _summarize_bad_triangles(bad_rows),
        "mesh_problem_sets": mesh_sets,
        "construction_feature_resolution": (
            "CAD construction feature names are not encoded in the BRep. Use "
            "cad_faces_by_id.vtp and mapped checkMesh VTK/CSV artifacts to visually "
            "classify top CAD face IDs as loft, fillet, cap, or Boolean seam before "
            "changing geometry."
        ),
        "decision": decision,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "topology_report.json"
    markdown_path = out_dir / "topology_report.md"
    atomic_write_json(json_path, report)
    _write_markdown(markdown_path, report)
    return TopologyReportArtifacts(
        report_json_path=json_path,
        report_markdown_path=markdown_path,
    )
