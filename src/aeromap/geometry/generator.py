"""CadQuery Venturi-underfloor benchmark generator."""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path

import cadquery as cq
import numpy as np
import plotly.graph_objects as go
import trimesh
import yaml

from aeromap.constants import GEOMETRY_GENERATOR_VERSION
from aeromap.geometry.regions import (
    DIFFUSER_EXIT_X_M,
    THROAT_BASE_X_M,
    THROAT_HEIGHT_M,
    TUNNEL_FLOOR_Z_M,
    tunnel_design_metadata,
    write_surface_regions,
)
from aeromap.geometry.schema import GeometryArtifacts, GeometryMetrics
from aeromap.geometry.validate import validate_mesh
from aeromap.io import atomic_write_json, atomic_write_text, sha256_file
from aeromap.parameters import AeroParams
from aeromap.transforms import apply_ride_height_pitch

OUTER_HALF_WIDTH_M = 0.50
OUTER_EDGE_RADIUS_M = 0.030
OUTER_SIDE_TOP_Z_M = 0.078
OUTER_CENTER_CROWN_Z_M = 0.102
OUTER_END_SIDE_TOP_Z_M = 0.074
OUTER_END_CENTER_CROWN_Z_M = 0.092
MIN_TUNNEL_SOLID_LIGAMENT_M = 0.020
TUNNEL_HALF_WIDTH_M = 0.142
TUNNEL_CENTRES_Y_M = (-0.245, 0.245)
STABLE_REFERENCE_TUNNEL_HALF_WIDTH_M = 0.320
STABLE_REFERENCE_TUNNEL_CENTRES_Y_M = (0.0,)
STABLE_REFERENCE_INLET_HEIGHT_M = 0.055
STABLE_REFERENCE_THROAT_HEIGHT_M = 0.042


def _vec(x_pos: float, y_pos: float, z_pos: float) -> cq.Vector:
    return cq.Vector(float(x_pos), float(y_pos), float(z_pos))


def _line(start: cq.Vector, end: cq.Vector) -> cq.Edge:
    return cq.Edge.makeLine(start, end)


def _arc(start: cq.Vector, midpoint: cq.Vector, end: cq.Vector) -> cq.Edge:
    return cq.Edge.makeThreePointArc(start, midpoint, end)


def _wire_from_edges(edges: Sequence[cq.Edge]) -> cq.Wire:
    wire = cq.Wire.assembleEdges(list(edges))
    if not wire.IsClosed():
        message = "constructed CAD section wire is not closed"
        raise ValueError(message)
    return wire


def _tunnel_roof_height(x_mid: float, params: AeroParams) -> float:
    if params.geometry_family == "stable_reference":
        throat_x = THROAT_BASE_X_M + params.throat_offset_mm / 1000.0
        diffuser_length = DIFFUSER_EXIT_X_M - throat_x
        exit_height = (
            STABLE_REFERENCE_THROAT_HEIGHT_M
            + np.tan(
                np.deg2rad(params.diffuser_angle_deg),
            )
            * diffuser_length
        )
        if x_mid <= throat_x:
            blend = np.clip((x_mid - 0.20) / (throat_x - 0.20), 0.0, 1.0)
            return float(
                (1.0 - blend) * STABLE_REFERENCE_INLET_HEIGHT_M
                + blend * STABLE_REFERENCE_THROAT_HEIGHT_M,
            )

        blend = np.clip((x_mid - throat_x) / diffuser_length, 0.0, 1.0)
        return float((1.0 - blend) * STABLE_REFERENCE_THROAT_HEIGHT_M + blend * exit_height)

    throat_x = THROAT_BASE_X_M + params.throat_offset_mm / 1000.0
    inlet_height = 0.055
    diffuser_length = DIFFUSER_EXIT_X_M - throat_x
    exit_height = THROAT_HEIGHT_M + np.tan(np.deg2rad(params.diffuser_angle_deg)) * diffuser_length
    if x_mid <= throat_x:
        blend = np.clip((x_mid - 0.20) / (throat_x - 0.20), 0.0, 1.0)
        return float((1.0 - blend) * inlet_height + blend * THROAT_HEIGHT_M)

    blend = np.clip((x_mid - throat_x) / diffuser_length, 0.0, 1.0)
    return float((1.0 - blend) * THROAT_HEIGHT_M + blend * exit_height)


def _rounded_rect_wire(
    *,
    x_pos: float,
    y_center: float,
    half_width: float,
    z_min: float,
    z_max: float,
    radius: float,
) -> cq.Wire:
    y_min = y_center - half_width
    y_max = y_center + half_width
    if radius <= 0.0:
        points = [
            _vec(x_pos, y_min, z_min),
            _vec(x_pos, y_max, z_min),
            _vec(x_pos, y_max, z_max),
            _vec(x_pos, y_min, z_max),
        ]
        return _wire_from_edges(
            [
                _line(points[0], points[1]),
                _line(points[1], points[2]),
                _line(points[2], points[3]),
                _line(points[3], points[0]),
            ],
        )

    if radius >= min(half_width, (z_max - z_min) / 2.0):
        message = f"rounded rectangle radius {radius:g} m is infeasible"
        raise ValueError(message)

    root_half = radius / np.sqrt(2.0)
    bottom_right_center = (y_max - radius, z_min + radius)
    top_right_center = (y_max - radius, z_max - radius)
    top_left_center = (y_min + radius, z_max - radius)
    bottom_left_center = (y_min + radius, z_min + radius)

    p0 = _vec(x_pos, y_min + radius, z_min)
    p1 = _vec(x_pos, y_max - radius, z_min)
    p2 = _vec(x_pos, y_max, z_min + radius)
    p3 = _vec(x_pos, y_max, z_max - radius)
    p4 = _vec(x_pos, y_max - radius, z_max)
    p5 = _vec(x_pos, y_min + radius, z_max)
    p6 = _vec(x_pos, y_min, z_max - radius)
    p7 = _vec(x_pos, y_min, z_min + radius)

    return _wire_from_edges(
        [
            _line(p0, p1),
            _arc(
                p1,
                _vec(x_pos, bottom_right_center[0] + root_half, bottom_right_center[1] - root_half),
                p2,
            ),
            _line(p2, p3),
            _arc(
                p3,
                _vec(x_pos, top_right_center[0] + root_half, top_right_center[1] + root_half),
                p4,
            ),
            _line(p4, p5),
            _arc(
                p5,
                _vec(x_pos, top_left_center[0] - root_half, top_left_center[1] + root_half),
                p6,
            ),
            _line(p6, p7),
            _arc(
                p7,
                _vec(
                    x_pos,
                    bottom_left_center[0] - root_half,
                    bottom_left_center[1] - root_half,
                ),
                p0,
            ),
        ],
    )


def _tunnel_cutter(y_center: float, half_width: float, params: AeroParams) -> cq.Workplane:
    throat_x = THROAT_BASE_X_M + params.throat_offset_mm / 1000.0
    stations = [
        (-0.04, _tunnel_roof_height(0.18, params)),
        (0.48, _tunnel_roof_height(0.48, params)),
        (throat_x, _tunnel_roof_height(throat_x, params)),
        (1.28, _tunnel_roof_height(1.28, params)),
        (DIFFUSER_EXIT_X_M, _tunnel_roof_height(DIFFUSER_EXIT_X_M, params)),
        (2.04, _tunnel_roof_height(DIFFUSER_EXIT_X_M, params)),
    ]
    edge_radius_m = params.edge_radius_mm / 1000.0
    wires: list[cq.Wire] = []
    for x_pos, roof_z in stations:
        wires.append(
            _rounded_rect_wire(
                x_pos=x_pos,
                y_center=y_center,
                half_width=half_width,
                z_min=TUNNEL_FLOOR_Z_M,
                z_max=roof_z,
                radius=edge_radius_m,
            ),
        )
    return cq.Workplane("XY").add(cq.Solid.makeLoft(wires, ruled=True))


def _outer_station_heights(x_pos: float) -> tuple[float, float]:
    nose_tail_blend = max(
        np.clip(1.0 - x_pos / 0.28, 0.0, 1.0),
        np.clip((x_pos - 1.82) / 0.18, 0.0, 1.0),
    )
    side_top = (1.0 - nose_tail_blend) * OUTER_SIDE_TOP_Z_M + (
        nose_tail_blend * OUTER_END_SIDE_TOP_Z_M
    )
    center_top = (1.0 - nose_tail_blend) * OUTER_CENTER_CROWN_Z_M + (
        nose_tail_blend * OUTER_END_CENTER_CROWN_Z_M
    )
    return float(side_top), float(center_top)


def _upper_fairing_z(y_pos: float, *, side_top: float, center_top: float) -> float:
    span = min(abs(y_pos) / OUTER_HALF_WIDTH_M, 1.0)
    crown = np.cos(0.5 * np.pi * span) ** 2
    return float(side_top + (center_top - side_top) * crown)


def _outer_section_wire(x_pos: float) -> cq.Wire:
    side_top, center_top = _outer_station_heights(x_pos)
    edge_r = OUTER_EDGE_RADIUS_M
    y_inner = OUTER_HALF_WIDTH_M - edge_r

    upper_y = (-0.42, -0.30, -0.15, 0.0, 0.15, 0.30, 0.42)
    upper_points = [
        _vec(x_pos, y, _upper_fairing_z(y, side_top=side_top, center_top=center_top))
        for y in upper_y
    ]
    edge_theta = (np.pi / 6.0, np.pi / 3.0)
    right_edge_points = [
        _vec(
            x_pos,
            y_inner + edge_r * np.sin(theta),
            edge_r * (1.0 - np.cos(theta)),
        )
        for theta in edge_theta
    ]
    left_edge_points = [
        _vec(
            x_pos,
            -y_inner - edge_r * np.sin(theta),
            edge_r * (1.0 - np.cos(theta)),
        )
        for theta in reversed(edge_theta)
    ]

    p0 = _vec(x_pos, -y_inner, 0.0)
    p1 = _vec(x_pos, y_inner, 0.0)
    p2 = _vec(x_pos, OUTER_HALF_WIDTH_M, edge_r)
    p3 = _vec(x_pos, OUTER_HALF_WIDTH_M, side_top)
    p4 = _vec(x_pos, -OUTER_HALF_WIDTH_M, side_top)
    p5 = _vec(x_pos, -OUTER_HALF_WIDTH_M, edge_r)

    return _wire_from_edges(
        [
            _line(p0, p1),
            _line(p1, right_edge_points[0]),
            _line(right_edge_points[0], right_edge_points[1]),
            _line(right_edge_points[1], p2),
            _line(p2, p3),
            _line(p3, upper_points[-1]),
            _line(upper_points[-1], upper_points[-2]),
            _line(upper_points[-2], upper_points[-3]),
            _line(upper_points[-3], upper_points[-4]),
            _line(upper_points[-4], upper_points[-5]),
            _line(upper_points[-5], upper_points[-6]),
            _line(upper_points[-6], upper_points[-7]),
            _line(upper_points[-7], p4),
            _line(p4, p5),
            _line(p5, left_edge_points[0]),
            _line(left_edge_points[0], left_edge_points[1]),
            _line(left_edge_points[1], p0),
        ],
    )


def _outer_article_shell() -> cq.Workplane:
    station_x = (0.0, 0.10, 0.32, 0.80, 1.45, 1.82, 2.0)
    wires = [_outer_section_wire(x_pos) for x_pos in station_x]
    return cq.Workplane("XY").add(cq.Solid.makeLoft(wires, ruled=True))


def minimum_upper_ligament_m(params: AeroParams) -> float:
    """Return the minimum analytic solid ligament above the tunnel roof."""

    design_x = np.linspace(0.05, 1.95, 80)
    if params.geometry_family == "stable_reference":
        probe_y = np.linspace(
            -STABLE_REFERENCE_TUNNEL_HALF_WIDTH_M,
            STABLE_REFERENCE_TUNNEL_HALF_WIDTH_M,
            7,
        )
    else:
        probe_y = np.asarray(TUNNEL_CENTRES_Y_M, dtype=np.float64)
    values = []
    for x_pos in design_x:
        side_top, center_top = _outer_station_heights(float(x_pos))
        roof_z = _tunnel_roof_height(float(x_pos), params)
        for tunnel_y in probe_y:
            upper_z = _upper_fairing_z(float(tunnel_y), side_top=side_top, center_top=center_top)
            values.append(upper_z - roof_z)
    return float(np.min(values))


def build_article(params: AeroParams) -> cq.Workplane:
    """Build the untransformed body datum solid with CadQuery."""

    ligament = minimum_upper_ligament_m(params)
    if ligament < MIN_TUNNEL_SOLID_LIGAMENT_M:
        message = (
            f"upper fairing leaves only {ligament * 1000:g} mm over the tunnel roof; "
            f"minimum is {MIN_TUNNEL_SOLID_LIGAMENT_M * 1000:g} mm"
        )
        raise ValueError(message)

    article = _outer_article_shell()

    if params.geometry_family == "stable_reference":
        for y_center in STABLE_REFERENCE_TUNNEL_CENTRES_Y_M:
            article = article.cut(
                _tunnel_cutter(y_center, STABLE_REFERENCE_TUNNEL_HALF_WIDTH_M, params),
            )
        return article

    for y_center in TUNNEL_CENTRES_Y_M:
        article = article.cut(_tunnel_cutter(y_center, TUNNEL_HALF_WIDTH_M, params))

    return article


def _mesh_from_cadquery(article: cq.Workplane) -> trimesh.Trimesh:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "article.stl"
        cq.exporters.export(article, str(tmp_path), tolerance=0.001, angularTolerance=0.05)
        mesh = trimesh.load_mesh(tmp_path, process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def _write_preview(mesh: trimesh.Trimesh, glb_path: Path, html_path: Path) -> None:
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(glb_path, file_type="glb")

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color="#8fb7c9",
                opacity=1.0,
                flatshading=True,
            ),
        ],
    )
    fig.update_layout(
        scene={
            "aspectmode": "data",
            "xaxis_title": "x m",
            "yaxis_title": "y m",
            "zaxis_title": "z m",
        },
        margin={"l": 0, "r": 0, "t": 24, "b": 0},
        title="AeroCliff canonical Venturi-underfloor benchmark",
    )
    fig.write_html(html_path, include_plotlyjs="cdn")


def generate_geometry(params: AeroParams, output_dir: Path) -> GeometryArtifacts:
    """Generate transformed STL, previews, params, metrics, and validation JSON."""

    case_dir = output_dir / params.case_id()
    geometry_dir = case_dir / "geometry"
    geometry_dir.mkdir(parents=True, exist_ok=True)

    article = build_article(params)
    body_mesh = _mesh_from_cadquery(article)
    mesh = body_mesh.copy()
    mesh.vertices = apply_ride_height_pitch(
        np.asarray(mesh.vertices, dtype=np.float64),
        ride_height_mm=params.ride_height_mm,
        pitch_deg=params.pitch_deg,
    )
    mesh.fix_normals()

    step_path = geometry_dir / "article_body_datum.step"
    cq.exporters.export(article, str(step_path))

    stl_path = geometry_dir / "article.stl"
    mesh.export(stl_path, file_type="stl")

    validation = validate_mesh(mesh)
    metrics = validation.metrics
    design = tunnel_design_metadata(params)
    if metrics is None:
        metrics = GeometryMetrics(
            watertight=False,
            winding_consistent=False,
            body_count=0,
            vertex_count=len(mesh.vertices),
            face_count=len(mesh.faces),
            bounds_min_m=(
                float(mesh.bounds[0][0]),
                float(mesh.bounds[0][1]),
                float(mesh.bounds[0][2]),
            ),
            bounds_max_m=(
                float(mesh.bounds[1][0]),
                float(mesh.bounds[1][1]),
                float(mesh.bounds[1][2]),
            ),
            volume_m3=float(mesh.volume),
            surface_area_m2=float(mesh.area),
            min_ground_clearance_m=float(np.min(mesh.vertices[:, 2])),
            diffuser_region_x_m=(1.05, 1.92),
            throat_x_m=design.throat_x_m,
            left_tunnel_half_width_m=design.tunnel_half_width_m,
            right_tunnel_half_width_m=design.tunnel_half_width_m,
            diffuser_exit_roof_height_m=design.diffuser_exit_roof_height_m,
            generator_version=GEOMETRY_GENERATOR_VERSION,
        )
    else:
        metrics = metrics.model_copy(
            update={
                "throat_x_m": design.throat_x_m,
                "left_tunnel_half_width_m": design.tunnel_half_width_m,
                "right_tunnel_half_width_m": design.tunnel_half_width_m,
                "diffuser_exit_roof_height_m": design.diffuser_exit_roof_height_m,
            },
        )
    validation = validation.model_copy(update={"metrics": metrics})

    regions_json_path = geometry_dir / "surface_regions.json"
    regions_vtp_path = geometry_dir / "surface_regions.vtp"
    write_surface_regions(
        mesh,
        params,
        regions_json_path,
        regions_vtp_path,
        classification_mesh=body_mesh,
    )

    preview_glb_path = geometry_dir / "preview.glb"
    preview_html_path = geometry_dir / "preview.html"
    _write_preview(mesh, preview_glb_path, preview_html_path)

    params_path = case_dir / "params.json"
    params_yaml_path = case_dir / "params.yaml"
    metrics_path = geometry_dir / "geometry_metrics.json"
    hashes_path = geometry_dir / "hashes.json"
    validation_path = geometry_dir / "validation.json"
    atomic_write_json(params_path, params.model_dump())
    atomic_write_text(params_yaml_path, yaml.safe_dump(params.model_dump(), sort_keys=True))
    atomic_write_json(metrics_path, metrics.model_dump(mode="json"))
    atomic_write_json(validation_path, validation.model_dump(mode="json"))
    atomic_write_json(
        hashes_path,
        {
            "article_body_datum.step": sha256_file(step_path),
            "article.stl": sha256_file(stl_path),
            "surface_regions.json": sha256_file(regions_json_path),
            "surface_regions.vtp": sha256_file(regions_vtp_path),
            "params.json": sha256_file(params_path),
            "params.yaml": sha256_file(params_yaml_path),
            "geometry_metrics.json": sha256_file(metrics_path),
            "validation.json": sha256_file(validation_path),
        },
    )

    return GeometryArtifacts(
        case_id=params.case_id(),
        step_path=step_path,
        stl_path=stl_path,
        regions_json_path=regions_json_path,
        regions_vtp_path=regions_vtp_path,
        preview_glb_path=preview_glb_path,
        preview_html_path=preview_html_path,
        params_yaml_path=params_yaml_path,
        params_path=params_path,
        metrics_path=metrics_path,
        hashes_path=hashes_path,
        validation_path=validation_path,
        validation=validation,
    )
