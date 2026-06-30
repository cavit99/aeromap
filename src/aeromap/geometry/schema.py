"""Geometry schemas."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class GeometryMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    watertight: bool
    winding_consistent: bool
    body_count: int
    vertex_count: int
    face_count: int
    bounds_min_m: tuple[float, float, float]
    bounds_max_m: tuple[float, float, float]
    volume_m3: float
    surface_area_m2: float
    min_ground_clearance_m: float
    diffuser_region_x_m: tuple[float, float]
    throat_x_m: float | None = None
    left_tunnel_half_width_m: float | None = None
    right_tunnel_half_width_m: float | None = None
    diffuser_exit_roof_height_m: float | None = None
    generator_version: str


class GeometryValidation(BaseModel):
    model_config = ConfigDict(frozen=True)

    valid: bool
    reasons: tuple[str, ...]
    metrics: GeometryMetrics | None = None


class GeometryArtifacts(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    step_path: Path
    stl_path: Path
    regions_json_path: Path
    regions_vtp_path: Path
    preview_glb_path: Path
    preview_html_path: Path
    params_yaml_path: Path
    params_path: Path
    metrics_path: Path
    hashes_path: Path
    validation_path: Path
    validation: GeometryValidation
