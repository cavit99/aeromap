"""CFD schemas."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PatchLayerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    patch: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    n_surface_layers: int = Field(ge=0)


class RefinementBox(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    level: int = Field(ge=0)

    @model_validator(mode="after")
    def _bounds_are_ordered(self) -> Self:
        if any(lo >= hi for lo, hi in zip(self.bounds_min, self.bounds_max, strict=True)):
            message = "refinement box bounds_min entries must be below bounds_max entries"
            raise ValueError(message)
        return self


class SpanRefinement(BaseModel):
    model_config = ConfigDict(frozen=True)

    surface: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    level: int = Field(ge=0)
    cells_across_span: int = Field(gt=0)
    distance_m: float = Field(default=1000.0, gt=0.0)


class MeshConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    block_cells: tuple[int, int, int] = (42, 16, 16)
    max_global_cells: int = Field(default=300_000, gt=0)
    surface_level: tuple[int, int] = (1, 2)
    feature_angle_deg: float = Field(default=30.0, gt=0.0, le=180.0)
    implicit_feature_snap: bool = False
    explicit_feature_snap: bool = True
    snap_solve_iterations: int = Field(default=30, gt=0)
    add_layers: bool = True
    n_surface_layers: int = Field(default=3, gt=0)
    layer_relative_sizes: bool = True
    layer_expansion_ratio: float = Field(default=1.2, ge=1.0)
    first_layer_thickness: float | None = Field(default=None, gt=0.0)
    final_layer_thickness: float = Field(default=0.35, gt=0.0)
    min_layer_thickness: float = Field(default=0.10, gt=0.0)
    max_thickness_to_medial_ratio: float = Field(default=0.3, gt=0.0)
    layer_feature_angle_deg: float = Field(default=60.0, gt=0.0, le=180.0)
    layer_slip_feature_angle_deg: float | None = Field(default=None, gt=0.0, le=180.0)
    layer_n_relaxed_iter: int | None = Field(default=None, ge=0)
    layer_n_medial_axis_iter: int | None = Field(default=None, ge=0)
    layer_additional_reporting: bool = False
    layer_n_grow: int = Field(default=0, ge=0)
    layer_n_buffer_cells_no_extrude: int = Field(default=0, ge=0)
    patch_layers: tuple[PatchLayerConfig, ...] = ()
    n_cells_between_levels: int = Field(default=2, gt=0)
    refinement_boxes: tuple[RefinementBox, ...] = ()
    span_refinements: tuple[SpanRefinement, ...] = ()

    @field_validator("block_cells")
    @classmethod
    def _block_cells_positive(cls, value: tuple[int, int, int]) -> tuple[int, int, int]:
        if any(cell <= 0 for cell in value):
            message = "block_cells entries must be positive"
            raise ValueError(message)
        return value

    @field_validator("surface_level")
    @classmethod
    def _surface_level_ordered(cls, value: tuple[int, int]) -> tuple[int, int]:
        level_min, level_max = value
        if level_min < 0 or level_max < 0:
            message = "surface_level entries must be non-negative"
            raise ValueError(message)
        if level_min > level_max:
            message = "surface_level minimum must be <= maximum"
            raise ValueError(message)
        return value


class SurfaceExportConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: Literal["cadquery_current", "gmsh_occ_g0_no_healing"] = "cadquery_current"
    openfoam_patch_mode: Literal[
        "single_patch",
        "gate2b_core_transition",
        "critical_underfloor",
    ] = "single_patch"
    transition_band_width_m: float = Field(default=0.024, gt=0.0)
    gmsh_path: Path | None = None
    mesh_size_min_m: float = Field(default=0.001, gt=0.0)
    mesh_size_max_m: float = Field(default=0.012, gt=0.0)
    mesh_algorithm: Literal["front2d"] = "front2d"
    mesh_optimize: int = Field(default=1, ge=0, le=1)

    @model_validator(mode="after")
    def _mesh_sizes_are_ordered(self) -> Self:
        if self.mesh_size_min_m > self.mesh_size_max_m:
            message = "mesh_size_min_m must be <= mesh_size_max_m"
            raise ValueError(message)
        return self


class SolverConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_iterations: int = Field(default=250, gt=0)
    write_interval: int = Field(default=50, gt=0)
    force_window: int = Field(default=50, gt=0)


class QualityConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_class: Literal["NON_CAMPAIGN_ENGINEERING_SMOKE", "CAMPAIGN_REFERENCE_CFD"] = (
        "CAMPAIGN_REFERENCE_CFD"
    )
    mesh_quality_fatal: bool = True
    extended_diagnostics_required: bool = True
    extended_diagnostics_fatal: bool = False
    region_mapping_min_coverage: float = Field(default=0.995, gt=0.0, le=1.0)
    region_mapping_max_distance_face_scale: float = Field(default=1.0, gt=0.0)


class CfdConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    profile: str = "smoke"
    mesh: MeshConfig = MeshConfig()
    surface_export: SurfaceExportConfig = SurfaceExportConfig()
    solver: SolverConfig = SolverConfig()
    quality: QualityConfig = QualityConfig()


class CfdCaseArtifacts(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    simulation_id: str
    case_dir: Path
    openfoam_dir: Path
    stl_path: Path
    manifest_path: Path
    run_script_path: Path
