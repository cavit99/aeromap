"""Typed CFD data sample schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

SAMPLE_SCHEMA_VERSION = "aerocliff_sample_v0.2.0"
CONVERTER_VERSION = "gate3_converter_v0.2.0"


class VtkWorkflowManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "aerocliff_vtp_vtu_workflow_v0.1.0"
    geometry_format: Literal["STL"] = "STL"
    surface_format: Literal["VTP"] = "VTP"
    volume_format: Literal["VTU"] = "VTU"
    geometry_adapter: str = "aeromap.data.vtk_workflow.load_geometry_stl"
    surface_adapter: str
    volume_adapter: str
    required_volume_cell_id_array: str = "cellID"
    geometry_path: str | None = None
    surface_path: str
    volume_path: str
    semantics: dict[str, str]


class FoamToVtkDecompositionReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_polyhedra_decomposed: int = Field(ge=0)
    child_tetrahedra: int = Field(ge=0)
    child_pyramids: int = Field(ge=0)
    exported_child_cells: int = Field(ge=0)
    net_exported_cell_increase: int


class DuplicatedChildFieldReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    association: Literal["cell"] = "cell"
    duplicated_source_cells_checked: int = Field(ge=0)
    max_abs_spread: float = Field(ge=0.0)
    tolerance: float = Field(ge=0.0)
    passed: bool

    @model_validator(mode="after")
    def _must_pass(self) -> Self:
        if not self.passed:
            msg = "duplicated child field validation must pass"
            raise ValueError(msg)
        return self


class VolumeCellProvenance(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "aerocliff_volume_cell_provenance_v0.1.0"
    source_openfoam_cell_count: int = Field(gt=0)
    exported_vtu_cell_count: int = Field(gt=0)
    cellid_array_name: str = "cellID"
    cellid_count: int = Field(gt=0)
    cellid_unique_source_count: int = Field(gt=0)
    cellid_missing_source_count: int = Field(ge=0)
    cellid_min: int = Field(ge=0)
    cellid_max: int = Field(ge=0)
    cellid_maps_all_exported_cells: bool
    cellid_covers_all_source_cells: bool
    duplicated_source_cell_count: int = Field(ge=0)
    duplicated_exported_child_cell_count: int = Field(ge=0)
    foam_to_vtk_decomposition: FoamToVtkDecompositionReport
    duplicated_child_field_validation: dict[str, DuplicatedChildFieldReport]
    exported_cell_handling: Literal["preserved_exported_vtu_cells"] = "preserved_exported_vtu_cells"
    source_reduction_semantics: str

    @model_validator(mode="after")
    def _cellid_is_complete(self) -> Self:
        if self.cellid_count != self.exported_vtu_cell_count:
            msg = "cellID count must match exported VTU cell count"
            raise ValueError(msg)
        if self.cellid_unique_source_count != self.source_openfoam_cell_count:
            msg = "unique cellID count must match source OpenFOAM cell count"
            raise ValueError(msg)
        if self.cellid_missing_source_count != 0:
            msg = "cellID must cover every source OpenFOAM cell"
            raise ValueError(msg)
        if not self.cellid_maps_all_exported_cells or not self.cellid_covers_all_source_cells:
            msg = "cellID mapping must cover all exported and source cells"
            raise ValueError(msg)
        if self.cellid_max >= self.source_openfoam_cell_count:
            msg = "cellID values must be within the source OpenFOAM cell count"
            raise ValueError(msg)
        return self


class FieldValidationCheck(BaseModel):
    model_config = ConfigDict(frozen=True)

    equation: str
    dimensional_array: str
    nondimensional_array: str
    max_abs_error: float = Field(ge=0.0)
    tolerance: float = Field(ge=0.0)
    passed: bool

    @model_validator(mode="after")
    def _must_pass(self) -> Self:
        if not self.passed:
            msg = f"field validation failed for {self.nondimensional_array}"
            raise ValueError(msg)
        return self


class FieldValidationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "aerocliff_field_validation_v0.1.0"
    checks: dict[str, FieldValidationCheck]

    @model_validator(mode="after")
    def _requires_checks(self) -> Self:
        if not self.checks:
            msg = "field validation report must contain at least one check"
            raise ValueError(msg)
        return self


class DataSampleArtifacts(BaseModel):
    model_config = ConfigDict(frozen=True)

    sample_id: str
    sample_dir: Path
    manifest_path: Path
    arrays_path: Path


class DataSampleManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = SAMPLE_SCHEMA_VERSION
    converter_version: str = CONVERTER_VERSION
    sample_id: str
    geometry_id: str
    state_id: str
    simulation_id: str
    attempt_id: str
    case_class: Literal["NON_CAMPAIGN_ENGINEERING_SMOKE", "CAMPAIGN_REFERENCE_CFD"]
    training_eligible: bool
    source_case_dir: str
    arrays_path: str
    arrays_sha256: str
    reference: dict[str, float]
    counts: dict[str, int]
    array_names: list[str]
    vtk_workflow: VtkWorkflowManifest
    volume_provenance: VolumeCellProvenance
    field_validation: FieldValidationReport
    loads: dict[str, Any]
    quality: dict[str, Any]
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _training_eligibility_matches_case_class(self) -> Self:
        if self.training_eligible and self.case_class != "CAMPAIGN_REFERENCE_CFD":
            msg = "only CAMPAIGN_REFERENCE_CFD samples can be training eligible"
            raise ValueError(msg)
        return self


class DataSample(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    manifest: DataSampleManifest
    arrays: dict[str, Any]
