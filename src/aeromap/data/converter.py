"""Convert post-processed OpenFOAM cases into immutable AeroCliff samples."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pyvista as pv

from aeromap.attempts import stable_id
from aeromap.constants import REF
from aeromap.data.schema import (
    CONVERTER_VERSION,
    SAMPLE_SCHEMA_VERSION,
    DataSampleArtifacts,
    DataSampleManifest,
    DuplicatedChildFieldReport,
    FieldValidationCheck,
    FieldValidationReport,
    FoamToVtkDecompositionReport,
    VolumeCellProvenance,
)
from aeromap.data.volume import duplicate_group_max_spread
from aeromap.data.vtk_workflow import load_volume_vtu, load_wall_vtp, workflow_manifest
from aeromap.io import atomic_write_json, sha256_file
from aeromap.transforms import (
    nondim_coords,
    nondim_velocity,
    nondim_wall_shear,
    pressure_coefficient,
)

ACCEPTED_SMOKE_SIMULATION_ID = "simulation_46a8d8375bde7c78"
ACCEPTED_SMOKE_SOURCE_OPENFOAM_CELLS = 21_536
ACCEPTED_SMOKE_EXPORTED_VTU_CELLS = 23_886
ACCEPTED_SMOKE_DECOMPOSED_POLYHEDRA = 168
ACCEPTED_SMOKE_DECOMPOSED_TETRAHEDRA = 1_688
ACCEPTED_SMOKE_DECOMPOSED_PYRAMIDS = 830
CELL_ID_ARRAY = "cellID"
DUPLICATED_CHILD_FIELDS = ("p", "U", "k", "omega", "nut")
FIELD_VALIDATION_TOLERANCE = 1.0e-12
DUPLICATED_FIELD_TOLERANCE = 1.0e-12


def _load_json(path: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))


def _resolve_case_path(case_dir: Path, value: str | None) -> Path:
    if value is None:
        msg = "required artifact path is missing"
        raise ValueError(msg)
    path = Path(value)
    if path.is_absolute() and path.exists():
        return path
    candidate = case_dir / path
    if candidate.exists():
        return candidate
    if case_dir.name in path.parts:
        case_index = path.parts.index(case_dir.name)
        case_relative = Path(*path.parts[case_index + 1 :])
        candidate = case_dir / case_relative
        if candidate.exists():
            return candidate
    parent_candidate = case_dir.parent / path
    if parent_candidate.exists():
        return parent_candidate
    msg = f"case artifact does not exist: {value}"
    raise FileNotFoundError(msg)


def _cell_array(dataset: pv.DataSet, name: str) -> np.ndarray:
    if name not in dataset.cell_data:
        msg = f"missing required cell array {name!r}"
        raise ValueError(msg)
    return np.asarray(dataset.cell_data[name])


def _field_array(dataset: pv.DataSet, name: str) -> tuple[np.ndarray, str]:
    if name in dataset.point_data:
        return np.asarray(dataset.point_data[name]), "point"
    if name in dataset.cell_data:
        return np.asarray(dataset.cell_data[name]), "cell"
    msg = f"missing required field array {name!r}"
    raise ValueError(msg)


def _cell_normals(poly: pv.PolyData) -> np.ndarray:
    with_normals = poly.compute_normals(
        cell_normals=True,
        point_normals=False,
        auto_orient_normals=False,
        consistent_normals=False,
    )
    return np.asarray(with_normals.cell_data["Normals"], dtype=np.float64)


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.",
        suffix=".npz",
        dir=path.parent,
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
    try:
        payload = cast("Any", arrays)
        np.savez_compressed(tmp_path, **payload)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


CaseClass = Literal["NON_CAMPAIGN_ENGINEERING_SMOKE", "CAMPAIGN_REFERENCE_CFD"]


def _status_case_class(status: dict[str, Any], manifest: dict[str, Any]) -> CaseClass:
    accepted_scope = status.get("accepted_scope")
    if accepted_scope in ("NON_CAMPAIGN_ENGINEERING_SMOKE", "CAMPAIGN_REFERENCE_CFD"):
        return cast("CaseClass", accepted_scope)
    quality = manifest.get("cfd_config", {}).get("quality", {})
    case_class = quality.get("case_class")
    if case_class in ("NON_CAMPAIGN_ENGINEERING_SMOKE", "CAMPAIGN_REFERENCE_CFD"):
        return cast("CaseClass", case_class)
    msg = "case class is missing from status and manifest"
    raise ValueError(msg)


def _source_openfoam_cell_count(mesh_quality: dict[str, Any]) -> int:
    count = mesh_quality.get("cells")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        msg = "quality mesh.json must record positive source OpenFOAM cell count in 'cells'"
        raise ValueError(msg)
    return count


def _volume_cell_ids(volume: pv.UnstructuredGrid) -> np.ndarray:
    if CELL_ID_ARRAY not in volume.cell_data:
        msg = "volume VTU must carry foamToVTK cellID cell data"
        raise ValueError(msg)
    raw = np.asarray(volume.cell_data[CELL_ID_ARRAY]).reshape(-1)
    if raw.shape[0] != volume.n_cells:
        msg = f"cellID length {raw.shape[0]} does not match exported VTU cells {volume.n_cells}"
        raise ValueError(msg)
    if np.issubdtype(raw.dtype, np.integer):
        return raw.astype(np.int64)
    as_float = raw.astype(np.float64)
    if not np.all(np.isfinite(as_float)) or not np.allclose(as_float, np.rint(as_float)):
        msg = "cellID values must be finite integers"
        raise ValueError(msg)
    return np.rint(as_float).astype(np.int64)


def _duplicated_field_report(
    *,
    volume: pv.UnstructuredGrid,
    cell_ids: np.ndarray,
    field_name: str,
    duplicated_source_cell_count: int,
) -> DuplicatedChildFieldReport:
    values = np.asarray(volume.cell_data[field_name])
    if values.shape[0] != volume.n_cells:
        msg = f"duplicated child field {field_name!r} length does not match VTU cells"
        raise ValueError(msg)
    spread = duplicate_group_max_spread(values, cell_ids)
    return DuplicatedChildFieldReport(
        duplicated_source_cells_checked=duplicated_source_cell_count,
        max_abs_spread=spread,
        tolerance=DUPLICATED_FIELD_TOLERANCE,
        passed=spread <= DUPLICATED_FIELD_TOLERANCE,
    )


def _duplicated_child_field_validation(
    volume: pv.UnstructuredGrid,
    cell_ids: np.ndarray,
    duplicated_source_cell_count: int,
    simulation_id: str,
) -> dict[str, DuplicatedChildFieldReport]:
    if duplicated_source_cell_count == 0:
        return {}
    required = {"p", "U"}
    if simulation_id == ACCEPTED_SMOKE_SIMULATION_ID:
        required.update(DUPLICATED_CHILD_FIELDS)
    missing_required = sorted(field for field in required if field not in volume.cell_data)
    if missing_required:
        msg = (
            "duplicated exported child cells require cell-data field validation; "
            f"missing {missing_required}"
        )
        raise ValueError(msg)
    reports: dict[str, DuplicatedChildFieldReport] = {}
    for field_name in DUPLICATED_CHILD_FIELDS:
        if field_name in volume.cell_data:
            reports[field_name] = _duplicated_field_report(
                volume=volume,
                cell_ids=cell_ids,
                field_name=field_name,
                duplicated_source_cell_count=duplicated_source_cell_count,
            )
    return reports


def _volume_cell_provenance(
    volume: pv.UnstructuredGrid,
    *,
    source_openfoam_cell_count: int,
    simulation_id: str,
) -> VolumeCellProvenance:
    cell_ids = _volume_cell_ids(volume)
    exported_count = int(volume.n_cells)
    if np.any(cell_ids < 0) or np.any(cell_ids >= source_openfoam_cell_count):
        msg = "cellID values must map to source OpenFOAM cell IDs"
        raise ValueError(msg)
    unique_cell_ids = np.unique(cell_ids)
    expected_ids = np.arange(source_openfoam_cell_count, dtype=np.int64)
    missing_ids = np.setdiff1d(expected_ids, unique_cell_ids, assume_unique=True)
    child_counts = np.bincount(cell_ids, minlength=source_openfoam_cell_count)
    duplicated_source_mask = child_counts > 1
    duplicated_source_count = int(np.count_nonzero(duplicated_source_mask))
    duplicated_child_mask = duplicated_source_mask[cell_ids]
    duplicated_child_count = int(np.count_nonzero(duplicated_child_mask))
    duplicated_child_types = np.asarray(volume.celltypes, dtype=np.uint8)[duplicated_child_mask]
    tetrahedra = int(np.count_nonzero(duplicated_child_types == pv.CellType.TETRA))
    pyramids = int(np.count_nonzero(duplicated_child_types == pv.CellType.PYRAMID))

    provenance = VolumeCellProvenance(
        source_openfoam_cell_count=source_openfoam_cell_count,
        exported_vtu_cell_count=exported_count,
        cellid_count=int(cell_ids.shape[0]),
        cellid_unique_source_count=int(unique_cell_ids.shape[0]),
        cellid_missing_source_count=int(missing_ids.shape[0]),
        cellid_min=int(np.min(cell_ids)),
        cellid_max=int(np.max(cell_ids)),
        cellid_maps_all_exported_cells=bool(cell_ids.shape[0] == exported_count),
        cellid_covers_all_source_cells=bool(missing_ids.shape[0] == 0),
        duplicated_source_cell_count=duplicated_source_count,
        duplicated_exported_child_cell_count=duplicated_child_count,
        foam_to_vtk_decomposition=FoamToVtkDecompositionReport(
            source_polyhedra_decomposed=duplicated_source_count,
            child_tetrahedra=tetrahedra,
            child_pyramids=pyramids,
            exported_child_cells=duplicated_child_count,
            net_exported_cell_increase=exported_count - source_openfoam_cell_count,
        ),
        duplicated_child_field_validation=_duplicated_child_field_validation(
            volume,
            cell_ids,
            duplicated_source_count,
            simulation_id,
        ),
        source_reduction_semantics=(
            "Raw VTU cells are preserved in arrays. Any volume reduction over exported "
            "cell fields must aggregate by volume_cell_id/cellID to source OpenFOAM cells; "
            "raw exported VTU cell count is not the source mesh cell count."
        ),
    )
    _validate_accepted_smoke_provenance(simulation_id, provenance)
    return provenance


def _validate_accepted_smoke_provenance(
    simulation_id: str, provenance: VolumeCellProvenance
) -> None:
    if simulation_id != ACCEPTED_SMOKE_SIMULATION_ID:
        return
    expected = {
        "source_openfoam_cell_count": ACCEPTED_SMOKE_SOURCE_OPENFOAM_CELLS,
        "exported_vtu_cell_count": ACCEPTED_SMOKE_EXPORTED_VTU_CELLS,
        "source_polyhedra_decomposed": ACCEPTED_SMOKE_DECOMPOSED_POLYHEDRA,
        "child_tetrahedra": ACCEPTED_SMOKE_DECOMPOSED_TETRAHEDRA,
        "child_pyramids": ACCEPTED_SMOKE_DECOMPOSED_PYRAMIDS,
    }
    actual = {
        "source_openfoam_cell_count": provenance.source_openfoam_cell_count,
        "exported_vtu_cell_count": provenance.exported_vtu_cell_count,
        "source_polyhedra_decomposed": (
            provenance.foam_to_vtk_decomposition.source_polyhedra_decomposed
        ),
        "child_tetrahedra": provenance.foam_to_vtk_decomposition.child_tetrahedra,
        "child_pyramids": provenance.foam_to_vtk_decomposition.child_pyramids,
    }
    mismatches = [
        f"{key}: expected {expected[key]}, got {actual[key]}"
        for key in expected
        if actual[key] != expected[key]
    ]
    missing_fields = [
        field
        for field in DUPLICATED_CHILD_FIELDS
        if field not in provenance.duplicated_child_field_validation
    ]
    if missing_fields:
        mismatches.append(f"missing duplicated child field reports: {missing_fields}")
    if mismatches:
        msg = "accepted smoke volume provenance mismatch: " + "; ".join(mismatches)
        raise ValueError(msg)


def _validation_check(
    *,
    equation: str,
    dimensional_array: str,
    nondimensional_array: str,
    actual: np.ndarray,
    expected: np.ndarray,
) -> FieldValidationCheck:
    error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected)))) if actual.size else 0.0
    return FieldValidationCheck(
        equation=equation,
        dimensional_array=dimensional_array,
        nondimensional_array=nondimensional_array,
        max_abs_error=error,
        tolerance=FIELD_VALIDATION_TOLERANCE,
        passed=error <= FIELD_VALIDATION_TOLERANCE,
    )


def _field_validation_report(arrays: dict[str, np.ndarray]) -> FieldValidationReport:
    checks = {
        "surface_points_nd": _validation_check(
            equation="surface_points_m / L_ref",
            dimensional_array="surface_points_m",
            nondimensional_array="surface_points_nd",
            actual=arrays["surface_points_nd"],
            expected=nondim_coords(arrays["surface_points_m"]),
        ),
        "surface_cp": _validation_check(
            equation="rho * surface_pressure_kinematic / q_inf",
            dimensional_array="surface_pressure_kinematic",
            nondimensional_array="surface_cp",
            actual=arrays["surface_cp"],
            expected=pressure_coefficient(REF.rho_kg_m3 * arrays["surface_pressure_kinematic"]),
        ),
        "surface_cf": _validation_check(
            equation="rho * surface_wall_shear_kinematic / q_inf",
            dimensional_array="surface_wall_shear_kinematic",
            nondimensional_array="surface_cf",
            actual=arrays["surface_cf"],
            expected=nondim_wall_shear(REF.rho_kg_m3 * arrays["surface_wall_shear_kinematic"]),
        ),
        "volume_points_nd": _validation_check(
            equation="volume_points_m / L_ref",
            dimensional_array="volume_points_m",
            nondimensional_array="volume_points_nd",
            actual=arrays["volume_points_nd"],
            expected=nondim_coords(arrays["volume_points_m"]),
        ),
        "volume_cp": _validation_check(
            equation="rho * volume_pressure_kinematic / q_inf",
            dimensional_array="volume_pressure_kinematic",
            nondimensional_array="volume_cp",
            actual=arrays["volume_cp"],
            expected=pressure_coefficient(REF.rho_kg_m3 * arrays["volume_pressure_kinematic"]),
        ),
        "volume_velocity_nd": _validation_check(
            equation="volume_velocity / U_inf",
            dimensional_array="volume_velocity",
            nondimensional_array="volume_velocity_nd",
            actual=arrays["volume_velocity_nd"],
            expected=nondim_velocity(arrays["volume_velocity"]),
        ),
    }
    return FieldValidationReport(checks=checks)


def _arrays_from_exports(
    surface: pv.PolyData, volume: pv.UnstructuredGrid
) -> dict[str, np.ndarray]:
    surface_pressure_kinematic = _cell_array(surface, "p").astype(np.float64).reshape(-1)
    surface_shear_kinematic = _cell_array(surface, "wallShearStress").astype(np.float64)
    surface_region_ids = _cell_array(surface, "surface_region_id").astype(np.int32).reshape(-1)
    surface_region_names = np.asarray(_cell_array(surface, "surface_region")).astype(str)
    surface_pressure_pa = REF.rho_kg_m3 * surface_pressure_kinematic
    surface_shear_pa = REF.rho_kg_m3 * surface_shear_kinematic

    volume_pressure, pressure_association = _field_array(volume, "p")
    volume_velocity, velocity_association = _field_array(volume, "U")
    volume_pressure_pa = REF.rho_kg_m3 * np.asarray(volume_pressure, dtype=np.float64).reshape(-1)
    volume_velocity = np.asarray(volume_velocity, dtype=np.float64)

    return {
        "surface_points_m": np.asarray(surface.points, dtype=np.float64),
        "surface_points_nd": nondim_coords(np.asarray(surface.points, dtype=np.float64)),
        "surface_faces": np.asarray(surface.faces, dtype=np.int64),
        "surface_cell_normals": _cell_normals(surface),
        "surface_pressure_kinematic": surface_pressure_kinematic,
        "surface_cp": pressure_coefficient(surface_pressure_pa),
        "surface_wall_shear_kinematic": surface_shear_kinematic,
        "surface_cf": nondim_wall_shear(surface_shear_pa),
        "surface_region_id": surface_region_ids,
        "surface_region_name": surface_region_names,
        "surface_local_face_area_m2": _cell_array(surface, "local_face_area_m2").astype(
            np.float64,
        ),
        "volume_points_m": np.asarray(volume.points, dtype=np.float64),
        "volume_points_nd": nondim_coords(np.asarray(volume.points, dtype=np.float64)),
        "volume_cells": np.asarray(volume.cells, dtype=np.int64),
        "volume_celltypes": np.asarray(volume.celltypes, dtype=np.uint8),
        "volume_cell_id": _volume_cell_ids(volume),
        "volume_pressure_kinematic": np.asarray(volume_pressure, dtype=np.float64).reshape(-1),
        "volume_cp": pressure_coefficient(volume_pressure_pa),
        "volume_velocity": volume_velocity,
        "volume_velocity_nd": nondim_velocity(volume_velocity),
        "volume_pressure_association": np.asarray([pressure_association]),
        "volume_velocity_association": np.asarray([velocity_association]),
    }


def convert_case_to_sample(case_dir: Path, out_dir: Path) -> DataSampleArtifacts:
    """Convert one post-processed CFD case to a device-neutral data sample."""

    case_dir = case_dir.resolve()
    manifest = _load_json(case_dir / "manifest.json")
    status = _load_json(case_dir / "quality" / "status.json")
    artifacts = status.get("artifacts", {})
    if not isinstance(artifacts, dict):
        msg = "status.json artifacts must be a mapping"
        raise TypeError(msg)

    surface_path = _resolve_case_path(case_dir, artifacts.get("mapped_wall_vtp"))
    volume_path = _resolve_case_path(case_dir, artifacts.get("volume_vtu"))
    geometry_path: Path | None = None
    cfd_surface = manifest.get("cfd_surface", {})
    if isinstance(cfd_surface, dict) and isinstance(cfd_surface.get("stl_path"), str):
        geometry_path = _resolve_case_path(case_dir, cfd_surface["stl_path"])
    convergence = _load_json(_resolve_case_path(case_dir, artifacts.get("convergence")))
    force_integration = _load_json(_resolve_case_path(case_dir, artifacts.get("force_integration")))
    mesh_quality = _load_json(_resolve_case_path(case_dir, artifacts.get("mesh")))
    region_mapping = _load_json(case_dir / "quality" / "region_mapping.json")

    case_class = _status_case_class(status, manifest)
    accepted = status.get("accepted")
    if not isinstance(accepted, bool):
        msg = "status.json accepted must be a boolean"
        raise TypeError(msg)
    training_eligible = accepted and case_class == "CAMPAIGN_REFERENCE_CFD"
    sample_id = stable_id(
        "sample",
        {
            "schema_version": SAMPLE_SCHEMA_VERSION,
            "converter_version": CONVERTER_VERSION,
            "simulation_id": manifest["simulation_id"],
            "case_class": case_class,
        },
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = out_dir / sample_id
    arrays_path = sample_dir / "arrays.npz"
    manifest_path = sample_dir / "manifest.json"
    if sample_dir.exists():
        msg = (
            f"sample {sample_id} already exists at {sample_dir}; "
            "refusing to overwrite immutable sample"
        )
        raise FileExistsError(msg)

    surface = load_wall_vtp(surface_path)
    volume = load_volume_vtu(volume_path)
    source_openfoam_cell_count = _source_openfoam_cell_count(mesh_quality)
    volume_provenance = _volume_cell_provenance(
        volume,
        source_openfoam_cell_count=source_openfoam_cell_count,
        simulation_id=str(manifest["simulation_id"]),
    )
    arrays = _arrays_from_exports(surface, volume)
    field_validation = _field_validation_report(arrays)

    staged_dir = Path(tempfile.mkdtemp(prefix=f".{sample_id}.", dir=out_dir))
    staged_arrays_path = staged_dir / arrays_path.name
    staged_manifest_path = staged_dir / manifest_path.name
    try:
        _write_npz_atomic(staged_arrays_path, arrays)

        sample_manifest = DataSampleManifest(
            sample_id=sample_id,
            geometry_id=manifest["geometry_id"],
            state_id=manifest["state_id"],
            simulation_id=manifest["simulation_id"],
            attempt_id=str(manifest.get("attempt_id", manifest["case_id"])),
            case_class=case_class,
            training_eligible=training_eligible,
            source_case_dir=str(case_dir),
            arrays_path=str(arrays_path.relative_to(sample_dir)),
            arrays_sha256=sha256_file(staged_arrays_path),
            reference={
                "l_ref_m": REF.l_ref_m,
                "a_ref_m2": REF.a_ref_m2,
                "u_inf_m_s": REF.u_inf_m_s,
                "rho_kg_m3": REF.rho_kg_m3,
                "q_inf_pa": REF.q_inf_pa,
                "p_inf_pa": REF.p_inf_pa,
            },
            counts={
                "surface_points": int(arrays["surface_points_m"].shape[0]),
                "surface_faces": int(arrays["surface_pressure_kinematic"].shape[0]),
                "volume_points": int(arrays["volume_points_m"].shape[0]),
                "volume_cells": int(arrays["volume_celltypes"].shape[0]),
                "volume_exported_vtu_cells": volume_provenance.exported_vtu_cell_count,
                "volume_source_openfoam_cells": volume_provenance.source_openfoam_cell_count,
                "volume_unique_cell_ids": volume_provenance.cellid_unique_source_count,
            },
            array_names=sorted(arrays),
            vtk_workflow=workflow_manifest(surface_path, volume_path, geometry_path=geometry_path),
            volume_provenance=volume_provenance,
            field_validation=field_validation,
            loads={
                "convergence": convergence,
                "force_integration": force_integration,
            },
            quality={
                "status": status,
                "mesh": mesh_quality,
                "region_mapping": region_mapping,
            },
            provenance={
                "git_sha": manifest.get("git_sha", "unknown"),
                "case_id": manifest.get("case_id"),
                "surface_export_id": manifest.get("surface_export_id"),
                "cfd_config": manifest.get("cfd_config"),
            },
        )
        atomic_write_json(staged_manifest_path, sample_manifest.model_dump(mode="json"))
        staged_dir.replace(sample_dir)
    finally:
        if staged_dir.exists():
            shutil.rmtree(staged_dir)
    return DataSampleArtifacts(
        sample_id=sample_id,
        sample_dir=sample_dir,
        manifest_path=manifest_path,
        arrays_path=arrays_path,
    )
